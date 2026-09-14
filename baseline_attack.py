
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import load_npz_data, ROC_AUC_Result_logshow,path_exists,ROC_AUC_Result_logshow_with_auc,custom_collate
import random
from torch.utils.data import DataLoader, TensorDataset
from Data import ClientDataset,ClientDatasetWithMember
import os
import utils
from utils import load_npz_data
from CSModels import PublicLayer,PrivateLayer
from tqdm import tqdm
import matplotlib.pyplot as plt
from copy import copy
import scipy
from sklearn import metrics
import json
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score, precision_score, recall_score
from scipy import stats


def _build_public_model(args):
    model = PublicLayer(args)
    return model




class OutputComponent(nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=64, dropout=0.2):
        super(OutputComponent, self).__init__()
        self.FC = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
            nn.ReLU(),
            nn.Dropout(p=dropout))

    def forward(self, x):
        x = self.FC(x)
        return x


class LabelComponent(nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=64, dropout=0.2):
        super(LabelComponent, self).__init__()
        self.FC = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
            nn.ReLU(),
            nn.Dropout(p=dropout))

    def forward(self, x):
        x = self.FC(x)
        return x


class LossComponent(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, output_size=64, dropout=0.2):
        super(LossComponent, self).__init__()
        self.FC = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.FC(x)
        return x


class GradientComponent(nn.Module):
    def __init__(self, input_channels, next_layer_size, output_size,dropout=0.2):
        super(GradientComponent, self).__init__()
        
        kernels = 4
        
        self.conv = nn.Conv2d(input_channels, kernels, (1, next_layer_size), stride=1)
        
        self.relu1 = nn.ReLU()
        
        self.dropout1 = nn.Dropout(p=dropout)
        
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(kernels * next_layer_size * 55, 128)
        
        self.relu2 = nn.ReLU()
        
        self.dropout2 = nn.Dropout(p=dropout)
        
        self.fc2 = nn.Linear(128, 64)
        
        self.relu3 = nn.ReLU()
        
        self.dropout3 = nn.Dropout(p=dropout)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.flatten(x)
        
        x = self.fc1(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        x = self.fc2(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        
        return x



class EncoderComponent(nn.Module):
    def __init__(self, input_size, hidden_size1=256, hidden_size2=128, hidden_size3=64, output_size=2, dropout=0.2):
        super(EncoderComponent, self).__init__()
        self.FC = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=hidden_size1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=hidden_size1, out_features=hidden_size2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=hidden_size2, out_features=hidden_size3),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=hidden_size3, out_features=output_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.FC(x)
        return x


class DecoderComponent(nn.Module):
    def __init__(self, input_size, hidden_size=64, output_size=4, dropout=0.2):
        super(DecoderComponent, self).__init__()
        self.FC = nn.Sequential(
            nn.Linear(in_features=1, out_features=hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=hidden_size, out_features=output_size),
        )

    def forward(self, x):
        x = self.FC(x)
        return x


class SP19:

    def __init__(self, args,data_size):
        self.args = args
        self.server_model=self.public_model_loader(model_nums=10)
        self.middle_outs={}
        self.gradients=[]
        self.conv_channels=[]
        self.lr=0.005
        self.epochs = 10
        self.batch_size=1
        self.gradient_layer_nums = 1
        self.gradient_conv=None
        self.attack_client_idx=0
        self.train_size=data_size
        self.test_size=data_size
        self.train_loader = self.load_dataloader()
        self.test_loader = self.load_test_loader()
        self.component = ['output','label','loss','gradient_conv','encoder','decoder']
        self.model_path = self.args.model_path+'/'+self.args.model+'/'
        path_exists(self.model_path)

    def public_model_loader(self,model_nums):

        model_folder = self.args.model_path + '/' +self.args.model+'/' + self.args.dataset + '/server_model'
        model_list = nn.ModuleList()
        model_files = [f'server_{f}.pth' for f in range(self.args.training_round)]
        model_files = model_files[-model_nums:]
        for model_file in model_files:
            model_path = os.path.join(model_folder, model_file)
            model_pth = torch.load(model_path)
            model = _build_public_model(self.args)
            model.load_state_dict(model_pth)
            model.to(self.args.device)
            model.eval()
            model_list.append(model)
            print(f'load model {model_file}')
        print(f'load {len(model_list)} models')
        return model_list
    
    def load_dataloader(self):
        data_size=self.train_size
        args = self.args
        data_path = args.data_path + '/'+args.dataset
        datas, labels = load_npz_data(data_path + '/train_non_iid.npz')
        data, label = datas[self.attack_client_idx], labels[self.attack_client_idx]
        data = np.array(data)
        label = np.array(label)
        non_member_datas, non_member_labels = load_npz_data(data_path + '/test_non_iid.npz')
        non_member_data,non_member_label =  non_member_datas[self.attack_client_idx], non_member_labels[self.attack_client_idx] 

        total_samples = len(non_member_label)-1
        random.seed(self.args.random_seed)
        random_indices = np.random.choice(total_samples, data_size, replace=False)
        sampled_data1, sampled_label1 = data[random_indices], label[random_indices]
        sampled_data2, sampled_label2 = non_member_data[random_indices], non_member_label[random_indices]
        data = np.concatenate((sampled_data1,sampled_data2))
        label = np.concatenate((sampled_label1,sampled_label2))
        membership = data_size*[1]+data_size*[0]
        dataset = ClientDatasetWithMember(data, label,membership)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        return dataloader
    
    def load_test_loader(self):
        data_size=self.test_size
        args = self.args
        data_path = args.data_path + '/'+args.dataset
        datas, labels = load_npz_data(data_path + '/train_non_iid.npz')
        data, label = np.delete(datas,self.attack_client_idx,axis=0), np.delete(labels,self.attack_client_idx,axis=0)
        data = np.concatenate(data)
        label = np.concatenate(label)
        data = np.array(data)
        label = np.array(label)
        non_member_datas, non_member_labels = load_npz_data(data_path + '/test_non_iid.npz')
        non_member_data,non_member_label =  np.delete(non_member_datas,self.attack_client_idx,axis=0),np.delete(non_member_labels,self.attack_client_idx,axis=0)
        non_member_data=np.concatenate(non_member_data)
        non_member_label=np.concatenate(non_member_label)
        non_member_data = np.array(non_member_data)
        non_member_label=np.array(non_member_label)

        total_samples = len(label)
        random.seed(self.args.random_seed)
        random_indices = np.random.choice(total_samples, data_size, replace=False)
        sampled_data1, sampled_label1 = data[random_indices], label[random_indices]
        sampled_data2, sampled_label2 = non_member_data[random_indices], non_member_label[random_indices]
        data = np.concatenate((sampled_data1,sampled_data2))
        label = np.concatenate((sampled_label1,sampled_label2))
        membership = data_size*[1]+data_size*[0]
        dataset = ClientDatasetWithMember(data, label,membership)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        return dataloader

    def hook_fn(self, module, input, output):
        self.middle_outs[module] = output
        self.conv_channels.append(output.shape)
    

    def init_layers(self, attacker_number=0):
        model_num = len(self.server_model)
        dummy_input = torch.randn(1, 3, 32, 32) 
        dummy_label = torch.randint(0,10,(1,))
        data = dummy_input.to(self.args.device)
        label = dummy_label.to(self.args.device)
        loss_one_sample = []
        out_one_sample = []
        label_one_sample = []
        gradient_one_sample=[]


        for model in self.server_model:
            for name,layer in model.named_modules():
                if isinstance(layer,nn.ReLU):
                    layer.register_forward_hook(self.hook_fn)

            loss_fn = nn.CrossEntropyLoss()
            model.zero_grad(set_to_none=True)
            out = model(data)
            loss = loss_fn(out,label)
            loss.backward()
            _, predicted = torch.max(out, 1,keepdim=False) 
            loss = torch.tensor([loss.item()])
            loss_one_sample.append(loss)

            out_one_sample.append(out.detach())

            one_hot_label = torch.nn.functional.one_hot(predicted,out.size(1))
            label_one_sample.append(one_hot_label)

            modules = list(model.named_modules())
            last_n_layers = modules[-self.gradient_layer_nums:]
            for name, module in last_n_layers:
                if hasattr(module, 'weight') and module.weight is not None and module.weight.grad is not None:
                    self.gradients.append(module.weight.grad.view(1,1,10,64))
            gradient_one_sample.append(self.gradients[-1])
            self.gradients=[]

        
        self.loss = LossComponent(input_size=1*model_num,output_size=64*model_num)
        self.loss.to(self.args.device)
        loss_one_sample = torch.stack(loss_one_sample,dim=1).to(self.args.device)
        encoder_input = self.loss(loss_one_sample)


        self.output = OutputComponent(input_size=out.shape[1]*model_num,output_size=64*model_num)
        self.output.to(self.args.device)
        out_one_sample = torch.cat(out_one_sample,dim=1).to(self.args.device)
        output = self.output(out_one_sample)
        encoder_input = torch.cat((encoder_input,output),dim=1)

        self.label = LabelComponent(input_size=10*model_num,output_size=64*model_num)
        self.label.to(self.args.device)
        label_one_sample  = torch.cat(label_one_sample,dim=1).float().to(self.args.device)
        label_out = self.label(label_one_sample)
        encoder_input = torch.cat((encoder_input,label_out),dim=1)
        self.gradient_conv = GradientComponent(input_channels=model_num,
                                                    output_size=64*model_num,
                                                    next_layer_size=10)
        self.gradient_conv.to(self.args.device)
        gradient_one_sample  = torch.cat(gradient_one_sample,dim=1).to(self.args.device)
        gradient_out = self.gradient_conv(gradient_one_sample)
        encoder_input = torch.cat((encoder_input,gradient_out),dim=1)

        self.encoder = EncoderComponent(input_size=encoder_input.shape[1],output_size=2)
        self.encoder.to(self.args.device)
        encoder_out = self.encoder(encoder_input)
        

    def attack_model_train(self):
        model_path = self.model_path+self.args.dataset
        self.init_layers(self.attack_client_idx)
        loss_ce = nn.CrossEntropyLoss()
        loss_bce = nn.BCELoss(reduction='mean')
        optimizer = torch.optim.SGD([
            {'params': self.loss.parameters(), 'lr': self.lr},
            {'params': self.output.parameters(), 'lr': self.lr},
            {'params': self.label.parameters(), 'lr': self.lr},
            {'params': self.encoder.parameters(), 'lr': self.lr},
            {'params': self.gradient_conv.parameters(), 'lr': self.lr},
        ])
        self.loss.train()
        self.output.train()
        self.label.train()
        self.encoder.train()
        self.gradient_conv.train()
        for model in self.server_model:
                for name,layer in model.named_modules():
                    if isinstance(layer,nn.ReLU):
                        layer.register_forward_hook(self.hook_fn)

        for epoch in range(self.epochs):
            running_loss=0
            with tqdm(total=self.train_size*2//self.batch_size,desc=f'Epoch {epoch+1}/{self.epochs}', unit="batch") as pbar:
                for i,(data,label,member) in enumerate(self.train_loader):
                    data=data.to(self.args.device)
                    label=label.to(self.args.device)
                    member = member.to(self.args.device)

                    self.loss.zero_grad()
                    self.output.zero_grad()
                    self.label.zero_grad()
                    self.encoder.zero_grad()
                    self.gradient_conv.zero_grad()

                    loss_one_sample = []
                    out_one_sample = []
                    label_one_sample = []
                    gradient_one_sample=[]

                    for model in self.server_model:
                        loss_fn = nn.CrossEntropyLoss()
                        model.zero_grad()
                        out = model(data)
                        loss = loss_fn(out,label)
                        loss.backward()
                        _, predicted = torch.max(out, 1,keepdim=False) 
                        loss = torch.tensor([loss.item()])
                        loss_one_sample.append(loss)
                        

                        out_one_sample.append(out.detach())

                        one_hot_label = torch.nn.functional.one_hot(predicted,out.size(1))
                        label_one_sample.append(one_hot_label)

                        modules = list(model.named_modules())
                        last_n_layers = modules[-self.gradient_layer_nums:]
                        for name, module in last_n_layers:
                            if hasattr(module, 'weight') and module.weight is not None and module.weight.grad is not None:
                                self.gradients.append(module.weight.grad.view(1,1,10,64))
                        gradient_one_sample.append(self.gradients[-1])
                        self.gradients=[]

                    loss_one_sample = torch.stack(loss_one_sample,dim=1).to(self.args.device)
                    encoder_input = self.loss(loss_one_sample)

                    out_one_sample = torch.cat(out_one_sample,dim=1).to(self.args.device)
                    output = self.output(out_one_sample)
                    encoder_input = torch.cat((encoder_input,output),dim=1)

                    label_one_sample  = torch.cat(label_one_sample,dim=1).float().to(self.args.device)
                    label_out = self.label(label_one_sample)
                    encoder_input = torch.cat((encoder_input,label_out),dim=1)

                    gradient_one_sample  = torch.cat(gradient_one_sample,dim=1).to(self.args.device)
                    gradient_out = self.gradient_conv(gradient_one_sample)
                    encoder_input = torch.cat((encoder_input,gradient_out),dim=1)

                    encoder_input = encoder_input.to(self.args.device)
                    mem_predict = self.encoder(encoder_input)
                    mem_predict = torch.softmax(mem_predict,dim=1)
                    loss_b = loss_ce(mem_predict,member)
                    loss_b.backward()
                    optimizer.step()
                    running_loss += loss_b.item()
                    avg_loss = running_loss / (i+1)
                    pbar.set_postfix(loss=avg_loss)
                    pbar.update(1)
        print('save_models')
        torch.save(self.gradient_conv,model_path+'/our_model/gradient_conv.pth')
        torch.save(self.loss,model_path+'/our_model/loss.pth')
        torch.save(self.output,model_path+'/our_model/output.pth')
        torch.save(self.label,model_path+'/our_model/label.pth')
        torch.save(self.encoder,model_path+'/our_model/encoder.pth')
                
    def attack(self,train=True):
        model_path = self.model_path+self.args.dataset
        if train is True:
            self.attack_model_train()
        else:
            self.gradient_conv = torch.load(model_path+'/our_model/gradient_conv.pth')
            self.gradient_conv.to(self.args.device)
            self.gradient_conv.eval()
            self.loss = torch.load(model_path+'/our_model/loss.pth')
            self.loss.to(self.args.device)
            self.loss.eval()
            self.output = torch.load(model_path+'/our_model/output.pth')
            self.output.to(self.args.device)
            self.output.eval()
            self.label = torch.load(model_path+'/our_model/label.pth')
            self.label.to(self.args.device)
            self.label.eval()
            self.encoder = torch.load(model_path+'/our_model/encoder.pth')
            self.encoder.to(self.args.device)
            self.encoder.eval()
        
        for model in self.server_model:
                for name,layer in model.named_modules():
                    if isinstance(layer,nn.ReLU):
                        layer.register_forward_hook(self.hook_fn)


        loss_fn = nn.CrossEntropyLoss()
        correct=0
        total=0
        predict,ground_truth,scores=[],[],[]
        for i,(data,label,member) in enumerate(tqdm(self.test_loader,desc='test')):
            loss_one_sample = []
            out_one_sample = []
            label_one_sample = []
            gradient_one_sample=[]

            data = data.to(self.args.device)
            label = label.to(self.args.device)
            member = member.to(self.args.device)

            for model in self.server_model:
                model.train()
                loss_fn = nn.CrossEntropyLoss()
                out = model(data)
                loss = loss_fn(out,label)
                loss.backward()
                _, predicted = torch.max(out, 1,keepdim=False) 
                loss = torch.tensor([loss.item()])
                loss_one_sample.append(loss)

                out_one_sample.append(out.detach())

                one_hot_label = torch.nn.functional.one_hot(predicted,out.size(1))
                label_one_sample.append(one_hot_label)

                modules = list(model.named_modules())
                last_n_layers = modules[-self.gradient_layer_nums:]
                for name, module in last_n_layers:
                    if hasattr(module, 'weight') and module.weight is not None and module.weight.grad is not None:
                        self.gradients.append(module.weight.grad.view(1,1,10,64))
                gradient_one_sample.append(self.gradients[-1])
                self.gradients=[]


            loss_one_sample = torch.stack(loss_one_sample,dim=1).to(self.args.device)
            encoder_input = self.loss(loss_one_sample)

            out_one_sample = torch.cat(out_one_sample,dim=1).to(self.args.device)
            output = self.output(out_one_sample)
            encoder_input = torch.cat((encoder_input,output),dim=1)

            label_one_sample  = torch.cat(label_one_sample,dim=1).float().to(self.args.device)
            label_out = self.label(label_one_sample)
            encoder_input = torch.cat((encoder_input,label_out),dim=1)

            gradient_one_sample  = torch.cat(gradient_one_sample,dim=1).to(self.args.device)
            gradient_out = self.gradient_conv(gradient_one_sample)
            encoder_input = torch.cat((encoder_input,gradient_out),dim=1)

            encoder_input = encoder_input.to(self.args.device)
            mem_predict = self.encoder(encoder_input)
            mem_predict = torch.softmax(mem_predict,dim=1)
            score = mem_predict[0,1]/(mem_predict[0,0])
            predicted_label = torch.argmax(mem_predict)
            correct += (predicted_label == member).sum().item()
            total+=len(member)
            predict.append(int(predicted_label.item()))
            scores.append(score.detach().cpu().item())
            ground_truth.append(member.item())
        print(f'Global Model test Acc:{correct / total}----')
        tpr = ROC_AUC_Result_logshow(ground_truth,scores,False)
        


class USENIX2024:

    def __init__(self, args,data_size):
        self.args = args
        self.data_size=data_size
        self.thre_data_size=50
        self.attack_client_idx = 0
        self.client_dataloader,self.thre_dataloader = self.load_client_dataloader(self.data_size+self.thre_data_size)
        self.attack_round = 300
        self.private_model_list = self.private_model_loader(self.attack_client_idx)
        self.public_dataloader = self.load_public_dataloader(data_size=self.data_size)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        

    def compute_weights(self,t):
        weights = []
        for u in range(1, t + 1,self.args.client_num//self.args.participant):
            weight = 6 * (2 * t * u - t**2 + 1) / (t**4 - t**2)
            weights.append(weight)
        return weights


    def compute_slopes(self,losses, t):
        weights = self.compute_weights(t)
        slopes = []
        for sample_losses in losses:
            slope = sum(w * c for w, c in zip(weights, sample_losses[:t]))
            slopes.append(slope)
        return slopes


    def compute_threshold(self, losses, rounds):
        slopes = self.compute_slopes(losses, rounds)
        threshold = np.percentile(slopes, 50)
        return threshold
    

    def free_training_attack(self,losses,threshold,rounds):
        slopes = self.compute_slopes(losses, rounds)
        predicted_labels = slopes < threshold
        return predicted_labels,slopes
        

    
    def load_client_dataloader(self, total_data_size):
        args = self.args
        random.seed(self.args.random_seed)
        
        cal_size = self.thre_data_size
        eval_size = self.data_size
        
        assert eval_size > 0, "Total data size must be larger than threshold data size!"

        data_path = args.data_path + '/' + args.dataset + '/' + args.model + '/' + args.data_split
        datas, labels = load_npz_data(data_path + '/train_non_iid.npz')
        non_member_data, non_member_label = load_npz_data(data_path + '/test_non_iid.npz')

        c_idx = self.attack_client_idx
        client_member_data = datas[c_idx]
        client_member_label = labels[c_idx]
        client_non_member_data = non_member_data[c_idx]
        client_non_member_label = non_member_label[c_idx]

        indices_member = np.arange(len(client_member_data))
        random.shuffle(indices_member)
        indices_non_member = np.arange(len(client_non_member_data))
        random.shuffle(indices_non_member)

        sampled_idx_m_total = indices_member[:total_data_size]
        sampled_idx_nm_total = indices_non_member[:total_data_size]

        
        cal_idx_m = sampled_idx_m_total[:cal_size]
        cal_idx_nm = sampled_idx_nm_total[:cal_size]
        
        eval_idx_m = sampled_idx_m_total[cal_size:]
        eval_idx_nm = sampled_idx_nm_total[cal_size:]

        def create_dataset(idx_m, idx_nm):
            d_m = client_member_data[idx_m]
            l_m = client_member_label[idx_m]
            d_nm = client_non_member_data[idx_nm]
            l_nm = client_non_member_label[idx_nm]
            
            data_concat = np.concatenate((d_m, d_nm))
            label_concat = np.concatenate((l_m, l_nm))
            membership = [1] * len(d_m) + [0] * len(d_nm)
            
            return ClientDatasetWithMember(data_concat, label_concat, membership)

        dataset_cal = create_dataset(cal_idx_m, cal_idx_nm)
        dataset_eval = create_dataset(eval_idx_m, eval_idx_nm)

        print(f"Loaded Datasets: Eval Size={len(dataset_eval)} (Target), Calibration Size={len(dataset_cal)} (Threshold)")

        thre_dataloader = DataLoader(dataset_cal, batch_size=1, shuffle=True)
        client_dataloader = DataLoader(dataset_eval, batch_size=1, shuffle=True)

        return client_dataloader, thre_dataloader

    def load_public_dataloader(self, data_size):
        args = self.args
        random.seed(self.args.random_seed)
        data_path = args.data_path + '/'+args.dataset+'/'+args.model+'/'+args.data_split
        datas, labels = load_npz_data(data_path + '/train_non_iid.npz')
        data = [client_data for client_idx, client_data in enumerate(datas) if client_idx == self.attack_client_idx]
        label = [client_label for client_idx, client_label in enumerate(labels) if client_idx == self.attack_client_idx]
        data = np.concatenate(data)
        label = np.concatenate(label)
        data = np.array(data)
        label = np.array(label)
        non_member_datas, non_member_labels = load_npz_data(data_path + '/test_non_iid.npz')
        non_member_data,non_member_label =  np.delete(non_member_datas,self.attack_client_idx,axis=0),np.delete(non_member_labels,self.attack_client_idx,axis=0)
        non_member_data=np.concatenate(non_member_data)
        non_member_label=np.concatenate(non_member_label)
        non_member_data = np.array(non_member_data)
        non_member_label=np.array(non_member_label)

        total_samples = len(label)
        actual_size = min(total_samples, data_size)

        if total_samples < data_size:
            print(f"[Warning] 请求 public_data={data_size}, 但数据集只有 {total_samples}。将使用全部数据。")

        random_indices = np.random.choice(total_samples, actual_size, replace=False)
        sampled_data1, sampled_label1 = data[random_indices], label[random_indices]
        sampled_data2, sampled_label2 = non_member_data[random_indices], non_member_label[random_indices]
        data = np.concatenate((sampled_data1,sampled_data2))
        label = np.concatenate((sampled_label1,sampled_label2))
        membership = data_size*[1]+data_size*[0]
        dataset = ClientDatasetWithMember(data, label,membership)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        return dataloader

    def public_model_loader(self):
        model_folder = self.args.model_path + '/' +self.args.model+'/' + self.args.dataset + '/server_model'
        model_list = nn.ModuleList()
        model_files = [f'server_{f}.pth' for f in range(self.args.training_round)]
        for model_file in model_files:
            model_path = os.path.join(model_folder, model_file)
            model_pth = torch.load(model_path)
            model = _build_public_model(self.args)
            model.load_state_dict(model_pth)
            model.to(self.args.device)
            model_list.append(model)
            print(f'load model {model_file}')
        print(f'load {len(model_list)} models')
        return model_list
    
    
    def private_model_loader(self,client_idx):
        model_folder = self.args.model_path + '/' +self.args.model+'/' + self.args.dataset + '/client_model/'
        model_list = nn.ModuleList()
        model_files = [
            f'client_{client_idx}_{f}.pth'
            for f in range(self.args.training_round)
            if os.path.exists(model_folder+f'client_{client_idx}_{f}.pth')
        ]
        for model_file in model_files:
            model_path = os.path.join(model_folder, model_file)
            model_pth = torch.load(model_path)
            model = _build_public_model(self.args)
            model.load_state_dict(model_pth)
            model.to(self.args.device)
            model.eval()
            model_list.append(model)
            print(f'load model {model_file}')
        print(f'load {len(model_list)} models')
        return model_list


    def client_attack(self):
        losses = []
        conf=[]
        ground_truth = []
        for data, label,member in tqdm(self.public_dataloader,desc='free training attack'):
            temp1 = []
            temp2=[]
            for public_model in self.public_model_list:
                data = data.to(self.args.device)
                label = label.to(self.args.device)
                public_model.eval()
                output = public_model(data)
                loss = self.loss_fn(output,label)
                confidences = torch.softmax(output,dim=1)
                confidence = torch.max(confidences)
                temp1.append(loss.detach().cpu().item())
                temp2.append(confidence.cpu().item())
            losses.append(temp1)
            conf.append(temp2)
            ground_truth.append(member.item())
        print('client attack')
        print('loss:')
        slopes = self.compute_slopes(losses, self.attack_round)
        slopes = [-i for i in slopes]
        tpr = ROC_AUC_Result_logshow(ground_truth,slopes,False)
        tpr = ROC_AUC_Result_logshow(ground_truth,slopes,True)

        self.plot_histograms(slopes,ground_truth)
        return tpr
    
    def server_attack(self):
        losses = []
        ground_truth = []
        for data, label,member in tqdm(self.client_dataloader,desc='free training attack'):
            temp1 = []
            temp2=[]
            for private_model in self.private_model_list:
                data = data.to(self.args.device)
                label = label.to(self.args.device)
                private_model.eval()
                output = private_model(data)
                loss = self.loss_fn(output,label)
                confidences = torch.softmax(output,dim=1)
                confidence = torch.max(confidences)
                temp1.append(loss.detach().cpu().item())
                temp2.append(confidence.cpu().item())
            losses.append(temp1)
            ground_truth.append(member.item())

        slopes = self.compute_slopes(losses, self.attack_round)
        slopes = [-i for i in slopes]
        tpr = ROC_AUC_Result_logshow(ground_truth,slopes,False)

        self.plot_histograms(slopes,ground_truth)
        return tpr
    
    
    def attack(self,type):
        if type == 'client':
            self.client_attack()
        elif type == 'server':
            self.server_attack()
        else:
            print('wrong type')
        
    
    def plot_histograms(self, x, y):
        save_dir = './plots'
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist([xi for xi, yi in zip(x, y) if yi == 0], bins=30, alpha=0.5, label='nonmember', color='blue', edgecolor='black')

        ax.hist([xi for xi, yi in zip(x, y) if yi == 1], bins=30, alpha=0.5, label='member', color='red', edgecolor='black')

        ax.set_title('Distribution of Data by Labels')
        ax.set_xlabel('Data Value')
        ax.set_ylabel('Frequency')

        ax.legend()
        plt.cla()

        

class ICLR2023:

    def __init__(self, args, attack_client_idx, total_eval_size=600):
        self.args = args
        self.attack_client_idx = attack_client_idx
        self.device = args.device
        self.data_size = total_eval_size

        self.model_template = _build_public_model(args).to(self.device)
        
        self.client_dataloader= self.load_client_dataloader(total_eval_size)
        self.eval_loader = self.client_dataloader
        
        self.eval_scores_sum = np.zeros(len(self.eval_loader.dataset))

    def load_client_dataloader(self, data_size):
        args = self.args
        random.seed(self.args.random_seed)

        data_path = args.data_path + '/'+args.dataset+'/'+args.model+'/'+args.data_split
        datas, labels = load_npz_data(data_path + '/train_non_iid.npz')
        non_member_data, non_member_label = load_npz_data(data_path + '/test_non_iid.npz')

        all_indices = np.arange(len(datas[self.attack_client_idx]))
        random.shuffle(all_indices)
        sampled_indices1 = all_indices[:int(data_size)]
        sampled_indices2 = all_indices[:int(data_size)]

        sampled_data1 = datas[self.attack_client_idx][sampled_indices1]
        sampled_label1 = labels[self.attack_client_idx][sampled_indices1]
        sampled_data2 = non_member_data[self.attack_client_idx][sampled_indices2]
        sampled_label2 = non_member_label[self.attack_client_idx][sampled_indices2]

        assert len(sampled_data1) == len(sampled_label1), "Sampled data and labels are mismatched!"
        assert len(sampled_data2) == len(sampled_label2), "Sampled data and labels are mismatched!"

        membership = [1] * len(sampled_data1) + [0] * len(sampled_data2)
        data = np.concatenate((sampled_data1, sampled_data2))
        label = np.concatenate((sampled_label1, sampled_label2))

        assert len(data) == len(label), "Data and labels have mismatched lengths!"

        dataset = ClientDatasetWithMember(data, label, membership)

        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        return dataloader

    def load_models_and_get_update(self, round_idx):
        server_path = os.path.join(
            self.args.model_path, self.args.model, self.args.dataset, 
            'server_model', f'server_{round_idx}.pth'
        )
        
        client_path = os.path.join(
            self.args.model_path, self.args.model, self.args.dataset, 
            'client_model', f'client_{self.attack_client_idx}_{round_idx}.pth'
        )
        
        if not os.path.exists(server_path):
            raise FileNotFoundError(f"Server model not found: {server_path}")
        if not os.path.exists(client_path):
            raise FileNotFoundError(f"Client model not found: {client_path}")

        try:
            server_state = torch.load(server_path, map_location=self.device)
            self.model_template.load_state_dict(server_state)
            self.model_template.eval()
            
            server_vec = nn.utils.parameters_to_vector(self.model_template.parameters()).detach()
            
            client_state = torch.load(client_path, map_location=self.device)
            
            
            self.model_template.load_state_dict(client_state)
            client_vec = nn.utils.parameters_to_vector(self.model_template.parameters()).detach()
            
            delta_gamma = client_vec - server_vec
            
            self.model_template.load_state_dict(server_state)
            
            return self.model_template, delta_gamma

        except RuntimeError as e:
            print(f"[Load Error] 模型结构不匹配或显存不足: {e}")
            return None, None

    def compute_scores_for_loader(self, loader, model, update_vec):
        scores = []
        update_vec_np = update_vec.cpu().numpy()
        norm_update = np.linalg.norm(update_vec_np)
        
        if norm_update == 0:
            return np.zeros(len(loader.dataset))

        criterion = nn.CrossEntropyLoss()

        for data, target, _ in loader:
            for i in range(len(data)):
                img = data[i:i+1].to(self.device)
                lbl = target[i:i+1].to(self.device)
                
                model.zero_grad()
                
                for p in model.parameters(): 
                    p.requires_grad = True
                
                out = model(img)
                loss = criterion(out, lbl)
                
                loss.backward()
                
                grads = []
                for param in model.parameters():
                    if param.grad is not None:
                        grads.append(param.grad.view(-1))
                
                if len(grads) > 0:
                    grad_vec = torch.cat(grads).detach().cpu().numpy()
                    norm_grad = np.linalg.norm(grad_vec)
                    
                    if norm_grad == 0:
                        scores.append(0.0)
                    else:
                        sim = np.dot(grad_vec, update_vec_np) / (norm_grad * norm_update)
                        scores.append(sim)
                else:
                    scores.append(0.0)
                    
        return np.array(scores)

    def attack(self):
        start_round = 0 
        end_round = self.args.training_round
        valid_rounds = 0
        
        print(f"Starting Attack on WHOLE models ({start_round} -> {end_round})...")

        step = self.args.client_num//self.args.participant

            
        for r in tqdm(range(start_round, end_round, step), desc="Attacking Rounds"):
            try:
                model, delta_gamma = self.load_models_and_get_update(r)
                
                if model is None: continue 
                
                
                eval_scores = self.compute_scores_for_loader(self.eval_loader, model, delta_gamma)
                self.eval_scores_sum += eval_scores
                
                valid_rounds += 1
                
            except FileNotFoundError:
                continue
            except Exception as e:
                print(f"Error in round {r}: {e}")
                continue

        if valid_rounds == 0:
            print("Error: No valid rounds found!")
            return 0

        final_eval_scores = self.eval_scores_sum / valid_rounds


        eval_gt = []
        for _, _, m in self.eval_loader:
            eval_gt.extend(m.numpy())
        eval_gt = np.array(eval_gt)
        auc = ROC_AUC_Result_logshow(eval_gt, final_eval_scores,True)
        return auc
    

class Arxiv2025:
    def __init__(self,args):
        self.args = args
        self.MODE = 'test'
        self.attack_modes=["cosine attack","grad diff","loss based","grad norm"]
        self.epochs=list(range(self.args.client_num//self.args.participant,self.args.training_round,self.args.client_num//self.args.participant))
        self.p_folder=args.model_path + '/' + args.model+'/'+ args.dataset + '/our_model/arxiv/'
        self.PATH=self.p_folder+"/client_{}_round_{}.pkl"
        self.p=self.PATH
        self.save_dir='log_file/'+args.model+'/'+args.dataset+'/'+'arxiv/'
        self.device = self.args.device
        self.SEED = args.random_seed
        self.MAX_K=10
        self.mix_length = 1800
        self.select_mode=1
        self.select_method='outlier'
        self.SHADOW_NUM=4
        path_exists(self.save_dir)

    @ torch.no_grad()
    def hinge_loss_fn(self,x,y):
        x,y=copy.deepcopy(x).cuda(),copy.deepcopy(y).cuda()
        mask=torch.eye(x.shape[1],device="cuda")[y].bool()
        tmp1=x[mask]
        x[mask]=-1e10
        tmp2=torch.max(x,dim=1)[0]
        return (tmp1-tmp2).cpu().numpy()

    def extract_hinge_loss(self,i):
        val_dict={}
        val_index=i["val_index"]
        val_hinge_index=self.hinge_loss_fn(i["val_res"]["logit"] , i["val_res"]["labels"] )
        for j,k in zip(val_index,val_hinge_index):
            if j in val_dict:
                val_dict[j].append(k)
            else:
                val_dict[j]=[k]

        train_dict={}
        train_index=i["train_index"]
        train_hinge_index=self.hinge_loss_fn(i["train_res"]["logit"] , i["train_res"]["labels"] )
        for j,k in zip(train_index,train_hinge_index):
            if j in train_dict:
                train_dict[j].append(k)
            else:
                train_dict[j]=[k]
        
        test_dict={}
        test_index=i["test_index"]
        test_hinge_index=self.hinge_loss_fn(i["test_res"]["logit"] , i["test_res"]["labels"] )
        for j,k in zip(test_index,test_hinge_index):
            if j in test_dict:
                test_dict[j].append(k)
            else:
                test_dict[j]=[k]

        return (val_dict,train_dict,test_dict)

    
    def ce_loss_fn(self,x,y):
        loss_fn=torch.nn.CrossEntropyLoss(reduction='none')
        return loss_fn(x,y)
    
    def plot_auc(self,name,target_val_score,target_train_score,epoch): 


        fpr, tpr, thresholds = metrics.roc_curve(torch.cat( [torch.zeros_like(target_val_score),torch.ones_like(target_train_score)] ).cpu().numpy(), torch.cat([target_val_score,target_train_score]).cpu().numpy())
        auc=metrics.auc(fpr, tpr)
        log_tpr,log_fpr=np.log10(tpr),np.log10(fpr)
        log_tpr[log_tpr<-5]=-5
        log_fpr[log_fpr<-5]=-5
        log_fpr=(log_fpr+5)/5.0
        log_tpr=(log_tpr+5)/5.0
        log_auc=metrics.auc( log_fpr,log_tpr )

        tprs={}
        for fpr_thres in [10, 1, 0.1,0.02,0.01,0.001,0.0001]:
            tpr_index = np.sum(fpr<fpr_thres)
            tprs[str(fpr_thres)]=tpr[tpr_index-1]
        return auc,log_auc,tprs

    def lira_attack_ldh_cosine(self,f,epch,K, save_dir, extract_fn=None,attack_mode="cos"):
        print('******************************************************')
        print('************','Epch:',epch,' attack_mode:',attack_mode,'**************')
        print('******************************************************')
        accs=[]
        training_res=[]
        for i in range(K):
            filepath = f.format(i, epch)
            print(filepath)
            if os.path.exists(filepath):
                training_res.append(torch.load(filepath))
                accs.append(training_res[-1]["test_acc"])
        
        target_idx=self.args.arxiv_client[0]
        val_idx = self.args.arxiv_client[1]
        target_res=training_res[target_idx]
        shadow_res=training_res[val_idx:]
        if attack_mode=="cos":
            target_train_loss = target_res["train_cos"].clone().detach().cpu().numpy()
            if self.MODE=="test":
                target_test_loss = target_res["test_cos"].clone().detach().cpu().numpy()
            elif self.MODE=="val":
                mix_test_loss = target_res["mix_cos"].clone().detach().cpu().numpy()
            elif self.MODE =='mix':
                random_indices = torch.randperm(target_res["test_cos"].shape[0])
                target_test_loss = target_res["test_cos"][random_indices[:self.mix_length]]
                target_test_loss = torch.tensor(target_test_loss).cpu().numpy()
                mix_test_loss = torch.tensor(target_res["mix_cos"]).cpu().numpy()
                mix_test_loss = np.concatenate([target_test_loss,mix_test_loss],axis=0)
                print('mix_test_loss shape:',mix_test_loss.shape)
                target_test_loss = mix_test_loss

        if attack_mode=="diff":
            target_train_loss=target_res["train_diffs"].cpu().numpy()
            if self.MODE=="test":
                target_test_loss=target_res["test_diffs"].cpu().numpy()
            elif self.MODE=="val":
                target_test_loss=target_res["val_diffs"].cpu().numpy()
        if attack_mode == 'loss':
            target_train_loss = -self.ce_loss_fn(target_res["train_res"]["logit"] , target_res["train_res"]["labels"] ).cpu().numpy()
            if self.MODE=="test":
                target_test_loss=-self.ce_loss_fn(target_res["test_res"]["logit"] , target_res["test_res"]["labels"] ).cpu().numpy()
            elif self.MODE=="val":
                target_test_loss=-self.ce_loss_fn(target_res["val_res"]["logit"] , target_res["val_res"]["labels"] ).cpu().numpy()


        shadow_train_losses=[]
        shadow_test_losses=[]
        if attack_mode=="cos":
            for i in shadow_res:
                shadow_train_losses.append( i["train_cos"].clone().detach().cpu().numpy() )
                if self.MODE=="val":
                    shadow_test_losses.append(torch.tensor(i["val_cos"]).cpu().numpy() )
                elif self.MODE=="test":
                    shadow_test_losses.append(i["test_cos"].clone().detach().cpu().numpy())

        elif attack_mode=="diff":
            for i in shadow_res:
                shadow_train_losses.append( i["train_diffs"].cpu().numpy() )
                if self.MODE=="val":
                    shadow_test_losses.append(i["val_diffs"].cpu().numpy() )
                elif self.MODE=="test":
                    shadow_test_losses.append(i["test_diffs"].cpu().numpy() )
        elif attack_mode=="loss":
             for i in shadow_res:
                shadow_train_losses.append(-self.ce_loss_fn(i["train_res"]["logit"] , i["train_res"]["labels"]).cpu().numpy() )
                if self.MODE=="val":
                    shadow_test_losses.append(-self.ce_loss_fn(i["val_res"]["logit"], i["val_res"]["labels"]).cpu().numpy() )
                elif self.MODE=="test":
                    shadow_test_losses.append(-self.ce_loss_fn(i["test_res"]["logit"], i["test_res"]["labels"]).cpu().numpy() )


        shadow_train_losses_stack=np.vstack( shadow_train_losses )
        shadow_test_losses_stack=np.vstack( shadow_test_losses )
        print('attack_mode:',attack_mode)

        if self.select_mode == 1 and attack_mode =='cos':
            tmps=[]
            means=[]
            client_ids=[]
            
            if self.select_method == 'outlier':
                train_mu_out=np.zeros_like(shadow_train_losses_stack.mean(axis=0))
                train_var_out=np.zeros_like(shadow_train_losses_stack.var(axis=0)+1e-8)
                print('**************',train_mu_out.shape)
                test_mu_out=np.zeros_like(shadow_test_losses_stack.mean(axis=0))
                test_var_out=np.zeros_like(shadow_test_losses_stack.var(axis=0)+1e-8)

                for j in range(0,shadow_train_losses_stack.shape[1]):
                    mask = shadow_train_losses_stack[:,j] < shadow_train_losses_stack[:,j].mean(axis=0) + 3*shadow_train_losses_stack[:,j].std(axis=0)
                    sel_mdm = shadow_train_losses_stack[:,j][mask]
                    if j %2000==0:
                        print(' train outlier view:')
                        print(shadow_train_losses_stack[:,j])
                        print(target_train_loss[j])
                        print('sel_mdm.shape', sel_mdm.shape)
                    if sel_mdm.shape[0]==0:
                        if j % 50 == 0:
                            print('outlier view:')
                            print('mask:',mask)
                            print(shadow_train_losses_stack[:,j])
                            print(target_train_loss[j])
                        sel_mdm=np.array([np.min(shadow_train_losses_stack[:,j])])
                    train_mu_out[j] = np.mean(sel_mdm, axis=0)
                    train_var_out[j] = np.var(sel_mdm, axis=0)+1e-8
                
                for j in range(0,shadow_test_losses_stack.shape[1]):
                    mask = shadow_test_losses_stack[:,j] < shadow_test_losses_stack[:,j].mean(axis=0) + 3*shadow_test_losses_stack[:,j].std(axis=0)
                    sel_mdm = shadow_test_losses_stack[:,j][mask]
                    if j % 10==0:
                        print('outlier view:')
                        print(shadow_test_losses_stack[:,j])
                        print(target_test_loss[j])
                        print(mask)
                        print('sel_mdm.shape', sel_mdm.shape)
                    
                    if j %2000==0:
                        print('test outlier view:')
                        print(shadow_test_losses_stack[:,j])
                        print(target_test_loss[j])
                        print('sel_mdm.shape', sel_mdm.shape)
                    if sel_mdm.shape[0]==0:
                        if j % 50==0:
                            print('outlier view:')
                            print(shadow_test_losses_stack[:,j])
                            print(target_test_loss[j])
                            print('sel_mdm.shape', sel_mdm.shape)
                        sel_mdm=np.array([np.min(shadow_test_losses_stack[:,j])])
                    test_mu_out[j] = np.mean(sel_mdm, axis=0)
                    test_var_out[j] = np.var(sel_mdm, axis=0)+1e-8    

        if attack_mode != 'cos'or self.select_mode == 0 or (self.select_method != 'mean_per' and self.select_method != 'outlier'):
            
            train_mu_out=shadow_train_losses_stack.mean(axis=0)
            train_var_out=shadow_train_losses_stack.std(axis=0)+1e-8

            test_mu_out=shadow_test_losses_stack.mean(axis=0)
            test_var_out=shadow_test_losses_stack.std(axis=0)+1e-8


        train_l_out=scipy.stats.norm.cdf(target_train_loss,train_mu_out,train_var_out)
        test_l_out=scipy.stats.norm.cdf(target_test_loss,test_mu_out,test_var_out)
        result = list(np.concatenate((train_l_out,test_l_out),axis=0))
        gt = [1]*train_l_out.shape[0]+[0]*test_l_out.shape[0]
        auc,tpr = ROC_AUC_Result_logshow_with_auc(gt,result,False)
        return auc,tpr,(train_l_out,test_l_out)

    def cos_attack(self,f,K,epch,attack_mode,extract_fn=None):
        reverse=False
        accs=[]
        target_res=torch.load(f.format(self.args.arxiv_client[0],epch))
        tprs=None
        print(attack_mode)

        if attack_mode =="cosine attack":
            if self.MODE=="test":
                val_liratios=target_res['test_cos']
            elif self.MODE=="val":
                val_liratios=target_res['val_cos']
            elif self.MODE=='mix':
                random_indices = torch.randperm(target_res["test_cos"].shape[0])
                val_liratios = target_res["test_cos"][random_indices[:self.mix_length]]
                val_liratios = torch.tensor(val_liratios)
                mix_test_loss = torch.tensor(target_res["mix_cos"])
                mix_test_loss = torch.cat([val_liratios,mix_test_loss],axis=0)
                val_liratios = mix_test_loss

            val_liratios=np.array([ i.cpu().item() for i in val_liratios ])
            val_liratios = np.nan_to_num(val_liratios, nan=0.0, posinf=1.0, neginf=-1.0)
            train_liratios=target_res['train_cos']
            train_liratios=np.array([ i.cpu().item() for i in train_liratios ])
            train_liratios = np.nan_to_num(train_liratios, nan=0.0, posinf=1.0, neginf=-1.0)
            scores = np.concatenate((train_liratios,val_liratios),axis=0)
            gt = [1]*train_liratios.shape[0]+[0]*val_liratios.shape[0]
            auc,tpr = ROC_AUC_Result_logshow_with_auc(gt,scores,reverse)
    

        elif attack_mode =="grad diff":
            if self.MODE=="test":
                val_liratios=target_res['test_diffs']
            elif self.MODE=="val":
                val_liratios=target_res['val_diffs']
            elif self.MODE=='mix':
                random_indices = torch.randperm(target_res["test_diffs"].shape[0])
                val_liratios = target_res["test_diffs"][random_indices[:self.mix_length]]
                val_liratios = val_liratios.clone().detach() 
                mix_test_loss = target_res["mix_diffs"].clone().detach()
                mix_test_loss = torch.cat([val_liratios,mix_test_loss],axis=0)
                val_liratios = mix_test_loss
            val_liratios=np.array([ i.cpu().item() for i in val_liratios ])
            train_liratios=target_res['train_diffs']
            train_liratios=np.array([ i.cpu().item() for i in train_liratios ])
            scores = np.concatenate((train_liratios,val_liratios),axis=0)
            gt = [1]*train_liratios.shape[0]+[0]*val_liratios.shape[0]
            auc,tpr = ROC_AUC_Result_logshow_with_auc(gt,scores,reverse)
        elif attack_mode =="grad norm":
            if self.MODE=="test":
                val_liratios=target_res['test_grad_norm']
            elif self.MODE=="val":
                val_liratios=target_res['val_grad_norm']
            elif self.MODE=='mix':
                random_indices = torch.randperm(target_res["test_grad_norm"].shape[0])
                val_liratios = target_res["test_grad_norm"][random_indices[:self.mix_length]]
                val_liratios = -val_liratios.clone().detach()
                mix_test_loss = -target_res["mix_grad_norm"].clone().detach()
                mix_test_loss = torch.cat([val_liratios,mix_test_loss],axis=0)
                val_liratios = mix_test_loss
            val_liratios=-np.array([ i.cpu().item() for i in val_liratios ])
            train_liratios=target_res['train_grad_norm']
            train_liratios=-np.array([ i.cpu().item() for i in train_liratios ])
            scores = np.concatenate((train_liratios,val_liratios),axis=0)
            gt = [1]*train_liratios.shape[0]+[0]*val_liratios.shape[0]
            auc,tpr = ROC_AUC_Result_logshow_with_auc(gt,scores,reverse)
        elif attack_mode =="loss based":
            if self.MODE=="test":
                val_liratios=target_res["test_res"]["loss"]
            elif self.MODE=="val":
                val_liratios=target_res["val_res"]["loss"]
            elif self.MODE =='mix':
                random_indices = torch.randperm(target_res["test_res"]["logit"].shape[0])
                val_liratios =target_res["test_res"]["loss"][random_indices[:self.mix_length]]
                mix_test_loss=target_res["mix_res"]["loss"]
                mix_test_loss = np.concatenate([val_liratios,mix_test_loss],axis=0)
                val_liratios = mix_test_loss

            train_liratios=target_res["train_res"]["loss"]
            scores = np.concatenate((train_liratios,val_liratios),axis=0)
            gt = [1]*train_liratios.shape[0]+[0]*val_liratios.shape[0]
            auc,tpr = ROC_AUC_Result_logshow_with_auc(gt,scores,reverse)

        return tpr,auc,(train_liratios, val_liratios)

    def attack(self):
        lira_scores=[]
        lira_loss_scores=[]
        common_scores=[]
        other_scores={}

        scores={k:[] for k in self.attack_modes}
        scores["lira"]=[]
        scores["lira_loss"]=[]
        single_score={k:0 for k in self.attack_modes}
        single_score["lira"]=0
        single_score["lira_loss"]=0
        reses_lira=[]
        reses_lira_loss=[]
        reses_common={k:[] for k in self.attack_modes}
        avg_scores={k:None for k in self.attack_modes}
        avg_scores["lira"]=None
        avg_scores["lira_loss"]=None

        auc_dict={k:[] for k in self.attack_modes}
        auc_dict["lira"]=[]
        auc_dict["lira_loss"]=[]

        for epch in self.epochs:
            cos_auc,cos_tpr,cos_score=self.lira_attack_ldh_cosine(self.p,epch,self.MAX_K,self.save_dir, extract_fn=self.extract_hinge_loss,attack_mode='diff') 
            loss_auc,loss_tpr,loss_score=self.lira_attack_ldh_cosine(self.p,epch,self.MAX_K,self.save_dir, extract_fn=self.extract_hinge_loss,attack_mode='loss') 
                
            scores["lira"].append(cos_tpr)
            scores["lira_loss"].append(loss_tpr)
            auc_dict["lira"].append(cos_auc)
            auc_dict["lira_loss"].append(loss_auc)


            for attack_mode in self.attack_modes:
                common_score=self.cos_attack(self.p,0,epch,attack_mode,extract_fn=self.extract_hinge_loss) 
                reses_common[attack_mode].append(common_score[-1])
                scores[attack_mode].append(common_score[0])
                auc_dict[attack_mode].append(common_score[1])

            lira_scores.append(cos_tpr)
            lira_loss_scores.append(loss_tpr)
            common_scores.append(common_score[0])

            reses_lira.append(cos_score)
            reses_lira_loss.append(loss_score)

        for attack_mode in self.attack_modes:
            sorted_id = sorted(range(len(scores[attack_mode])), key=lambda k: scores[attack_mode][k], reverse=True)
            single_score[attack_mode]=(scores[attack_mode][sorted_id[0]])
            single_score[f'single {attack_mode}_auc'] = auc_dict[attack_mode][sorted_id[0]]


        for attack_mode in ['lira', 'lira_loss']:
            sorted_id = sorted(range(len(scores[attack_mode])), key=lambda k: scores[attack_mode][k], reverse=True)
            single_score[attack_mode]=(scores[attack_mode][sorted_id[0]])
            single_score[f'single {attack_mode}_auc'] = auc_dict[attack_mode][sorted_id[0]]

        print('------------ ----------------- -------------  ')
        print('------------ ---Best attack--- -------------  ')
        print('------------ ----------------- -------------  ')

        for attack_mode in ['lira', 'lira_loss']:
            auc = single_score[f'single {attack_mode}_auc']
            tpr = single_score[attack_mode]
            print(f'Best {attack_mode} auc:{auc}\ntpr:{tpr}')
        for attack_mode in self.attack_modes:
            auc = single_score[f'single {attack_mode}_auc']
            tpr = single_score[attack_mode]
            print(f'Best {attack_mode} auc:{auc}\ntpr:{tpr}')
        
        print('------------ ----------------- -------------  ')
        print('------------ Sequential attack -------------  ')
        print('------------ ----------------- -------------  ')

        reses=reses_lira
        train_score=np.vstack([ i[0].reshape(1,-1) for i in reses]).mean(axis=0)
        test_score=np.vstack([ i[1].reshape(1,-1) for i in reses]).mean(axis=0)
        scores = np.concatenate((train_score,test_score),axis=0)
        gt = [1]*train_score.shape[0]+[0]*test_score.shape[0]
        auc,tpr = ROC_AUC_Result_logshow_with_auc(gt,scores,True)
        if auc<0.5:
            auc,tpr = ROC_AUC_Result_logshow_with_auc(gt,scores,False)
        print(f"averaged_lira_grad tprs:{tpr} \n auc:{auc}")
        avg_scores["lira"]=tpr
        other_scores["lira_auc"]=[auc]


        reses=reses_lira_loss
        train_score=np.vstack([ i[0].reshape(1,-1) for i in reses]).mean(axis=0)
        test_score=np.vstack([ i[1].reshape(1,-1) for i in reses]).mean(axis=0)

        train_fscore=train_score[0:300]
        test_fscore=test_score[0:300]

        train_vscore=train_score[300:350]
        test_vscore=test_score[300:350]

        scores = np.concatenate((train_fscore,test_fscore),axis=0)
        v_scores = np.concatenate((train_vscore,test_vscore),axis=0)
        gt = [1]*train_fscore.shape[0]+[0]*test_fscore.shape[0]
        v_gt=[1]*train_vscore.shape[0]+[0]*test_vscore.shape[0]
        auc,tpr = ROC_AUC_Result_logshow_with_auc(gt,scores,False)
        if auc<0.5:
            auc,tpr = ROC_AUC_Result_logshow_with_auc(gt,scores,False)
        print(f"averaged_lira_loss tprs:{tpr} \n auc:{auc}")
        avg_scores["lira_loss"]=tpr
        other_scores["lira_loss_auc"]=[auc]


        reses=reses_common["cosine attack"]
        train_score=np.vstack([ i[0] for i in reses]).mean(axis=0)
        test_score=np.vstack([ i[1] for i in reses]).mean(axis=0)

        train_fscore=train_score[0:300]
        test_fscore=test_score[0:300]

        train_vscore=train_score[300:350]
        test_vscore=test_score[300:350]












    def fig_out(self,x_axis_data, log_path, d,avg_d=None,single_score=None, other_scores=None,accs=None): 
        colors={
            "cosine attack":"r",
            "grad diff":"g",
            "loss based":"b",
            "grad norm":(242/256, 159/256, 5/256),
            "lira":"y",
            "log_lira":"k",
            "lira_loss":'purple'
                }
        labels_per_epoch = {
            "cosine attack":"Grad-Cosine",
            "grad diff":"Grad-Diff",
            "loss based":"Blackbox-Loss",
            "grad norm":"Grad-Norm"
        }
        labels_temporal = {
            "cosine attack":"Avg-Cosine",
            "loss based":"Loss-Series",
            "lira":"FedMIA-II",
            "lira_loss":"FedMIA-I"
        }
        fig = plt.figure(figsize=(6.5, 6.5), dpi=200)
        fig.subplots_adjust(top=0.91,
                            bottom=0.160,
                            left=0.180,
                            right=0.9,
                            hspace=0.2,
                            wspace=0.2)
        for k in labels_per_epoch.keys():
            print(k, d[k])
            plt.plot(x_axis_data[0:len(d[k])], d[k], linewidth=1, label=labels_per_epoch[k], color=colors[k])
        plt.legend(loc=3)  

        plt.xlim(-2, 305)
        my_x_ticks = np.arange(0, 302, 50)
        plt.xticks(my_x_ticks,size=14)
        if avg_d:
            for k in labels_temporal.keys():
                if avg_d[k]:    
                    plt.hlines([avg_d[k]["0.001"]],xmin=0,xmax=300,label=labels_temporal[k],color=colors[k])

        plt.legend(prop={'size': 10})
        plt.xlabel('Epoch',fontsize=14,fontdict={'size': 14})
        plt.ylabel('TPR@FPR=0.001',fontsize=14,fontdict={'size': 14})
        plt.grid(axis='both')

        pdf_path=self.PATH.split("/")[0:-1]
        pdf_path="/".join(pdf_path)+f"/attack_fig_{self.select_mode}_{self.select_method}_n{self.SHADOW_NUM}_s{self.SEED}.pdf"
        
        print('fig saved in', pdf_path)
        plt.savefig(pdf_path)

        log_path=log_path+f"/attack_score_{self.select_mode}_{self.select_method}_n{self.SHADOW_NUM}_s{self.SEED}.log"
        with open(log_path,"w") as f:
            json.dump({"avg_d":avg_d,"single_score":single_score,"other_scores":other_scores,"accs":accs},f, indent=4)

class MBA:
    def __init__(self,args):
        self.args=args
        self.attack_client_idx=0
        self.client_model=self.load_client_model()
        self.server_model = self.load_server_model()
        self.data_size=1000
        self.thre_data_size=50
        self.client_dataloader,self.thre_dataloader = self.load_client_dataloader(self.data_size+self.thre_data_size)
        self.other_dataloader = self.load_otherloader(data_size=self.data_size)

    def load_client_model(self):
        model_folder = self.args.model_path + '/' +self.args.model+'/' + self.args.dataset + '/client_model'
        model_file = f"client_{self.attack_client_idx}_{self.args.training_round-self.args.client_num//self.args.participant}.pth"
        model_path = os.path.join(model_folder, model_file)
        model_pth = torch.load(model_path)
        model = _build_public_model(self.args)
        model.load_state_dict(model_pth)
        model.to(self.args.device)
        print(f'load model {model_file}')
        return model
    
    def load_server_model(self):
        model_folder = self.args.model_path + '/' +self.args.model+'/' + self.args.dataset + '/server_model'
        model_file = f"server_{self.args.training_round-1}.pth"
        model_path = os.path.join(model_folder, model_file)
        model_pth = torch.load(model_path)
        model = _build_public_model(self.args)
        model.load_state_dict(model_pth)
        model.to(self.args.device)
        print(f'load model {model_file}')
        return model
        

    def load_client_dataloader(self, total_data_size):
        args = self.args
        random.seed(self.args.random_seed)
        
        cal_size = self.thre_data_size
        eval_size = self.data_size
        
        assert eval_size > 0, "Total data size must be larger than threshold data size!"

        data_path = args.data_path + '/' + args.dataset + '/' + args.model + '/' + args.data_split
        datas, labels = load_npz_data(data_path + '/train_non_iid.npz')
        non_member_data, non_member_label = load_npz_data(data_path + '/test_non_iid.npz')

        c_idx = self.attack_client_idx
        client_member_data = datas[c_idx]
        client_member_label = labels[c_idx]
        client_non_member_data = non_member_data[c_idx]
        client_non_member_label = non_member_label[c_idx]

        indices_member = np.arange(len(client_member_data))
        random.shuffle(indices_member)
        indices_non_member = np.arange(len(client_non_member_data))
        random.shuffle(indices_non_member)

        sampled_idx_m_total = indices_member[:total_data_size]
        sampled_idx_nm_total = indices_non_member[:total_data_size]

        
        cal_idx_m = sampled_idx_m_total[:cal_size]
        cal_idx_nm = sampled_idx_nm_total[:cal_size]
        
        eval_idx_m = sampled_idx_m_total[cal_size:]
        eval_idx_nm = sampled_idx_nm_total[cal_size:]

        def create_dataset(idx_m, idx_nm):
            d_m = client_member_data[idx_m]
            l_m = client_member_label[idx_m]
            d_nm = client_non_member_data[idx_nm]
            l_nm = client_non_member_label[idx_nm]
            
            data_concat = np.concatenate((d_m, d_nm))
            label_concat = np.concatenate((l_m, l_nm))
            membership = [1] * len(d_m) + [0] * len(d_nm)
            
            return ClientDatasetWithMember(data_concat, label_concat, membership)

        dataset_cal = create_dataset(cal_idx_m, cal_idx_nm)
        dataset_eval = create_dataset(eval_idx_m, eval_idx_nm)

        print(f"Loaded Datasets: Eval Size={len(dataset_eval)} (Target), Calibration Size={len(dataset_cal)} (Threshold)")

        thre_dataloader = DataLoader(dataset_cal, batch_size=1, shuffle=True)
        client_dataloader = DataLoader(dataset_eval, batch_size=1, shuffle=True)

        return client_dataloader, thre_dataloader
    
    def load_otherloader(self,data_size):
        args = self.args
        data_path = args.data_path + '/'+args.dataset+'/'+args.model+'/'+args.data_split
        datas, labels = load_npz_data(data_path + '/train_non_iid.npz')
        data = [client_data for client_idx, client_data in enumerate(datas) if client_idx == self.attack_client_idx]
        label = [client_label for client_idx, client_label in enumerate(labels) if client_idx == self.attack_client_idx]
        data = np.concatenate(data)
        label = np.concatenate(label)
        data = np.array(data)
        label = np.array(label)
        non_member_datas, non_member_labels = load_npz_data(data_path + '/test_non_iid.npz')
        non_member_data,non_member_label =  np.delete(non_member_datas,self.attack_client_idx,axis=0),np.delete(non_member_labels,self.attack_client_idx,axis=0)
        non_member_data=np.concatenate(non_member_data)
        non_member_label=np.concatenate(non_member_label)
        non_member_data = np.array(non_member_data)
        non_member_label=np.array(non_member_label)
        nonmember_total_samples = len(non_member_label)
        member_total_samples = len(label)
        min_number = min(nonmember_total_samples,member_total_samples)
        random.seed(self.args.random_seed)
        actual_size = min(member_total_samples, data_size)

        if member_total_samples < data_size:
            print(f"[Warning] 请求 public_data={data_size}, 但数据集只有 {member_total_samples}。将使用全部数据。")

        random_indices = np.random.choice(member_total_samples, actual_size, replace=False)
        sampled_data,sampled_label = data[random_indices],label[random_indices]
        sampled_data_n, sampled_label_n = non_member_data[random_indices], non_member_label[random_indices]
        sampled_data = np.concatenate((sampled_data, sampled_data_n))
        sampled_label = np.concatenate((sampled_label,sampled_label_n))
        membership = data_size*[1]+data_size*[0]
        dataset = ClientDatasetWithMember(sampled_data, sampled_label,membership)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        return dataloader
    

    def clipDataTopX(self,dataToClip, top=3):
        res = [sorted(s, reverse=True)[0:top] for s in
            dataToClip]
        return np.array(res)

    def MetricBasedAttacking(self,pre_vectors_from_targetmodel, member_labels, classification_y,
                         metric="max"):
        pre_vectors_from_targetmodel = np.array(pre_vectors_from_targetmodel)
        classification_y = np.array(classification_y)
        if (metric == "max"):
            pre_vectors_from_targetmodel = self.clipDataTopX(pre_vectors_from_targetmodel, top=1)
            metrics = pre_vectors_from_targetmodel

        if (metric == "loss"):
            log_loss_list = []
            temp_labels = range(len(pre_vectors_from_targetmodel[0]))
            for i in range(len(member_labels)):
                log_loss_list.append(log_loss([classification_y[i]], [pre_vectors_from_targetmodel[i]],
                                            labels=temp_labels))
            metrics = log_loss_list

        if (metric == "sd"):
            sd = np.std(pre_vectors_from_targetmodel, axis=1)
            metrics = sd

        if (metric == "entropy"):
            negative_logs = -np.log(
                np.maximum(pre_vectors_from_targetmodel, 1e-30))
            entropys = np.sum(np.multiply(pre_vectors_from_targetmodel, negative_logs),
                            axis=1)
            metrics = entropys

        if (metric == "mentropy"):
            neg_log_probs = -np.log(np.maximum(pre_vectors_from_targetmodel, 1e-30))
            reverse_probs = 1 - pre_vectors_from_targetmodel
            neg_log_reverse_probs = -np.log(np.maximum(reverse_probs, 1e-30))
            modified_probs = np.copy(pre_vectors_from_targetmodel)
            modified_probs[range(classification_y.shape[0]), classification_y] = reverse_probs[
                range(classification_y.shape[0]), classification_y]
            modified_log_probs = np.copy(neg_log_reverse_probs)
            modified_log_probs[range(classification_y.shape[0]), classification_y] = neg_log_probs[
                range(classification_y.shape[0]), classification_y]
            mentropys = np.sum(np.multiply(modified_probs, modified_log_probs), axis=1)
            metrics = mentropys

        if (metric == "correctness"):
            pre_class_label = np.argmax(pre_vectors_from_targetmodel, axis=1)
            temp = pre_class_label - classification_y
            pre_member_label = np.int64(
                temp == 0)
            metrics = pre_member_label

        return metrics
    

    def attack(self,metric_flag):
        outputs,predicts,ground_truth=[],[],[]
        private_model = self.client_model
        private_model.eval()
        for data, label,member in tqdm(self.client_dataloader,desc='metric based attack'):
            data = data.to(self.args.device)
            label = label.to(self.args.device)
            output = private_model(data)
            confidences = torch.softmax(output,dim=1)
            confidences = confidences.squeeze(0)
            outputs.append(confidences.detach().cpu())
            predict = torch.max(confidences)
            predicts.append(predict.cpu().item())
            ground_truth.append(member.item())
        metrics = self.MetricBasedAttacking(outputs,predicts,ground_truth,metric=metric_flag)
        t_outputs,t_predicts,t_ground_truth=[],[],[]
        for data, label,member in tqdm(self.thre_dataloader,desc='thresold metrics compute'):
            data = data.to(self.args.device)
            label = label.to(self.args.device)
            output = private_model(data)
            confidences = torch.softmax(output,dim=1)
            confidences = confidences.squeeze(0)
            t_outputs.append(confidences.detach().cpu())
            predict = torch.max(confidences)
            t_predicts.append(predict.cpu().item())
            t_ground_truth.append(member.item())
        t_metrics = self.MetricBasedAttacking(t_outputs,t_predicts,t_ground_truth,metric=metric_flag)
        print(f'attack type:{metric_flag},attack Perspective:server')
        tpr = ROC_AUC_Result_logshow(ground_truth,metrics,False)
        tpr = ROC_AUC_Result_logshow(ground_truth,metrics,True)
        acc = utils.calculate_acc(metrics,ground_truth,t_metrics,t_ground_truth,'best_acc')
        acc = utils.calculate_acc(metrics,ground_truth,t_metrics,t_ground_truth,'percentile',75)


    def attack_client(self,metric_flag):
        outputs,predicts,ground_truth=[],[],[]
        private_model = self.server_model
        private_model.eval()
        for data, label,member in tqdm(self.other_dataloader,desc='metric based attack'):
            data = data.to(self.args.device)
            label = label.to(self.args.device)
            output = private_model(data)
            confidences = torch.softmax(output,dim=1)
            confidences = confidences.squeeze(0)
            outputs.append(confidences.detach().cpu())
            predict = torch.max(confidences)
            predicts.append(predict.cpu().item())
            ground_truth.append(member.item())
        metrics = self.MetricBasedAttacking(outputs,predicts,ground_truth,metric=metric_flag)
        print(f'attack type:{metric_flag},attack Perspective:client')
        tpr = ROC_AUC_Result_logshow(ground_truth,metrics,False)
        tpr = ROC_AUC_Result_logshow(ground_truth,metrics,True)

class EnhancedMIA:
    def __init__(self,args):
        self.args=args
        self.attack_client_idx=0
        self.client_model=self.load_client_model()
        self.data_size=300
        self.client_dataloader = self.load_client_dataloader(data_size=self.data_size)
        self.other_dataloader = self.load_otherloader(data_size=self.data_size*2)
        self.num_distilled = 16
        self.train_flag=True
        self.distill_epoch=50

    def load_client_model(self):
        model_folder = self.args.model_path + '/' +self.args.model+'/' + self.args.dataset + '/client_model'
        model_file = f"client_{self.attack_client_idx}_{self.args.training_round-4}.pth"
        model_path = os.path.join(model_folder, model_file)
        model_pth = torch.load(model_path)
        model = _build_public_model(self.args)
        model.load_state_dict(model_pth)
        model.to(self.args.device)
        print(f'load model {model_file}')
        return model
    
    def load_server_model(self):
        model_folder = self.args.model_path + '/' +self.args.model+'/' + self.args.dataset + '/server_model'
        model_file = f"server_{self.args.training_round-1}.pth"
        model_path = os.path.join(model_folder, model_file)
        model_pth = torch.load(model_path)
        model = _build_public_model(self.args)
        model.load_state_dict(model_pth)
        model.to(self.args.device)
        print(f'load model {model_file}')
        return model
    
    def load_client_dataloader(self, data_size):
        args = self.args
        random.seed(self.args.random_seed)

        data_path = args.data_path + '/'+args.dataset+'/'+args.model+'/'+args.data_split
        datas, labels = load_npz_data(data_path + '/train_non_iid.npz')
        non_member_data, non_member_label = load_npz_data(data_path + '/test_non_iid.npz')

        all_indices = np.arange(len(datas[self.attack_client_idx]))
        random.shuffle(all_indices)
        sampled_indices1 = all_indices[:int(data_size)]
        sampled_indices2 = all_indices[:int(data_size)]

        sampled_data1 = datas[self.attack_client_idx][sampled_indices1]
        sampled_label1 = labels[self.attack_client_idx][sampled_indices1]
        sampled_data2 = non_member_data[self.attack_client_idx][sampled_indices2]
        sampled_label2 = non_member_label[self.attack_client_idx][sampled_indices2]

        assert len(sampled_data1) == len(sampled_label1), "Sampled data and labels are mismatched!"
        assert len(sampled_data2) == len(sampled_label2), "Sampled data and labels are mismatched!"

        membership = [1] * len(sampled_data1) + [0] * len(sampled_data2)
        data = np.concatenate((sampled_data1, sampled_data2))
        label = np.concatenate((sampled_label1, sampled_label2))

        assert len(data) == len(label), "Data and labels have mismatched lengths!"

        dataset = ClientDatasetWithMember(data, label, membership)

        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

        return dataloader
    
    def load_otherloader(self,data_size):
        args = self.args
        data_path = args.data_path + '/'+args.dataset+'/'+args.model+'/'+args.data_split
        non_member_datas, non_member_labels = load_npz_data(data_path + '/test_non_iid.npz')
        non_member_data,non_member_label =  np.delete(non_member_datas,self.attack_client_idx,axis=0),np.delete(non_member_labels,self.attack_client_idx,axis=0)
        non_member_data=np.concatenate(non_member_data)
        non_member_label=np.concatenate(non_member_label)
        non_member_data = np.array(non_member_data)
        non_member_label=np.array(non_member_label)
        nonmember_total_samples = len(non_member_label)
        random.seed(self.args.random_seed)
        random_indices = np.random.choice(nonmember_total_samples, data_size, replace=False)
        sampled_data_n, sampled_label_n = non_member_data[random_indices], non_member_label[random_indices]
        membership = data_size*[0]
        dataset = ClientDatasetWithMember(sampled_data_n, sampled_label_n,membership)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        return dataloader
    
    def attack_d(self):
        num_distilled=self.num_distilled
        device=self.args.device
        self.client_model.to(device)
        self.client_model.eval()

        pop_data = []
        pop_targets = []
        for x, y,_ in self.other_dataloader:
            pop_data.append(x)
            pop_targets.append(y)
        pop_data = torch.cat(pop_data, dim=0).to(device)
        pop_targets = torch.cat(pop_targets, dim=0).to(device)

        with torch.no_grad():
            soft_labels = self.client_model(pop_data)

        distilled_models = []
        for i in range(num_distilled):
            model = self._train_distilled_model(
                pop_data, soft_labels, 
                epochs=self.distill_epoch, lr=0.001, 
                seed=i, device=device,train_flag=self.train_flag,model_id=i
            )
            distilled_models.append(model)

        predictions = []
        scores = []
        gt=[]
        distilled_losses = []
        target_losses=[]

        criterion = nn.CrossEntropyLoss(reduction='none')

        for x, y, m in tqdm(self.client_dataloader):
            x = x.to(device)
            y = torch.tensor([y]).to(device)
            gt.append(m.item())

            with torch.no_grad():
                logits_target = self.client_model(x)
                target_loss = criterion(logits_target, y).item() 
                target_losses.append(target_loss)

            one_sample_distilled_losses = []
            for model in distilled_models:
                model.eval()
                with torch.no_grad():
                    logits_d = model(x)
                    loss_d = criterion(logits_d, y).item()
                    one_sample_distilled_losses.append(loss_d)
            distilled_losses.append(one_sample_distilled_losses)

        mean=np.mean(distilled_losses,axis=1)
        std=np.std(distilled_losses,axis=1)
        score=1-scipy.stats.norm.cdf(target_losses,mean,std)
        tpr,auc=ROC_AUC_Result_logshow_with_auc(gt,score,reverse=False)

        return predictions, scores

    def _train_distilled_model(self, inputs, soft_targets, epochs=10, lr=1e-3, seed=0, device='cpu', train_flag=False, model_id=0):
        import os

        model_path = os.path.join(self.args.model_path,self.args.model,self.args.dataset,'enhancedMIA',f'distilled_model_{model_id}.pth')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        torch.manual_seed(seed)
        np.random.seed(seed)

        if hasattr(self.client_model, 'config'):
            student_model = type(self.client_model)(**self.client_model.config)
        else:
            student_model = type(self.client_model)(self.args)

        student_model.to(device)

        if not train_flag:
            if os.path.exists(model_path):
                student_model.load_state_dict(torch.load(model_path, map_location=device))
                student_model.eval()
                print(f'loading model {model_id}')
                return student_model
            else:
                raise FileNotFoundError(f"未找到预训练模型: {model_path}，请先设置 train_flag=True 进行训练并保存。")

        optimizer = torch.optim.Adam(student_model.parameters(), lr=lr)
        criterion = nn.KLDivLoss(reduction='batchmean')

        dataset = TensorDataset(inputs, soft_targets)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        student_model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0

            for x_batch, soft_y_batch in loader:
                x_batch = x_batch.to(device)
                soft_y_batch = soft_y_batch.to(device)

                optimizer.zero_grad()
                logits = student_model(x_batch)
                log_probs = torch.log_softmax(logits, dim=1)
                target_probs = torch.softmax(soft_y_batch, dim=1)

                loss = criterion(log_probs, target_probs)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            print(f"Distill Epoch [{epoch+1}/{epochs}], Avg Loss: {avg_loss:.6f}")

        torch.save(student_model.state_dict(), model_path)
        print(f'training model {model_id}')

        return student_model
    
class CSF18:
    
    def __init__(self, args):
        self.args=args
        self.attack_client_idx=0
        self.client_model=self.load_client_model()
        self.data_size=1000
        self.client_dataloader = self.load_client_dataloader(data_size=self.data_size)

    def load_client_dataloader(self, data_size):
        args = self.args
        random.seed(self.args.random_seed)

        data_path = args.data_path + '/'+args.dataset+'/'+args.model+'/'+args.data_split
        datas, labels = load_npz_data(data_path + '/train_non_iid.npz')
        non_member_data, non_member_label = load_npz_data(data_path + '/test_non_iid.npz')

        all_indices = np.arange(len(datas[self.attack_client_idx]))
        random.shuffle(all_indices)
        sampled_indices1 = all_indices[:int(data_size)]
        sampled_indices2 = all_indices[:int(data_size)]

        sampled_data1 = datas[self.attack_client_idx][sampled_indices1]
        sampled_label1 = labels[self.attack_client_idx][sampled_indices1]
        sampled_data2 = non_member_data[self.attack_client_idx][sampled_indices2]
        sampled_label2 = non_member_label[self.attack_client_idx][sampled_indices2]

        assert len(sampled_data1) == len(sampled_label1), "Sampled data and labels are mismatched!"
        assert len(sampled_data2) == len(sampled_label2), "Sampled data and labels are mismatched!"

        membership = [1] * len(sampled_data1) + [0] * len(sampled_data2)
        data = np.concatenate((sampled_data1, sampled_data2))
        label = np.concatenate((sampled_label1, sampled_label2))

        assert len(data) == len(label), "Data and labels have mismatched lengths!"

        dataset = ClientDatasetWithMember(data, label, membership)

        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        return dataloader
    
    def load_client_model(self):
        model_folder = self.args.model_path + '/' +self.args.model+'/' + self.args.dataset + '/client_model'
        model_file = f"client_{self.attack_client_idx}_{self.args.training_round-self.args.client_num//self.args.participant}.pth"
        model_path = os.path.join(model_folder, model_file)
        model_pth = torch.load(model_path)
        model = _build_public_model(self.args)
        model.load_state_dict(model_pth)
        model.to(self.args.device)
        print(f'load model {model_file}')
        return model


    def attack(self):
        self.client_model.eval()
        
        all_scores = []
        true_memberships = []
        
        device = self.args.device if hasattr(self.args, 'device') else 'cpu'
        self.client_model.to(device)

        with torch.no_grad():
            for data, label, membership in self.client_dataloader:
                data = data.to(device)
                label = label.to(device)
                
                outputs = self.client_model(data)
                
                _, predicted = torch.max(outputs.data, 1)
                batch_scores = (predicted == label).float()
                
                all_scores.extend(batch_scores.cpu().numpy())
                true_memberships.extend(membership.cpu().numpy())
        
        auc,tpr=ROC_AUC_Result_logshow_with_auc(true_memberships,all_scores,False)
        acc=accuracy_score(true_memberships,all_scores)
        print(f"acc:{acc}")

        return {
            "scores": all_scores,
            "labels": true_memberships
        }