# 这个文件是实现三篇baseline的

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
from sklearn.metrics import precision_score, f1_score, recall_score, accuracy_score

"""
《Comprehensive Privacy Analysis of Deep Learning》
该论文设计了多个攻击组件观察模型的输出差异，进行成员推理
包括：
output component
label component
loss component
gradient component（包括conv层和FC层）
encoder component
decoder component
"""


class OutputComponent(nn.Module):
    """
    分析模型的输出和中间层输出
    """
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
    """
    必须经过onehot向量化
    """
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
    """
    损失默认形状为1
    """
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
    """
    原文中kernels写的1000
    """
    def __init__(self, input_channels, next_layer_size, output_size,dropout=0.2):
        super(GradientComponent, self).__init__()
        
        kernels = 4
        
        # 卷积层
        self.conv = nn.Conv2d(input_channels, kernels, (1, next_layer_size), stride=1)
        
        # ReLU 激活函数
        self.relu1 = nn.ReLU()
        
        # Dropout 层
        self.dropout1 = nn.Dropout(p=dropout)
        
        # Flatten 层
        self.flatten = nn.Flatten()
        
        # 全连接层
        self.fc1 = nn.Linear(kernels * next_layer_size * 55, 128)
        
        # ReLU 激活函数
        self.relu2 = nn.ReLU()
        
        # Dropout 层
        self.dropout2 = nn.Dropout(p=dropout)
        
        # 第二个全连接层
        self.fc2 = nn.Linear(128, 64)
        
        # ReLU 激活函数
        self.relu3 = nn.ReLU()
        
        # Dropout 层
        self.dropout3 = nn.Dropout(p=dropout)
    
    def forward(self, x):
        # 卷积层 + ReLU + Dropout
        x = self.conv(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        # Flatten
        x = self.flatten(x)
        
        # 第一个全连接层 + ReLU + Dropout
        x = self.fc1(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        # 第二个全连接层 + ReLU + Dropout
        x = self.fc2(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        
        return x



class EncoderComponent(nn.Module):
    def __init__(self, input_size, hidden_size1=256, hidden_size2=128, hidden_size3=64, output_size=2, dropout=0.2):
        super(EncoderComponent, self).__init__()
        # 定义4个全连接层，并使用Dropout
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
        # 定义两个全连接层和一个Dropout层
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
    """
    component是一个list，表示该方法所包括的组件
    可选择：output,label,loss,gradient_conv,encoder,decoder
    """

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
        """
        模型加载器，将文件夹下面所有的public模型都加载进同一个modulelist中
        :return:
        """

        model_folder = self.args.model_path + '/' +self.args.model+'/' + self.args.dataset + '/server_model'
        model_list = nn.ModuleList()
        # model_files = [f for f in os.listdir(model_folder) if f.startswith('sercer_') and f.endswith('./pth')]
        # model_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        model_files = [f'server_{f}.pth' for f in range(self.args.training_round)]
        model_files = model_files[-model_nums:]
        for model_file in model_files:
            model_path = os.path.join(model_folder, model_file)
            model_pth = torch.load(model_path)  # 加载模型
            model = PublicLayer(self.args)
            model.load_state_dict(model_pth)
            model.to(self.args.device)
            model.eval()
            model_list.append(model)  # 将模型添加到 ModuleList 中
            print(f'load model {model_file}')
        print(f'load {len(model_list)} models')
        return model_list
    
    def load_dataloader(self):
        """
        构建训练的
        :param data_size: 构建出的数据集的大小，其中成员和非成员要求是一样多的，为了数据集的均衡，都是data_size
        :return: dataloader
        """
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
        # 3. 根据随机索引采样数据
        sampled_data1, sampled_label1 = data[random_indices], label[random_indices]
        sampled_data2, sampled_label2 = non_member_data[random_indices], non_member_label[random_indices]
        data = np.concatenate((sampled_data1,sampled_data2))
        label = np.concatenate((sampled_label1,sampled_label2))
        membership = data_size*[1]+data_size*[0]
        dataset = ClientDatasetWithMember(data, label,membership)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        return dataloader
    
    def load_test_loader(self):
        """
        构建测试的数据集
        :param data_size: 构建出的数据集的大小，其中成员和非成员要求是一样多的，为了数据集的均衡，都是data_size
        :return: dataloader
        """
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
        # 3. 根据随机索引采样数据
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
        # print(f"{module.__class__.__name__} 输出大小: {output.shape}")
    
    # def backward_hook_fn(self,module, grad_input, grad_output):
    #     self.gradients[module] = grad_output[0].clone().detach()
    #     # print(f"{module.__class__.__name__} 梯度大小: {grad_output[0].shape}")

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
                    # layer.register_full_backward_hook(self.backward_hook_fn)

            loss_fn = nn.CrossEntropyLoss()
            model.zero_grad(set_to_none=True)  # 仅重置梯度，提高效率
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
        # loss_mse = nn.MSELoss()
        optimizer = torch.optim.SGD([
            {'params': self.loss.parameters(), 'lr': self.lr},
            {'params': self.output.parameters(), 'lr': self.lr},
            {'params': self.label.parameters(), 'lr': self.lr},
            {'params': self.encoder.parameters(), 'lr': self.lr},
            {'params': self.gradient_conv.parameters(), 'lr': self.lr},
        ])
        # optimizer = torch.optim.Adam([
        #     {'params': self.loss.parameters(), 'lr': self.lr},
        #     {'params': self.output.parameters(), 'lr': self.lr},
        #     {'params': self.label.parameters(), 'lr': self.lr},
        #     {'params': self.encoder.parameters(), 'lr': self.lr},
        #     {'params': self.gradient_conv.parameters(), 'lr': self.lr},
        # ], lr=self.lr, betas=(0.9, 0.999), eps=1e-8)
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
                    # mem_predict = torch.sigmoid(mem_predict)
                    mem_predict = torch.softmax(mem_predict,dim=1)
                    # member = member.view(1,-1).float()
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
            # mem_predict = torch.sigmoid(mem_predict)
            # predicted_label = (mem_predict >= 0.5).float()
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
    """
    USENIX2024<Efficient Privacy Auditing in Federated Learning>
    该论文的攻击叫做FreeTrainingAttack
    注意一定要写model.eval()否则会导致loss和confidence的计算不准确
    """

    def __init__(self, args,data_size):
        self.args = args
        self.data_size=data_size
        self.thre_data_size=50
        self.attack_client_idx = 0
        self.client_dataloader,self.thre_dataloader = self.load_client_dataloader(self.data_size+self.thre_data_size)
        self.attack_round = self.args.training_round
        # self.public_model_list = self.public_model_loader()
        self.private_model_list = self.private_model_loader(self.attack_client_idx)
        self.public_dataloader = self.load_public_dataloader(data_size=self.data_size)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        

    def compute_weights(self,t):
        """
        计算时间权重 w_u
        :param t: 当前轮次 t
        :return: 权重列表
        """
        weights = []
        step=self.args.client_num//self.args.participant
        for u in range(1, t + 1,step):
            weight = 6 * (2 * t * u - t**2 + 1) / (t**4 - t**2)
            weights.append(weight)
        return weights


    def compute_slopes(self,losses, t):
        # 将权重wu和置信度cu相乘，计算得到斜率
        weights = self.compute_weights(t)
        slopes = []
        for sample_losses in losses:
            slope = sum(w * c for w, c in zip(weights, sample_losses[:t]))
            slopes.append(slope)
        return slopes


    def compute_threshold(self, losses, rounds):
        # 使用分位数计算斜率
        slopes = self.compute_slopes(losses, rounds)
        threshold = np.percentile(slopes, 50)
        return threshold
    

    def free_training_attack(self,losses,threshold,rounds):
        # 和阈值比较，因为成员的斜率应该更小，所以此处比较用小于
        slopes = self.compute_slopes(losses, rounds)
        predicted_labels = slopes < threshold
        return predicted_labels,slopes
        

    
    def load_client_dataloader(self, total_data_size):
        """
        数据加载与切分
        
        参数:
        - total_data_size: 需要加载的总样本数 (即 self.data_size + self.thre_data_size)
        
        返回:
        - client_dataloader: 用于攻击评估的主数据集 (大小为 total - thre)
        - thre_dataloader:   用于确定阈值的校准数据集 (大小为 self.thre_data_size)
        """
        args = self.args
        random.seed(self.args.random_seed)
        
        # 确定切分大小
        # 校准集大小 (例如 50)
        cal_size = self.thre_data_size
        # 评估集大小 (例如 300)
        eval_size = self.data_size
        
        assert eval_size > 0, "Total data size must be larger than threshold data size!"

        # 1. 加载原始数据 (Members 和 Non-Members)
        data_path = args.data_path + '/' + args.dataset + '/' + args.model + '/' + args.data_split
        # datas: [client_num, data_len, ...]
        datas, labels = load_npz_data(data_path + '/train_non_iid.npz')
        non_member_data, non_member_label = load_npz_data(data_path + '/test_non_iid.npz')

        # 2. 获取当前攻击客户端的数据
        c_idx = self.attack_client_idx
        client_member_data = datas[c_idx]
        client_member_label = labels[c_idx]
        client_non_member_data = non_member_data[c_idx]
        client_non_member_label = non_member_label[c_idx]

        # 3. 生成随机索引并打乱
        # Member 索引
        indices_member = np.arange(len(client_member_data))
        random.shuffle(indices_member)
        # Non-Member 索引
        indices_non_member = np.arange(len(client_non_member_data))
        random.shuffle(indices_non_member)

        # 4. 采样总数据 (Members 和 Non-Members 各取 total_data_size 个)
        # 注意：这里假设源数据量足够，如果不够需要加 min() 保护
        sampled_idx_m_total = indices_member[:total_data_size]
        sampled_idx_nm_total = indices_non_member[:total_data_size]

        # 5. 切分索引: 校准集 (Calibration) vs 评估集 (Evaluation)
        
        # --- 校准集索引 (前 thre_data_size 个) ---
        cal_idx_m = sampled_idx_m_total[:cal_size]
        cal_idx_nm = sampled_idx_nm_total[:cal_size]
        
        # --- 评估集索引 (剩下的) ---
        eval_idx_m = sampled_idx_m_total[cal_size:]
        eval_idx_nm = sampled_idx_nm_total[cal_size:]

        # 6. 构建数据集辅助函数
        def create_dataset(idx_m, idx_nm):
            # 提取 Member 数据
            d_m = client_member_data[idx_m]
            l_m = client_member_label[idx_m]
            # 提取 Non-Member 数据
            d_nm = client_non_member_data[idx_nm]
            l_nm = client_non_member_label[idx_nm]
            
            # 合并
            data_concat = np.concatenate((d_m, d_nm))
            label_concat = np.concatenate((l_m, l_nm))
            # 生成成员性标签: Member=1, Non-Member=0
            membership = [1] * len(d_m) + [0] * len(d_nm)
            
            return ClientDatasetWithMember(data_concat, label_concat, membership)

        # 7. 创建 Dataset 对象
        dataset_cal = create_dataset(cal_idx_m, cal_idx_nm)     # 校准用 (50 + 50)
        dataset_eval = create_dataset(eval_idx_m, eval_idx_nm)  # 评估用 (300 + 300)

        print(f"Loaded Datasets: Eval Size={len(dataset_eval)} (Target), Calibration Size={len(dataset_cal)} (Threshold)")

        # 8. 创建 DataLoader
        # 为了方便测试设置 batch=1
        thre_dataloader = DataLoader(dataset_cal, batch_size=1, shuffle=True)
        client_dataloader = DataLoader(dataset_eval, batch_size=1, shuffle=True)

        # 返回顺序：先返回主评估集，再返回校准集
        return client_dataloader, thre_dataloader

    def load_public_dataloader(self, data_size):
        """
        构建除了攻击客户端之外其他客户端的数据集
        """
        args = self.args
        random.seed(self.args.random_seed)
        data_path = args.data_path + '/'+args.dataset+'/'+args.model+'/'+args.data_split
        datas, labels = load_npz_data(data_path + '/train_non_iid.npz')
        data = [client_data for client_idx, client_data in enumerate(datas) if client_idx == self.attack_client_idx]
        label = [client_label for client_idx, client_label in enumerate(labels) if client_idx == self.attack_client_idx]
        # data, label = np.delete(datas,self.attack_client_idx,axis=0), np.delete(labels,self.attack_client_idx,axis=0)
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
        random_indices = np.random.choice(total_samples, data_size, replace=False)
        # 3. 根据随机索引采样数据
        sampled_data1, sampled_label1 = data[random_indices], label[random_indices]
        sampled_data2, sampled_label2 = non_member_data[random_indices], non_member_label[random_indices]
        data = np.concatenate((sampled_data1,sampled_data2))
        label = np.concatenate((sampled_label1,sampled_label2))
        membership = data_size*[1]+data_size*[0]
        dataset = ClientDatasetWithMember(data, label,membership)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        return dataloader

    def public_model_loader(self):
        """
        模型加载器，将文件夹下面所有的public模型都加载进同一个modulelist中
        :return:
        """

        model_folder = self.args.model_path + '/' +self.args.model+'/' + self.args.dataset + '/server_model'
        model_list = nn.ModuleList()
        # model_files = [f for f in os.listdir(model_folder) if f.startswith('sercer_') and f.endswith('./pth')]
        # model_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        model_files = [f'server_{f}.pth' for f in range(self.args.training_round)]
        for model_file in model_files:
            model_path = os.path.join(model_folder, model_file)
            model_pth = torch.load(model_path)  # 加载模型
            model = PublicLayer(self.args)
            model.load_state_dict(model_pth)
            model.to(self.args.device)
            model_list.append(model)  # 将模型添加到 ModuleList 中
            print(f'load model {model_file}')
        print(f'load {len(model_list)} models')
        return model_list
    
    
    def private_model_loader(self,client_idx):
        """
        模型加载器，将文件夹下面所有的public模型都加载进同一个modulelist中
        :return:
        """
        model_folder = self.args.model_path + '/' +self.args.model+'/' + self.args.dataset + '/client_model/'
        model_list = nn.ModuleList()
        model_files = [
            f'client_{client_idx}_{f}.pth'
            for f in range(self.args.training_round)
            if os.path.exists(model_folder+f'client_{client_idx}_{f}.pth')
        ]
        for model_file in model_files:
            model_path = os.path.join(model_folder, model_file)
            model_pth = torch.load(model_path)  # 加载模型
            model = PublicLayer(self.args)
            model.load_state_dict(model_pth)
            model.to(self.args.device)
            model.eval()
            model_list.append(model)  # 将模型添加到 ModuleList 中
            print(f'load model {model_file}')
        print(f'load {len(model_list)} models')
        return model_list


    def client_attack(self):
        """
        客户端视角发起的攻击，对global model使用FTA
        数据集使用除自己以外其他客户端数据
        """
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
        # print('confidence:')
        # slopes = self.compute_slopes(conf, self.attack_round)
        # tpr = ROC_AUC_Result_logshow(ground_truth,slopes,False)

        self.plot_histograms(slopes,ground_truth)
        return tpr
    
    def server_attack(self):
        """
        服务器端视角发起的攻击，对单独的客户端使用FTA
        数据集使用被攻击者的数据集，也就是单一客户端
        """
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
        # t_losses=[]
        # t_ground_truth=[]
        # for data, label,member in tqdm(self.thre_dataloader,desc='threshold dataloader'):
        #     temp1 = []
        #     temp2=[]
        #     for private_model in self.private_model_list:
        #         data = data.to(self.args.device)
        #         label = label.to(self.args.device)
        #         private_model.eval()
        #         output = private_model(data)
        #         loss = self.loss_fn(output,label)
        #         confidences = torch.softmax(output,dim=1)
        #         confidence = torch.max(confidences)
        #         temp1.append(loss.detach().cpu().item())
        #         temp2.append(confidence.cpu().item())
        #     t_losses.append(temp1)
        #     t_ground_truth.append(member.item())
        # print("server attack")
        # print('loss:')
        # t_slopes = self.compute_slopes(t_losses, self.attack_round)
        # t_slopes = [-i for i in t_slopes]
        tpr = ROC_AUC_Result_logshow(ground_truth,slopes,False)
        # acc = utils.calculate_acc(slopes,ground_truth,t_slopes,t_ground_truth,'best_acc')
        metrics=utils.get_best_metrics(ground_truth,slopes)
        # print('confidence:')
        # slopes = self.compute_slopes(conf, self.attack_round)
        # tpr = ROC_AUC_Result_logshow(ground_truth,slopes,False)

        # self.plot_histograms(slopes,ground_truth)
        return tpr
    
    
    def attack(self,type):
        """
        :type:攻击视角 client server
        """
        if type == 'client':
            self.client_attack()
        elif type == 'server':
            self.server_attack()
        else:
            print('wrong type')
        
    
    def plot_histograms(self, x, y):
        """
        绘制不同标签的分布直方图
        :param x: 数据列表或数组
        :param y: 标签列表或数组，取值为 0 或 1
        """
        # 创建一个图形和坐标轴
        save_dir = './plots'
        fig, ax = plt.subplots(figsize=(10, 6))

        # 绘制标签为 0 的数据的直方图
        ax.hist([xi for xi, yi in zip(x, y) if yi == 0], bins=30, alpha=0.5, label='nonmember', color='blue', edgecolor='black')

        # 绘制标签为 1 的数据的直方图
        ax.hist([xi for xi, yi in zip(x, y) if yi == 1], bins=30, alpha=0.5, label='member', color='red', edgecolor='black')

        # 添加标题和轴标签
        ax.set_title('Distribution of Data by Labels')
        ax.set_xlabel('Data Value')
        ax.set_ylabel('Frequency')

        # 显示图例
        ax.legend()
        plt.savefig(f'{save_dir}/USENIX2024.png', dpi=300)
        # 显示图形
        plt.cla()


class ICLR2023:
    """
    全历史 ICLR2023 攻击 (针对完整模型保存版)
    
    [cite_start]Paper Reference: [cite: 194-203] "Attacks using multiple communication rounds"
    """

    def __init__(self, args, attack_client_idx, total_eval_size=600):
        self.args = args
        self.attack_client_idx = attack_client_idx
        self.device = args.device
        self.data_size = total_eval_size
        
        # 1. 实例化一个模型模板 (用于加载权重计算梯度)
        # 请确保 TargetModel 是你训练时使用的那个包含所有层的类
        self.model_template = PublicLayer(args).to(self.device)
        
        # 2. 准备数据
        self.client_dataloader= self.load_client_dataloader(total_eval_size)
        self.eval_loader = self.client_dataloader
        
        self.eval_scores_sum = np.zeros(len(self.eval_loader.dataset))

    def load_client_dataloader(self, data_size):
        """
        数据加载
        数据来源是攻击者控制的0号客户端，使用训练集和测试集构建成员和非成员作为训练集。
        """
        args = self.args
        random.seed(self.args.random_seed)

        # 加载训练数据和测试数据
        data_path = args.data_path + '/'+args.dataset+'/'+args.model+'/'+args.data_split
        datas, labels = load_npz_data(data_path + '/train_non_iid.npz')
        non_member_data, non_member_label = load_npz_data(data_path + '/test_non_iid.npz')

        # 生成随机索引
        all_indices = np.arange(len(datas[self.attack_client_idx]))  # 获取所有的索引
        random.shuffle(all_indices)  # 打乱索引
        sampled_indices1 = all_indices[:int(data_size)]  # 从训练数据中采样
        sampled_indices2 = all_indices[:int(data_size)]  # 从非成员测试数据中采样

        # 使用索引提取数据和标签
        sampled_data1 = datas[self.attack_client_idx][sampled_indices1]
        sampled_label1 = labels[self.attack_client_idx][sampled_indices1]
        sampled_data2 = non_member_data[self.attack_client_idx][sampled_indices2]
        sampled_label2 = non_member_label[self.attack_client_idx][sampled_indices2]

        # 确保标签和数据的长度匹配
        assert len(sampled_data1) == len(sampled_label1), "Sampled data and labels are mismatched!"
        assert len(sampled_data2) == len(sampled_label2), "Sampled data and labels are mismatched!"

        # 创建成员性标签
        membership = [1] * len(sampled_data1) + [0] * len(sampled_data2)
        data = np.concatenate((sampled_data1, sampled_data2))
        label = np.concatenate((sampled_label1, sampled_label2))

        # 确保数据与标签的维度一致
        assert len(data) == len(label), "Data and labels have mismatched lengths!"

        # 构建数据集
        dataset = ClientDatasetWithMember(data, label, membership)

        # 为了方便测试设置batch=1
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        return dataloader

    def load_models_and_get_update(self, round_idx):
        """
        加载第 round_idx 轮的模型，并计算 ΔΓ = W_client - W_server
        """
        # 路径构建 (请根据你实际的保存路径修改)
        # Server Model Path
        server_path = os.path.join(
            self.args.model_path, self.args.model, self.args.dataset, 
            'server_model', f'server_{round_idx}.pth'
        )
        
        # Client Model Path (名字没改，指向 client_model 文件夹)
        # 假设文件名格式是 client_{idx}_{round}.pth
        client_path = os.path.join(
            self.args.model_path, self.args.model, self.args.dataset, 
            'client_model', f'client_{self.attack_client_idx}_{round_idx}.pth'
        )
        
        # 检查文件是否存在
        if not os.path.exists(server_path):
            raise FileNotFoundError(f"Server model not found: {server_path}")
        if not os.path.exists(client_path):
            # 如果这轮该客户端没参与训练，可能就没有文件
            raise FileNotFoundError(f"Client model not found: {client_path}")

        try:
            # 1. 加载 Global Model (W_S)
            # 使用 model_template 加载权重
            server_state = torch.load(server_path, map_location=self.device)
            self.model_template.load_state_dict(server_state)
            self.model_template.eval()
            
            # 将 Global Model 参数展平为向量
            # 必须 detach，否则会占用计算图内存
            server_vec = nn.utils.parameters_to_vector(self.model_template.parameters()).detach()
            
            # 2. 加载 Client Model (W_C)
            # 为了计算差值，我们需要临时加载 Client 权重
            client_state = torch.load(client_path, map_location=self.device)
            
            # 这里有个技巧：我们不需要实例化两个模型对象。
            # 我们可以先加载 Server 算完 vector，再加载 Client 算 vector。
            # 但为了后续计算梯度，我们需要保留 Server Model 的状态在 self.model_template 中。
            
            # 所以，先用一个临时变量或者重新 load 一次来获取 Client Vector
            self.model_template.load_state_dict(client_state)
            client_vec = nn.utils.parameters_to_vector(self.model_template.parameters()).detach()
            
            # 3. 计算 ΔΓ (Update Vector)
            # ΔΓ = W_client - W_server
            delta_gamma = client_vec - server_vec
            
            # 4. 恢复 self.model_template 为 Server Model
            # 因为论文攻击是计算样本在 Global Model 上的梯度 ∇L(x; W_S)
            self.model_template.load_state_dict(server_state)
            
            return self.model_template, delta_gamma

        except RuntimeError as e:
            print(f"[Load Error] 模型结构不匹配或显存不足: {e}")
            return None, None

    def compute_scores_for_loader(self, loader, model, update_vec):
        """
        计算 loader 中所有样本的梯度与 update_vec 的余弦相似度
        """
        scores = []
        update_vec_np = update_vec.cpu().numpy()
        norm_update = np.linalg.norm(update_vec_np)
        
        if norm_update == 0:
            return np.zeros(len(loader.dataset))

        criterion = nn.CrossEntropyLoss()

        # 遍历数据
        for data, target, _ in loader:
            # 针对 Batch 中的每个样本单独计算梯度 (Per-sample gradient)
            # 这非常慢，但符合论文定义。如果显存够大，可以尝试 Opacus 库加速。
            for i in range(len(data)):
                img = data[i:i+1].to(self.device)
                lbl = target[i:i+1].to(self.device)
                
                # 清空梯度
                model.zero_grad()
                
                # 开启梯度记录 (哪怕是 eval 模式)
                for p in model.parameters(): 
                    p.requires_grad = True
                
                # Forward
                out = model(img)
                loss = criterion(out, lbl)
                
                # Backward
                loss.backward()
                
                # 提取完整模型的梯度并展平
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
                        # 计算余弦相似度
                        sim = np.dot(grad_vec, update_vec_np) / (norm_grad * norm_update)
                        scores.append(sim)
                else:
                    scores.append(0.0)
                    
        return np.array(scores)

    def attack(self):
        """
        执行攻击主循环
        """
        start_round = 0 
        end_round = self.args.training_round
        valid_rounds = 0
        
        print(f"Starting Attack on WHOLE models ({start_round} -> {end_round})...")

        # 步长可调整，比如每隔几轮采一次样以节省时间
        step = self.args.client_num//self.args.participant

            
        for r in tqdm(range(start_round, end_round, step), desc="Attacking Rounds"):
            try:
                # 1. 加载模型并计算差值
                model, delta_gamma = self.load_models_and_get_update(r)
                
                if model is None: continue 
                
                
                # 3. 计算 Evaluation Set 分数
                eval_scores = self.compute_scores_for_loader(self.eval_loader, model, delta_gamma)
                self.eval_scores_sum += eval_scores
                
                valid_rounds += 1
                
            except FileNotFoundError:
                # print(f"Round {r} files not found, skipping.")
                continue
            except Exception as e:
                print(f"Error in round {r}: {e}")
                continue

        if valid_rounds == 0:
            print("Error: No valid rounds found!")
            return 0

        # --- 计算平均分 ---
        final_eval_scores = self.eval_scores_sum / valid_rounds


        # 提取 Ground Truth (假设 loader 里的 member 标签是第三个返回值)
        eval_gt = []
        for _, _, m in self.eval_loader:
            eval_gt.extend(m.numpy())
        eval_gt = np.array(eval_gt)
        auc = ROC_AUC_Result_logshow(eval_gt, final_eval_scores,True) # 注意：sklearn需要(y_true, y_score)
        metrics = utils.get_best_metrics(eval_gt, final_eval_scores)
        return auc

class Arxiv2025:
    def __init__(self,args,test_size):
        self.args = args
        self.test_size=test_size
        self.MODE = 'test'
        self.attack_modes=["cosine attack","grad diff","loss based","grad norm"]
        self.epochs=list(range(4,self.args.training_round,self.args.client_num//self.args.participant))
        self.p_folder=args.model_path + '/' + args.model+'/'+ args.dataset + '/our_model/arxiv/'
        self.PATH=self.p_folder+"/client_{}_round_{}.pkl"
        self.p=self.PATH
        self.save_dir='log_file/'+args.model+'/'+args.dataset+'/'+'arxiv/'
        self.device = self.args.device
        self.SEED = args.random_seed
        self.MAX_K=20
        self.mix_length = 1800
        self.select_mode=1 #or 1
        self.select_method='outlier' # outlier
        self.SHADOW_NUM=4 # OR 4

        self.data_size=500
        path_exists(self.save_dir)

    @ torch.no_grad()
    def hinge_loss_fn(self,x,y):
        x,y=copy.deepcopy(x).cuda(),copy.deepcopy(y).cuda()
        mask=torch.eye(x.shape[1],device="cuda")[y].bool()
        tmp1=x[mask]
        x[mask]=-1e10
        tmp2=torch.max(x,dim=1)[0]
        # print(tmp1.shape,tmp2.shape)
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
        # print('target_val_score.shape:',target_val_score.shape)
        # indices = random.sample([i for i in range(0,target_val_score.shape[0])], target_train_score.shape[0])
        # target_val_score = torch.index_select(target_val_score, 0, torch.tensor(indices))
        # print('after sampling target_val_score.shape:',target_val_score.shape)


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
        """
        LIRA (Label Inference via Reconstruction Attack) 攻击函数，用于对模型进行攻击，计算损失、梯度、并进行分析。
        该攻击使用不同的攻击模式（如cosine距离、梯度差异、损失）来进行推断。
        
        :param f: 文件格式字符串，包含模型文件路径
        :param epch: 当前训练的epoch
        :param K: 训练结果的数量（即攻击时的client数量）
        :param save_dir: 保存结果的目录
        :param extract_fn: 可选的提取函数，用于从模型中提取特定信息
        :param attack_mode: 攻击模式（"cos"、"diff"、"loss"等）
        """
        print('******************************************************')
        print('************','Epch:',epch,' attack_mode:',attack_mode,'**************')
        print('******************************************************')
        accs=[]
        training_res=[]
        for i in range(K):
            filepath = f.format(i, epch)
            if os.path.exists(filepath):
                training_res.append(torch.load(filepath))
            # 否则静默跳过
                accs.append(training_res[-1]["test_acc"])
        
        target_idx=self.args.arxiv_client[0]
        val_idx = self.args.arxiv_client[1]
        target_res=training_res[target_idx]
        shadow_res=training_res[val_idx:]
        #print(target_res["train_cos"])
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
        # print('shadow_train_losses_stack:',shadow_train_losses_stack.shape)
        print('attack_mode:',attack_mode)

        if self.select_mode == 1 and attack_mode =='cos':
            # print('***********first in*************')
            tmps=[]
            means=[]
            client_ids=[]
            
            if self.select_method == 'outlier':
                # shadow_mdm_stack = np.vstack(shadow_train_losses_stack, shadow_test_losses_stack)
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
                    # sel_mdm = np.sort(shadow_test_losses_stack[:,j])[2:3+SHADOW_NUM]
                    # sel_mdm = shadow_test_losses_stack[:,j][mask]
                    
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

        ## 计算均值和方差，以备分布估计
        if attack_mode != 'cos'or self.select_mode == 0 or (self.select_method != 'mean_per' and self.select_method != 'outlier'):
            
            train_mu_out=shadow_train_losses_stack.mean(axis=0)
            train_var_out=shadow_train_losses_stack.std(axis=0)+1e-8

            test_mu_out=shadow_test_losses_stack.mean(axis=0)
            test_var_out=shadow_test_losses_stack.std(axis=0)+1e-8


        train_l_out=scipy.stats.norm.cdf(target_train_loss,train_mu_out,train_var_out)[:self.data_size]
        test_l_out=scipy.stats.norm.cdf(target_test_loss,test_mu_out,test_var_out)[:self.data_size]
        # auc,log_auc,tprs=self.plot_auc("lira",torch.tensor(test_l_out),torch.tensor(train_l_out),epch)
        result = list(np.concatenate((train_l_out,test_l_out),axis=0))
        gt = [1]*train_l_out.shape[0]+[0]*test_l_out.shape[0]
        auc,tpr = ROC_AUC_Result_logshow_with_auc(gt,result,True)
        metircs= utils.get_best_metrics(gt,result)
        # return accs,tprs,auc,log_auc,(train_l_out,test_l_out)
        return auc,tpr,(train_l_out,test_l_out)

    def cos_attack(self,f,K,epch,attack_mode,extract_fn=None):
        # return tpr,auc,(train_liratios, val_liratios)
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
                # print('mix_test_loss shape:',mix_test_loss.shape)
                val_liratios = mix_test_loss
            # print(val_liratios)

            val_liratios=np.array([ i.cpu().item() for i in val_liratios ])
            val_liratios = np.nan_to_num(val_liratios, nan=0.0, posinf=1.0, neginf=-1.0)
            train_liratios=target_res['train_cos']
            train_liratios=np.array([ i.cpu().item() for i in train_liratios ])
            train_liratios = np.nan_to_num(train_liratios, nan=0.0, posinf=1.0, neginf=-1.0)
            # auc,log_auc,tprs=self.plot_auc("cos_attack",torch.tensor(val_liratios),torch.tensor(train_liratios),epch)
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
                # print('mix_test_loss shape:',mix_test_loss.shape)
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
                # print('mix_test_loss shape:',mix_test_loss.shape)
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
                # print('mix_test_loss shape:',mix_test_loss.shape)
                val_liratios = mix_test_loss

            # val_liratios=np.array([ i.cpu().item() for i in val_liratios ])
            train_liratios=target_res["train_res"]["loss"]
            # train_liratios=np.array([ i.cpu().item() for i in train_liratios ])
            scores = np.concatenate((train_liratios,val_liratios),axis=0)
            gt = [1]*train_liratios.shape[0]+[0]*val_liratios.shape[0]
            auc,tpr = ROC_AUC_Result_logshow_with_auc(gt,scores,reverse)

        return tpr,auc,(train_liratios, val_liratios)

    def attack(self):
        lira_scores=[]
        lira_loss_scores=[]
        common_scores=[]
        other_scores={}

        ## 记录TPR@FPR=0.001
        scores={k:[] for k in self.attack_modes}
        scores["lira"]=[]
        scores["lira_loss"]=[]
        ## 记录所有epoch的TPR@FPR=0.01中最大的
        single_score={k:0 for k in self.attack_modes}
        single_score["lira"]=0
        single_score["lira_loss"]=0
        ## 记录每轮 lira的mem和non-mem的cdf, 即(train_l_out,test_l_out) 
        reses_lira=[]
        reses_lira_loss=[]
        ## 记录其他attack mode的 (val_liratios,train_liratios)
        reses_common={k:[] for k in self.attack_modes}
        ## 记录所有轮数cdf avg后AUC攻击的TPR得分
        avg_scores={k:None for k in self.attack_modes}
        avg_scores["lira"]=None
        avg_scores["lira_loss"]=None

        auc_dict={k:[] for k in self.attack_modes}
        auc_dict["lira"]=[]
        auc_dict["lira_loss"]=[]

        for epch in self.epochs:
            # try:
            cos_auc,cos_tpr,cos_score=self.lira_attack_ldh_cosine(self.p,epch,self.MAX_K,self.save_dir, extract_fn=self.extract_hinge_loss,attack_mode='diff') 
            loss_auc,loss_tpr,loss_score=self.lira_attack_ldh_cosine(self.p,epch,self.MAX_K,self.save_dir, extract_fn=self.extract_hinge_loss,attack_mode='loss') 
                
                # the above function retruns: accs, tprs, auc, log_auc, (train_l_out,test_l_out) 
                ## log_auc: 基于log_lira所得的auc
            # except ValueError:
            #     print("ValueError")
            #     continue
            scores["lira"].append(cos_tpr)
            scores["lira_loss"].append(loss_tpr)
            auc_dict["lira"].append(cos_auc)
            auc_dict["lira_loss"].append(loss_auc)


            # lira_score=lira_attack(p,epch,K=9,extract_fn=extract_hinge_loss)
            for attack_mode in self.attack_modes:
                common_score=self.cos_attack(self.p,0,epch,attack_mode,extract_fn=self.extract_hinge_loss) 
                # the above function return:  accs, tprs, auc, log_auc, (val_liratios,train_liratios)
                reses_common[attack_mode].append(common_score[-1])
                scores[attack_mode].append(common_score[0])
                auc_dict[attack_mode].append(common_score[1])

            lira_scores.append(cos_tpr)
            lira_loss_scores.append(loss_tpr)
            common_scores.append(common_score[0]) # 为最后一个loss based的common_score, 但似乎这个list没啥用

            reses_lira.append(cos_score) # 当下epoch的 (train_l_out,test_l_out) 
            reses_lira_loss.append(loss_score)

        for attack_mode in self.attack_modes:
            sorted_id = sorted(range(len(scores[attack_mode])), key=lambda k: scores[attack_mode][k], reverse=True)
            single_score[attack_mode]=(scores[attack_mode][sorted_id[0]])
            single_score[f'single {attack_mode}_auc'] = auc_dict[attack_mode][sorted_id[0]]


        for attack_mode in ['lira', 'lira_loss']:
            sorted_id = sorted(range(len(scores[attack_mode])), key=lambda k: scores[attack_mode][k], reverse=True)
            single_score[attack_mode]=(scores[attack_mode][sorted_id[0]])
            single_score[f'single {attack_mode}_auc'] = auc_dict[attack_mode][sorted_id[0]]

        # print('------------ ----------------- -------------  ')
        # print('------------ ---Best attack--- -------------  ')
        # print('------------ ----------------- -------------  ')

        # for attack_mode in ['lira', 'lira_loss']:
        #     auc = single_score[f'single {attack_mode}_auc']
        #     tpr = single_score[attack_mode]
        #     print(f'Best {attack_mode} auc:{auc}\ntpr:{tpr}')
        # for attack_mode in self.attack_modes:
        #     auc = single_score[f'single {attack_mode}_auc']
        #     tpr = single_score[attack_mode]
        #     print(f'Best {attack_mode} auc:{auc}\ntpr:{tpr}')
        
        # print('------------ ----------------- -------------  ')
        # print('------------ Sequential attack -------------  ')
        # print('------------ ----------------- -------------  ')

        reses=reses_lira
        train_score=np.vstack([ i[0].reshape(1,-1) for i in reses]).mean(axis=0)
        test_score=np.vstack([ i[1].reshape(1,-1) for i in reses]).mean(axis=0)
        scores = np.concatenate((train_score,test_score),axis=0)
        gt = [1]*train_score.shape[0]+[0]*test_score.shape[0]
        auc,tpr = ROC_AUC_Result_logshow_with_auc(gt,scores,True)
        if auc<0.5:
            auc,tpr = ROC_AUC_Result_logshow_with_auc(gt,scores,False)
        # print(f"averaged_lira_grad tprs:{tpr} \n auc:{auc}")
        avg_scores["lira"]=tpr
        other_scores["lira_auc"]=[auc]


        reses=reses_lira_loss
        train_score=np.vstack([ i[0].reshape(1,-1) for i in reses]).mean(axis=0)
        test_score=np.vstack([ i[1].reshape(1,-1) for i in reses]).mean(axis=0)

        train_fscore=train_score[0:self.test_size]
        test_fscore=test_score[0:self.test_size]



        scores = np.concatenate((train_fscore,test_fscore),axis=0)
        gt = [1]*train_fscore.shape[0]+[0]*train_fscore.shape[0]

        print("*************FedMIA***************")
        auc,tpr = ROC_AUC_Result_logshow_with_auc(gt,scores,False)
        utils.plot_log_roc(gt,scores,name='FedMIA.png')
        # if auc<0.5:
        #     auc,tpr = ROC_AUC_Result_logshow_with_auc(gt,scores,False)
        print(f"averaged_lira_loss tprs:{tpr} \n auc:{auc}")
        metircs= utils.get_best_metrics(gt,scores)
        # acc = utils.calculate_acc(scores,gt,v_scores,v_gt,'percentile',75)
        avg_scores["lira_loss"]=tpr
        other_scores["lira_loss_auc"]=[auc]


        # reses=reses_common["cosine attack"]
        # # print(reses)
        # # print(len(reses),len(reses[0]))
        # # assert 0
        # train_score=np.vstack([ i[0] for i in reses]).mean(axis=0)
        # test_score=np.vstack([ i[1] for i in reses]).mean(axis=0)

        # train_fscore=train_score[0:300]
        # test_fscore=test_score[0:300]

        # train_vscore=train_score[300:350]
        # test_vscore=test_score[300:350]

        # print('***********COS_MIA************')
        # scores = np.concatenate((train_fscore,test_fscore),axis=0)
        # v_scores = np.concatenate((train_vscore,test_vscore),axis=0)
        # gt = [1]*train_fscore.shape[0]+[0]*test_fscore.shape[0]
        # v_gt=[1]*train_vscore.shape[0]+[0]*test_vscore.shape[0]
        # auc,tpr = ROC_AUC_Result_logshow_with_auc(gt,scores,False)

        # avg_scores["cosine attack"]=tpr
        # other_scores["cos_attack_auc"]=[auc]
        # print(f"averaged cosine attack tprs:{tpr} \n auc:{auc}")

        # acc = utils.calculate_acc(scores,gt,v_scores,v_gt,'percentile',75)


        # reses=reses_common["grad diff"]
        # # print(reses)
        # # print(len(reses),len(reses[0]))
        # # assert 0
        # train_score=np.vstack([ i[0].reshape(1,-1) for i in reses]).mean(axis=0)
        # test_score=np.vstack([ i[1].reshape(1,-1) for i in reses]).mean(axis=0)
        # print('***********check avg grad diff attack:')
        # scores = np.concatenate((train_score,test_score),axis=0)
        # gt = [1]*train_score.shape[0]+[0]*test_score.shape[0]
        # auc,tpr = ROC_AUC_Result_logshow_with_auc(gt,scores,True)
        # avg_scores["grad diff"]=tpr
        # other_scores["grad_diff_auc"]=[auc]
        # print(f"averaged_diff tprs:{tpr} \n auc:{auc}")


        # reses=reses_common["grad norm"]
        # train_score=-np.vstack([ i[0].reshape(1,-1) for i in reses]).mean(axis=0)
        # test_score=-np.vstack([ i[1].reshape(1,-1) for i in reses]).mean(axis=0)
        # scores = np.concatenate((train_score,test_score),axis=0)
        # gt = [1]*train_score.shape[0]+[0]*test_score.shape[0]
        # auc,tpr = ROC_AUC_Result_logshow_with_auc(gt,scores,True)
        # avg_scores["grad norm"]=tpr
        # other_scores["grad_norm_auc"]=[auc]
        # print(f"averaged grad norm tprs:{tpr} \n auc:{auc}")


        # reses=reses_common["loss based"]
        # train_score=np.vstack([ i[0].reshape(1,-1) for i in reses]).mean(axis=0)
        # test_score=np.vstack([ i[1].reshape(1,-1) for i in reses]).mean(axis=0)
        # print('***********check avg loss based attack:')
        # scores = np.concatenate((train_score,test_score),axis=0)
        # gt = [1]*train_score.shape[0]+[0]*test_score.shape[0]
        # auc,tpr = ROC_AUC_Result_logshow_with_auc(gt,scores,False)
        # avg_scores["loss based"]=tpr
        # other_scores["loss_based_auc"]=[auc]
        # print(f"averaged_loss tprs:{tpr} \n auc:{auc}")


        # self.fig_out(self.epochs,self.save_dir,scores,avg_scores,single_score, other_scores,final_acc)

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
        # plt.plot(x_axis_data, common_score,'bo-', linewidth=1, color='#2E8B57', label=r'Baseline')
        plt.legend(loc=3)  

        plt.xlim(-2, 305)
        my_x_ticks = np.arange(0, 302, 50)
        plt.xticks(my_x_ticks,size=14)
        if avg_d:
            for k in labels_temporal.keys():
                if avg_d[k]:    
                    plt.hlines([avg_d[k]["0.001"]],xmin=0,xmax=300,label=labels_temporal[k],color=colors[k])

        plt.legend(prop={'size': 10})
        plt.xlabel('Epoch',fontsize=14,fontdict={'size': 14})  # x_label
        plt.ylabel('TPR@FPR=0.001',fontsize=14,fontdict={'size': 14})  # y_label
        plt.grid(axis='both')

        pdf_path=self.PATH.split("/")[0:-1]
        pdf_path="/".join(pdf_path)+f"/attack_fig_{self.select_mode}_{self.select_method}_n{self.SHADOW_NUM}_s{self.SEED}.pdf"
        
        # pdf_path="/".join(pdf_path)+"/attack9_val_mode_positive_plus.pdf"
        # attack9_val_mode_positive_plus_select_mean_<<.pdf
        print('fig saved in', pdf_path)
        plt.savefig(pdf_path)

        # print("log_path0:",log_path)
        # log_path=log_path+f"/def{defence}2_0.85_k{MAX_K}_{seed}_attack.log"
        log_path=log_path+f"/attack_score_{self.select_mode}_{self.select_method}_n{self.SHADOW_NUM}_s{self.SEED}.log"
        # print("log_path:",log_path)
        with open(log_path,"w") as f:
            json.dump({"avg_d":avg_d,"single_score":single_score,"other_scores":other_scores,"accs":accs},f, indent=4)
        # assert 0

class MBA:
    def __init__(self,args):
        self.args=args
        self.attack_client_idx=0
        self.client_model=self.load_client_model()
        self.server_model = self.load_server_model()
        self.data_size=300
        self.thre_data_size=50
        self.client_dataloader,self.thre_dataloader = self.load_client_dataloader(self.data_size+self.thre_data_size)
        self.other_dataloader = self.load_otherloader(data_size=self.data_size)

    def load_client_model(self):
        model_folder = self.args.model_path + '/' +self.args.model+'/' + self.args.dataset + '/client_model'
        # model_files = [f for f in os.listdir(model_folder) if f.startswith('sercer_') and f.endswith('./pth')]
        # model_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        model_file = f"client_{self.attack_client_idx}_{self.args.training_round-4}.pth"
        model_path = os.path.join(model_folder, model_file)
        model_pth = torch.load(model_path)  # 加载模型
        model = PublicLayer(self.args)
        model.load_state_dict(model_pth)
        model.to(self.args.device)
        print(f'load model {model_file}')
        return model
    
    def load_server_model(self):
        model_folder = self.args.model_path + '/' +self.args.model+'/' + self.args.dataset + '/server_model'
        model_file = f"server_{self.args.training_round-1}.pth"
        model_path = os.path.join(model_folder, model_file)
        model_pth = torch.load(model_path)  # 加载模型
        model = PublicLayer(self.args)
        model.load_state_dict(model_pth)
        model.to(self.args.device)
        print(f'load model {model_file}')
        return model
        

    def load_client_dataloader(self, total_data_size):
        """
        数据加载与切分
        
        参数:
        - total_data_size: 需要加载的总样本数 (即 self.data_size + self.thre_data_size)
        
        返回:
        - client_dataloader: 用于攻击评估的主数据集 (大小为 total - thre)
        - thre_dataloader:   用于确定阈值的校准数据集 (大小为 self.thre_data_size)
        """
        args = self.args
        random.seed(self.args.random_seed)
        
        # 确定切分大小
        # 校准集大小 (例如 50)
        cal_size = self.thre_data_size
        # 评估集大小 (例如 300)
        eval_size = self.data_size
        
        assert eval_size > 0, "Total data size must be larger than threshold data size!"

        # 1. 加载原始数据 (Members 和 Non-Members)
        data_path = args.data_path + '/' + args.dataset + '/' + args.model + '/' + args.data_split
        # datas: [client_num, data_len, ...]
        datas, labels = load_npz_data(data_path + '/train_non_iid.npz')
        non_member_data, non_member_label = load_npz_data(data_path + '/test_non_iid.npz')

        # 2. 获取当前攻击客户端的数据
        c_idx = self.attack_client_idx
        client_member_data = datas[c_idx]
        client_member_label = labels[c_idx]
        client_non_member_data = non_member_data[c_idx]
        client_non_member_label = non_member_label[c_idx]

        # 3. 生成随机索引并打乱
        # Member 索引
        indices_member = np.arange(len(client_member_data))
        random.shuffle(indices_member)
        # Non-Member 索引
        indices_non_member = np.arange(len(client_non_member_data))
        random.shuffle(indices_non_member)

        # 4. 采样总数据 (Members 和 Non-Members 各取 total_data_size 个)
        # 注意：这里假设源数据量足够，如果不够需要加 min() 保护
        sampled_idx_m_total = indices_member[:total_data_size]
        sampled_idx_nm_total = indices_non_member[:total_data_size]

        # 5. 切分索引: 校准集 (Calibration) vs 评估集 (Evaluation)
        
        # --- 校准集索引 (前 thre_data_size 个) ---
        cal_idx_m = sampled_idx_m_total[:cal_size]
        cal_idx_nm = sampled_idx_nm_total[:cal_size]
        
        # --- 评估集索引 (剩下的) ---
        eval_idx_m = sampled_idx_m_total[cal_size:]
        eval_idx_nm = sampled_idx_nm_total[cal_size:]

        # 6. 构建数据集辅助函数
        def create_dataset(idx_m, idx_nm):
            # 提取 Member 数据
            d_m = client_member_data[idx_m]
            l_m = client_member_label[idx_m]
            # 提取 Non-Member 数据
            d_nm = client_non_member_data[idx_nm]
            l_nm = client_non_member_label[idx_nm]
            
            # 合并
            data_concat = np.concatenate((d_m, d_nm))
            label_concat = np.concatenate((l_m, l_nm))
            # 生成成员性标签: Member=1, Non-Member=0
            membership = [1] * len(d_m) + [0] * len(d_nm)
            
            return ClientDatasetWithMember(data_concat, label_concat, membership)

        # 7. 创建 Dataset 对象
        dataset_cal = create_dataset(cal_idx_m, cal_idx_nm)     # 校准用 (50 + 50)
        dataset_eval = create_dataset(eval_idx_m, eval_idx_nm)  # 评估用 (300 + 300)

        print(f"Loaded Datasets: Eval Size={len(dataset_eval)} (Target), Calibration Size={len(dataset_cal)} (Threshold)")

        # 8. 创建 DataLoader
        # 为了方便测试设置 batch=1
        thre_dataloader = DataLoader(dataset_cal, batch_size=1, shuffle=True)
        client_dataloader = DataLoader(dataset_eval, batch_size=1, shuffle=True)

        # 返回顺序：先返回主评估集，再返回校准集
        return client_dataloader, thre_dataloader
    
    def load_otherloader(self,data_size):
        """
        构建测试的数据集，测试集都采样于非攻击者的其他客户端。
        :param data_size: 构建出的数据集的大小，其中成员和非成员要求是一样多的，为了数据集的均衡，都是data_size
        :return: dataloader
        """
        args = self.args
        data_path = args.data_path + '/'+args.dataset+'/'+args.model+'/'+args.data_split
        datas, labels = load_npz_data(data_path + '/train_non_iid.npz')
        data = [client_data for client_idx, client_data in enumerate(datas) if client_idx == self.attack_client_idx]
        label = [client_label for client_idx, client_label in enumerate(labels) if client_idx == self.attack_client_idx]
        # data, label = np.delete(datas,self.attack_client_idx,axis=0), np.delete(labels,self.attack_client_idx,axis=0)
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
        random_indices = np.random.choice(min_number, data_size, replace=False)
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
            dataToClip]  # 对于数据集中的每个样本，都取概率值最大的前三，reverse=True是逆序排列，也就是从大到小。Sorted函数返回值是排序后的一个数组。
        return np.array(res)

    def MetricBasedAttacking(self,pre_vectors_from_targetmodel, member_labels, classification_y,
                         metric="max"):  # member_labels是指成员或非成员，2分类的标签。targetX是目标模型的预测结果
        # classification_y是真实的分类标签（非预测），例如CIFA10就是0-9的类别标签，用于计算loss这个metric，其他的metric不需要
        # pre_vectors_from_targetmodel已经进行了softmax，所以是0-1区间的。
        pre_vectors_from_targetmodel = np.array(pre_vectors_from_targetmodel)
        classification_y = np.array(classification_y)
        if (metric == "max"):  # 预测向量最大值
            pre_vectors_from_targetmodel = self.clipDataTopX(pre_vectors_from_targetmodel, top=1)  # 切成最大值了。
            # pre_vectors_from_targetmodel是每个样本的目标模型预测输出向量中的最大值（1个值）了。
            # ROC_AUC_Result_logshow(member_labels,pre_vectors_from_targetmodel,reverse=False) #pre_vectors_from_targetmodel是0-1区间的。
            metrics = pre_vectors_from_targetmodel

        if (metric == "loss"):  # 预测向量的交叉熵损失值   #实现错误，需要的是分类标签计算logloss，而不是成员非成员的标签。
            # Scikit-learn中提供了交叉熵损失的计算方法log_loss
            log_loss_list = []  # 存储每条样本的交叉熵loss值。
            temp_labels = range(len(pre_vectors_from_targetmodel[0]))  # 要生成目标模型分类的类别，否则log_loss会从real_labels中统计类别数量，就对不上了。
            for i in range(len(member_labels)):
                log_loss_list.append(log_loss([classification_y[i]], [pre_vectors_from_targetmodel[i]],
                                            labels=temp_labels))  # targetX是目标模型的预测结果
            # ROC_AUC_Result_logshow(member_labels,log_loss_list,reverse=True)
            metrics = log_loss_list

        if (metric == "sd"):  # 预测向量的标准差
            # 计算每个样本（行）的标准差
            sd = np.std(pre_vectors_from_targetmodel, axis=1)  # axis=1表示计算每行的标准差
            # ROC_AUC_Result_logshow(member_labels,sd,reverse=False)
            metrics = sd

        if (metric == "entropy"):  # 预测向量的熵
            # 计算每个样本（行）的熵
            negative_logs = -np.log(
                np.maximum(pre_vectors_from_targetmodel, 1e-30))  # 对pre_vectors_from_targetmodel矩阵中的每个元素都取log值，再取负号。
            entropys = np.sum(np.multiply(pre_vectors_from_targetmodel, negative_logs),
                            axis=1)  # np.multiply是对应元素相乘，不是矩阵乘法  #axis=1表示每行求和，因此计算结果就是熵。
            # ROC_AUC_Result_logshow(member_labels,entropys,reverse=True)
            metrics = entropys

        if (metric == "mentropy"):  # 预测向量的修正熵
            # 计算每个样本（行）的修正熵
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
            # ROC_AUC_Result_logshow(member_labels,mentropys,reverse=True)
            metrics = mentropys

        if (metric == "correctness"):  # 预测分类正确性   分类标签从0开始编号
            pre_class_label = np.argmax(pre_vectors_from_targetmodel, axis=1)  # 每行取出最大值对应的下标，也就是类别。
            temp = pre_class_label - classification_y  # 相减，为0则表示猜中了。
            pre_member_label = np.int64(
                temp == 0)  # temp==0是判断每个元素是否为0，0则为True，再将True用int64()转化为int，也就是1。其他非0值，是False，int64()后就会被转化为0
            # ROC_AUC_Result_logshow(member_labels,pre_member_label,reverse=False) #pre_vectors_from_targetmodel是0-1区间的。
            metrics = pre_member_label

        return metrics
    

    def attack(self,metric_flag):
        """
        : metric_flag: entropy mentropy
        """
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
        metrics = utils.get_best_metrics(ground_truth,metrics)



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
        # model_files = [f for f in os.listdir(model_folder) if f.startswith('sercer_') and f.endswith('./pth')]
        # model_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        model_file = f"client_{self.attack_client_idx}_{self.args.training_round-4}.pth"
        model_path = os.path.join(model_folder, model_file)
        model_pth = torch.load(model_path)  # 加载模型
        model = PublicLayer(self.args)
        model.load_state_dict(model_pth)
        model.to(self.args.device)
        print(f'load model {model_file}')
        return model
    
    def load_server_model(self):
        model_folder = self.args.model_path + '/' +self.args.model+'/' + self.args.dataset + '/server_model'
        model_file = f"server_{self.args.training_round-1}.pth"
        model_path = os.path.join(model_folder, model_file)
        model_pth = torch.load(model_path)  # 加载模型
        model = PublicLayer(self.args)
        model.load_state_dict(model_pth)
        model.to(self.args.device)
        print(f'load model {model_file}')
        return model
    
    def load_client_dataloader(self, data_size):
        """
        数据加载
        数据来源是攻击者控制的0号客户端，使用训练集和测试集构建成员和非成员作为训练集。
        """
        args = self.args
        random.seed(self.args.random_seed)

        # 加载训练数据和测试数据
        data_path = args.data_path + '/'+args.dataset+'/'+args.model+'/'+args.data_split
        datas, labels = load_npz_data(data_path + '/train_non_iid.npz')
        non_member_data, non_member_label = load_npz_data(data_path + '/test_non_iid.npz')

        # 生成随机索引
        all_indices = np.arange(len(datas[self.attack_client_idx]))  # 获取所有的索引
        random.shuffle(all_indices)  # 打乱索引
        sampled_indices1 = all_indices[:int(data_size)]  # 从训练数据中采样
        sampled_indices2 = all_indices[:int(data_size)]  # 从非成员测试数据中采样

        # 使用索引提取数据和标签
        sampled_data1 = datas[self.attack_client_idx][sampled_indices1]
        sampled_label1 = labels[self.attack_client_idx][sampled_indices1]
        sampled_data2 = non_member_data[self.attack_client_idx][sampled_indices2]
        sampled_label2 = non_member_label[self.attack_client_idx][sampled_indices2]

        # 确保标签和数据的长度匹配
        assert len(sampled_data1) == len(sampled_label1), "Sampled data and labels are mismatched!"
        assert len(sampled_data2) == len(sampled_label2), "Sampled data and labels are mismatched!"

        # 创建成员性标签
        membership = [1] * len(sampled_data1) + [0] * len(sampled_data2)
        data = np.concatenate((sampled_data1, sampled_data2))
        label = np.concatenate((sampled_label1, sampled_label2))

        # 确保数据与标签的维度一致
        assert len(data) == len(label), "Data and labels have mismatched lengths!"

        # 构建数据集
        dataset = ClientDatasetWithMember(data, label, membership)

        # 为了方便测试设置batch=1
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

        return dataloader
    
    def load_otherloader(self,data_size):
        """
        构建测试的数据集，测试集都采样于非攻击者的其他客户端。
        :param data_size: 构建出的数据集的大小，其中成员和非成员要求是一样多的，为了数据集的均衡，都是data_size
        :return: dataloader
        """
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
        """
        对一组目标样本执行 Attack D。
        Returns:
            predictions: list of bool, True 表示判定为成员
            scores: list of (target_loss, threshold)
        """
        num_distilled=self.num_distilled
        device=self.args.device
        self.client_model.to(device)
        self.client_model.eval()

        # Step 1: 用 other_dataloader 作为辅助数据（population data）
        pop_data = []
        pop_targets = []
        for x, y,_ in self.other_dataloader:
            pop_data.append(x)
            pop_targets.append(y)
        pop_data = torch.cat(pop_data, dim=0).to(device)
        pop_targets = torch.cat(pop_targets, dim=0).to(device)

        # Step 2: 用 client_model 生成软标签（logits 或 softmax）
        with torch.no_grad():
            soft_labels = self.client_model(pop_data)  # shape: [N, num_classes]

        # Step 3: 训练 num_distilled 个蒸馏模型
        distilled_models = []
        for i in range(num_distilled):
            model = self._train_distilled_model(
                pop_data, soft_labels, 
                epochs=self.distill_epoch, lr=0.001, 
                seed=i, device=device,train_flag=self.train_flag,model_id=i
            )
            distilled_models.append(model)

        # Step 4: 对每个目标样本进行判定
        predictions = []
        scores = []
        gt=[]
        distilled_losses = []
        target_losses=[]

        criterion = nn.CrossEntropyLoss(reduction='none')  # 逐样本 loss

        for x, y, m in tqdm(self.client_dataloader):
            x = x.to(device)  # [1, C, H, W]
            y = torch.tensor([y]).to(device)  # [1]
            gt.append(m.item())

            # 目标模型在该样本上的损失
            with torch.no_grad():
                logits_target = self.client_model(x)
                target_loss = criterion(logits_target, y).item() 
                target_losses.append(target_loss)

            # 所有蒸馏模型在该样本上的损失
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

    # -----------------------------
    # 辅助函数：训练一个蒸馏模型
    # -----------------------------
    def _train_distilled_model(self, inputs, soft_targets, epochs=10, lr=1e-3, seed=0, device='cpu', train_flag=False, model_id=0):
        """
        使用输入和软标签训练一个学生模型（结构与 client_model 相同）
        如果 train_flag=False，则从磁盘加载已保存的模型（编号为 model_id）
        """
        import os

        model_path = os.path.join(self.args.model_path,self.args.model,self.args.dataset,'enhancedMIA',f'distilled_model_{model_id}.pth')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        torch.manual_seed(seed)
        np.random.seed(seed)

        # 克隆模型结构（假设 client_model 是 nn.Module）
        if hasattr(self.client_model, 'config'):
            student_model = type(self.client_model)(**self.client_model.config)
        else:
            # 如果没有 config，尝试无参初始化（需确保可行）
            student_model = type(self.client_model)(self.args)

        student_model.to(device)

        if not train_flag:
            # 加载已保存的模型
            if os.path.exists(model_path):
                student_model.load_state_dict(torch.load(model_path, map_location=device))
                student_model.eval()  # 设置为评估模式
                print(f'loading model {model_id}')
                return student_model
            else:
                raise FileNotFoundError(f"未找到预训练模型: {model_path}，请先设置 train_flag=True 进行训练并保存。")

        # ===== 训练流程 =====
        optimizer = torch.optim.Adam(student_model.parameters(), lr=lr)
        criterion = nn.KLDivLoss(reduction='batchmean')

        # 准备 dataset
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

        # 保存训练好的模型
        torch.save(student_model.state_dict(), model_path)
        print(f'training model {model_id}')

        return student_model
    
class CSF18:
    """
    实现了论文中的 Adversary 1 (Bounded Loss Function)。
    针对分类任务的 0-1 Loss 场景。
    """
    
    def __init__(self, args):
        self.args=args
        self.attack_client_idx=0
        self.client_model=self.load_client_model()
        self.data_size=300
        self.client_dataloader = self.load_client_dataloader(data_size=self.data_size)

    def load_client_dataloader(self, data_size):
        """
        数据加载
        数据来源是攻击者控制的0号客户端，使用训练集和测试集构建成员和非成员作为训练集。
        """
        args = self.args
        random.seed(self.args.random_seed)

        # 加载训练数据和测试数据
        data_path = args.data_path + '/'+args.dataset+'/'+args.model+'/'+args.data_split
        datas, labels = load_npz_data(data_path + '/train_non_iid.npz')
        non_member_data, non_member_label = load_npz_data(data_path + '/test_non_iid.npz')

        # 生成随机索引
        all_indices = np.arange(len(datas[self.attack_client_idx]))  # 获取所有的索引
        random.shuffle(all_indices)  # 打乱索引
        sampled_indices1 = all_indices[:int(data_size)]  # 从训练数据中采样
        sampled_indices2 = all_indices[:int(data_size)]  # 从非成员测试数据中采样

        # 使用索引提取数据和标签
        sampled_data1 = datas[self.attack_client_idx][sampled_indices1]
        sampled_label1 = labels[self.attack_client_idx][sampled_indices1]
        sampled_data2 = non_member_data[self.attack_client_idx][sampled_indices2]
        sampled_label2 = non_member_label[self.attack_client_idx][sampled_indices2]

        # 确保标签和数据的长度匹配
        assert len(sampled_data1) == len(sampled_label1), "Sampled data and labels are mismatched!"
        assert len(sampled_data2) == len(sampled_label2), "Sampled data and labels are mismatched!"

        # 创建成员性标签
        membership = [1] * len(sampled_data1) + [0] * len(sampled_data2)
        data = np.concatenate((sampled_data1, sampled_data2))
        label = np.concatenate((sampled_label1, sampled_label2))

        # 确保数据与标签的维度一致
        assert len(data) == len(label), "Data and labels have mismatched lengths!"

        # 构建数据集
        dataset = ClientDatasetWithMember(data, label, membership)

        # 为了方便测试设置batch=1
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        return dataloader
    
    def load_client_model(self):
        model_folder = self.args.model_path + '/' +self.args.model+'/' + self.args.dataset + '/client_model'
        # model_files = [f for f in os.listdir(model_folder) if f.startswith('sercer_') and f.endswith('./pth')]
        # model_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        model_file = f"client_{self.attack_client_idx}_{self.args.training_round-4}.pth"
        model_path = os.path.join(model_folder, model_file)
        model_pth = torch.load(model_path)  # 加载模型
        model = PublicLayer(self.args)
        model.load_state_dict(model_pth)
        model.to(self.args.device)
        print(f'load model {model_file}')
        return model


    def attack(self):
        """
        获取攻击得分 (Loss-based Score)。
        
        原理:
        根据论文 ，对于神经网络模型，虽然执行分类任务，
        但攻击者通常使用模型的损失函数值 (如 Cross-Entropy) 作为攻击特征。
        
        攻击逻辑 (Adversary 2 / Threshold):
        - Score = Sample Loss (样本损失)
        - 判定逻辑: 如果 Score <= Threshold (阈值)，则判定为成员 (Member)；
                   如果 Score > Threshold，则判定为非成员 (Non-Member)。
                   (因为训练数据的 Loss 通常比非训练数据更小 )
        
        返回:
        - results: 包含 'scores' (损失值列表) 和 'labels' (真实成员性标签)
        """
        self.client_model.eval()
        
        all_scores = []      # 存储每个样本的 Loss 值 (得分)
        true_memberships = [] # 真实的成员性标签
        
        device = self.args.device if hasattr(self.args, 'device') else 'cpu'
        self.client_model.to(device)

        with torch.no_grad():
            for data, label, membership in self.client_dataloader:
                data = data.to(device)
                label = label.to(device)
                
                # 1. 模型前向传播
                outputs = self.client_model(data)
                
                # 2. 计算每个样本的 Loss 作为得分
                # 使用 reduction='none' 以获取 batch 中每个样本的独立 Loss
                # 论文中提到使用 Cross-Entropy Loss 
                _, predicted = torch.max(outputs.data, 1)
                batch_scores = (predicted == label).float()
                # batch_losses = F.cross_entropy(outputs, label, reduction='none')
                
                # 3. 收集结果
                # 注意：这里返回的是 Loss。Loss 越小，越像成员。
                # 如果后续使用 sklearn 的 roc_auc_score，可能需要取负数 (-Loss) 作为 "Membership Score"
                all_scores.extend(batch_scores.cpu().numpy())
                true_memberships.extend(membership.cpu().numpy())
        
        auc,tpr=ROC_AUC_Result_logshow_with_auc(true_memberships,all_scores,False)
        acc=accuracy_score(true_memberships,all_scores)
        precision = precision_score(true_memberships,all_scores, zero_division=0)
        recall = recall_score(true_memberships,all_scores, zero_division=0)
        f1 = f1_score(true_memberships,all_scores, zero_division=0)

        print(f"--- Evaluation Metrics ---")
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1-Score : {f1:.4f}")

        return {
            "scores": all_scores,       # 攻击得分 (Loss)
            "labels": true_memberships  # 真实标签 (1=Member, 0=Non-Member)
        }
    

class ICLR2023:
    """
    全历史 ICLR2023 攻击 (针对完整模型保存版)
    
    [cite_start]Paper Reference: [cite: 194-203] "Attacks using multiple communication rounds"
    """

    def __init__(self, args, attack_client_idx, total_eval_size=600):
        self.args = args
        self.attack_client_idx = attack_client_idx
        self.device = args.device
        self.data_size = total_eval_size
        
        # 1. 实例化一个模型模板 (用于加载权重计算梯度)
        # 请确保 TargetModel 是你训练时使用的那个包含所有层的类
        self.model_template = PublicLayer(args).to(self.device)
        
        # 2. 准备数据
        self.client_dataloader= self.load_client_dataloader(total_eval_size)
        self.eval_loader = self.client_dataloader
        
        self.eval_scores_sum = np.zeros(len(self.eval_loader.dataset))

    def load_client_dataloader(self, data_size):
        """
        数据加载
        数据来源是攻击者控制的0号客户端，使用训练集和测试集构建成员和非成员作为训练集。
        """
        args = self.args
        random.seed(self.args.random_seed)

        # 加载训练数据和测试数据
        data_path = args.data_path + '/'+args.dataset+'/'+args.model+'/'+args.data_split
        datas, labels = load_npz_data(data_path + '/train_non_iid.npz')
        non_member_data, non_member_label = load_npz_data(data_path + '/test_non_iid.npz')

        # 生成随机索引
        all_indices = np.arange(len(datas[self.attack_client_idx]))  # 获取所有的索引
        random.shuffle(all_indices)  # 打乱索引
        sampled_indices1 = all_indices[:int(data_size)]  # 从训练数据中采样
        sampled_indices2 = all_indices[:int(data_size)]  # 从非成员测试数据中采样

        # 使用索引提取数据和标签
        sampled_data1 = datas[self.attack_client_idx][sampled_indices1]
        sampled_label1 = labels[self.attack_client_idx][sampled_indices1]
        sampled_data2 = non_member_data[self.attack_client_idx][sampled_indices2]
        sampled_label2 = non_member_label[self.attack_client_idx][sampled_indices2]

        # 确保标签和数据的长度匹配
        assert len(sampled_data1) == len(sampled_label1), "Sampled data and labels are mismatched!"
        assert len(sampled_data2) == len(sampled_label2), "Sampled data and labels are mismatched!"

        # 创建成员性标签
        membership = [1] * len(sampled_data1) + [0] * len(sampled_data2)
        data = np.concatenate((sampled_data1, sampled_data2))
        label = np.concatenate((sampled_label1, sampled_label2))

        # 确保数据与标签的维度一致
        assert len(data) == len(label), "Data and labels have mismatched lengths!"

        # 构建数据集
        dataset = ClientDatasetWithMember(data, label, membership)

        # 为了方便测试设置batch=1
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        return dataloader

    def load_models_and_get_update(self, round_idx):
        """
        加载第 round_idx 轮的模型，并计算 ΔΓ = W_client - W_server
        """
        # 路径构建 (请根据你实际的保存路径修改)
        # Server Model Path
        server_path = os.path.join(
            self.args.model_path, self.args.model, self.args.dataset, 
            'server_model', f'server_{round_idx}.pth'
        )
        
        # Client Model Path (名字没改，指向 client_model 文件夹)
        # 假设文件名格式是 client_{idx}_{round}.pth
        client_path = os.path.join(
            self.args.model_path, self.args.model, self.args.dataset, 
            'client_model', f'client_{self.attack_client_idx}_{round_idx}.pth'
        )
        
        # 检查文件是否存在
        if not os.path.exists(server_path):
            raise FileNotFoundError(f"Server model not found: {server_path}")
        if not os.path.exists(client_path):
            # 如果这轮该客户端没参与训练，可能就没有文件
            raise FileNotFoundError(f"Client model not found: {client_path}")

        try:
            # 1. 加载 Global Model (W_S)
            # 使用 model_template 加载权重
            server_state = torch.load(server_path, map_location=self.device)
            self.model_template.load_state_dict(server_state)
            self.model_template.eval()
            
            # 将 Global Model 参数展平为向量
            # 必须 detach，否则会占用计算图内存
            server_vec = nn.utils.parameters_to_vector(self.model_template.parameters()).detach()
            
            # 2. 加载 Client Model (W_C)
            # 为了计算差值，我们需要临时加载 Client 权重
            client_state = torch.load(client_path, map_location=self.device)
            
            # 这里有个技巧：我们不需要实例化两个模型对象。
            # 我们可以先加载 Server 算完 vector，再加载 Client 算 vector。
            # 但为了后续计算梯度，我们需要保留 Server Model 的状态在 self.model_template 中。
            
            # 所以，先用一个临时变量或者重新 load 一次来获取 Client Vector
            self.model_template.load_state_dict(client_state)
            client_vec = nn.utils.parameters_to_vector(self.model_template.parameters()).detach()
            
            # 3. 计算 ΔΓ (Update Vector)
            # ΔΓ = W_client - W_server
            delta_gamma = client_vec - server_vec
            
            # 4. 恢复 self.model_template 为 Server Model
            # 因为论文攻击是计算样本在 Global Model 上的梯度 ∇L(x; W_S)
            self.model_template.load_state_dict(server_state)
            
            return self.model_template, delta_gamma

        except RuntimeError as e:
            print(f"[Load Error] 模型结构不匹配或显存不足: {e}")
            return None, None

    def compute_scores_for_loader(self, loader, model, update_vec):
        """
        计算 loader 中所有样本的梯度与 update_vec 的余弦相似度
        """
        scores = []
        update_vec_np = update_vec.cpu().numpy()
        norm_update = np.linalg.norm(update_vec_np)
        
        if norm_update == 0:
            return np.zeros(len(loader.dataset))

        criterion = nn.CrossEntropyLoss()

        # 遍历数据
        for data, target, _ in loader:
            # 针对 Batch 中的每个样本单独计算梯度 (Per-sample gradient)
            # 这非常慢，但符合论文定义。如果显存够大，可以尝试 Opacus 库加速。
            for i in range(len(data)):
                img = data[i:i+1].to(self.device)
                lbl = target[i:i+1].to(self.device)
                
                # 清空梯度
                model.zero_grad()
                
                # 开启梯度记录 (哪怕是 eval 模式)
                for p in model.parameters(): 
                    p.requires_grad = True
                
                # Forward
                out = model(img)
                loss = criterion(out, lbl)
                
                # Backward
                loss.backward()
                
                # 提取完整模型的梯度并展平
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
                        # 计算余弦相似度
                        sim = np.dot(grad_vec, update_vec_np) / (norm_grad * norm_update)
                        scores.append(sim)
                else:
                    scores.append(0.0)
                    
        return np.array(scores)

    def attack(self):
        """
        执行攻击主循环
        """
        start_round = 0 
        end_round = self.args.training_round
        valid_rounds = 0
        
        print(f"Starting Attack on WHOLE models ({start_round} -> {end_round})...")

        # 步长可调整，比如每隔几轮采一次样以节省时间
        step = self.args.client_num//self.args.participant

            
        for r in tqdm(range(start_round, end_round, step), desc="Attacking Rounds"):
            try:
                # 1. 加载模型并计算差值
                model, delta_gamma = self.load_models_and_get_update(r)
                
                if model is None: continue 
                
                
                # 3. 计算 Evaluation Set 分数
                eval_scores = self.compute_scores_for_loader(self.eval_loader, model, delta_gamma)
                self.eval_scores_sum += eval_scores
                
                valid_rounds += 1
                
            except FileNotFoundError:
                # print(f"Round {r} files not found, skipping.")
                continue
            except Exception as e:
                print(f"Error in round {r}: {e}")
                continue

        if valid_rounds == 0:
            print("Error: No valid rounds found!")
            return 0

        # --- 计算平均分 ---
        final_eval_scores = self.eval_scores_sum / valid_rounds


        # 提取 Ground Truth (假设 loader 里的 member 标签是第三个返回值)
        eval_gt = []
        for _, _, m in self.eval_loader:
            eval_gt.extend(m.numpy())
        eval_gt = np.array(eval_gt)
        auc = ROC_AUC_Result_logshow(eval_gt, final_eval_scores,True) # 注意：sklearn需要(y_true, y_score)
        return auc