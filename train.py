import copy
import random
import os
import CSModels
from Data import ClientDataset, load_npz_data, non_iid_dirichlet_split,random_uniform_split
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import sys
import torch
import copy
from utils import path_exists
import torch.nn.functional as F
from torch.func import grad, vmap

import logging
logger = logging.getLogger(__name__)


class FederatedLearning:
    def __init__(self, args):
        self.args = args
        self.datas, self.labels = self.split_data(self.args, path=args.data_path + '/' + args.dataset + '/full.npz')
        self.global_model = self.build_global_model()
        self.client_models = []

        
        self.train_data, self.train_label, self.test_data, self.test_label = self.split_train_test_data(True,self.datas,self.labels,self.args.split_ratio)

    
    def split_data(self, args, path):
        data, label = load_npz_data(path)
        data = data.astype(np.float32)
        label = label.astype(np.int64)

        if args.data_split == 'uniform':
            datas, labels = random_uniform_split(data, label, self.args.client_num,
                                                seed=args.random_seed)
        else:
            datas, labels = non_iid_dirichlet_split(data, label, self.args.client_num, self.args.alpha,
                                                    seed=args.random_seed)

        sample_counts = [len(d) for d in datas]

        max_samples = max(sample_counts) if sample_counts else 0
        min_samples = min(sample_counts) if sample_counts else 0
        print(f"max:{max_samples}  min:{min_samples}")
        return datas, labels
    
    def split_train_test_data(self,flag, data_list, label_list, x):
        if flag is True:

            train_data, test_data = [], []
            train_labels, test_labels = [], []
            
            for data, labels in zip(data_list, label_list):
                data = np.array(data)
                labels = np.array(labels)
                
                assert len(data) == len(labels), "数据和标签的长度不一致！"
                
                train_size = int(len(data) * x)
                
                indices = np.arange(len(data))
                np.random.shuffle(indices)
                shuffled_data = data[indices]
                shuffled_labels = labels[indices]
                
                train_data.append(shuffled_data[:train_size])
                train_labels.append(shuffled_labels[:train_size])
                test_data.append(shuffled_data[train_size:])
                test_labels.append(shuffled_labels[train_size:])
            self.save_split_data(train_data,train_labels,'train_non_iid.npz')
            self.save_split_data(test_data,test_labels,'test_non_iid.npz')
        else:
            train_data,train_labels = self.load_split_data('train_non_iid.npz')
            test_data, test_labels = self.load_split_data('test_non_iid.npz')
        return train_data, train_labels, test_data, test_labels




    

    def save_split_data(self, datas, labels, name):
        args = self.args
        all_client_data_dict = {}
        num_clients = len(datas)

        for client_idx in range(num_clients):
            client_data = datas[client_idx]
            client_labels = labels[client_idx]

            client_data_np = np.array(client_data)
            client_labels_np = np.array(client_labels)

            data_key = f'client_{client_idx}_data'
            labels_key = f'client_{client_idx}_labels'

            all_client_data_dict[data_key] = client_data_np
            all_client_data_dict[labels_key] = client_labels_np

        save_dir = os.path.join(args.data_path, args.dataset, args.model, args.data_split)
        path_exists(save_dir)

        full_save_path = os.path.join(save_dir, f'{name}')

        np.savez(full_save_path, **all_client_data_dict)

    def load_split_data(self, name):
        args = self.args
        full_load_path = os.path.join(args.data_path, args.dataset, args.model, args.data_split, f'{name}')

        loaded_data = np.load(full_load_path)

        loaded_split_data_list = []
        loaded_split_labels_list = []

        client_indices = set()
        for key in loaded_data.files:
            if key.startswith('client_') and key.endswith('_data'):
                try:
                    idx_str = key[len('client_'):-len('_data')]
                    client_idx = int(idx_str)
                    client_indices.add(client_idx)
                except ValueError:
                    pass

        num_clients = len(client_indices)
        sorted_client_indices = sorted(list(client_indices))

        for client_idx in sorted_client_indices:
             data_key = f'client_{client_idx}_data'
             labels_key = f'client_{client_idx}_labels'

             loaded_split_data_list.append(loaded_data[data_key])
             loaded_split_labels_list.append(loaded_data[labels_key])

        loaded_data.close()

        return loaded_split_data_list, loaded_split_labels_list

    def build_dataloader(self, data, label, idx,shuffle=True, drop_last=True):
        dataset = ClientDataset(data[idx], label[idx])
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=shuffle, pin_memory=False,
                                drop_last=drop_last)
        return dataloader
    
    def build_mixloader(self, data, label, idx, num_samples_per_partition):
        selected_data = []
        selected_labels = []

        for i, (partition_data, partition_label) in enumerate(zip(data, label)):
            if i != idx:
                total_samples = partition_data.shape[0]
                sampled_indices = np.random.choice(total_samples, num_samples_per_partition, replace=False)
                
                part_data = torch.tensor(partition_data[sampled_indices]) if isinstance(partition_data, np.ndarray) else partition_data[sampled_indices]
                part_label = torch.tensor(partition_label[sampled_indices]) if isinstance(partition_label, np.ndarray) else partition_label[sampled_indices]
                
                selected_data.append(part_data)
                selected_labels.append(part_label)

        selected_data = torch.cat(selected_data, dim=0)
        selected_labels = torch.cat(selected_labels, dim=0)

        dataset = ClientDataset(selected_data, selected_labels)
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False, pin_memory=False, drop_last=True)

        return dataloader

    def build_global_model(self):
        Server = CSModels.PublicLayer(self.args)
        Server.to(self.args.device)
        return Server

    def build_private_model(self):
        Private = CSModels.PrivateLayer(self.args)
        return Private

    def aggregate_params(self, client_params_list):
        aggregated_params = {}
        total_num = self.args.client_num
        for param_key in client_params_list[0].keys():
            aggregated_params[param_key] = sum(
                client_params_list[i][param_key]
                for i in range(len(client_params_list)))*(1 / len(client_params_list)
                )
        logging.info('params aggregate')
        return aggregated_params

    def update_global_model(self, parameters):
        logging.info('update global model')
        self.global_model.load_state_dict(parameters)

    def update_client_model(self, idx, model, aggregated_parameters):
        logging.info(f'update client public model:{idx+1}')
        model.load_state_dict(aggregated_parameters)

    def train_FL_models(self):
        args = self.args
        client_num = args.client_num
        participant = args.participant
        device = args.device

        path = args.model_path + '/' + args.model+'/'+ args.dataset + '/'
        server_path = path + 'server_model'
        client_path = path + 'client_model'
        arxiv_path = path+'our_model/arxiv/'
        path_exists(server_path)
        path_exists(client_path)




        lr = args.lr
        for round in range(args.training_round):
            if args.steplr is True and round % args.lr_step == 0:
                lr = lr*args.lr_gamma
                print(f"-----this round lr:{lr}-----")
            items = list(range(client_num))
            if self.args.random_client_mode is True:
                idx_cur_round=random.sample(items,participant)
            else:
                start_idx = (round*participant)%client_num
                if start_idx+participant<=client_num:
                    idx_cur_round=items[start_idx:start_idx+participant]
                else:
                    remain = participant-(client_num-start_idx)
                    idx_cur_round=items[start_idx:]+items[:remain]
            self.client_models.clear()
            for idx in idx_cur_round:
                client_model = self.build_private_model()
                self.update_client_model(idx, client_model, self.global_model.state_dict())
                client_model.to(self.args.device)
                self.client_models.append(client_model)
                
            for i,c_idx in enumerate(idx_cur_round):
                train_loader = self.build_dataloader(self.train_data,self.train_label,c_idx)
                test_loader = self.build_dataloader(self.test_data, self.test_label,c_idx,shuffle=False)
                client_model = self.client_models[i]
                for param in client_model.parameters():
                    param.requires_grad = True
                if args.optimizer == 'Adam':
                    optimizer = optim.Adam(
                        client_model.parameters(),
                        lr=lr,
                        betas=(0.9, 0.999),
                    )

                elif args.optimizer == 'SGD':
                    optimizer = optim.SGD(
                        client_model.parameters(),
                        lr=lr,
                        momentum=0.9,
                        weight_decay=1e-4
                    )
                else:
                    logging.warning('optimizer is wrong')
                    sys.exit(1)

                loss_fn = nn.CrossEntropyLoss()
                loss_noreduce = nn.CrossEntropyLoss(reduction='none')
                client_model.train()
                for epoch in range(args.epochs):
                    running_loss, correct, total = 0, 0, 0
                    for _, (data, label) in enumerate(train_loader):
                        optimizer.zero_grad()
                        data = data.to(device)
                        label = label.to(device)

                        outputs = client_model(data)
                        loss = loss_fn(outputs, label)

                        loss.backward()

                        optimizer.step()
                        _, predicted = torch.max(outputs, 1)
                        total += label.size(0)
                        correct += (predicted == label).sum().item()
                        running_loss += loss.item()
                    print(
                        f"round:{round + 1}, client:{c_idx + 1}, Epoch {epoch + 1}, Acc:{correct / total}, Loss: {running_loss / len(train_loader)}")
                if c_idx in self.args.save_client_model_idx:
                    torch.save(client_model.state_dict(), client_path + f'/client_{c_idx}_{round}.pth')

                del optimizer
                

                if args.arxiv_save is True:
                    with torch.no_grad():
                        total, correct, loss_meter= 0, 0,0
                        for _, (x, y) in enumerate(test_loader):
                            x = x.to(device)
                            y = y.to(device)
                            outputs = client_model(x)
                            loss_meter += F.cross_entropy(outputs, y, reduction='sum').item()
                            _, predicted = torch.max(outputs, 1)
                            total += y.size(0)
                            correct += (predicted == y).sum().item()
                    acc = correct/total
                    loss_meter /= total
                    save_dict={}
                    save_dict['test_acc']=acc
                    save_dict["test_loss"]=loss_meter
                    test_loader_for_arxiv_train = self.build_dataloader(self.test_data,self.test_label,self.args.arxiv_client[0],shuffle=False)
                    test_res = self.get_all_losses(test_loader_for_arxiv_train,client_model,loss_noreduce,self.args.device)
                    save_dict['test_index']=c_idx
                    save_dict['test_res']=test_res

                    train_loader_for_arxiv_train = self.build_dataloader(self.train_data,self.train_label,self.args.arxiv_client[0],shuffle=False)
                    train_res = self.get_all_losses(train_loader_for_arxiv_train,client_model,loss_noreduce,self.args.device)
                    save_dict['train_index']=self.args.arxiv_client[0]
                    save_dict['train_res']=train_res

                    train_loader_for_arxiv_val = self.build_dataloader(self.train_data,self.train_label,self.args.arxiv_client[1],shuffle=False)
                    val_res = self.get_all_losses(train_loader_for_arxiv_val,client_model,loss_noreduce,self.args.device)
                    save_dict['val_index']=self.args.arxiv_client[1]
                    save_dict["val_res"]=val_res

                    model_grads= []
                    for name, local_param in client_model.named_parameters():
                        if local_param.requires_grad == True:
                            para_diff =  self.global_model.state_dict()[name] - client_model.state_dict()[name]
                            model_grads.append(para_diff.detach().cpu().flatten())
                    model_grads=torch.cat(model_grads,-1)
                    cos_model = self.build_global_model()
                    cos_model= cos_model.to(self.args.device)
                    cos_model.load_state_dict(self.global_model.state_dict())
                    train_cos,train_diffs, train_norm,val_cos, val_diffs,val_norm,test_cos, test_diffs,test_norm, mix_cos, mix_diffs,mix_norm=self.get_all_cos_functorch(
                                                                    cos_model, 
                                                                    test_loader,
                                                                    train_loader, 
                                                                    None,
                                                                    None,
                                                                    model_grads, 
                                                                    lr,
                                                                    self.args.optimizer)
                    save_dict['train_cos']=train_cos
                    save_dict['val_cos']=val_cos
                    save_dict['test_cos']=test_cos
                    save_dict['mix_cos']=mix_cos
                    save_dict['train_diffs']=train_diffs
                    save_dict['val_diffs']=val_diffs
                    save_dict['test_diffs']=test_diffs
                    save_dict['mix_diffs']=mix_diffs
                    save_dict['train_grad_norm']=train_norm
                    save_dict['val_grad_norm']=val_norm
                    save_dict['test_grad_norm']=test_norm
                    save_dict['mix_grad_norm']=mix_norm
                    if not os.path.exists(arxiv_path):
                        os.makedirs(arxiv_path)
                        print('MIA Score Saved in:', os.path.join(arxiv_path))
                    torch.save(save_dict, os.path.join(arxiv_path+f'client_{c_idx}_round_{round}.pkl'))


            self.global_model.train()
            param_list = []
            for i in range(len(self.client_models)):
                param_list.append(self.client_models[i].state_dict())
            global_param = self.aggregate_params(param_list)
            self.update_global_model(global_param)


            self.global_model.eval()
            for i in range(client_num):
                test_loader = self.build_dataloader(self.test_data, self.test_label,i)
                with torch.no_grad():
                    total, correct = 0, 0
                    for _, (x, y) in enumerate(test_loader):
                        x = x.to(device)
                        y = y.to(device)
                        outputs = self.global_model(x)
                        _, predicted = torch.max(outputs, 1)
                        total += y.size(0)
                        correct += (predicted == y).sum().item()
                    print(f"----round:{round + 1},  Global Model test Acc(client:{i+1}):{correct / total}----")
            torch.save(self.global_model.state_dict(), server_path + f'/server_{round}.pth')

        logging.info('finish training FL model')


    def get_all_losses(self, dataloader, model, criterion, device,req_logits=False):
        model.eval()
        losses = []
        logits = []
        labels = []

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                losses.append(loss.cpu().numpy())
                logits.append(outputs.cpu())
                labels.append(targets.cpu())

        losses = np.concatenate(losses)
        logits = torch.cat(logits)
        labels = torch.cat(labels)
        return {"loss":losses,"logit":logits,"labels":labels}
    

    def get_all_cos_functorch(self, cos_model, test_loader, train_loader, mix_loader, DP_INIT_LOADER,model_grads, lr, optim_choice):
        device = self.args.device
        cos_model = cos_model.to(device)
        
        for module in cos_model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        
        if optim_choice == "SGD":
            optimizer = optim.SGD(cos_model.parameters(), lr=lr, momentum=0.9)
        else:
            optimizer = optim.Adam(cos_model.parameters(), lr=lr, betas=(0.9, 0.999))
        
        train_cos, train_diffs, train_norm = self.get_cos_score_functorch(train_loader, cos_model, device, model_grads)
        test_cos, test_diffs, test_norm = self.get_cos_score_functorch(test_loader, cos_model, device, model_grads)
        
        return train_cos, train_diffs, train_norm, None, None, None, test_cos, test_diffs, test_norm, None,None,None
    
    def get_cos_score_functorch_origin(self, dataloader, model, device, model_grads):
        
        model.train()
        
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                module.eval()
        
        model.half() 
        
        def compute_loss(params, buffers, x, y):
            output = torch.func.functional_call(model, (params, buffers), x.unsqueeze(0))
            return F.cross_entropy(output, y.unsqueeze(0))
        
        params = {k: v.detach().half() for k, v in model.named_parameters()}
        buffers = {k: v.detach().half() for k, v in model.named_buffers()}
        
        ft_compute_grad = grad(compute_loss)
        batch_ft_compute_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0))
        
        all_cos, all_diffs, all_norms = [], [], []
        
        model_grads = model_grads.to(device).half().view(1, -1)
        
        for x, y in dataloader:
            x, y = x.to(device).half(), y.to(device)
            
            batch_grads = batch_ft_compute_grad(params, buffers, x, y)
            
            flat_grads = torch.cat([g.view(g.shape[0], -1) for g in batch_grads.values()], dim=1)
            
            cos_sim = F.cosine_similarity(flat_grads, model_grads, dim=1)
            diff = torch.norm(flat_grads - model_grads, p=2, dim=1)
            norm = torch.norm(flat_grads, p=2, dim=1)
            
            all_cos.extend(cos_sim.float().cpu().tolist())
            all_diffs.extend(diff.float().cpu().tolist())
            all_norms.extend(norm.float().cpu().tolist())
        
        model.eval().float()
        
        return torch.tensor(all_cos), torch.tensor(all_diffs), torch.tensor(all_norms)

    def get_cos_score_functorch(self, dataloader, model, device, model_grads):
        model.eval()
        
        def compute_loss(params, x, y):
            output = torch.func.functional_call(model, params, x.unsqueeze(0))
            return F.cross_entropy(output, y.unsqueeze(0))
        
        params = {k: v.detach() for k, v in model.named_parameters()}
        buffers = {k: v.detach() for k, v in model.named_buffers()}
        
        ft_compute_grad = grad(compute_loss)
        batch_ft_compute_grad = vmap(ft_compute_grad, in_dims=(None, 0, 0))
        
        all_cos, all_diffs, all_norms = [], [], []
        model_grads = model_grads.to(device)
        
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            
            batch_grads = batch_ft_compute_grad(params, x, y)
            
            flat_grads = torch.cat([g.view(g.shape[0], -1) for g in batch_grads.values()], dim=1)
            
            cos_sim = F.cosine_similarity(flat_grads, model_grads.view(1, -1), dim=1)
            diff = torch.norm(flat_grads - model_grads, p=2, dim=1)
            norm = torch.norm(flat_grads, p=2, dim=1)
            
            all_cos.extend(cos_sim.cpu().tolist())
            all_diffs.extend(diff.cpu().tolist())
            all_norms.extend(norm.cpu().tolist())
        
        return torch.tensor(all_cos), torch.tensor(all_diffs), torch.tensor(all_norms)