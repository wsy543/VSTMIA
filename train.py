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
from utils import path_exists, set_seed

import logging
logger = logging.getLogger(__name__)


class FederatedLearning:
    def __init__(self, args):
        self.args = args
        set_seed(self.args.random_seed)
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
