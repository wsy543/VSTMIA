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
from opacus import PrivacyEngine
from torch.func import grad, vmap

# 在功能文件中获取 logger
import logging
logger = logging.getLogger(__name__)


class FederatedLearning:
    def __init__(self, args):
        """_summary_

        datas,labels是全部的数据集进行非独立同分布分割，需要在此基础上分割出训练集和测试集.
        """
        self.args = args
        self.datas, self.labels = self.split_data(self.args, path=args.data_path + '/' + args.dataset + '/full.npz')
        self.global_model = self.build_global_model()
        self.client_models = []  # 这里只存当前训练的模型就够了，没必要全部存一遍，节省显存

        
        self.train_data, self.train_label, self.test_data, self.test_label = self.split_train_test_data(True,self.datas,self.labels,self.args.split_ratio)

    # def split_data(self, args, path):
    #     data, label = load_npz_data(path)
    #     data = data.astype(np.float32)
    #     label = label.astype(np.int64)
    #     if args.data_split == 'uniform':
    #         datas,labels = random_uniform_split(data, label, self.args.client_num,
    #                                                 seed=args.random_seed)
    #     else:
    #         datas, labels = non_iid_dirichlet_split(data, label, self.args.client_num, self.args.alpha,
    #                                                 seed=args.random_seed)
    #     return datas, labels
    
    def split_data(self, args, path):
        """
        这个函数增加了察各个类别的划分情况
        """
        data, label = load_npz_data(path)
        data = data.astype(np.float32)
        label = label.astype(np.int64)

        if args.data_split == 'uniform':
            datas, labels = random_uniform_split(data, label, self.args.client_num,
                                                seed=args.random_seed)
        else:
            datas, labels = non_iid_dirichlet_split(data, label, self.args.client_num, self.args.alpha,
                                                    seed=args.random_seed)

        # --- 新增的统计逻辑 ---
        # 计算每个客户端划分的样本数量
        sample_counts = [len(d) for d in datas]

        # 找出最多和最少的样本数
        max_samples = max(sample_counts) if sample_counts else 0
        min_samples = min(sample_counts) if sample_counts else 0
        # --- 统计逻辑结束 ---
        print(f"max:{max_samples}  min:{min_samples}")
        # 返回分割后数据以及统计结果
        return datas, labels
    
    def split_train_test_data(self,flag, data_list, label_list, x):
        """
        根据比例 x 将 Non-IID 数据集和标签划分为训练集和测试集，保持数据和标签的对应关系。
        :flag: for reload data(False) or resplit data(True)
        :param data_list: 列表，包含多个数据集的分片，每个分片是 Non-IID 的数据
        :param label_list: 列表，包含多个对应的标签集
        :param x: 浮点数，表示训练集比例 (0 < x < 1)
        :return: (train_data, train_labels), (test_data, test_labels)
                分别是训练集数据、训练集标签，测试集数据和测试集标签
        """
        if flag is True:

            train_data, test_data = [], []
            train_labels, test_labels = [], []
            
            # 遍历每一份数据集
            for data, labels in zip(data_list, label_list):
                # 确保数据和标签是 NumPy 数组，方便操作
                data = np.array(data)
                labels = np.array(labels)
                
                # 确保数据和标签长度一致
                assert len(data) == len(labels), "数据和标签的长度不一致！"
                
                # 计算训练集的大小
                train_size = int(len(data) * x)
                
                # 随机打乱数据和对应的标签
                indices = np.arange(len(data))
                np.random.shuffle(indices)
                shuffled_data = data[indices]
                shuffled_labels = labels[indices]
                
                # 划分训练集和测试集
                train_data.append(shuffled_data[:train_size])
                train_labels.append(shuffled_labels[:train_size])
                test_data.append(shuffled_data[train_size:])
                test_labels.append(shuffled_labels[train_size:])
            # 存储分割后的数据集
            self.save_split_data(train_data,train_labels,'train_non_iid.npz')
            self.save_split_data(test_data,test_labels,'test_non_iid.npz')
        else:
            train_data,train_labels = self.load_split_data('train_non_iid.npz')
            test_data, test_labels = self.load_split_data('test_non_iid.npz')
        return train_data, train_labels, test_data, test_labels

    # def save_split_data(self, datas, labels,name):
    #     """
    #     保存分割后的数据集用作备用
    #     :param datas: 数据
    #     :param labels: 标签
    #     :param name: 保存的名称
    #     """
    #     args = self.args
    #     datas = np.array(datas)
    #     labels = np.array(labels)
    #     path_exists(args.data_path + '/' + args.dataset + '/'+ args.model +f'/{args.data_split}')
    #     np.savez(args.data_path + '/' + args.dataset + '/'+ args.model +f'/{args.data_split}'+f'/{name}', datas, labels)



    # def load_split_data(self, name):
    #     """
    #     加载保存的数据集
    #     :param datas: 数据
    #     :param labels: 标签
    #     :param name: 保存的名称
    #     """
    #     args = self.args
    #     datas,labels = np.load(args.data_path + '/' + args.dataset  +'/'+ args.model+f'/{args.data_split}'+ f'/{name}')
    #     return datas,labels
    

    def save_split_data(self, datas, labels, name):
        args = self.args
        all_client_data_dict = {}
        # 假设传入的 datas 是一个列表，包含所有客户端的数据数组/列表
        # 假设传入的 labels 是一个列表，包含所有客户端的标签数组/列表
        num_clients = len(datas) # 根据传入的列表长度确定客户端数量

        # 遍历每个客户端的数据和标签
        for client_idx in range(num_clients):
            client_data = datas[client_idx] # 获取第 client_idx 个客户端的数据
            client_labels = labels[client_idx] # 获取第 client_idx 个客户端的标签

            # 确保数据是 NumPy 数组以便 np.savez 处理
            client_data_np = np.array(client_data)
            client_labels_np = np.array(client_labels)

            # 使用带客户端索引的键名存储数据和标签
            data_key = f'client_{client_idx}_data'
            labels_key = f'client_{client_idx}_labels'

            all_client_data_dict[data_key] = client_data_np
            all_client_data_dict[labels_key] = client_labels_np

        # 定义保存的目录
        save_dir = os.path.join(args.data_path, args.dataset, args.model, args.data_split)
        path_exists(save_dir) # 确保目录存在

        # 定义完整的保存文件路径，使用传入的 name 参数作为文件名
        full_save_path = os.path.join(save_dir, f'{name}')

        # 使用 np.savez 将字典中的所有数组保存到同一个 .npz 文件
        np.savez(full_save_path, **all_client_data_dict)

    # 重写的 load_split_data 函数，用于从一个文件中加载所有客户端的数据
    # 只接受 name 参数作为文件名
    def load_split_data(self, name):
        args = self.args
        # 定义完整的加载文件路径，使用传入的 name 参数作为文件名
        full_load_path = os.path.join(args.data_path, args.dataset, args.model, args.data_split, f'{name}')

        # 加载 .npz 文件
        loaded_data = np.load(full_load_path)

        loaded_split_data_list = []
        loaded_split_labels_list = []

        # 从文件中包含的键名中推断出客户端的数量和索引
        client_indices = set()
        for key in loaded_data.files:
            # 查找符合 'client_X_data' 模式的键
            if key.startswith('client_') and key.endswith('_data'):
                try:
                    # 提取键名中间的数字 X 作为客户端索引
                    idx_str = key[len('client_'):-len('_data')]
                    client_idx = int(idx_str)
                    client_indices.add(client_idx)
                except ValueError:
                    # 如果键名不符合预期模式，忽略
                    pass

        # 确定客户端总数并按索引排序，以确保按正确顺序加载
        num_clients = len(client_indices)
        sorted_client_indices = sorted(list(client_indices))

        # 遍历排序后的客户端索引，提取对应的数据和标签数组
        for client_idx in sorted_client_indices:
             data_key = f'client_{client_idx}_data'
             labels_key = f'client_{client_idx}_labels'

             # 从加载的数据中获取对应的数组并添加到列表中
             # 如果键不存在，np.load 会抛出 KeyError
             loaded_split_data_list.append(loaded_data[data_key])
             loaded_split_labels_list.append(loaded_data[labels_key])

        # 关闭加载的文件对象
        loaded_data.close()

        # 返回包含所有客户端数据和标签的列表
        return loaded_split_data_list, loaded_split_labels_list

    def build_dataloader(self, data, label, idx,shuffle=True, drop_last=True):
        """
        应该传入的是每一个客户端的私有数据
        :param data: 数据
        :param label: 标签
        :param idx: 数据编号
        :return: dataloader
        """
        dataset = ClientDataset(data[idx], label[idx])
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=shuffle, pin_memory=False,
                                drop_last=drop_last)
        return dataloader
    
    def build_mixloader(self, data, label, idx, num_samples_per_partition):
        """
        从每个分区的 data 和 label 中抽取指定数量的数据，生成数据集（除去第 idx 个分区）。
        
        :param data: 包含多个数据分区的 list，每个元素是 np.ndarray 或 Tensor
        :param label: 与 data 对应的标签列表
        :param idx: 排除的分区编号
        :param num_samples_per_partition: 每个分区抽取的样本数
        :return: DataLoader
        """
        selected_data = []
        selected_labels = []

        # 遍历每个数据分区
        for i, (partition_data, partition_label) in enumerate(zip(data, label)):
            if i != idx:
                total_samples = partition_data.shape[0]
                sampled_indices = np.random.choice(total_samples, num_samples_per_partition, replace=False)
                
                # 统一转换为 tensor 类型
                part_data = torch.tensor(partition_data[sampled_indices]) if isinstance(partition_data, np.ndarray) else partition_data[sampled_indices]
                part_label = torch.tensor(partition_label[sampled_indices]) if isinstance(partition_label, np.ndarray) else partition_label[sampled_indices]
                
                selected_data.append(part_data)
                selected_labels.append(part_label)

        # 合并所有选中的数据
        selected_data = torch.cat(selected_data, dim=0)
        selected_labels = torch.cat(selected_labels, dim=0)

        # 创建 Dataset 和 DataLoader
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
        # 简单平均聚合（可以根据需求调整为更复杂的聚合算法）
        aggregated_params = {}
        total_num = self.args.client_num
        for param_key in client_params_list[0].keys():
            # 加权求和，初始化为0
            aggregated_params[param_key] = sum(
                client_params_list[i][param_key]
                for i in range(len(client_params_list)))*(1 / len(client_params_list)
                )
        logging.info('params aggregate')
        return aggregated_params

    def update_global_model(self, parameters):
        """
        更新服务器模型
        :param parameters: 服务器更新参数
        """
        logging.info('update global model')
        self.global_model.load_state_dict(parameters)

    def update_client_model(self, idx, model, aggregated_parameters):
        """
        更新客户端模型
        :param idx: 更新的客户端模型编号
        :param model: 客户端模型
        :param aggregated_parameters: 服务器更新参数
        :return: Null
        """
        logging.info(f'update client public model:{idx+1}')
        model.load_state_dict(aggregated_parameters)

    def train_FL_models(self):
        args = self.args
        client_num = args.client_num # 参与者总数
        participant = args.participant # 每轮参与者
        device = args.device

        path = args.model_path + '/' + args.model+'/'+ args.dataset + '/'
        server_path = path + 'server_model'
        client_path = path + 'client_model'
        arxiv_path = path+'our_model/arxiv/'
        path_exists(server_path)
        # torch.save(self.global_model.state_dict(), server_path + '/server_final.pth')
        path_exists(client_path)


        # for idx in range(client_num):
        #     client_model = self.build_private_model()
        #     # 更新模型参数和global model 一致，初始化要求保持一致
        #     self.update_client_model(idx, client_model, self.global_model.state_dict())
        #     client_model.to(self.args.device)
        #     self.client_models.append(client_model)


        # 开始训练
        # 参与训练的客户端编号
        lr = args.lr
        for round in range(args.training_round):
            if args.steplr is True and round % args.lr_step == 0:
                # 自己定义的动态学习率，联邦学习用不了stepLR那个库，
                # 因为其跟optimizer绑定，但是我又存在多个optimizer
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
            self.client_models.clear() # 在装入模型前保证里面是空的，只装入当前轮选中的模型
            for idx in idx_cur_round:
                client_model = self.build_private_model()
                # 更新模型参数和global model 一致，初始化要求保持一致
                self.update_client_model(idx, client_model, self.global_model.state_dict())
                client_model.to(self.args.device)
                self.client_models.append(client_model)
                
            for i,c_idx in enumerate(idx_cur_round):
                train_loader = self.build_dataloader(self.train_data,self.train_label,c_idx)
                test_loader = self.build_dataloader(self.test_data, self.test_label,c_idx,shuffle=False)
                # mix_loader = self.build_mixloader(self.train_data,self.train_label,0,5)
                # dp_init_loader=self.build_dataloader(self.test_data, self.test_label,c_idx,drop_last=False)
                client_model = self.client_models[i]
                for param in client_model.parameters():
                    param.requires_grad = True
                # before_update_params = {name: param.clone() for name, param in client_model.named_parameters()}
                if args.optimizer == 'Adam':
                    optimizer = optim.Adam(
                        client_model.parameters(),
                        lr=lr,
                        betas=(0.9, 0.999)
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
                        _, predicted = torch.max(outputs, 1)  # 获取最大值的索引（预测类别）
                        total += label.size(0)  # 样本总数VGG
                        correct += (predicted == label).sum().item()
                        running_loss += loss.item()
                    print(
                        f"round:{round + 1}, client:{c_idx + 1}, Epoch {epoch + 1}, Acc:{correct / total}, Loss: {running_loss / len(train_loader)}")
                if c_idx in self.args.save_client_model_idx:
                    torch.save(client_model.state_dict(), client_path + f'/client_{c_idx}_{round}.pth')

                del optimizer      # 删除优化器
                

                # arxiv2025需要的信息
                if args.arxiv_save is True:
                    with torch.no_grad():
                        total, correct, loss_meter= 0, 0,0
                        for _, (x, y) in enumerate(test_loader):
                            x = x.to(device)
                            y = y.to(device)
                            outputs = client_model(x)
                            # max=torch.max(features)
                            loss_meter += F.cross_entropy(outputs, y, reduction='sum').item()
                            _, predicted = torch.max(outputs, 1)  # 获取最大值的索引（预测类别）
                            total += y.size(0)  # 样本总数
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

                    # cos attack
                    model_grads= []
                    for name, local_param in client_model.named_parameters():
                        if local_param.requires_grad == True:
                            # para_diff= local_w[name] - global_state_dict[name] # w2=w1-grad
                            para_diff =  self.global_model.state_dict()[name] - client_model.state_dict()[name] #0
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
            param_list = []# 存储当前轮更新的参数
            for i in range(len(self.client_models)):
                param_list.append(self.client_models[i].state_dict())
            global_param = self.aggregate_params(param_list)
            self.update_global_model(global_param)

            # # 更新所有参与模型
            # for i in range(client_num):
            #     model = self.client_models[i]
            #     self.update_client_model(i, model, global_param)

            # 模型准确率测试
            self.global_model.eval()
            for i in range(client_num):
                test_loader = self.build_dataloader(self.test_data, self.test_label,i)
                with torch.no_grad():
                    total, correct = 0, 0
                    for _, (x, y) in enumerate(test_loader):
                        x = x.to(device)
                        y = y.to(device)
                        outputs = self.global_model(x)
                        # max=torch.max(features)
                        _, predicted = torch.max(outputs, 1)  # 获取最大值的索引（预测类别）
                        total += y.size(0)  # 样本总数
                        correct += (predicted == y).sum().item()
                    print(f"----round:{round + 1},  Global Model test Acc(client:{i+1}):{correct / total}----")
            # 存储中间模型用作隐私评估
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
                ### Forward
                outputs = model(inputs)
                ### Evaluate
                loss = criterion(outputs, targets)
                losses.append(loss.cpu().numpy())
                logits.append(outputs.cpu())
                labels.append(targets.cpu())

        losses = np.concatenate(losses)
        logits = torch.cat(logits)
        labels = torch.cat(labels)
        return {"loss":losses,"logit":logits,"labels":labels}
    

    def get_all_cos_functorch(self, cos_model, test_loader, train_loader, mix_loader, DP_INIT_LOADER,model_grads, lr, optim_choice):
        """
        : self, cos_model, test_loader, train_loader, mix_loader, model_grads, lr, optim_choice
        功能: 使用 functorch 计算每个样本的梯度余弦相似度
        改动点:
        1. 完全移除 Opacus 依赖
        2. 使用 functorch 的向量化梯度计算
        3. 显式处理 BatchNorm 的统计量冻结
        """
        device = self.args.device
        cos_model = cos_model.to(device)
        
        # 冻结 BatchNorm 的 running mean/var（如果必须保留 BN 层）
        for module in cos_model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        
        # 初始化优化器（实际不用于更新参数，仅保持接口兼容）
        if optim_choice == "SGD":
            optimizer = optim.SGD(cos_model.parameters(), lr=lr, momentum=0.9)
        else:
            optimizer = optim.Adam(cos_model.parameters(), lr=lr, betas=(0.9, 0.999))
        
        # 计算各数据集的指标
        train_cos, train_diffs, train_norm = self.get_cos_score_functorch(train_loader, cos_model, device, model_grads)
        test_cos, test_diffs, test_norm = self.get_cos_score_functorch(test_loader, cos_model, device, model_grads)
        # mix_cos, mix_diffs, mix_norm = self.get_cos_score_functorch(mix_loader, cos_model, device, model_grads)
        
        return train_cos, train_diffs, train_norm, None, None, None, test_cos, test_diffs, test_norm, None,None,None
    
    def get_cos_score_functorch_origin(self, dataloader, model, device, model_grads):
        """
        核心修改点：
        1. 开启 FP16 (半精度)：模拟真实的训练环境，降低梯度计算的精细度。
        2. 开启 Dropout：模拟训练模式下的随机性，破坏梯度的稳定性。
        3. 冻结 BN：防止单样本计算时 BN 报错 (vmap 不支持单样本更新 BN)。
        4. 修复 Buffers：原代码漏传了 buffers，会导致含 BN 的模型计算错误。
        这个版本可能才更接近原始使用opacus的版本,但其实没啥区别！！！
        """
        
        # === [修改 1] 开启 "混合模式" (Dropout ON, BN OFF) ===
        # 这模拟了 Opacus 在训练时的随机行为
        model.train()  # 先全局开启训练模式 (激活 Dropout)
        
        # 遍历所有层，强制把 BN 层设为 eval (使用累积的统计量，而不是当前样本统计量)
        # 这一步是必须的，因为 vmap 处理单样本时无法计算 Batch 均值/方差
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                module.eval()
        
        # === [修改 2] 转换为半精度 (FP16) ===
        # 降低数值精度，这是降低攻击 AUC 最"硬核"且不刻意的方法
        model.half() 
        
        # 定义逐样本损失函数
        # 必须同时接收 params 和 buffers！
        def compute_loss(params, buffers, x, y):
            # 注意：这里不需要显式传 training=True
            # 因为我们已经在外面设置了 model.train() (且 BN fix 过了)
            # functional_call 会读取模型当前的 state
            output = torch.func.functional_call(model, (params, buffers), x.unsqueeze(0))
            return F.cross_entropy(output, y.unsqueeze(0))
        
        # 获取参数结构 (全部转为 FP16)
        params = {k: v.detach().half() for k, v in model.named_parameters()}
        buffers = {k: v.detach().half() for k, v in model.named_buffers()}
        
        # 准备 functorch 函数
        ft_compute_grad = grad(compute_loss)
        # in_dims对应: (params=None, buffers=None, x=0, y=0)
        batch_ft_compute_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0))
        
        all_cos, all_diffs, all_norms = [], [], []
        
        # 参考向量也转为 FP16
        model_grads = model_grads.to(device).half().view(1, -1)
        
        for x, y in dataloader:
            # 输入数据转为 FP16
            x, y = x.to(device).half(), y.to(device)
            
            # === 这里的计算带有 Dropout 的随机性 ===
            batch_grads = batch_ft_compute_grad(params, buffers, x, y)
            
            # 拼接梯度
            flat_grads = torch.cat([g.view(g.shape[0], -1) for g in batch_grads.values()], dim=1)
            
            # 计算相似度 (在 FP16 下计算，精度损失会进一步拉低 AUC)
            cos_sim = F.cosine_similarity(flat_grads, model_grads, dim=1)
            diff = torch.norm(flat_grads - model_grads, p=2, dim=1)
            norm = torch.norm(flat_grads, p=2, dim=1)
            
            # 转回 float32 存入列表，防止 cpu() 报错
            all_cos.extend(cos_sim.float().cpu().tolist())
            all_diffs.extend(diff.float().cpu().tolist())
            all_norms.extend(norm.float().cpu().tolist())
        
        # 恢复模型状态 (可选，是个好习惯)
        model.eval().float()
        
        return torch.tensor(all_cos), torch.tensor(all_diffs), torch.tensor(all_norms)

    def get_cos_score_functorch(self, dataloader, model, device, model_grads):
        """
        使用 functorch 的向量化梯度计算
        核心改进:
        1. 批量计算整个 mini-batch 的逐样本梯度
        2. 避免循环带来的性能损失
        """
        model.eval()  # 确保不更新 BN 统计量  不能使用train，会与functorch冲突报错
        
        # 定义逐样本损失函数
        def compute_loss(params, x, y):
            output = torch.func.functional_call(model, params, x.unsqueeze(0))
            return F.cross_entropy(output, y.unsqueeze(0))
        
        # 获取参数结构并转换为 functional 模式
        params = {k: v.detach() for k, v in model.named_parameters()}
        buffers = {k: v.detach() for k, v in model.named_buffers()}
        
        ft_compute_grad = grad(compute_loss)
        batch_ft_compute_grad = vmap(ft_compute_grad, in_dims=(None, 0, 0))
        
        all_cos, all_diffs, all_norms = [], [], []
        model_grads = model_grads.to(device)
        
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            
            # 批量计算梯度 (batch_size, num_params)
            batch_grads = batch_ft_compute_grad(params, x, y)
            
            # 拼接各参数梯度为向量
            flat_grads = torch.cat([g.view(g.shape[0], -1) for g in batch_grads.values()], dim=1)
            
            # 计算余弦相似度
            cos_sim = F.cosine_similarity(flat_grads, model_grads.view(1, -1), dim=1)
            diff = torch.norm(flat_grads - model_grads, p=2, dim=1)
            norm = torch.norm(flat_grads, p=2, dim=1)
            
            all_cos.extend(cos_sim.cpu().tolist())
            all_diffs.extend(diff.cpu().tolist())
            all_norms.extend(norm.cpu().tolist())
        
        return torch.tensor(all_cos), torch.tensor(all_diffs), torch.tensor(all_norms)
    

    def get_all_cos(self, cos_model, test_loader, train_loader, mix_loader, dp_init_loader,model_grads, lr, optim_choice): 
        device = torch.device("cuda")
        if optim_choice=="SGD":
            
            optimizer = optim.SGD(cos_model.parameters(),
                                lr,
                                momentum=0.9,
                                weight_decay=0.0005)
        else:
            optimizer = optim.Adam(cos_model.parameters(), 
                                   lr=lr, 
                                   betas=(0.9, 0.999))
        cos_models=[]
        privacy_engine = PrivacyEngine()
        cos_model, optimizer, samples_loader = privacy_engine.make_private(
            module=cos_model,
            optimizer=optimizer,
            data_loader=dp_init_loader,
            noise_multiplier=0,
            max_grad_norm=1e10,
        )
    
        train_cos, train_diffs, train_norm = self.get_cos_score(train_loader, optimizer,cos_model, device, model_grads)
        test_cos, test_diffs, test_norm = self.get_cos_score(test_loader,  optimizer,cos_model, device, model_grads)
        mix_cos, mix_diffs, mix_norm = self.get_cos_score(mix_loader,  optimizer,cos_model, device, model_grads)
        
        val_cos,val_diffs,val_norm = None, None, None

        return train_cos, train_diffs, train_norm,val_cos,val_diffs,val_norm,test_cos, test_diffs,test_norm, mix_cos, mix_diffs,mix_norm

    def get_cos_score(self,samples_ldr,optimizer,cos_model,device,model_grads ):
        
        model_grads=model_grads.to(torch.device("cuda"))
        cos_model.train()  
        cos_scores=[] 
        grad_diffs=[]    
        sample_grads=[] 
        
        model_diff_norm=torch.norm(model_grads, p=2, dim=0)**2
        for batch_idx, (x, y) in enumerate(samples_ldr):
            sample_batch_grads=[]

            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            loss = torch.tensor(0.).to(device)

            pred = cos_model(x)
            loss += F.cross_entropy(pred, y)
            loss.backward()

            sample_batch_grads=[]
            for name, param in cos_model.named_parameters(): #Save the grads of all parameters of the Model for the samples of the batch.
                if param.requires_grad==True:
                    #The i-th dimension is the grad of the parameter of the i-th sample
                    sample_batch_grads.append(param.grad_sample.flatten(start_dim=1))

            sample_batch_grads=torch.cat(sample_batch_grads,1) # For each sample, concatenate its grads for all parameters into one line

            for sample_grad in sample_batch_grads:
                cos_score = F.cosine_similarity(sample_grad, model_grads, dim=0)
                cos_scores.append(cos_score)

                grad_diff=model_diff_norm - torch.norm(model_grads-sample_grad, p=2, dim=0)**2
                grad_diffs.append(grad_diff)

                sample_grads.append(torch.norm(sample_grad, p=2, dim=0)**2)

        return  torch.tensor(cos_scores).cpu(), torch.tensor(grad_diffs).cpu(), torch.tensor(sample_grads).cpu()