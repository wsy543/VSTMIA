# 使用信心波动检测成员关系，理论依据是模型会对刚学习过的样本记忆更清楚
# 而联邦学习刚好能够将样本划分成多次学习，这样可以区分样本是否存在于刚提交的更新中，成员的波动按理来说更大
# 尝试验证
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import random
from utils import load_npz_data
from torch.utils.data import DataLoader
from CSModels import ClientModel,PublicLayer,PrivateLayer
from Data import ClientDataset,ClientDatasetWithMember,TensorDataset
import os
from utils import ROC_AUC_Result_logshow,path_exists,custom_collate
from torch.optim.lr_scheduler import StepLR
from sklearn.cluster import KMeans,Birch,AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import math
from sklearn.decomposition import PCA
from scipy.stats import multivariate_normal
from sklearn.metrics import pairwise_distances
import seaborn as sns
from scipy.stats import norm
from sklearn.metrics import pairwise_kernels
from fix import fix
from scipy.spatial.distance import mahalanobis
import numpy as np
import pickle
from scipy.stats import chi2

class ours:
    def __init__(self,args,size):
        self.args=args
        self.batch_size=1
        self.train_size=size
        self.lr=self.args.lr
        self.attack_client_idx=self.args.attacker_client_idx
        self.collusion_client_idx=self.args.collusion_client_idx
        self.feature_dim=5
        self.d_model=0 # 不需要调了，我写成了自动获取的
        self.loss_fn = torch.nn.CrossEntropyLoss()
        # self.attacker_models = self.attacker_models[1:] #跳过第一轮模型
        # self.attacker_idx = self.attacker_idx[1:]
        self.global_models = self.server_model_loader()
        self.train_loader = self.load_trainloader(data_size=self.train_size)
        self.classfy_dataset_size = 5
        self.classfy_loader = self.load_classfyloader(data_size=self.classfy_dataset_size)# 很小的数据集，全是非成员，用来辅助kmeans数据分类
        self.test_loader = self.load_dataloader(data_size=self.train_size)
        self.other_loader = self.load_otherloader(data_size = self.train_size)
        self.model_path=self.args.model_path+'/'+self.args.model+'/'+self.args.dataset+'/our_model/'
        path_exists(self.model_path)
        

    @staticmethod
    def random_samples(data, sample_num, seed,replace=False):
        random.seed(seed)
        random_indices = np.random.choice(data.shape[0], sample_num, replace=replace)
        return data[random_indices]
    
    def model_eval(self,model,dataloader,idx):
        model.eval()
        with torch.no_grad():
            total, correct = 0, 0
            for _, (x, y, z) in enumerate(dataloader):
                x = x.to(self.args.device)
                y = y.to(self.args.device)
                outputs = model(x)
                # max=torch.max(features)
                _, predicted = torch.max(outputs, 1)  # 获取最大值的索引（预测类别）
                total += y.size(0)  # 样本总数
                correct += (predicted == y).sum().item()
            print(f"----model:{idx}, Acc{correct / total}----")

    def server_model_loader(self):
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
            model.eval()
            model_list.append(model)  # 将模型添加到 ModuleList 中
            print(f'load model {model_file}')
        print(f'load {len(model_list)} models')
        return model_list
    
    def client_model_loader(self,client_idx,bias=0):
        """
        模型加载器，将文件夹下面所有的public模型都加载进同一个modulelist中
        :return:
        """
        model_folder = self.args.model_path + '/' +self.args.model+'/' + self.args.dataset + '/client_model/'
        model_list = nn.ModuleList()
        model_files = [
            f'client_{client_idx}_{f}.pth'
            for f in range(self.args.training_round-bias)
            if os.path.exists(model_folder+f'client_{client_idx}_{f}.pth')
        ]
        idx_list = [
            f
            for f in range(self.args.training_round-bias)
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
        return model_list,idx_list
    
    
    def load_trainloader(self, data_size):
        """
        为每个共谋客户端分别创建并返回一个Dataloader。
        返回一个包含多个Dataloader的列表。
        """
        args = self.args
        data_path = f"{args.data_path}/{args.dataset}/{args.model}/{args.data_split}"
        
        # 加载完整的训练集和测试集数据
        all_member_datas, all_member_labels = load_npz_data(f"{data_path}/train_non_iid.npz")
        all_non_member_datas, all_non_member_labels = load_npz_data(f"{data_path}/test_non_iid.npz")

        # 初始化一个空列表，用于存储每个客户端的dataloader
        dataloader_list = []

        # 遍历指定的每个共谋客户端索引
        for client_idx in self.collusion_client_idx:
            # --- 1. 为当前客户端提取成员和非成员数据 ---
            # 成员数据（来自训练集）
            member_data = all_member_datas[client_idx]
            member_label = all_member_labels[client_idx]
            
            # 非成员数据（来自测试集）
            non_member_data = all_non_member_datas[client_idx]
            non_member_label = all_non_member_labels[client_idx]

            # --- 2. 为当前客户端进行数据采样 ---
            member_total_samples = len(member_label)
            nonmember_total_samples = len(non_member_label)

            # 确保采样数量不超过该客户端拥有的样本数
            sample_size = min(data_size, member_total_samples, nonmember_total_samples)
            if sample_size < data_size:
                print(f"Warning: For client {client_idx}, requested data_size {data_size} is too large. "
                    f"Using smaller sample size of {sample_size}.")

            if sample_size == 0:
                print(f"Info: Client {client_idx} has no data available for sampling, skipping.")
                continue # 跳过这个客户端，不为它创建dataloader

            random.seed(self.args.random_seed)
            random_indices = np.random.choice(member_total_samples, sample_size, replace=False)
            non_random_indices = np.random.choice(nonmember_total_samples, sample_size, replace=False)

            sampled_data1, sampled_label1 = member_data[random_indices], member_label[random_indices]
            sampled_data2, sampled_label2 = non_member_data[non_random_indices], non_member_label[non_random_indices]
            
            final_data = np.concatenate((sampled_data1, sampled_data2))
            final_label = np.concatenate((sampled_label1, sampled_label2))
            membership = [1] * sample_size + [0] * sample_size
            
            # --- 3. 为当前客户端创建Dataset和DataLoader ---
            dataset = ClientDatasetWithMember(final_data, final_label, membership)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
            
            # --- 4. 将创建好的dataloader添加到列表中 ---
            dataloader_list.append(dataloader)

        return dataloader_list
        

    def load_classfyloader(self,data_size):
        """
        少量用于聚类的数据集，来自于除目标客户端外的其他客户端的非成员
        """
        args = self.args
        data_path = args.data_path + '/'+args.dataset+'/'+args.model+'/'+args.data_split
        # datas, labels = load_npz_data(data_path + '/train_non_iid.npz')
        # data, label = np.delete(datas,self.attack_client_idx), np.delete(labels,self.attack_client_idx)
        # data = np.concatenate(data)
        # label = np.concatenate(label)
        # data = np.array(data)
        # label = np.array(label)
        non_member_datas, non_member_labels = load_npz_data(data_path + '/test_non_iid.npz')
        non_member_data = [client_data for client_idx, client_data in enumerate(non_member_datas) if client_idx != self.attack_client_idx]
        non_member_label = [client_label for client_idx, client_label in enumerate(non_member_labels) if client_idx != self.attack_client_idx]
        # non_member_data,non_member_label =  np.delete(non_member_datas,self.attack_client_idx,axis=0),np.delete(non_member_labels,self.attack_client_idx,axis=0)
        non_member_data=np.concatenate(non_member_data)
        non_member_label=np.concatenate(non_member_label)
        non_member_data = np.array(non_member_data)
        non_member_label=np.array(non_member_label)
        nonmember_total_samples = len(non_member_label)-1
        random.seed(self.args.random_seed)
        non_random_indices = np.random.choice(nonmember_total_samples, data_size, replace=False)
        sampled_data, sampled_label = non_member_data[non_random_indices], non_member_label[non_random_indices]
        membership = data_size*[0]
        dataset = ClientDatasetWithMember(sampled_data, sampled_label,membership)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        return dataloader
    

    def load_otherloader(self,data_size):
        """
        客户端视角下的测试集，来自除攻击者外的所有客户端
        """
        args = self.args
        data_path = args.data_path + '/'+args.dataset+'/'+args.model+'/'+args.data_split
        datas, labels = load_npz_data(data_path + '/train_non_iid.npz')
        data = [client_data for client_idx, client_data in enumerate(datas) if client_idx != self.attack_client_idx]
        label = [client_label for client_idx, client_label in enumerate(labels) if client_idx != self.attack_client_idx]
        # data, label = np.delete(datas,self.attack_client_idx,axis=0), np.delete(labels,self.attack_client_idx,axis=0)
        data = np.concatenate(data)
        label = np.concatenate(label)
        data = np.array(data)
        label = np.array(label)
        non_member_datas, non_member_labels = load_npz_data(data_path + '/test_non_iid.npz')
        non_member_data = [client_data for client_idx, client_data in enumerate(non_member_datas) if client_idx != self.attack_client_idx]
        non_member_label = [client_label for client_idx, client_label in enumerate(non_member_labels) if client_idx != self.attack_client_idx]
        # non_member_data,non_member_label =  np.delete(non_member_datas,self.attack_client_idx,axis=0),np.delete(non_member_labels,self.attack_client_idx,axis=0)
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
    
    
    def load_dataloader(self, data_size):
        """
        服务器视角下的测试集，来自目标客户端
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


    def make_loader_for_vlm(self):
        train = self.train_loader[0]
        attacker_models, attacker_idx = self.client_model_loader(0)
        save_dir = f'./plot/vlm_data/{self.args.model}/{self.args.dataset}/'
        criterion = nn.CrossEntropyLoss(reduction='none')

        # 创建保存目录
        os.makedirs(save_dir + "member", exist_ok=True)
        os.makedirs(save_dir + "nonmember", exist_ok=True)

        # ✅ 新增：用于保存所有样本 loss 序列的列表
        all_member_losses = []     # 每个元素是 shape [num_models] 的 numpy array
        all_nonmember_losses = []

        for i, (data, label, member) in enumerate(tqdm(train)):
            data, label = data.to(self.args.device), label.to(self.args.device)
            losses = []

            with torch.no_grad():
                for model in attacker_models:
                    output = model(data)
                    batch_losses = criterion(output, label)
                    losses.append(batch_losses.cpu().numpy())  # list of [B]

            losses = np.array(losses).T  # [B, num_models]

            for j in range(data.size(0)):
                # 🖼️ 保留画图逻辑
                plt.figure(figsize=(6.22, 2.67))
                plt.plot(losses[j], marker='o',markersize=2,linewidth=1)
                plt.xlabel('Model Index')
                plt.ylabel('Loss')
                plt.grid(True)
                plt.tight_layout()

                if member[j].item() == 1:
                    plt.savefig(os.path.join(save_dir, f'member/train_sample_{i * data.size(0) + j}.png'))
                    # ✅ 保存 loss 序列到成员列表
                    all_member_losses.append(losses[j])  # shape: [num_models]
                else:
                    plt.savefig(os.path.join(save_dir, f'nonmember/train_sample_{i * data.size(0) + j}.png'))
                    # ✅ 保存 loss 序列到非成员列表
                    all_nonmember_losses.append(losses[j])

                plt.close()

        # 💾 保存所有成员/非成员的 loss 序列为 .pkl 文件
        with open(os.path.join(save_dir, 'member_losses.pkl'), 'wb') as f:
            pickle.dump(all_member_losses, f)

        with open(os.path.join(save_dir, 'nonmember_losses.pkl'), 'wb') as f:
            pickle.dump(all_nonmember_losses, f)

        fix(save_dir)

        print(f"✅ Saved {len(all_member_losses)} member samples and {len(all_nonmember_losses)} non-member samples.")
        
