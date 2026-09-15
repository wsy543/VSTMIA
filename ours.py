import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import random
from utils import load_npz_data, set_seed
from torch.utils.data import DataLoader
from CSModels import ClientModel,PublicLayer,PrivateLayer
from Data import ClientDataset,ClientDatasetWithMember,TensorDataset
import os
from utils import path_exists
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
        set_seed(self.args.random_seed)
        self.batch_size=1
        self.train_size=size
        self.lr=self.args.lr
        self.attack_client_idx=self.args.attacker_client_idx
        self.collusion_client_idx=self.args.collusion_client_idx
        self.feature_dim=5
        self.d_model=0
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.global_models = self.server_model_loader()
        self.train_loader = self.load_trainloader(data_size=self.train_size)
        self.classfy_dataset_size = 5
        self.classfy_loader = self.load_classfyloader(data_size=self.classfy_dataset_size)
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
                _, predicted = torch.max(outputs, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
            print(f"----model:{idx}, Acc{correct / total}----")

    def server_model_loader(self):
        model_folder = self.args.model_path + '/' +self.args.model+'/' + self.args.dataset + '/server_model'
        model_list = nn.ModuleList()
        model_files = [f'server_{f}.pth' for f in range(self.args.training_round)]
        for model_file in model_files:
            model_path = os.path.join(model_folder, model_file)
            model_pth = torch.load(model_path)
            model = PublicLayer(self.args)
            model.load_state_dict(model_pth)
            model.to(self.args.device)
            model.eval()
            model_list.append(model)
            print(f'load model {model_file}')
        print(f'load {len(model_list)} models')
        return model_list
    
    def client_model_loader(self,client_idx,bias=0):
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
            model_pth = torch.load(model_path)
            model = PublicLayer(self.args)
            model.load_state_dict(model_pth)
            model.to(self.args.device)
            model.eval()
            model_list.append(model)
            print(f'load model {model_file}')
        print(f'load {len(model_list)} models')
        return model_list,idx_list
    
    
    def load_trainloader(self, data_size):
        args = self.args
        data_path = f"{args.data_path}/{args.dataset}/{args.model}/{args.data_split}"
        
        all_member_datas, all_member_labels = load_npz_data(f"{data_path}/train_non_iid.npz")
        all_non_member_datas, all_non_member_labels = load_npz_data(f"{data_path}/test_non_iid.npz")

        dataloader_list = []

        for client_idx in self.collusion_client_idx:
            member_data = all_member_datas[client_idx]
            member_label = all_member_labels[client_idx]
            
            non_member_data = all_non_member_datas[client_idx]
            non_member_label = all_non_member_labels[client_idx]

            member_total_samples = len(member_label)
            nonmember_total_samples = len(non_member_label)

            sample_size = min(data_size, member_total_samples, nonmember_total_samples)
            if sample_size < data_size:
                print(f"Warning: For client {client_idx}, requested data_size {data_size} is too large. "
                    f"Using smaller sample size of {sample_size}.")

            if sample_size == 0:
                print(f"Info: Client {client_idx} has no data available for sampling, skipping.")
                continue

            random.seed(self.args.random_seed)
            random_indices = np.random.choice(member_total_samples, sample_size, replace=False)
            non_random_indices = np.random.choice(nonmember_total_samples, sample_size, replace=False)

            sampled_data1, sampled_label1 = member_data[random_indices], member_label[random_indices]
            sampled_data2, sampled_label2 = non_member_data[non_random_indices], non_member_label[non_random_indices]
            
            final_data = np.concatenate((sampled_data1, sampled_data2))
            final_label = np.concatenate((sampled_label1, sampled_label2))
            membership = [1] * sample_size + [0] * sample_size
            
            dataset = ClientDatasetWithMember(final_data, final_label, membership)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
            
            dataloader_list.append(dataloader)

        return dataloader_list
        

    def load_classfyloader(self,data_size):
        args = self.args
        data_path = args.data_path + '/'+args.dataset+'/'+args.model+'/'+args.data_split
        non_member_datas, non_member_labels = load_npz_data(data_path + '/test_non_iid.npz')
        non_member_data = [client_data for client_idx, client_data in enumerate(non_member_datas) if client_idx != self.attack_client_idx]
        non_member_label = [client_label for client_idx, client_label in enumerate(non_member_labels) if client_idx != self.attack_client_idx]
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
        args = self.args
        data_path = args.data_path + '/'+args.dataset+'/'+args.model+'/'+args.data_split
        datas, labels = load_npz_data(data_path + '/train_non_iid.npz')
        data = [client_data for client_idx, client_data in enumerate(datas) if client_idx != self.attack_client_idx]
        label = [client_label for client_idx, client_label in enumerate(labels) if client_idx != self.attack_client_idx]
        data = np.concatenate(data)
        label = np.concatenate(label)
        data = np.array(data)
        label = np.array(label)
        non_member_datas, non_member_labels = load_npz_data(data_path + '/test_non_iid.npz')
        non_member_data = [client_data for client_idx, client_data in enumerate(non_member_datas) if client_idx != self.attack_client_idx]
        non_member_label = [client_label for client_idx, client_label in enumerate(non_member_labels) if client_idx != self.attack_client_idx]
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


    def make_loader_for_vlm(self):
        train = self.train_loader[0]
        attacker_models, attacker_idx = self.client_model_loader(0)
        save_dir = f'./plot/vlm_data/{self.args.model}/{self.args.dataset}/'
        criterion = nn.CrossEntropyLoss(reduction='none')

        os.makedirs(save_dir + "member", exist_ok=True)
        os.makedirs(save_dir + "nonmember", exist_ok=True)

        all_member_losses = []
        all_nonmember_losses = []

        for i, (data, label, member) in enumerate(tqdm(train)):
            data, label = data.to(self.args.device), label.to(self.args.device)
            losses = []

            with torch.no_grad():
                for model in attacker_models:
                    output = model(data)
                    batch_losses = criterion(output, label)
                    losses.append(batch_losses.cpu().numpy())

            losses = np.array(losses).T

            for j in range(data.size(0)):
                plt.figure(figsize=(self.args.plot_width, self.args.plot_height), dpi=self.args.plot_dpi)
                plt.plot(losses[j], marker='o',markersize=2,linewidth=1)
                plt.xlabel('Training Round (Model Index)')
                plt.ylabel('Loss')
                plt.grid(True)
                plt.tight_layout()

                if member[j].item() == 1:
                    plt.savefig(os.path.join(save_dir, f'member/train_sample_{i * data.size(0) + j}.png'))
                    all_member_losses.append(losses[j])
                else:
                    plt.savefig(os.path.join(save_dir, f'nonmember/train_sample_{i * data.size(0) + j}.png'))
                    all_nonmember_losses.append(losses[j])

                plt.close()

        with open(os.path.join(save_dir, 'member_losses.pkl'), 'wb') as f:
            pickle.dump(all_member_losses, f)

        with open(os.path.join(save_dir, 'nonmember_losses.pkl'), 'wb') as f:
            pickle.dump(all_nonmember_losses, f)

        fix(save_dir)

        print(f"✅ Saved {len(all_member_losses)} member samples and {len(all_nonmember_losses)} non-member samples.")
        
