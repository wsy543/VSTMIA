from torch.utils.data import Dataset
import numpy as np
from collections import defaultdict
import torch


class ClientDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]


def load_npz_data(data_name):
    with np.load(data_name,allow_pickle=True) as f:
        train_x, train_y = [f[i] for i in f.files]
    return train_x, train_y


class ClientDatasetWithMember(Dataset):
    def __init__(self, data, targets,member):
        self.data = data
        self.targets = targets
        self.member=member

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.targets[index],self.member[index]

def random_uniform_split(dataset, labels, num_splits, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    num_samples = len(dataset)
    
    indices = np.random.permutation(num_samples)
    dataset = np.array(dataset)[indices]
    labels = np.array(labels)[indices]
    
    samples_per_client = num_samples // num_splits

    split_data = [[] for _ in range(num_splits)]
    split_labels = [[] for _ in range(num_splits)]
    
    for i in range(num_splits):
        start_idx = i * samples_per_client
        end_idx = (i + 1) * samples_per_client
        split_data[i] = dataset[start_idx:end_idx]
        split_labels[i] = labels[start_idx:end_idx]

    return split_data, split_labels

def non_iid_dirichlet_split(dataset, labels, num_splits, alpha, seed=None):
    if seed is not None:
        np.random.seed(seed)
    num_classes = len(np.unique(labels))
    label_indices = defaultdict(list)

    for i, label in enumerate(labels):
        label_indices[label].append(i)

    class_split_ratios = np.random.dirichlet([alpha] * num_splits, num_classes)

    split_data = [[] for _ in range(num_splits)]
    split_labels = [[] for _ in range(num_splits)]

    for cls, indices in label_indices.items():
        np.random.shuffle(indices)
        class_size = len(indices)
        split_sizes = (class_split_ratios[cls] * class_size).astype(int)

        diff = class_size - np.sum(split_sizes)
        if diff > 0:
            split_sizes[np.argmax(split_sizes)] += diff

        current_idx = 0
        for split_idx, size in enumerate(split_sizes):
            split_data[split_idx].extend([dataset[i] for i in indices[current_idx:current_idx + size]])
            split_labels[split_idx].extend([labels[i] for i in indices[current_idx:current_idx + size]])
            current_idx += size

    return split_data, split_labels


class TensorDataset(Dataset):
    def __init__(self, *tensors):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors), \
            "所有张量的第一个维度必须相同"
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].size(0)

