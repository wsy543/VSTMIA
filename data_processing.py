import torch
import torchvision
import torchvision.transforms as transforms
import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import glob
import ssl
import random

from utils import path_exists

ssl._create_default_https_context = ssl._create_unverified_context


def save_subset_as_npz(subset, save_path):
    images = []
    labels = []

    for data, target in subset:
        images.append(data.numpy())
        labels.append(target)

    np.savez(save_path, images=np.array(images), labels=np.array(labels))
    print(f"Saved dataset to {save_path}")


def load_and_split_stl10(split_ratio, download_flag, save_dir="./saved_data"):
    transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize((0.4467, 0.4398, 0.4066), (0.2603, 0.2565, 0.2712))
    ])

    train_dataset = torchvision.datasets.STL10(root='./datas/stl10', split='train', download=download_flag,
                                              transform=transform)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_subset_as_npz(train_dataset, os.path.join(save_dir, 'full.npz'))


def load_and_process_location(root_dir, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(f"正在从 {root_dir} 搜索数据文件...")

    files = glob.glob(os.path.join(root_dir, "*"))
    data_file = None
    for f in files:
        if f.endswith(".txt") or f.endswith(".csv") or f.endswith(".data"):
            data_file = f
            break

    if not data_file:
        print(f"错误: 在 {root_dir} 下未找到 .txt/.csv/.data 数据文件")
        return

    print(f"找到文件: {data_file}，正在读取...")

    try:
        df = pd.read_csv(data_file, header=None, quotechar='"', sep=None, engine='python')
    except Exception as e:
        print(f"读取失败，尝试强制使用逗号分隔... {e}")
        df = pd.read_csv(data_file, header=None, quotechar='"', sep=',')

    print(f"原始数据加载完成，形状: {df.shape}")

    raw_labels = df.iloc[:, 0].values
    raw_features = df.iloc[:, 1:].values

    print("正在编码标签...")
    le = LabelEncoder()
    labels = le.fit_transform(raw_labels)

    print("正在转换特征格式...")
    features = raw_features.astype(np.float32)

    input_dim = features.shape[1]
    num_classes = len(le.classes_)

    print(f"处理完成!")
    print(f"-> 特征维度 (Input Size): {input_dim}")
    print(f"-> 类别数量 (Num Classes): {num_classes}")
    print(f"-> 样本总数: {len(labels)}")

    save_path = os.path.join(save_dir, 'full.npz')
    np.savez(save_path, datas=features, labels=labels)
    print(f"数据已保存到: {save_path}")


def process_data(args,download=False):
    random_seed = getattr(args, 'random_seed', 123)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    split_ratio = args.split_ratio
    if args.dataset == 'STL10':
        save_path='./datas/STL10'
        path_exists(save_path)
        load_and_split_stl10(split_ratio,download,save_path)
    elif args.dataset == 'location':
        save_path='./datas/location'
        load_and_process_location(root_dir='./datas/location_offical/',save_dir=save_path)
    else:
        print('dataset is error')
        sys.exit(1)
