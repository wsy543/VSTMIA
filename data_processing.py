import torch
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader
import os
from utils import path_exists
import numpy as np
import sys
import os
import pandas as pd
from Data import load_npz_data,ISICSplitDataset
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset
from collections import Counter
import re
import ssl
import glob
from sklearn.preprocessing import LabelEncoder
import random

# 全局取消证书验证，解决 SSL: CERTIFICATE_VERIFY_FAILED 错误
ssl._create_default_https_context = ssl._create_unverified_context


def save_subset_as_npz(subset, save_path):
    """
    将 PyTorch Subset 数据集保存为 npz 格式。
    :param subset: PyTorch Subset 数据集
    :param save_path: 保存路径
    """
    images = []
    labels = []

    # 迭代 Subset 以提取数据和标签
    for data, target in subset:
        images.append(data.numpy())  # 将图像数据转为 NumPy 数组
        labels.append(target)  # 标签通常是整数

    # 将数据和标签转换为 NumPy 数组，并保存为 .npz 文件
    np.savez(save_path, images=np.array(images), labels=np.array(labels))
    print(f"Saved dataset to {save_path}")

def save_subset_as_npz_text(dataset, file_path):
    """
    将数据集保存为 .npz 文件
    :param dataset: 包含文本和标签的元组 (文本, 标签)
    :param file_path: 保存文件路径
    """
    texts, labels = dataset
    np.savez(file_path, texts=texts, labels=labels)


def load_and_split_cifar10(split_ratio, download_flag,save_dir="./saved_data"):
    """
    加载并分割 CIFAR-10 数据集，进行标准化，并根据 split_ratio 分割为两部分。
    然后将两部分数据分别保存到 .npz 文件中。
    :param split_ratio: 介于 0 和 1 之间的浮点数，表示第一个部分的数据比例
    :param save_dir: 保存数据的目录
    """

    # 定义标准化的变换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))  # CIFAR-10 数据集的均值和标准差
    ])

    # 加载 CIFAR-10 数据集
    train_dataset = torchvision.datasets.CIFAR10(root='./datas/cifar10-offical', train=True, download=download_flag,
                                                 transform=transform)
    save_subset_as_npz(train_dataset,os.path.join(save_dir,'full.npz'))

    # 根据 split_ratio 计算每个部分的样本数量
    total_size = len(train_dataset)
    part1_size = int(total_size * split_ratio)
    part2_size = total_size - part1_size

    # 将数据集分割为两部分
    part1_dataset, part2_dataset = random_split(train_dataset, [part1_size, part2_size])

    # 确保保存目录存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 保存分割后的数据集为 .npz 格式
    save_subset_as_npz(part1_dataset, os.path.join(save_dir, 'train.npz'))
    save_subset_as_npz(part2_dataset, os.path.join(save_dir, 'test.npz'))

def load_and_split_cifar100(split_ratio, download_flag,save_dir="./saved_data"):
    """
    加载并分割数据集，进行标准化，并根据 split_ratio 分割为两部分。
    然后将两部分数据分别保存到 .npz 文件中。
    :param split_ratio: 介于 0 和 1 之间的浮点数，表示第一个部分的数据比例
    :param save_dir: 保存数据的目录
    """

    # 定义标准化的变换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))  # CIFAR-10 数据集的均值和标准差
    ])

    # 加载 CIFAR-10 数据集
    train_dataset = torchvision.datasets.CIFAR100(root='./datas/cifar100-offical', train=True, download=download_flag,
                                                 transform=transform)
    save_subset_as_npz(train_dataset,os.path.join(save_dir,'full.npz'))


    # # 根据 split_ratio 计算每个部分的样本数量
    # total_size = len(train_dataset)
    # part1_size = int(total_size * split_ratio)
    # part2_size = total_size - part1_size

    # # 将数据集分割为两部分
    # part1_dataset, part2_dataset = random_split(train_dataset, [part1_size, part2_size])

    # # 确保保存目录存在
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)

    # # 保存分割后的数据集为 .npz 格式
    # save_subset_as_npz(part1_dataset, os.path.join(save_dir, 'train.npz'))
    # save_subset_as_npz(part2_dataset, os.path.join(save_dir, 'test.npz'))

def load_and_split_gtsrb(split_ratio, download_flag, save_dir="./saved_data"):
    """
    加载并分割数据集，进行标准化，并根据 split_ratio 分割为两部分。
    然后将两部分数据分别保存到 .npz 文件中。
    :param split_ratio: 介于 0 和 1 之间的浮点数，表示第一个部分的数据比例
    :param save_dir: 保存数据的目录
    """

    # 定义标准化的变换
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))  # CIFAR-10 数据集的均值和标准差
    ])

    # 加载 CIFAR-10 数据集
    train_dataset = torchvision.datasets.GTSRB(root='./datas/GTSRB-offical', split='train', download=download_flag,
                                                 transform=transform)
    save_subset_as_npz(train_dataset,os.path.join(save_dir,'full.npz'))


def load_and_split_cinic10(split_ratio, download_flag, save_dir="./saved_data"):
    """
    加载并分割数据集，进行标准化，并根据 split_ratio 分割为两部分。
    然后将两部分数据分别保存到 .npz 文件中。
    :param split_ratio: 介于 0 和 1 之间的浮点数，表示第一个部分的数据比例
    :param save_dir: 保存数据的目录
    """

    # 定义标准化的变换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))  # CIFAR-10 数据集的均值和标准差
    ])
    paths = ["shadowTrain.npz", "shadowTest.npz", "targetTest.npz", "targetTrain.npz"]
    all_images = []
    all_labels = []

    # 逐个加载每个 npz 文件
    for path in paths:
        images,labels = load_npz_data('./datas/cinic10-offical/'+path)
        
        # 假设 npz 文件中键名分别为 'images' 和 'labels'
        
        # 将数据和标签添加到列表中
        all_images.append(images)
        all_labels.append(labels)

    # 将所有数据和标签沿第一个维度进行拼接
    merged_images = np.concatenate(all_images, axis=0)
    merged_labels = np.concatenate(all_labels, axis=0)

    # 保存合并后的数据到一个新的 npz 文件
    np.savez(save_dir+"/full.npz", images=merged_images,labels=merged_labels)

def load_and_split_20newsgroups(split_ratio, download_flag, save_dir="./saved_data"):
    """
    
    """
    # 加载 20 Newsgroups 数据集
    newsgroups_data = fetch_20newsgroups(subset='all',download_if_missing=True)
    # 使用 TfidfVectorizer 将文本转化为 TF-IDF 特征向量
    vectorizer = TfidfVectorizer(stop_words='english',max_features=500)
    texts_tfidf = vectorizer.fit_transform(newsgroups_data.data)

    # 保存整个数据集为 .npz 格式
    save_subset_as_npz_text((texts_tfidf.toarray(), newsgroups_data.target), os.path.join(save_dir, 'full.npz'))

    print(f"数据已保存到: {os.path.join(save_dir, 'full.npz')}")


def process_dbpedia_local_tfidf(root_dir, save_dir="./processed_dbpedia_tfidf"):
    """
    从位于指定根目录 (root_dir) 中的本地 CSV 文件
    (DBPEDIA_train.csv, DBPEDIA_test.csv, DBPEDIA_val.csv) 加载 DBpedia Ontology Dataset，
    将文本处理为 TF-IDF 特征，并将处理后的特征和标签保存到 NPZ 文件中。

    Args:
        root_dir (str): 包含 DBpedia CSV 文件的目录的路径。
                        该目录下应包含 DBPEDIA_train.csv, DBPEDIA_test.csv, DBPEDIA_val.csv。
        save_dir (str): 保存输出 NPZ 文件的目录。默认值是脚本所在的当前目录下的 processed_dbpedia_tfidf 文件夹。
    """
    # 创建保存目录如果不存在

    # --- 1. 从本地 CSV 文件加载数据 ---
    # 根据提供的 root_dir 参数构建完整的 CSV 文件路径
    train_csv_path = os.path.join(root_dir, 'DBPEDIA_train.csv')
    test_csv_path = os.path.join(root_dir, 'DBPEDIA_test.csv')
    val_csv_path = os.path.join(root_dir, 'DBPEDIA_val.csv')
    # 注意：DBP_wiki_data.csv 在这个标准处理流程中不被使用，除非你有特定需求。
    # dbp_wiki_data_csv_path = os.path.join(root_dir, 'DBP_wiki_data.csv') # Path to the other file

    print(f"正在从本地目录 {root_dir} 中的 CSV 文件加载数据...")

    # 使用 pandas 加载每个 CSV 文件到 DataFrame
    try:
        train_df = pd.read_csv(train_csv_path)
        test_df = pd.read_csv(test_csv_path)
        val_df = pd.read_csv(val_csv_path) # 加载验证集数据

        print("CSV 文件加载成功。")
        print(f"训练集样本数: {len(train_df)}, 测试集样本数: {len(test_df)}, 验证集样本数: {len(val_df)}")

    except FileNotFoundError as e:
        print(f"错误：未找到 CSV 文件: {e}。请确保 DBPEDIA_train.csv, DBPEDIA_test.csv, 和 DBPEDIA_val.csv 文件都位于指定的目录 {root_dir} 中。")
        return # 如果文件缺失，终止函数执行
    except Exception as e:
        print(f"加载 CSV 文件时发生其他错误: {e}")
        return


    # --- 2. 准备用于 TF-IDF 的文本和标签数据 ---
    # 将训练集、测试集和验证集的文本数据合并到一个列表中
    # 假设这些 CSV 文件中，文本内容列的列名是 'text'，标签列的列名是 'label'。
    # ***请务必检查你下载的 CSV 文件，确认实际的列名是否是 'text' 和 'label'！如果不是，请修改下面的代码。***
    try:
        # 合并所有分割的文本列表
        all_texts = train_df['text'].tolist() + test_df['text'].tolist() + val_df['text'].tolist()
        label = 'l2'
        # 合并所有分割的标签列表并转换为 NumPy 数组
        all_labels = train_df[label].tolist() + test_df[label].tolist() + val_df[label].tolist()
        # --- 将字符串标签 ('l2' 中的值) 映射到整数索引 ---
        # 查找 'l2' 列中所有唯一的字符串标签
        unique_l2_labels = sorted(list(set(all_labels))) # 获取唯一的标签并排序，确保映射一致

        # 创建一个从字符串标签到整数索引的字典映射
        # 例如 {'Station': 0, 'Building': 1, 'Organisation': 2, ...}
        l2_label_to_int = {label: i for i, label in enumerate(unique_l2_labels)}

        print(f"在 'l2' 列中找到 {len(unique_l2_labels)} 个唯一的字符串标签。")
        print(f"字符串标签到整数的映射示例: {list(l2_label_to_int.items())[:10]}...") # 打印前10个映射示例
        print(f"所有唯一的字符串标签: {unique_l2_labels}") # 打印所有唯一的字符串标签

        # 将所有字符串标签 (all_l2_labels 列表中的值) 转换为对应的整数标签列表
        all_integer_labels = [l2_label_to_int[label] for label in all_labels]

        # 将整数标签列表转换为 NumPy 数组
        all_labels_np = np.array(all_integer_labels) # 存储最终的整数标签 NumPy 数组并 train, test, val 的文本和标签数据。总样本数: {len(all_texts)}")

    except KeyError as e:
        print(f"错误：CSV 文件中缺少必需的列: {e}。请检查你的 CSV 文件是否包含名为 'text' 和 'label' 的列。")
        return
    except Exception as e:
        print(f"合并文本和标签数据时发生错误: {e}")
        return


    # --- 3. 将文本转换为 TF-IDF 特征向量 ---
    print("正在将合并后的文本转换为 TF-IDF 特征向量...")
    # 初始化 TfidfVectorizer
    # max_features 参数可以控制最终特征向量的维度，这里设置为 5000 或 10000 是示例值。
    # stop_words='english' 用于移除英文停用词。
    # 你可以根据需要调整 max_features 和其他参数。
    tfidf_max_features = 5000 # 示例：设置最大特征数为 10000
    vectorizer = TfidfVectorizer(stop_words='english', max_features=tfidf_max_features)

    # fit_transform 会在所有合并的文本数据 (all_texts) 上进行拟合 (学习词汇表和 IDF 权重)
    # 并同时将文本转换为 TF-IDF 特征矩阵
    # 结果 texts_tfidf 是一个稀疏矩阵 (sparse matrix)
    texts_tfidf = vectorizer.fit_transform(all_texts)

    print(f"TF-IDF 特征向量创建完成。生成的特征矩阵形状: {texts_tfidf.shape}")
    # 形状是 (总样本数, 特征维度，即 min(实际词汇量, max_features))


    # --- 4. 准备保存的数据 ---
    # 将稀疏的 TF-IDF 特征矩阵转换为密集的 NumPy 数组
    # 注意：如果总样本数和 max_features 都很大，这个转换可能会消耗大量内存
    print("正在将稀疏的 TF-IDF 矩阵转换为密集数组...")
    texts_tfidf_dense = texts_tfidf.toarray()
    print("转换为密集数组完成。")


    # --- 5. 将整个数据集 (TF-IDF 特征和标签) 保存为 .npz 文件 ---
    # 保存的 NPZ 文件将包含所有 train, test, val 的数据
    output_file_name = 'full.npz' # 输出 NPZ 文件的文件名
    output_file_path = os.path.join(save_dir, output_file_name) # 构建完整的输出文件路径

    print(f"正在保存处理后的数据到 NPZ 文件: {output_file_path}...")
    data_to_save = (texts_tfidf_dense, all_labels_np) # 准备要保存的数据元组
    save_subset_as_npz_text(data_to_save, output_file_path) # 调用保存函数

    print(f"数据处理和保存流程执行完毕。")
    print(f"输出的 NPZ 文件保存在: {output_file_path}")

def save_as_npz(dataset, file_path):
    data = []
    targets = []
    for inputs, labels in DataLoader(dataset, batch_size=100, shuffle=False):
        data.append(inputs.numpy())
        targets.append(labels.numpy())
    data = np.concatenate(data, axis=0)
    targets = np.concatenate(targets, axis=0)
    np.savez(file_path, data=data, targets=targets)


def load_and_process_tiny_imagenet(root_dir, save_dir="./saved_data"):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # 调整尺寸为 32x32
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载训练集、验证集和测试集
    train_dir = os.path.join(root_dir, 'train')
    val_dir = os.path.join(root_dir, 'val')
    test_dir = os.path.join(root_dir, 'test')

    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    # val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)
    # test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

    # 合并数据集
    full_dataset = torch.utils.data.ConcatDataset([train_dataset])

    # 保存合并后的数据集
    save_as_npz(full_dataset, os.path.join(save_dir, 'full.npz'))
    print(f"数据已保存到: {save_dir}")

def load_oct2017(root_dir,save_dir):
    """
    从本地目录加载 OCT2017 数据集的训练集、验证集和测试集，
    应用图像变换，并将所有分割的数据合并到一个单个的 PyTorch Dataset 对象中。
    所有四个类别的数据都会被包含在内。

    Args:
        root_dir (str): OCT2017 数据集在本地存放的根目录路径。
                        这个目录应该直接包含 'train', 'val', 'test' 子文件夹。
                        例如：'./path/to/your/OCT2017_Dataset/'。

    Returns:
        torch.utils.data.Dataset: 一个单个的 PyTorch Dataset 对象，包含来自
                                  训练集、验证集和测试集的所有样本（跨所有类别），
                                  并且已经应用了图像变换。
                                  如果加载过程中发生错误，返回 None。
    """
    # --- 定义图像预处理 Transforms ---
    # OCT 图像是灰度的，你需要根据你的模型输入需求来处理通道数 (1 或 3)。
    # Resize 尺寸可以调整。Normalization 的均值和标准差最好根据数据集自身统计，这里使用 ImageNet 的作为常见起点。
    print("定义图像预处理 Transforms...")
    image_size = (224,224) # 示例尺寸 (高, 宽)，可以根据你的模型输入需求修改
    transform = transforms.Compose([
        # OCT 图像是灰度的，但如果你使用预训练的 RGB 模型，可能需要将其转换为 3 个重复的通道。
        # 如果你的模型只需要 1 通道输入，使用 transforms.Grayscale(num_output_channels=1)。
        transforms.Grayscale(num_output_channels=3), # 示例：将灰度图转换为 3 个相同的通道
        transforms.Resize(image_size), # 缩放图像到指定尺寸
        transforms.ToTensor(),       # 将 PIL Image 转换为 Tensor (值范围通常变为 [0, 1])，形状为 (通道, 高, 宽)
        # 标准化 (可选)，使用 ImageNet 的均值和标准差作为示例。最好使用数据集自身的均值和标准差。
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # --- 使用 ImageFolder 分别加载训练集、验证集和测试集 ---
    # ImageFolder 期望数据集的目录结构是：root_dir/分割名称/类别名称/图片文件.jpg
    # 例如：./my_oct_data_root/train/CNV/image1.jpg
    # 你指定的 root_dir 应该包含 'train', 'val', 'test' 这三个子文件夹。
    train_dir = os.path.join(root_dir, 'train')
    val_dir = os.path.join(root_dir, 'val')
    test_dir = os.path.join(root_dir, 'test')

    print(f"\n正在从本地目录加载 OCT2017 数据集的不同分割: {root_dir}")
    print(f"  训练集路径: {train_dir}")
    print(f"  验证集路径: {val_dir}")
    print(f"  测试集路径: {test_dir}")

    try:
        # 使用 ImageFolder 加载每个分割。ImageFolder 会自动检测子文件夹作为类别，并分配整数标签。
        train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
        val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)
        test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

        print("\nOCT2017 数据集各分割成功加载到 Dataset 对象。")
        print(f"训练集样本总数: {len(train_dataset)}")
        print(f"验证集样本总数: {len(val_dataset)}")
        print(f"测试集样本总数: {len(test_dataset)}")

        # ImageFolder 加载的类别和整数映射是基于文件夹名称的，通常是字母顺序
        print(f"检测到的类别名称: {train_dataset.classes}") # 例如 ['CNV', 'DME', 'DRUSEN', 'NORMAL']
        print(f"类别到整数索引的映射: {train_dataset.class_to_idx}") # 例如 {'CNV': 0, 'DME': 1, ...}

        # --- 使用 ConcatDataset 将所有分割的数据集合并到一个 Dataset 中 ---
        print("\n正在使用 ConcatDataset 合并训练集、验证集和测试集...")
        # ConcatDataset 将多个 Dataset 对象连接起来，形成一个单一的 Dataset
        # 当你从 combined_dataset 中获取样本时，它会依次从 train_dataset, val_dataset, test_dataset 中取样
        combined_dataset = ConcatDataset([train_dataset, val_dataset, test_dataset])

        print(f"合并后的单个 Dataset 创建成功，共包含 {len(combined_dataset)} 个样本。")
        save_as_npz(combined_dataset, os.path.join(save_dir, 'full.npz'))
        print(f"数据已保存到: {save_dir}")
        # 这个 combined_dataset 对象现在包含了来自所有分割和所有类别的数据。
        # 当你从 combined_dataset 中获取一个样本时，它会返回一个元组 (image_tensor, label)。
        # 图像 Tensor 已经应用了你定义的 Transforms。
        # 标签 label 是 ImageFolder 分配的整数标签 (0, 1, 2, 或 3)。

    except FileNotFoundError as e:
        print(f"错误：未找到 OCT2017 数据集的目录或其下的 'train', 'val', 'test' 子文件夹: {e}。请检查你传入的 root_dir 是否正确，以及其中是否包含这三个子文件夹。")
        return None # 加载失败时返回 None
    except Exception as e:
        print(f"加载 OCT2017 数据集时发生其他错误: {e}")
        return None # 加载失败时返回 None

def load_and_split_texas(root_dir,save_dir):
    # 假设文件名是 'feats' 和 'labels'，根据实际情况修改
    datas = np.load(root_dir+'texas100.npz')
    data,label = datas['features'],datas['labels']
    hard_label = np.argmax(label, axis=1)
    np.savez(os.path.join(save_dir+'full.npz'),datas=data,labels=hard_label)

def load_and_split_purchase(root_dir, save_dir):
    """
    读取 root_dir 下的所有 parquet 文件，合并后保存为 full.npz
    """
    # 确保保存目录存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(f"正在从 {root_dir} 读取 Parquet 文件...")

    # 1. 使用 glob 匹配所有 parquet 文件 (包括 train, test, validation 分片)
    # 你的截图中文件位于子目录或根目录，这里假设 root_dir 指向包含 .parquet 文件的文件夹
    parquet_files = glob.glob(os.path.join(root_dir, "*.parquet"))
    
    if not parquet_files:
        print(f"错误: 在 {root_dir} 下未找到任何 .parquet 文件")
        return

    # 2. 读取并合并所有分片文件
    # pandas 的 read_parquet 依赖 pyarrow 或 fastparquet，请确保已安装 (pip install pyarrow)
    df_list = [pd.read_parquet(f) for f in parquet_files]
    full_df = pd.concat(df_list, ignore_index=True)

    print(f"数据加载完成，总行数: {len(full_df)}")
    labels = full_df['label'].to_numpy()
    feature = full_df['feature'].to_numpy()
    feature = np.vstack(feature).astype(np.float32)
    print(f"正在保存到 {save_dir}/full.npz ...")
    print(f"特征形状: {feature.shape}, 标签形状: {labels.shape}")
    # 5. 保存为 .npz
    # 保持你原来的 key 命名: 'datas' 和 'labels'
    save_path = os.path.join(save_dir, 'full.npz')
    np.savez(save_path, datas=feature, labels=labels)
    print("保存成功！")

def load_and_process_yahoo_full_tfidf(root_dir, save_dir="./processed_yahoo_tfidf"):
    # 加载训练集和测试集 CSV 文件
    train_df = pd.read_csv(root_dir+'./train.csv')
    test_df = pd.read_csv(root_dir+'./test.csv')

    # 准备文本数据
    # 根据数据集特点，通常将问题标题、问题内容和最佳答案拼接起来作为完整的文本
    # 确保列名与您的 CSV 文件实际列名一致
    train_texts = train_df['question_title'].fillna('') + ' ' + train_df['question_content'].fillna('') + ' ' + train_df['best_answer'].fillna('')
    test_texts = test_df['question_title'].fillna('') + ' ' + test_df['question_content'].fillna('') + ' ' + test_df['best_answer'].fillna('')

    # 合并训练集和测试集的文本列表
    all_texts = train_texts.tolist() + test_texts.tolist()

    # 准备标签数据
    # 标签通常在 'Class Index' 列，值是 1 到 10
    # 将 1-based 的标签转换为 0-based (减去 1)
    train_labels = train_df['class_index'] - 1
    test_labels = test_df['class_index'] - 1

    # 合并训练集和测试集的标签列表并转换为 NumPy 数组
    all_labels = train_labels.tolist() + test_labels.tolist()
    all_labels_np = np.array(all_labels)

    # 将文本转换为 TF-IDF 特征向量
    # 您可以根据需要调整 max_features 等参数
    vectorizer = TfidfVectorizer(stop_words='english',max_features=5000) # 示例 max_features 设为 5000
    texts_tfidf = vectorizer.fit_transform(all_texts)

    # 将稀疏的 TF-IDF 矩阵转换为密集数组
    texts_tfidf_dense = texts_tfidf.toarray()

    # 创建保存目录如果不存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 保存处理后的数据到 NPZ 文件
    output_file_name = 'full.npz'
    save_path = os.path.join(save_dir, output_file_name)

    data_to_save = (texts_tfidf_dense, all_labels_np)
    save_subset_as_npz_text(data_to_save, save_path)

def simple_tokenizer(text):
    if isinstance(text, str):
        # 移除标点符号，转换为小写，按空格分割
        text = re.sub(r'[^\w\s]', '', text)
        return text.lower().split()
    return [] # 处理非字符串输入

def process_yahoo_for_embedding(root_dir, save_dir="./processed_yahoo_embedding", vocab_size=30000, max_sequence_length=500):
    # 加载训练集和测试集 CSV 文件
    train_df = pd.read_csv(root_dir+'./train.csv')
    test_df = pd.read_csv(root_dir+'./test.csv')

    # 准备文本数据
    # 根据数据集特点，通常将问题标题、问题内容和最佳答案拼接起来作为完整的文本
    # 确保列名与您的 CSV 文件实际列名一致
    train_texts = train_df['question_title'].fillna('') + ' ' + train_df['question_content'].fillna('') + ' ' + train_df['best_answer'].fillna('')
    test_texts = test_df['question_title'].fillna('') + ' ' + test_df['question_content'].fillna('') + ' ' + test_df['best_answer'].fillna('')

    # 合并训练集和测试集的文本列表
    all_texts = train_texts.tolist() + test_texts.tolist()

    # 准备标签数据
    # 标签通常在 'Class Index' 列，值是 1 到 10
    # 将 1-based 的标签转换为 0-based (减去 1)
    train_labels = train_df['class_index'] - 1
    test_labels = test_df['class_index'] - 1

    # 合并训练集和测试集的标签列表并转换为 NumPy 数组
    all_labels = train_labels.tolist() + test_labels.tolist()
    all_labels_np = np.array(all_labels)

    # 分词并构建词汇表
    all_tokens = []
    for text in all_texts:
        all_tokens.extend(simple_tokenizer(text))

    # 统计词频并保留最常见的词汇，加上 Unknown 和 Padding token
    # vocab_size 包括 Unknown 和 Padding token
    word_counts = Counter(all_tokens)
    most_common_words = word_counts.most_common(vocab_size - 2) # 预留给 Unknown 和 Padding
    vocabulary = {word: i + 2 for i, (word, count) in enumerate(most_common_words)} # 从 2 开始编号
    vocabulary['<PAD>'] = 0 # Padding token 索引为 0
    vocabulary['<UNK>'] = 1 # Unknown token 索引为 1

    # 将文本转换为整数序列
    all_sequences = []
    for text in all_texts:
        sequence = [vocabulary.get(word, vocabulary['<UNK>']) for word in simple_tokenizer(text)]
        all_sequences.append(sequence)

    # 序列填充和截断
    # 将所有序列统一到 max_sequence_length 长度
    processed_sequences = np.zeros((len(all_sequences), max_sequence_length), dtype=int)
    for i, sequence in enumerate(all_sequences):
        # 截断或填充序列
        if len(sequence) > max_sequence_length:
            processed_sequences[i, :] = sequence[:max_sequence_length] # 截断
        else:
            processed_sequences[i, :len(sequence)] = sequence # 填充 (默认为 0, 即 Padding token 索引)

    # 保存处理后的序列和标签到 NPZ 文件
    output_file_name = 'full.npz'
    save_path = os.path.join(save_dir, output_file_name)

    # 保存序列和标签
    np.savez(save_path, sequences=processed_sequences, labels=all_labels_np)
    print(f'{len(vocabulary)}, {max_sequence_length}')
    # 返回词汇表大小和序列长度，以便构建模型时使用
    return len(vocabulary), max_sequence_length

def load_and_process_imdb_full_tfidf_from_folders(imdb_root_dir, save_dir="./processed_imdb_tfidf_folders"):
    all_texts = []
    all_labels = []

    # 定义标签映射
    label_map = {'neg': 0, 'pos': 1}

    # 遍历 train 文件夹
    train_dir = os.path.join(imdb_root_dir, 'train')
    for sentiment_folder in ['neg', 'pos']:
        folder_path = os.path.join(train_dir, sentiment_folder)
        label = label_map[sentiment_folder]
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.txt'):
                file_path = os.path.join(folder_path, file_name)
                with open(file_path, 'r', encoding='utf-8') as f:
                    all_texts.append(f.read())
                    all_labels.append(label)

    # 遍历 test 文件夹
    test_dir = os.path.join(imdb_root_dir, 'test')
    for sentiment_folder in ['neg', 'pos']:
        folder_path = os.path.join(test_dir, sentiment_folder)
        label = label_map[sentiment_folder]
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.txt'):
                file_path = os.path.join(folder_path, file_name)
                with open(file_path, 'r', encoding='utf-8') as f:
                    all_texts.append(f.read())
                    all_labels.append(label)

    # 将标签列表转换为 NumPy 数组
    all_labels_np = np.array(all_labels)

    # 将文本转换为 TF-IDF 特征向量
    # 您可以根据需要调整 max_features 等参数
    vectorizer = TfidfVectorizer(stop_words='english',max_features=1000) # 示例 max_features 设为 5000
    texts_tfidf = vectorizer.fit_transform(all_texts)

    # 将稀疏的 TF-IDF 矩阵转换为密集数组
    texts_tfidf_dense = texts_tfidf.toarray()

    # 创建保存目录如果不存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 保存处理后的数据到 NPZ 文件
    output_file_name = 'full.npz'
    save_path = os.path.join(save_dir, output_file_name)

    data_to_save = (texts_tfidf_dense, all_labels_np)
    save_subset_as_npz_text(data_to_save, save_path)


def load_and_split_stl10(split_ratio, download_flag, save_dir="./saved_data"):
    """
    加载并分割 STL-10 数据集
    STL-10 原图 96x96，这里缩放到 32x32 以适配 CIFAR 模型
    """

    # 定义标准化的变换
    transform = transforms.Compose([
        transforms.Resize((96, 96)),  # 强制缩放到 32x32
        transforms.ToTensor(),
        # STL-10 专用的均值和标准差
        transforms.Normalize((0.4467, 0.4398, 0.4066), (0.2603, 0.2565, 0.2712))
    ])

    # 加载 STL-10 数据集
    # 注意：STL-10 使用 'split' 参数，'train' 仅包含 5000 张有标签图像
    # 如果需要更多数据用于无监督训练，可以使用 split='unlabeled'
    train_dataset = torchvision.datasets.STL10(root='./datas/stl10', split='train', download=download_flag,
                                               transform=transform)
    
    # 确保保存目录存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 保存
    save_subset_as_npz(train_dataset, os.path.join(save_dir, 'full.npz'))


def load_and_split_eurosat(split_ratio, download_flag, save_dir="./saved_data"):
    """
    加载并分割 EuroSAT 数据集
    EuroSAT 原图 64x64，这里缩放到 32x32 以适配 CIFAR 模型
    """

    # 定义标准化的变换
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # 强制缩放到 32x32
        transforms.ToTensor(),
        # EuroSAT (RGB) 的推荐均值和标准差
        transforms.Normalize((0.3443, 0.3802, 0.4076), (0.2019, 0.1370, 0.1151))
    ])

    # 加载 EuroSAT 数据集
    # 注意：EuroSAT 在 torchvision 中通常不分 train/test，下载的是全量数据
    train_dataset = torchvision.datasets.EuroSAT(root='./datas/eurosat', download=download_flag,
                                                 transform=transform)
    
    # 确保保存目录存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 保存
    save_subset_as_npz(train_dataset, os.path.join(save_dir, 'full.npz'))


def load_and_process_location(root_dir, save_dir):
    """
    读取 Location (Bangkok) 数据集
    格式假设: 无表头 CSV，第一列是 Label (可能带引号)，后续列是 Features
    """
    # 确保保存目录存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(f"正在从 {root_dir} 搜索数据文件...")

    # 1. 自动寻找目录下的 txt 或 csv 文件
    # 通常 bangkok 数据集文件名为 Bangkok.txt 或 similar
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

    # 2. 读取数据 (关键配置)
    # header=None: 告诉 pandas 第一行就是数据，不是标题
    # quotechar='"': 告诉 pandas 自动去除类似 "label" 的引号
    # sep=None, engine='python': 自动检测分隔符 (逗号或空格)，更稳健
    try:
        df = pd.read_csv(data_file, header=None, quotechar='"', sep=None, engine='python')
    except Exception as e:
        print(f"读取失败，尝试强制使用逗号分隔... {e}")
        df = pd.read_csv(data_file, header=None, quotechar='"', sep=',')

    print(f"原始数据加载完成，形状: {df.shape}")

    # 3. 分离标签和特征
    # 第 0 列是 Label
    raw_labels = df.iloc[:, 0].values
    # 第 1 列到最后一列是 Features
    raw_features = df.iloc[:, 1:].values

    # 4. 处理标签 (Label Encoding)
    # 必须把标签转为 0 ~ N-1 的整数，否则 CrossEntropyLoss 会报错
    print("正在编码标签...")
    le = LabelEncoder()
    labels = le.fit_transform(raw_labels)
    
    # 打印标签映射关系 (可选，调试用)
    # print(f"标签映射示例: 原始 {raw_labels[:3]} -> 编码 {labels[:3]}")

    # 5. 处理特征
    # 确保是 float32 类型的数值矩阵
    print("正在转换特征格式...")
    features = raw_features.astype(np.float32)

    # 6. 获取维度信息 (用于设置模型参数)
    input_dim = features.shape[1]
    num_classes = len(le.classes_)
    
    print(f"处理完成!")
    print(f"-> 特征维度 (Input Size): {input_dim}")
    print(f"-> 类别数量 (Num Classes): {num_classes}")
    print(f"-> 样本总数: {len(labels)}")

    # 7. 保存
    save_path = os.path.join(save_dir, 'full.npz')
    np.savez(save_path, datas=features, labels=labels)
    print(f"数据已保存到: {save_path}")

def process_data(args,download=False):
    # ===== 固定随机种子, 确保数据划分可复现 =====
    random_seed = getattr(args, 'random_seed', 123)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    split_ratio = args.split_ratio
    if args.dataset == 'CIFAR10':
        save_path = './datas/CIFAR10'
        path_exists(save_path)
        load_and_split_cifar10(split_ratio, download,save_path)
    elif args.dataset == 'CIFAR100':
        save_path='./datas/CIFAR100'
        path_exists(save_path)
        load_and_split_cifar100(split_ratio,download,save_path)
    elif args.dataset == 'GTSRB':
        save_path='./datas/GTSRB'
        path_exists(save_path)
        load_and_split_gtsrb(split_ratio,download,save_path)
    elif args.dataset == 'CINIC10':
        save_path = './datas/CINIC10'
        path_exists(save_path)
        load_and_split_cinic10(split_ratio,download,save_path)
    elif args.dataset == '20Newsgroups':
        save_path = './datas/20Newsgroups'
        path_exists(save_path)
        load_and_split_20newsgroups(split_ratio,download,save_path)
    elif args.dataset == 'tinyimagenet':
        save_path = './datas/tinyimagenet'
        path_exists(save_path)
        load_and_process_tiny_imagenet('./datas/Tinyimagenet-offical',save_path)
    elif args.dataset == 'texas':
        save_path='./datas/texas/'
        path_exists(save_path)
        load_and_split_texas(root_dir='./datas/texas-offical/',save_dir=save_path)

    elif args.dataset == 'purchase':
        save_path='./datas/purchase/'
        path_exists(save_path)
        load_and_split_purchase(root_dir='./datas/purchase-offical/',save_dir=save_path)
    elif args.dataset == 'DBP':
        save_path = './datas/DBP'
        path_exists(save_path)
        process_dbpedia_local_tfidf(root_dir='./datas/DBPEDIA-offical/',save_dir=save_path)
    elif args.dataset == 'OCT':
        save_path = './datas/OCT'
        path_exists(save_path)
        load_oct2017(root_dir='./datas/OCT-offical/',save_dir = save_path)
    elif args.dataset == 'yahoo':
        save_path = './datas/yahoo'
        path_exists(save_path)
        process_yahoo_for_embedding(root_dir='./datas/yahoo-offical/',
                                    save_dir = save_path,
                                    vocab_size=30000,
                                    max_sequence_length=500,
                                    )
    elif args.dataset == 'imdb':
        save_path = './datas/imdb'
        path_exists(save_path)
        load_and_process_imdb_full_tfidf_from_folders('./datas/imdb-offical/',save_dir=save_path)
    elif args.dataset == 'STL10':
        save_path='./datas/STL10'
        path_exists(save_path)
        load_and_split_stl10(split_ratio,download,save_path)
    elif args.dataset == 'EuroSAT':
        save_path='./datas/EuroSAT'
        path_exists(save_path)
        load_and_split_eurosat(split_ratio,download,save_path)
    elif args.dataset == 'location':
        save_path='./datas/location'
        load_and_process_location(root_dir='./datas/location_offical/',save_dir=save_path)

    elif args.dataset == 'purchase':
        save_path='./datas/purchase/'
        path_exists(save_path)
        load_and_split_purchase(root_dir='./datas/purchase-offical/',save_dir=save_path)
    else:
        print('dataset is error')
        sys.exit(1)


if __name__ == "__main__":
    # 使用示例
    split_ratio = 0.9
    save_path = './datas/CIFAR10'
    path_exists(save_path)
    load_and_split_cifar10(split_ratio, save_path)
