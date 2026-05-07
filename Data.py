from torch.utils.data import DataLoader, Dataset
import numpy as np
from collections import defaultdict
import torch
import os
import pandas as pd
from PIL import Image


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


def load_pth_data(data_name):
    dataset = torch.load(data_name)
    return dataset

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
    """
    使用均匀分布将数据集进行划分，每个客户端获得相等数量的数据。
    :param dataset: 数据集 (例如图像数据)
    :param labels: 数据集的标签
    :param num_splits: 要划分的客户端数量
    :param seed: 随机种子 (可选)
    :return: 划分后的数据和标签
    """
    if seed is not None:
        np.random.seed(seed)
    
    # 获取样本数
    num_samples = len(dataset)
    
    # 随机打乱数据和标签
    indices = np.random.permutation(num_samples)
    dataset = np.array(dataset)[indices]
    labels = np.array(labels)[indices]
    
    # 计算每个客户端分配的样本数量
    samples_per_client = num_samples // num_splits

    # 用于存储划分后的数据和标签
    split_data = [[] for _ in range(num_splits)]
    split_labels = [[] for _ in range(num_splits)]
    
    # 将数据均匀地划分到各个客户端
    for i in range(num_splits):
        start_idx = i * samples_per_client
        # end_idx = (i + 1) * samples_per_client if i != num_splits - 1 else num_samples
        end_idx = (i + 1) * samples_per_client # 为了保证所有客户端的数据集相同所以最后舍弃部分数据
        split_data[i] = dataset[start_idx:end_idx]
        split_labels[i] = labels[start_idx:end_idx]

    return split_data, split_labels

# 数据以non-iid方式划分成指定份数
def non_iid_dirichlet_split(dataset, labels, num_splits, alpha, seed=None):
    """
    使用狄利克雷分布将数据集进行非独立同分布划分
    :param dataset: 数据集 (e.g., 图像数据)
    :param labels: 数据集的标签 (e.g., 类别标签)
    :param num_splits: 要划分的客户端数量
    :param alpha: 狄利克雷分布的α参数，控制不均匀性 (值越小，分布越极端)
    :return: 按狄利克雷分布划分的数据和标签
    """
    if seed is not None:
        np.random.seed(seed)
    num_classes = len(np.unique(labels))  # 获取类别数
    label_indices = defaultdict(list)

    # 收集每个类别的样本索引
    for i, label in enumerate(labels):
        label_indices[label].append(i)

    # 每个类别的数据分配到不同客户端的比例矩阵
    class_split_ratios = np.random.dirichlet([alpha] * num_splits, num_classes)

    # 用于存储划分后的数据和标签
    split_data = [[] for _ in range(num_splits)]
    split_labels = [[] for _ in range(num_splits)]

    # 按类别分配数据
    for cls, indices in label_indices.items():
        np.random.shuffle(indices)  # 随机打乱每个类别的样本
        class_size = len(indices)  # 当前类别的样本数量
        # 按照狄利克雷分布的比例为每个客户端分配该类别的样本
        split_sizes = (class_split_ratios[cls] * class_size).astype(int)

        # 保证分配总数正确
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
    """Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Arguments:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """
    def __init__(self, *tensors):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors), \
            "所有张量的第一个维度必须相同"
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].size(0)


class ISICSplitDataset(Dataset):
    def __init__(self, data_root, split, transform=None):
        """
        Args:
            data_root (string): 包含所有 ISIC2018 Task 3 数据文件夹和文件的根目录
                                (例如 './ISIC2018_Task3/').
                                该目录下应包含如 'ISIC2018_Task3_Training_Input',
                                'ISIC2018_Task3_Training_GroundTruth.csv' 等子目录和文件。
            split (string): 数据集分割类型 ('train', 'validation', 'test').
            transform (callable, optional): 应用于图像的预处理 transforms。
        """
        self.data_root = data_root
        self.split = split
        self.transform = transform

        # --- 定义对应分割的图像目录和元数据文件路径 ---
        # 这些路径模式必须与你下载的 ISIC 2018 Task 3 文件夹结构相匹配
        if split == 'train':
            self.image_dir = os.path.join(data_root, 'ISIC2018_Task3_Training_Input')
            self.metadata_csv_path = os.path.join(data_root, 'ISIC2018_Task3_Training_GroundTruth.csv')
        elif split == 'validation':
            self.image_dir = os.path.join(data_root, 'ISIC2018_Task3_Validation_Input')
            self.metadata_csv_path = os.path.join(data_root, 'ISIC2018_Task3_Validation_GroundTruth.csv')
        elif split == 'test':
             self.image_dir = os.path.join(data_root, 'ISIC2018_Task3_Test_Input')
             # 测试集的 Ground Truth 文件可能不公开提供，这里处理这种情况
             self.metadata_csv_path = os.path.join(data_root, 'ISIC2018_Task3_Test_GroundTruth.csv') # 假设文件名，请确认
        else:
            raise ValueError(f"无效的分割类型: {split}. 请选择 'train', 'validation' 或 'test'。")


        # --- 加载和处理元数据 ---
        # 假设 CSV 文件的结构有 'image' 列和二值诊断类别列 (如 'MEL', 'NV', 'BCC' 等)
        # ISIC 2018 Task 3 分类任务的类别列通常是这 8 个：
        diagnosis_cols = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC', 'SCC']

        self.image_info = [] # 存储 (image_id, label) 元组的列表

        print(f"正在加载 {self.split} 分割的元数据文件: {self.metadata_csv_path}")
        try:
            # 尝试读取元数据 CSV
            metadata_df = pd.read_csv(self.metadata_csv_path)

            # 创建诊断名称到整数标签的映射 (根据 diagnosis_cols 的顺序)
            self.diagnosis_to_label = {diag: i for i, diag in enumerate(diagnosis_cols)}

            # 遍历元数据 DataFrame 的每一行
            for index, row in metadata_df.iterrows():
                image_id = row['image'] # 假设图像 ID 的列名是 'image'

                # --- 提取分类标签 ---
                label = -1 # 默认标签值

                # 查找值为 1 的诊断列
                current_diagnoses = [diag for diag in diagnosis_cols if row.get(diag, 0) == 1] # 使用 .get 安全访问列

                # 处理单标签样本 (通常是分类任务的目标)
                if len(current_diagnoses) == 1:
                     diagnosis_name = current_diagnoses[0]
                     label = self.diagnosis_to_label[diagnosis_name]
                elif self.split != 'test' and len(current_diagnoses) > 1:
                     # 对于训练集/验证集的多标签样本，如果在单标签任务中，通常跳过或特殊处理
                     # print(f"警告: {split} 分割中发现多标签样本 {image_id}: {current_diagnoses}。跳过。")
                     continue # 示例中跳过多标签样本

                elif self.split != 'test' and len(current_diagnoses) == 0:
                     # 对于训练集/验证集没有指定诊断的样本，通常跳过
                     # print(f"警告: {split} 分割中样本 {image_id} 没有指定诊断。跳过。")
                     continue # 示例中跳过无诊断样本
                # 对于测试集，即使没有 Ground Truth CSV，我们也需要将图像加入列表以便后续处理（如预测）
                # 如果有 Ground Truth CSV，则按上面的逻辑提取标签
                elif self.split == 'test':
                    if len(current_diagnoses) == 1:
                         diagnosis_name = current_diagnoses[0]
                         label = self.diagnosis_to_label[diagnosis_name]
                    else:
                         # 如果测试集有 Ground Truth 但不是标准的单标签，也跳过或按需处理
                         continue # 示例中跳过测试集中非标准单标签的样本


                # 将成功确定标签（或测试集且无 Ground Truth）的样本添加到 image_info 列表
                if label != -1 or (self.split == 'test' and self.metadata_csv_path is None):
                     # 对于测试集且无 Ground Truth 的情况，label 可以保持 -1 或 None
                     if self.split == 'test' and self.metadata_csv_path is None:
                         label = -1 # Or None, depending on how you handle test set without ground truth

                     self.image_info.append((image_id, label))

            print(f"加载完成。{self.split} 分割共找到 {len(self.image_info)} 个有效样本。")

        except FileNotFoundError:
            # 如果元数据文件不存在 (特别是测试集无 Ground Truth 时)，尝试只列出图像文件
            if self.split == 'test':
                 print(f"警告: 未找到 {self.split} 分割的元数据文件 ({self.metadata_csv_path})。将尝试仅加载输入图像文件。")
                 if os.path.exists(self.image_dir):
                     image_files = [f for f in os.listdir(self.image_dir) if f.endswith('.jpg')] # 假设图像格式是 .jpg
                     # 对于测试集无 Ground Truth，标签设为 -1 或 None
                     self.image_info = [(f.split('.')[0], -1) for f in image_files] # 存储 (image_id, -1)
                     print(f"在图像目录 {self.image_dir} 中找到 {len(self.image_info)} 张图像。")
                 else:
                     print(f"错误: 未找到 {self.split} 分割的图像目录: {self.image_dir}。")
                     self.image_info = [] # 没有数据
            else:
                 # 训练集或验证集元数据缺失是关键错误
                 print(f"错误: 未找到 {self.split} 分割的元数据文件: {self.metadata_csv_path}。无法加载 {self.split} 数据。")
                 self.image_info = [] # 没有数据

        except Exception as e:
            print(f"加载或处理 {self.split} 分割的元数据时发生错误: {e}")
            self.image_info = [] # 没有数据


    def __len__(self):
        # 返回数据集中样本的总数
        return len(self.image_info)

    def __getitem__(self, idx):
        # 根据索引获取一个样本的数据
        image_id, label = self.image_info[idx]

        # 构建图像文件的完整路径
        image_path = os.path.join(self.image_dir, image_id + '.jpg') # 假设图像格式是 .jpg

        # 加载图像
        image = Image.open(image_path).convert('RGB') # 加载图像并转换为 RGB 格式

        # 应用 transforms
        if self.transform:
            image = self.transform(image)

        # 返回图像 Tensor 和对应的标签 (整数标签)
        # 对于测试集无 Ground Truth，标签可能是 -1
        return image, label # 或者根据需要返回 torch.tensor(label, dtype=torch.long)