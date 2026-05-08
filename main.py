import argparse
import torch
from train import FederatedLearning
from data_processing import process_data
import logging
from utils import path_exists
from baseline_attack import ICLR2023,USENIX2024,SP19,Arxiv2025,MBA,EnhancedMIA,CSF18
import os
import ours
import subprocess
import test


def auto_select_gpu():
    """
    自动选择显存占用最少的GPU
    :return: GPU索引
    """
    # 优先使用 pynvml
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        best_gpu = 0
        max_free_memory = 0
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            free_memory = mem_info.free
            if free_memory > max_free_memory:
                max_free_memory = free_memory
                best_gpu = i
        pynvml.nvmlShutdown()
        logging.info(f"自动选择GPU {best_gpu}, 空闲显存: {max_free_memory / 1024**3:.2f} GB")
        return best_gpu
    except ImportError:
        pass

    # 备选方案: 使用 nvidia-smi
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        lines = result.stdout.strip().split('\n')
        best_gpu = 0
        max_free_memory = 0
        for line in lines:
            parts = line.split(',')
            idx = int(parts[0].strip())
            free_mem = int(parts[1].strip())
            if free_mem > max_free_memory:
                max_free_memory = free_mem
                best_gpu = idx
        logging.info(f"自动选择GPU {best_gpu}, 空闲显存: {max_free_memory / 1024:.2f} MB")
        return best_gpu
    except Exception:
        logging.warning("无法自动检测GPU, 使用GPU 0")
        return 0

def init_logging(args):
    """
    初始化log设置
    :param args:
    :return:
    """
    log_path='./log_file/' + args.model+'/'+args.dataset
    path_exists(log_path)
    logging.basicConfig(
        # filename=log_path + '/' + args.log_name,
        # filemode='w',
        format='%(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_path + '/' + args.log_name,mode='w'),  # 将日志写入文件
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )


def init_args():
    parser = argparse.ArgumentParser(description='VLMFLMIA parameters')
    parser.add_argument('--dataset', type=str, default='STL10',help='CIFAR10 CIFAR100 CINIC10 tinyimagenet 20Newsgroups yahoo OCT EuroSAT STL10 location texas')
    parser.add_argument('--client_num', type=str, default=5)# 20
    parser.add_argument('--data_split',type=str,default='uniform',help = 'uniform dirichlet')
    parser.add_argument('--model', type=str, default='mobilenet',help='alexnet resnet mobilenet densenet  nn textcnn')
    parser.add_argument('--save_path', type=str, default='./models')
    
    parser.add_argument('--random_client_mode',type=bool,default= False,help='确定客户端选择方式,随机或者是顺序选择 True是随机')
    parser.add_argument('--epochs', type=int, default=2, help='每个客户端在本地自己训练的epoch')# 5
    parser.add_argument('--batch_size', type=int, default=64) #
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--optimizer', type=str, default='SGD', help='SGD,Adam')
    parser.add_argument('--training_round', type=int, default=300, help='模型总的训练轮数')# 200
    parser.add_argument('--participant', type=int, default=5, help='每一轮的参与者数量')
    parser.add_argument('--attacker_client_idx',type=int,default=0)
    parser.add_argument('--collusion_client_idx',type=int,nargs="+",default=[1,2,3])
    parser.add_argument('--save_client_model_idx',type=int,nargs="+",default=[0,1],help='server save these clients` model')
    parser.add_argument('--arxiv_client',type=int,nargs="+",default=[0,1],help="arxiv2025 need two client for attack")
    parser.add_argument('--data_path', type=str, default='./datas')
    parser.add_argument('--model_path', type=str, default='./models',help='./models or ./uniform_models')
    parser.add_argument('-alpha', type=float, default=0.2, help='迪利克雷分布的参数,越大越均匀,TDSC24的论文中说alpha为100时基本均匀')
    
    parser.add_argument('--split_ratio', type=float, default=0.5)
    parser.add_argument('--random_seed', type=int, default=123)
    parser.add_argument('--log_name', type=str, default='train_models')
    parser.add_argument('-lr',type=float,default=0.005)# mobilenet为0.001，其他网络为0.005，文本为0.01
    parser.add_argument('--steplr',type=bool,default=False) # 图像数据集都没有使用
    parser.add_argument('--lr_gamma',type=float,default=0.99)
    parser.add_argument('--lr_step',type=int,default=1)
    parser.add_argument('--method',type=str,default='ours',help='arxiv,USENIX,fluctuate,arxiv,MBA,enhancedMIA, CSF18 ICLR')

    parser.add_argument('--data_process_flag', type=bool, default=True)# 这个开关很危险，慎重！重新对数据进行训练和测试集划分生成full文件
    parser.add_argument('--train_model', default= True)

    parser.add_argument('--arxiv_save',type=bool,default=True)
    return parser.parse_args()

# torch.backends.cudnn.enabled = False
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# /home/tcadb3090/anaconda3/envs/mamba/bin/python /home/tcadb3090/code/VLMMIA/FL_train/main.py

# 自动选择显存占用最少的GPU
selected_gpu = auto_select_gpu()
os.environ['CUDA_VISIBLE_DEVICES'] = str(selected_gpu)
if __name__ == '__main__':
    args = init_args()
    init_logging(args)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.data_process_flag is True:
        process_data(args,download=True)
    if args.train_model is True:
        FL = FederatedLearning(args)
        FL.train_FL_models()
    if args.method== 'ICLR':
        iclr=ICLR2023(args,0,500)
        iclr.attack()
    elif args.method == 'USENIX':
        usenix = USENIX2024(args=args,data_size=500)
        usenix.attack('server')
        # usenix.attack('client')
    elif args.method == 'SP':
        sp = SP19(args=args,data_size=350)
        sp.attack(train=True)
    elif args.method == 'ours':
        ours = ours.ours(args=args,size=1000)
        ours.make_loader_for_vlm()
        # 生成完图片后自动运行攻击
        test.run_attack(dataset=args.dataset, model=args.model, max_samples=1000)
    elif args.method == 'arxiv':
        ours = Arxiv2025(args=args,test_size=500)
        ours.attack()
    elif args.method == 'MBA':
        ours = MBA(args=args)
        # ours.attack('entropy')
        ours.attack('mentropy')
        # ours.attack_client('entropy')
        # ours.attack_client('mentropy')
    elif args.method == 'enhancedMIA':
        ours=EnhancedMIA(args)
        ours.attack_d()
    elif args.method == 'CSF18':
        ours=CSF18(args)
        ours.attack()

    else:
        None
    print(f'model:{args.model},dataset:{args.dataset}')
