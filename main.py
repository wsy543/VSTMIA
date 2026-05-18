# ===== 第一步: 自动选GPU (必须在 import torch 之前, 只用 stdlib) =====
import os
import subprocess
import time
import random
import argparse

# 随机延迟避免两个进程同时查询 nvidia-smi 的竞态
time.sleep(random.uniform(0, 2))

def get_best_gpu():
    """
    通过 nvidia-smi 命令自动检测显存剩余最多的显卡并返回其 ID。
    如果检测失败，默认返回 '0'。
    """
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        best_gpu_id = "0"
        max_free_memory = -1
        for line in result.stdout.strip().split('\n'):
            if line:
                gpu_id, free_memory = line.split(',')
                gpu_id = gpu_id.strip()
                free_memory = int(free_memory.strip())
                if free_memory > max_free_memory:
                    max_free_memory = free_memory
                    best_gpu_id = gpu_id
        print(f"Auto-selected GPU: {best_gpu_id} with free memory: {max_free_memory} MB")
        return best_gpu_id
    except Exception as e:
        print(f"Failed to auto-detect GPU memory: {e}, defaulting to GPU 0")
        return "0"

# 自动分配负载最低的显卡 — 必须在 import torch 之前设置!
os.environ['CUDA_VISIBLE_DEVICES'] = get_best_gpu()

# ===== 第二步: 现在才 import torch =====
import torch
from train import FederatedLearning
from data_processing import process_data
import logging
from utils import path_exists
from baseline_attack import ICLR2023,USENIX2024,SP19,Arxiv2025,MBA,EnhancedMIA,CSF18
import ours
import test

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
    parser.add_argument('--client_num', type=int, default=5)# 20
    parser.add_argument('--data_split',type=str,default='uniform',help = 'uniform dirichlet')
    parser.add_argument('--model', type=str, default='resnet',help='alexnet resnet mobilenet densenet  nn textcnn')
    parser.add_argument('--save_path', type=str, default='./models')
    
    parser.add_argument('--random_client_mode',type=bool,default= False,help='确定客户端选择方式,随机或者是顺序选择 True是随机')
    parser.add_argument('--epochs', type=int, default=2, help='每个客户端在本地自己训练的epoch')# 5
    parser.add_argument('--batch_size', type=int, default=64) #
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--optimizer', type=str, default='SGD', help='SGD,Adam')
    parser.add_argument('--training_round', type=int, default=200, help='模型总的训练轮数')# 200
    parser.add_argument('--participant', type=int, default=5, help='每一轮的参与者数量')
    parser.add_argument('--attacker_client_idx',type=int,default=0)
    parser.add_argument('--collusion_client_idx',type=int,nargs="+",default=[1,2])
    parser.add_argument('--save_client_model_idx',type=int,nargs="+",default=[0,1],help='server save these clients` model')
    parser.add_argument('--arxiv_client',type=int,nargs="+",default=[0,1],help="arxiv2025 need two client for attack")
    parser.add_argument('--data_path', type=str, default='./datas')
    parser.add_argument('--model_path', type=str, default='./models_main',help='./models or ./uniform_models')
    parser.add_argument('-alpha', type=float, default=0.2, help='迪利克雷分布的参数,越大越均匀,TDSC24的论文中说alpha为100时基本均匀')
    
    parser.add_argument('--split_ratio', type=float, default=0.5)
    parser.add_argument('--random_seed', type=int, default=123)
    parser.add_argument('--log_name', type=str, default='train_models')
    parser.add_argument('--lr',type=float,default=0.005)# mobilenet为0.001，其他网络为0.005，文本为0.01
    parser.add_argument('--steplr',type=bool,default=False) # 图像数据集都没有使用
    parser.add_argument('--lr_gamma',type=float,default=0.99)
    parser.add_argument('--lr_step',type=int,default=1)
    parser.add_argument('--method',type=str,default='ours',help='arxiv,USENIX,fluctuate,arxiv,MBA,enhancedMIA, CSF18 ICLR')

    parser.add_argument('--vlm_type', type=str, default='qwen3',
                        choices=['qwen3', 'qwen3_8b', 'gemma4', 'llama3.2',
                                 'internvl3.5', 'internvl3.5_2b', 'smolvlm2',
                                 'llavaov', 'glm4.1v', 'glm4.1vbase', 'ovis2.5'],
                        help='VLM 模型选择: qwen3(2B) / qwen3_8b(8B) / gemma4 / '
                             'llama3.2 / internvl3.5(8B) / internvl3.5_2b(2B) / '
                             'smolvlm2(2.2B) / llavaov(7B) / glm4.1v(9B-思考模型) / '
                             'glm4.1vbase(9B-基础模型) / ovis2.5(2B)')
    parser.add_argument('--vlm_path', type=str, default=None,
                        help='自定义 VLM 路径, 不指定则用预设路径')

    parser.add_argument('--data_process_flag', type=bool, default=False)# 这个开关很危险，慎重！重新对数据进行训练和测试集划分生成full文件
    parser.add_argument('--train_model', type=bool,default=False)

    parser.add_argument('--regenerate_plots', default=False, action='store_true',
                        help='是否重新生成 VLM 输入图片, 不加此参数则跳过生成直接攻击')
    # Phase 开关 (默认开启 Phase 2, 传 --no_phase2 关闭)
    parser.add_argument('--no_phase2', dest='phase2', action='store_false', default=True,
                        help='禁用 Phase 2 (Physics-based Analysis)')
    # 物理分析超参数
    parser.add_argument('--physics_alpha_cap', type=float, default=5.0,
                        help='物理特征权重上限 (adaptive_alpha 的 clamp 值), 默认 5.0')
    parser.add_argument('--calib_threshold', type=float, default=0.1,
                        help='校准池伪负样本的 VLM 分数阈值, 默认 0.1')
    parser.add_argument('--head_ratio', type=float, default=0.5,
                        help='头部轮数比例 (用于下降速率特征提取), 默认 0.5')
    parser.add_argument('--tail_ratio', type=float, default=0.5,
                        help='尾部轮数比例 (用于末期波动特征提取), 默认 0.5')
    
    # VLM 图片生成参数 (控制输入图像的尺寸, 影响 VLM 的视觉理解效果)
    parser.add_argument('--plot_width', type=float, default=6.22,
                        help='图片宽度 (英寸), 默认 6.0, 建议与 plot_height 相近以兼容方形 resize 的模型')
    parser.add_argument('--plot_height', type=float, default=2.67,
                        help='图片高度 (英寸), 默认 6.0')
    parser.add_argument('--plot_dpi', type=int, default=150,
                        help='图片分辨率 DPI, 默认 150')
    # qwen3是 (6.22, 2.67)

    # ========== 防御参数 ==========
    parser.add_argument('--defence', type=str, default='none',
                        choices=['none', 'dpsgd', 'mixupmmd', 'l2'],
                        help='选择防御方法: none / dpsgd / mixupmmd / l2, 默认不启用')
    parser.add_argument('--dp_clip_norm', type=float, default=50.0,
                        help='DP-SGD 梯度裁剪阈值, 默认 60.0 (匹配 shufflenet 梯度量级, 几乎不裁剪)')
    parser.add_argument('--dp_noise_multiplier', type=float, default=0.0015,
                        help='DP-SGD 噪声乘数, 默认 0.0005 (noise_std=0.03, 极轻度扰动)')
    parser.add_argument('--mixup_alpha', type=float, default=0.5,
                        help='MixupMMD 的 Beta 分布 alpha 参数, 默认 0.2')
    parser.add_argument('--mmd_lambda', type=float, default=0.1, 
                        help='MixupMMD 的 MMD 正则化权重, 默认 0.01')
    parser.add_argument('--l2_lambda', type=float, default=0.001,
                        help='L2 正则化系数 (weight_decay), 默认 0.001 (收敛推荐)')

    parser.add_argument('--arxiv_save',type=bool,default=True)
    return parser.parse_args()

# torch.backends.cudnn.enabled = False
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# /home/tcadb3090/anaconda3/envs/mamba/bin/python /home/tcadb3090/code/VLMMIA/FL_train/main.py

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
        if args.regenerate_plots:
            ours = ours.ours(args=args,size=1000)
            ours.make_loader_for_vlm()
        else:
            print("[Skip] 跳过图片生成, 直接使用已有图片进行攻击")
        # 生成完图片后自动运行攻击
        test.run_attack(dataset=args.dataset, model=args.model, max_samples=2000,
                        vlm_type=args.vlm_type, vlm_path=args.vlm_path,
                        enable_phase2=args.phase2,
                        alpha_cap=args.physics_alpha_cap,
                        calib_threshold=args.calib_threshold,
                        head_ratio=args.head_ratio,
                        tail_ratio=args.tail_ratio)
    elif args.method == 'arxiv':
        ours = Arxiv2025(args=args)
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
