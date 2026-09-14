import os
import subprocess
import time
import random
import argparse

time.sleep(random.uniform(0, 2))

def get_best_gpu():
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

os.environ['CUDA_VISIBLE_DEVICES'] = get_best_gpu()

import torch
from train import FederatedLearning
from data_processing import process_data
import logging
from utils import path_exists
from baseline_attack import ICLR2023,USENIX2024,SP19,Arxiv2025,MBA,EnhancedMIA,CSF18
import ours
import test

def init_logging(args):
    log_path='./log_file/' + args.model+'/'+args.dataset
    path_exists(log_path)
    logging.basicConfig(
        format='%(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_path + '/' + args.log_name,mode='w'),
            logging.StreamHandler()
        ]
    )


def init_args():
    parser = argparse.ArgumentParser(description='VLMFLMIA parameters')
    parser.add_argument('--dataset', type=str, default='STL10', choices=['STL10', 'location'],
                        help='STL10 location')
    parser.add_argument('--client_num', type=int, default=5)
    parser.add_argument('--data_split',type=str,default='uniform',help = 'uniform dirichlet')
    parser.add_argument('--model', type=str, default='resnet', choices=['resnet', 'nn'], help='resnet nn')
    parser.add_argument('--save_path', type=str, default='./models')
    
    parser.add_argument('--random_client_mode',type=bool,default= False,help='确定客户端选择方式,随机或者是顺序选择 True是随机')
    parser.add_argument('--epochs', type=int, default=2, help='每个客户端在本地自己训练的epoch')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--optimizer', type=str, default='SGD', help='SGD,Adam')
    parser.add_argument('--training_round', type=int, default=200, help='模型总的训练轮数')
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
    parser.add_argument('--lr',type=float,default=0.01)
    parser.add_argument('--steplr',type=bool,default=False)
    parser.add_argument('--lr_gamma',type=float,default=0.99)
    parser.add_argument('--lr_step',type=int,default=1)
    parser.add_argument('--method',type=str,default='ours',help='arxiv,USENIX,fluctuate,arxiv,MBA,enhancedMIA, CSF18 ICLR')

    parser.add_argument('--vlm_type', type=str, default='qwen3_2b',
                        choices=['qwen3_2b'],
                        help='VLM 模型选择: qwen3_2b(2B)')
    parser.add_argument('--vlm_path', type=str, default=None,
                        help='自定义 VLM 路径, 不指定则用预设路径')

    parser.add_argument('--data_process_flag', type=bool, default=False)
    parser.add_argument('--train_model', type=bool,default=False)

    parser.add_argument('--regenerate_plots', default=False, action='store_true',
                        help='是否重新生成 VLM 输入图片, 不加此参数则跳过生成直接攻击')
    parser.add_argument('--no_phase2', dest='phase2', action='store_false', default=True,
                        help='禁用 Phase 2 (Physics-based Analysis)')
    parser.add_argument('--physics_alpha_cap', type=float, default=5.0,
                        help='物理特征权重上限 (adaptive_alpha 的 clamp 值), 默认 5.0')
    parser.add_argument('--calib_threshold', type=float, default=0.1,
                        help='校准池伪负样本的 VLM 分数阈值, 默认 0.1')
    parser.add_argument('--head_ratio', type=float, default=0.5,
                        help='头部轮数比例 (用于下降速率特征提取), 默认 0.5')
    parser.add_argument('--tail_ratio', type=float, default=0.5,
                        help='尾部轮数比例 (用于末期波动特征提取), 默认 0.5')
    
    parser.add_argument('--plot_width', type=float, default=6.22,
                        help='图片宽度 (英寸), 默认 6.0, 建议与 plot_height 相近以兼容方形 resize 的模型')
    parser.add_argument('--plot_height', type=float, default=2.67,
                        help='图片高度 (英寸), 默认 6.0')
    parser.add_argument('--plot_dpi', type=int, default=150,
                        help='图片分辨率 DPI, 默认 150')

    parser.add_argument('--arxiv_save',type=bool,default=True)
    return parser.parse_args()


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
    elif args.method == 'SP':
        sp = SP19(args=args,data_size=350)
        sp.attack(train=True)
    elif args.method == 'ours':
        if args.regenerate_plots:
            ours = ours.ours(args=args,size=1000)
            ours.make_loader_for_vlm()
        else:
            print("[Skip] 跳过图片生成, 直接使用已有图片进行攻击")
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
        ours.attack('mentropy')
    elif args.method == 'enhancedMIA':
        ours=EnhancedMIA(args)
        ours.attack_d()
    elif args.method == 'CSF18':
        ours=CSF18(args)
        ours.attack()

    else:
        None
    print(f'model:{args.model},dataset:{args.dataset}')
