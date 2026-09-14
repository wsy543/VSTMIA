# VLM-MIA：基于视觉语言模型的联邦学习成员推理攻击

本项目实现了一套针对联邦学习（Federated Learning, FL）模型的成员推理攻击（Membership Inference Attack, MIA）流程：

1. 在 FL 训练过程中保存每一轮的服务器模型权重；
2. 用这些中间模型对每个样本计算逐轮 loss 序列，并渲染成 loss 曲线图片；
3. 使用视觉语言模型（VLM）观察曲线几何形态，判断样本是成员（Member）还是非成员（Non-member）；
4. 同时提取曲线上的物理动力学特征（早期下降速率、末期波动），用非参数经验分位数（eCDF）背景分布校准后，与 VLM 分数按自适应权重融合；
5. 输出 AUC、TPR@FPR 等评估指标与可视化报告。

```
数据准备 --> 联邦训练 --> loss 曲线渲染 --> VLM 打分 + 物理特征融合 --> 评估报告
```

## 1. 支持范围

| 项目 | 取值 |
|---|---|
| 数据集 | `STL10`（图像，10 类）、`location`（Bangkok 表格数据，30 类） |
| 模型 | `resnet`（ResNet-9-9-9）、`nn`（3 层 MLP） |
| VLM | `qwen3_2b`（Qwen3-VL-2B-Instruct） |
| 攻击方法 | `ours`（本项目方法），另保留 7 个基线方法 `ICLR` / `USENIX` / `SP` / `arxiv` / `MBA` / `enhancedMIA` / `CSF18` |

## 2. 环境要求

- Linux，Python >= 3.9（推荐 3.10）
- NVIDIA GPU（显存 >= 8 GB）：VLM 权重约 4.3 GB（bf16），联邦训练与攻击都需要 GPU
- 磁盘空间：STL10 原始数据约 2.6 GB + 处理后 `full.npz` 约 0.6 GB；VLM 权重约 4.3 GB

依赖安装：

```bash
pip install -r requirements.txt
```

## 3. 快速开始（一键运行）

```bash
bash run.sh --install        # 首次运行: 安装依赖
bash run.sh                  # 一键跑通完整流程 (STL10 + resnet + 200 轮 + 攻击)
```

`run.sh` 会自动完成以下 5 个步骤：

1. 检查 Python 与依赖；
2. 下载数据集（STL10 官方源 / location 数据集）；
3. 下载 VLM 权重到 `./vlm_2b`（默认优先 ModelScope，失败自动切换 HuggingFace）；
4. 执行完整流程：数据划分 → 联邦训练 → 生成 loss 曲线图片 → VLM 成员推理攻击；
5. 打印结果目录与关键输出文件。

其他常用用法：

```bash
bash run.sh --quick                              # 快速冒烟验证 (10 轮训练)
bash run.sh --dataset location --model nn        # location 数据集 + MLP 模型
bash run.sh --rounds 50 --clients 5              # 自定义训练轮数
bash run.sh --skip-data --skip-vlm               # 数据与权重都已就绪时
bash run.sh --vlm-source hf                      # 强制从 HuggingFace 下载 VLM
bash run.sh --help                               # 查看全部参数
```

### run.sh 参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--dataset NAME` | `STL10` | 数据集：`STL10` / `location` |
| `--model NAME` | `resnet` | 模型：`resnet` / `nn` |
| `--rounds N` | `200` | 联邦训练总轮数 |
| `--clients N` | `5` | 客户端总数 |
| `--participant N` | `5` | 每轮参与训练的客户端数 |
| `--epochs N` | `2` | 客户端本地训练 epoch 数 |
| `--lr F` | `0.005` | 学习率 |
| `--batch-size N` | `64` | 批大小 |
| `--quick` | 关闭 | 快速模式，等价于 `--rounds 10` |
| `--skip-data` | 关闭 | 跳过数据集下载 |
| `--skip-vlm` | 关闭 | 跳过 VLM 权重下载 |
| `--vlm-source S` | `auto` | VLM 下载源：`auto` / `modelscope` / `hf` |
| `--vlm-dir DIR` | `./vlm_2b` | VLM 权重目录 |
| `--install` | 关闭 | 运行前先安装 `requirements.txt` |

## 4. 手动分步运行

### 4.1 准备数据集

```bash
# STL10：脚本会自动从官网下载并解压到 ./datas/stl10
bash run.sh --dataset STL10 --skip-vlm --rounds 1 --quick   # 仅触发下载

# location (Bangkok)：下载 tgz 并解压为 ./datas/location_offical/bangkok.csv
wget https://raw.githubusercontent.com/privacytrustlab/datasets/master/dataset_location.tgz
tar -xzf dataset_location.tgz -C ./datas/location_offical/
mv ./datas/location_offical/bangkok ./datas/location_offical/bangkok.csv
```

### 4.2 准备 VLM 权重

```bash
python -c "from modelscope import snapshot_download; \
snapshot_download('Qwen/Qwen3-VL-2B-Instruct', local_dir='./vlm_2b')"
# 或 HuggingFace
python -c "from huggingface_hub import snapshot_download; \
snapshot_download('Qwen/Qwen3-VL-2B-Instruct', local_dir='./vlm_2b')"
```

### 4.3 运行完整流程

```bash
python main.py \
    --dataset STL10 --model resnet \
    --client_num 5 --participant 5 --training_round 200 --epochs 2 \
    --data_process_flag True \
    --train_model True \
    --regenerate_plots \
    --method ours \
    --vlm_type qwen3_2b --vlm_path ./vlm_2b
```

各开关含义：

| 参数 | 含义 |
|---|---|
| `--data_process_flag True` | 下载/读取数据集并生成 `full.npz`（仅首次需要） |
| `--train_model True` | 执行联邦训练，保存每轮服务器模型与指定客户端模型 |
| `--regenerate_plots` | 根据逐轮 loss 生成 VLM 输入图片；不加则复用已有图片直接攻击 |
| `--method ours` | 使用本项目攻击方法（其它可选值见第 1 节） |
| `--vlm_type qwen3_2b` | 使用 Qwen3-VL-2B-Instruct |
| `--vlm_path` | VLM 权重目录，默认取 `./vlm_2b` |

### 4.4 只运行攻击（模型与图片已存在）

```bash
python test.py --dataset STL10 --model resnet --max_samples 1000
```

## 5. 项目结构

| 文件 | 说明 |
|---|---|
| `main.py` | 主入口：参数解析、数据准备、联邦训练、攻击调度 |
| `run.sh` | 一键运行脚本（环境检查 + 数据下载 + 权重下载 + 全流程） |
| `data_processing.py` | 数据集下载与预处理，生成 `datas/{dataset}/full.npz` |
| `Data.py` | 数据集封装、客户端划分（uniform / dirichlet） |
| `train.py` | 联邦学习训练主循环（聚合、保存中间轮次模型） |
| `ours.py` | 生成 VLM 输入：逐轮 loss 曲线图片与 loss 序列 |
| `fix.py` | 汇总 member / nonmember 的 loss 序列为 `metrics_history_selected.pkl` |
| `test.py` | 攻击主流程：VLM 打分 → 物理特征校准 → 自适应融合 → 评估与绘图 |
| `CSModels.py` | 模型工厂（resnet / nn）与数据相关参数（分辨率、类别数） |
| `normalModel.py` | 网络结构实现（ResNet-9-9-9、MLP） |
| `baseline_attack.py` | 7 个成员推理基线攻击实现 |
| `utils.py` | 通用工具：npz 读取、ROC/AUC 指标与绘图 |

## 6. 输出说明

| 路径 | 内容 |
|---|---|
| `log_file/{model}/{dataset}/` | 训练与攻击日志 |
| `datas/{dataset}/full.npz` | 预处理后的完整数据集 |
| `datas/{dataset}/{model}/{data_split}/` | 按客户端划分后的 `train_non_iid.npz` / `test_non_iid.npz` |
| `models_main/{model}/{dataset}/server_model/` | 每轮服务器模型 `server_{round}.pth` |
| `models_main/{model}/{dataset}/client_model/` | 参与攻击的客户端模型 `client_{idx}_{round}.pth` |
| `plot/vlm_data/{model}/{dataset}/member|nonmember/` | VLM 输入图片（loss 曲线） |
| `plot/vlm_data/{model}/{dataset}/metrics_history_selected.pkl` | 每个样本的逐轮 loss 序列 |
| `reports_lira_lite/audit_details.json` | 每个样本的最终分数、VLM 分数与物理特征 |
| `reports_lira_lite/roc_final_loglog.png` | 对数坐标 ROC 曲线 |
| `reports_lira_lite/final_score_distribution.png` | 最终融合分数分布 |
| `reports_lira_lite/vlm_score_distribution_p1.png` | Phase 1 纯 VLM 分数分布 |
| `reports_lira_lite/2d_decision_space_plot.png` | VLM 分数与物理分数构成的二维决策空间 |
| `reports_lira_lite/hypothesis_validation_plots.png` | 物理特征假设验证图 |

## 7. 常见问题

**Q: 显存不足（CUDA out of memory）？**
VLM 采用 `device_map="auto"` 加载，会自动选择空闲显存最多的 GPU。若显存紧张，可先关闭其它占用 GPU 的进程；2B 模型在 8 GB 显存下可运行。

**Q: 想指定使用哪张 GPU？**
`main.py` 启动时会自动选择空闲显存最多的显卡，也可以用 `CUDA_VISIBLE_DEVICES=1 bash run.sh` 指定。

**Q: VLM 下载失败或很慢？**
ModelScope 下载失败时脚本会自动尝试 HuggingFace；也可以手动下载权重后放进 `./vlm_2b`（目录内需包含 `config.json` 与 `model.safetensors`），再用 `--skip-vlm` 跳过下载。

**Q: 重新运行会不会覆盖之前的结果？**
会。数据划分（`datas/`）、模型（`models_main/`）、曲线图片（`plot/vlm_data/`）与报告（`reports_lira_lite/`）都是按 `{dataset}/{model}` 覆盖写入的，需要保留历史结果请先备份。

**Q: 训练太慢，想快速验证流程是否可用？**
使用 `bash run.sh --quick`（10 轮训练）；模型收敛程度会影响攻击效果，正式实验请使用默认的 200 轮。
