#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

DATASET="STL10"
MODEL="resnet"
ROUNDS=200
CLIENTS=5
PARTICIPANT=5
EPOCHS=2
LR=""
BATCH_SIZE=64
SEED=123
QUICK=0
SKIP_DATA=0
SKIP_VLM=0
DO_INSTALL=0
VLM_SOURCE="auto"
VLM_DIR="./vlm_2b"
STL10_DIR="./datas/stl10"
LOCATION_URL="https://raw.githubusercontent.com/privacytrustlab/datasets/master/dataset_location.tgz"
LOCATION_RAW="./datas/location_offical/bangkok.csv"
VLM_REPO="Qwen/Qwen3-VL-2B-Instruct"

usage() {
    cat <<'EOF'
一键运行脚本

用法:
  bash run.sh [选项]

选项:
  --dataset NAME    数据集: STL10 (默认) 或 location
  --model NAME      模型: resnet (默认) 或 nn
  --rounds N        联邦训练轮数, 默认 200
  --clients N       客户端数量, 默认 5
  --participant N   每轮参与训练的客户端数量, 默认 5
  --epochs N        客户端本地训练轮数, 默认 2
  --lr F            学习率, 默认 0.005
  --batch-size N    批大小, 默认 64
  --quick           快速冒烟模式(10 轮训练)
  --skip-data       跳过数据集下载
  --skip-vlm        跳过 VLM 权重下载
  --vlm-source S    VLM 下载源: auto (默认) / modelscope / hf
  --vlm-dir DIR     VLM 权重保存目录, 默认 ./vlm_2b
  --install         先安装 requirements.txt 中的依赖
  -h, --help        显示帮助

示例:
  bash run.sh                                  # STL10 + resnet 完整流程
  bash run.sh --dataset location --model nn    # location + nn 完整流程
  bash run.sh --quick                          # 10 轮快速验证
  bash run.sh --install --quick                # 安装依赖并快速验证
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset) DATASET="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --rounds) ROUNDS="$2"; shift 2 ;;
        --clients) CLIENTS="$2"; shift 2 ;;
        --participant) PARTICIPANT="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --lr) LR="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --quick) QUICK=1; shift ;;
        --skip-data) SKIP_DATA=1; shift ;;
        --skip-vlm) SKIP_VLM=1; shift ;;
        --vlm-source) VLM_SOURCE="$2"; shift 2 ;;
        --vlm-dir) VLM_DIR="$2"; shift 2 ;;
        --install) DO_INSTALL=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "未知参数: $1"; usage; exit 1 ;;
    esac
done

if [[ "$QUICK" == "1" ]]; then
    ROUNDS=10
fi

if [[ -z "$LR" ]]; then
    LR="0.005"
fi

PYTHON="${PYTHON:-python}"

banner() {
    echo ""
    echo "=================================================================="
    echo "  $1"
    echo "=================================================================="
}

banner "步骤 1/5  检查 Python 环境"
"$PYTHON" - <<'PY'
import sys
if sys.version_info < (3, 9):
    raise SystemExit(f"需要 Python >= 3.9, 当前版本 {sys.version.split()[0]}")
print(f"Python {sys.version.split()[0]}  OK")
try:
    import torch
    print(f"torch {torch.__version__}  CUDA available: {torch.cuda.is_available()}")
except ImportError:
    print("[提示] 未检测到 torch, 请先执行: bash run.sh --install")
PY

if [[ "$DO_INSTALL" == "1" ]]; then
    banner "安装依赖 (requirements.txt)"
    "$PYTHON" -m pip install -r requirements.txt
fi

banner "步骤 2/5  检查依赖"
"$PYTHON" - <<'PY'
import importlib
missing = []
for name in ["torch", "torchvision", "transformers", "numpy", "scipy", "sklearn",
             "pandas", "matplotlib", "seaborn", "tqdm", "PIL"]:
    try:
        importlib.import_module(name)
    except ImportError:
        missing.append(name)
if missing:
    raise SystemExit("缺少依赖: " + ", ".join(missing) + "\n请执行: bash run.sh --install")
print("依赖检查通过")
PY

banner "步骤 3/5  准备数据集 (${DATASET})"
if [[ "$SKIP_DATA" == "1" ]]; then
    echo "已跳过数据集下载"
elif [[ "$DATASET" == "STL10" ]]; then
    if [[ -d "$STL10_DIR/stl10_binary" ]]; then
        echo "STL10 已存在: $STL10_DIR"
    else
        echo "开始下载 STL10 (约 2.6 GB, 官方源 ai.stanford.edu)..."
        "$PYTHON" - <<PY
import torchvision
torchvision.datasets.STL10(root="${STL10_DIR}", split="train", download=True)
print("STL10 下载完成: ${STL10_DIR}")
PY
    fi
elif [[ "$DATASET" == "location" ]]; then
    if [[ -f "$LOCATION_RAW" ]]; then
        echo "location 数据已存在: $LOCATION_RAW"
    else
        echo "开始下载 location (Bangkok) 数据集..."
        LOCATION_URL="$LOCATION_URL" LOCATION_RAW="$LOCATION_RAW" "$PYTHON" - <<'PY'
import os
import ssl
import tarfile
import urllib.request

url = os.environ["LOCATION_URL"]
target = os.environ["LOCATION_RAW"]
root = os.path.dirname(target)
os.makedirs(root, exist_ok=True)
tgz = os.path.join(root, "dataset_location.tgz")

try:
    urllib.request.urlretrieve(url, tgz)
except Exception:
    ctx = ssl._create_unverified_context()
    with urllib.request.urlopen(url, context=ctx) as resp, open(tgz, "wb") as out:
        out.write(resp.read())

with tarfile.open(tgz) as f:
    f.extractall(root)

for name in os.listdir(root):
    if name != "dataset_location.tgz" and not name.endswith(".csv"):
        os.replace(os.path.join(root, name), target)
        break
os.remove(tgz)
print(f"location 数据下载完成: {target}")
PY
    fi
else
    echo "未知数据集: $DATASET (可选: STL10 / location)"
    exit 1
fi

banner "步骤 4/5  准备 VLM 权重 (${VLM_REPO})"
if [[ "$SKIP_VLM" == "1" ]]; then
    echo "已跳过 VLM 下载"
else
    VLM_DIR="$VLM_DIR" VLM_SOURCE="$VLM_SOURCE" VLM_REPO="$VLM_REPO" "$PYTHON" - <<'PY'
import os
import sys

target = os.environ["VLM_DIR"]
repo = os.environ["VLM_REPO"]
source = os.environ["VLM_SOURCE"].lower()

has_config = os.path.isfile(os.path.join(target, "config.json"))
has_weight = os.path.isdir(target) and any(
    f.endswith(".safetensors") or f.endswith(".bin") for f in os.listdir(target)
)
if has_config and has_weight:
    print(f"VLM 权重已存在: {target}")
    sys.exit(0)

os.makedirs(target, exist_ok=True)
downloaded = False

if source in ("auto", "modelscope"):
    try:
        from modelscope import snapshot_download
        print(f"从 ModelScope 下载 {repo} ...")
        snapshot_download(repo, local_dir=target)
        downloaded = True
        print("ModelScope 下载完成")
    except Exception as exc:
        print(f"[警告] ModelScope 下载失败: {exc}")

if not downloaded and source in ("auto", "hf", "huggingface"):
    try:
        from huggingface_hub import snapshot_download
        print(f"从 HuggingFace 下载 {repo} ...")
        snapshot_download(repo, local_dir=target)
        downloaded = True
        print("HuggingFace 下载完成")
    except Exception as exc:
        print(f"[警告] HuggingFace 下载失败: {exc}")

if not downloaded:
    print("[错误] VLM 权重下载失败, 可手动下载后放入 " + target)
    sys.exit(1)
PY
fi

banner "步骤 5/5  运行完整流程 (${DATASET} + ${MODEL})"
echo "训练轮数: ${ROUNDS}, 客户端数: ${CLIENTS}, 每轮参与: ${PARTICIPANT}, 本地 epoch: ${EPOCHS}"
echo "VLM: qwen3_2b @ ${VLM_DIR}"
echo ""
echo "流程: 数据划分 -> 联邦训练 -> 生成 loss 曲线图片 -> VLM 成员推理攻击"
echo ""

"$PYTHON" main.py \
    --dataset "$DATASET" \
    --model "$MODEL" \
    --client_num "$CLIENTS" \
    --participant "$PARTICIPANT" \
    --training_round "$ROUNDS" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --batch_size "$BATCH_SIZE" \
    --random_seed "$SEED" \
    --data_process_flag True \
    --train_model True \
    --regenerate_plots \
    --method ours \
    --vlm_type qwen3_2b \
    --vlm_path "$VLM_DIR"

banner "运行结束"
echo "攻击报告目录: ./reports_lira_lite"
echo "  - audit_details.json          每个样本的最终分数与物理特征"
echo "  - roc_final_loglog.png        对数坐标 ROC 曲线"
echo "  - final_score_distribution.png 最终分数分布"
echo "  - vlm_score_distribution_p1.png VLM 分数分布"
echo "  - 2d_decision_space_plot.png  多模态决策空间"
echo "  - hypothesis_validation_plots.png 物理特征验证图"
echo "训练日志: ./log_file/${MODEL}/${DATASET}/"
echo "模型权重: ./models_main/${MODEL}/${DATASET}/"
echo "曲线图片: ./plot/vlm_data/${MODEL}/${DATASET}/"
