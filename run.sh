#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

DATASET="location"
MODEL="nn"
ROUNDS=200
CLIENTS=5
PARTICIPANT=5
EPOCHS=2
LR=0.01
BATCH_SIZE=64
SEED=123
QUICK=0
SKIP_DATA=0
SKIP_VLM=0
DO_INSTALL=0
DO_DATA_PROCESS=1
DO_TRAIN=1
DO_PLOTS=1
VLM_SOURCE="auto"
VLM_DIR="./vlm_2b"
STL10_DIR="./datas/stl10"
LOCATION_URL="https://raw.githubusercontent.com/privacytrustlab/datasets/master/dataset_location.tgz"
LOCATION_RAW="./datas/location_offical/bangkok.csv"
VLM_REPO="Qwen/Qwen3-VL-2B-Instruct"

usage() {
    cat <<'EOF'
One-click runner for the VLM-based membership inference pipeline

Usage:
  bash run.sh [options]

Options:
  --dataset NAME    dataset: location (default) or STL10
  --model NAME      model: nn (default) or resnet
  --rounds N        federated training rounds, default 200
  --clients N       number of clients, default 5
  --participant N   clients participating in each round, default 5
  --epochs N        local epochs per client, default 2
  --lr F            learning rate, default 0.01
  --batch-size N    batch size, default 64
  --quick           quick smoke test (10 training rounds)
  --skip-data       skip dataset download
  --skip-vlm        skip VLM checkpoint download
  --no-data-process skip data pre-processing (reuse existing datas/{dataset}/full.npz)
  --no-train        skip federated training (reuse existing models_main checkpoints)
  --no-plots        skip loss curve rendering (reuse existing plot/vlm_data images)
  --vlm-source S    VLM download source: auto (default) / modelscope / hf
  --vlm-dir DIR     directory of the VLM checkpoint, default ./vlm_2b
  --install         install dependencies from requirements.txt first
  -h, --help        show this help message

Examples:
  bash run.sh                                  # full pipeline: STL10 + resnet
  bash run.sh --dataset location --model nn    # full pipeline: location + nn
  bash run.sh --quick                          # quick run with 10 rounds
  bash run.sh --install --quick                # install dependencies then quick run
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
        --no-data-process) DO_DATA_PROCESS=0; shift ;;
        --no-train) DO_TRAIN=0; shift ;;
        --no-plots) DO_PLOTS=0; shift ;;
        --vlm-source) VLM_SOURCE="$2"; shift 2 ;;
        --vlm-dir) VLM_DIR="$2"; shift 2 ;;
        --install) DO_INSTALL=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

if [[ "$QUICK" == "1" ]]; then
    ROUNDS=10
fi

if [[ -z "$LR" ]]; then
    LR="0.01"
fi

PYTHON="${PYTHON:-python}"

banner() {
    echo ""
    echo "=================================================================="
    echo "  $1"
    echo "=================================================================="
}

banner "Step 1/5  Checking Python environment"
"$PYTHON" - <<'PY'
import sys
if sys.version_info < (3, 9):
    raise SystemExit(f"Python >= 3.9 is required, current version: {sys.version.split()[0]}")
print(f"Python {sys.version.split()[0]}  OK")
try:
    import torch
    print(f"torch {torch.__version__}  CUDA available: {torch.cuda.is_available()}")
except ImportError:
    print("[hint] torch not found, run: bash run.sh --install")
PY

if [[ "$DO_INSTALL" == "1" ]]; then
    banner "Installing dependencies (requirements.txt)"
    "$PYTHON" -m pip install -r requirements.txt
fi

banner "Step 2/5  Checking dependencies"
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
    raise SystemExit("Missing dependencies: " + ", ".join(missing) + "\nRun: bash run.sh --install")
print("All dependencies are available")
PY

banner "Step 3/5  Preparing dataset (${DATASET})"
if [[ "$SKIP_DATA" == "1" ]]; then
    echo "Dataset download skipped"
elif [[ "$DATASET" == "STL10" ]]; then
    if [[ -d "$STL10_DIR/stl10_binary" ]]; then
        echo "STL10 already available: $STL10_DIR"
    else
        echo "Downloading STL10 (~2.6 GB from ai.stanford.edu) ..."
        "$PYTHON" - <<PY
import torchvision
torchvision.datasets.STL10(root="${STL10_DIR}", split="train", download=True)
print("STL10 ready: ${STL10_DIR}")
PY
    fi
elif [[ "$DATASET" == "location" ]]; then
    if [[ -f "$LOCATION_RAW" ]]; then
        echo "location dataset already available: $LOCATION_RAW"
    else
        echo "Downloading the location (Bangkok) dataset ..."
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
print(f"location dataset ready: {target}")
PY
    fi
else
    echo "Unknown dataset: $DATASET (supported: STL10 / location)"
    exit 1
fi

banner "Step 4/5  Preparing VLM checkpoint (${VLM_REPO})"
if [[ "$SKIP_VLM" == "1" ]]; then
    echo "VLM download skipped"
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
    print(f"VLM checkpoint already available: {target}")
    sys.exit(0)

os.makedirs(target, exist_ok=True)
downloaded = False

if source in ("auto", "modelscope"):
    try:
        from modelscope import snapshot_download
        print(f"Downloading {repo} from ModelScope ...")
        snapshot_download(repo, local_dir=target)
        downloaded = True
        print("ModelScope download finished")
    except Exception as exc:
        print(f"[warning] ModelScope download failed: {exc}")

if not downloaded and source in ("auto", "hf", "huggingface"):
    try:
        from huggingface_hub import snapshot_download
        print(f"Downloading {repo} from HuggingFace ...")
        snapshot_download(repo, local_dir=target)
        downloaded = True
        print("HuggingFace download finished")
    except Exception as exc:
        print(f"[warning] HuggingFace download failed: {exc}")

if not downloaded:
    print("[error] Failed to download the VLM checkpoint, please download it manually into " + target)
    sys.exit(1)
PY
fi

banner "Step 5/5  Running the pipeline (${DATASET} + ${MODEL})"
echo "rounds: ${ROUNDS}, clients: ${CLIENTS}, participants per round: ${PARTICIPANT}, local epochs: ${EPOCHS}"
echo "VLM: qwen3_2b @ ${VLM_DIR}"
echo ""

MAIN_ARGS=(
    --dataset "$DATASET"
    --model "$MODEL"
    --client_num "$CLIENTS"
    --participant "$PARTICIPANT"
    --training_round "$ROUNDS"
    --epochs "$EPOCHS"
    --lr "$LR"
    --batch_size "$BATCH_SIZE"
    --random_seed "$SEED"
    --method ours
    --vlm_type qwen3_2b
    --vlm_path "$VLM_DIR"
)

if [[ "$DO_DATA_PROCESS" == "1" ]]; then
    MAIN_ARGS+=(--data_process_flag True)
    echo "stage: data pre-processing  ON"
else
    echo "stage: data pre-processing  SKIPPED (reuse existing datas/${DATASET}/full.npz)"
fi

if [[ "$DO_TRAIN" == "1" ]]; then
    MAIN_ARGS+=(--train_model True)
    echo "stage: federated training   ON"
else
    echo "stage: federated training   SKIPPED (reuse existing models_main/${MODEL}/${DATASET}/)"
fi

if [[ "$DO_PLOTS" == "1" ]]; then
    MAIN_ARGS+=(--regenerate_plots)
    echo "stage: loss curve rendering ON"
else
    echo "stage: loss curve rendering SKIPPED (reuse existing plot/vlm_data/${MODEL}/${DATASET}/)"
fi

echo "stage: VLM attack           ON (always)"
echo ""

"$PYTHON" main.py "${MAIN_ARGS[@]}"

banner "Finished"
echo "Attack report directory: ./reports_lira_lite"
echo "  - audit_details.json             per-sample final score and physics features"
echo "Training logs: ./log_file/${MODEL}/${DATASET}/"
echo "Model checkpoints: ./models_main/${MODEL}/${DATASET}/"
echo "Loss curve images: ./plot/vlm_data/${MODEL}/${DATASET}/"
