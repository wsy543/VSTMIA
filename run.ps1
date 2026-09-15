#Requires -Version 5.1

param(
    [string]$Dataset = 'location',
    [string]$Model = 'nn',
    [int]$Rounds = 200,
    [int]$Clients = 5,
    [int]$Participant = 5,
    [int]$Epochs = 2,
    [string]$Lr = '0.01',
    [int]$BatchSize = 64,
    [int]$Seed = 123,
    [switch]$Quick,
    [switch]$SkipData,
    [switch]$SkipVlm,
    [switch]$NoDataProcess,
    [switch]$NoTrain,
    [switch]$NoPlots,
    [switch]$Install,
    [string]$VlmSource = 'auto',
    [string]$VlmDir = './vlm_2b',
    [switch]$Help
)

$ErrorActionPreference = 'Stop'
Set-Location -Path $PSScriptRoot

$Stl10Dir = './datas/stl10'
$LocationUrl = 'https://raw.githubusercontent.com/privacytrustlab/datasets/master/dataset_location.tgz'
$LocationOut = './datas/location_offical/bangkok.csv'
$VlmRepo = 'Qwen/Qwen3-VL-2B-Instruct'

function Show-Usage {
    Write-Host @'
One-click runner for the VLM-based membership inference pipeline (Windows / PowerShell)

Usage:
  run.bat [options]

Options:
  -Dataset NAME     dataset: location (default) or STL10
  -Model NAME       model: nn (default) or resnet
  -Rounds N         federated training rounds, default 200
  -Clients N        number of clients, default 5
  -Participant N    clients participating in each round, default 5
  -Epochs N         local epochs per client, default 2
  -Lr F             learning rate, default 0.01
  -BatchSize N      batch size, default 64
  -Quick            quick smoke test (10 training rounds)
  -SkipData         skip dataset download
  -SkipVlm          skip VLM checkpoint download
  -NoDataProcess    skip data pre-processing (reuse existing datas/{dataset}/full.npz)
  -NoTrain          skip federated training (reuse existing models_main checkpoints)
  -NoPlots          skip loss curve rendering (reuse existing plot/vlm_data images)
  -VlmSource S      VLM download source: auto (default) / modelscope / hf
  -VlmDir DIR       directory of the VLM checkpoint, default ./vlm_2b
  -Install          install dependencies from requirements.txt first
  -Help             show this help message

Examples:
  run.bat                              # full pipeline: location + nn
  run_stl10.bat                        # full pipeline: STL10 + resnet
  run_stl10.bat -Quick                 # quick run with 10 rounds
  run_location.bat -NoTrain -NoPlots   # attack only with existing artifacts
'@
}

function Write-Banner([string]$Text) {
    Write-Host ""
    Write-Host "=================================================================="
    Write-Host "  $Text"
    Write-Host "=================================================================="
}

if ($Help) {
    Show-Usage
    exit 0
}

if ($Quick) {
    $Rounds = 10
}

$Python = if ($env:PYTHON) { $env:PYTHON } else { 'python' }

function Invoke-PythonCode([string]$Code) {
    $Code | & $Python -
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

Write-Banner "Step 1/5  Checking Python environment"
Invoke-PythonCode @'
import sys
if sys.version_info < (3, 9):
    raise SystemExit("Python >= 3.9 is required, current version: " + sys.version.split()[0])
print("Python " + sys.version.split()[0] + "  OK")
try:
    import torch
    print("torch " + torch.__version__ + "  CUDA available: " + str(torch.cuda.is_available()))
except ImportError:
    print("[hint] torch not found, run: run.bat -Install")
'@

if ($Install) {
    Write-Banner "Installing dependencies (requirements.txt)"
    & $Python -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

Write-Banner "Step 2/5  Checking dependencies"
Invoke-PythonCode @'
import importlib
missing = []
for name in ["torch", "torchvision", "transformers", "numpy", "scipy", "sklearn",
             "pandas", "matplotlib", "seaborn", "tqdm", "PIL"]:
    try:
        importlib.import_module(name)
    except ImportError:
        missing.append(name)
if missing:
    raise SystemExit("Missing dependencies: " + ", ".join(missing) + "\nRun: run.bat -Install")
print("All dependencies are available")
'@

Write-Banner "Step 3/5  Preparing dataset ($Dataset)"
if ($SkipData) {
    Write-Host "Dataset download skipped"
}
elseif ($Dataset -eq 'STL10') {
    if (Test-Path (Join-Path $Stl10Dir 'stl10_binary')) {
        Write-Host "STL10 already available: $Stl10Dir"
    }
    else {
        Write-Host "Downloading STL10 (~2.6 GB from ai.stanford.edu) ..."
        $env:STL10_DIR = $Stl10Dir
        Invoke-PythonCode @'
import os
import torchvision
root = os.environ["STL10_DIR"]
torchvision.datasets.STL10(root=root, split="train", download=True)
print("STL10 ready: " + root)
'@
    }
}
elseif ($Dataset -eq 'location') {
    if (Test-Path $LocationOut) {
        Write-Host "location dataset already available: $LocationOut"
    }
    else {
        Write-Host "Downloading the location (Bangkok) dataset ..."
        $env:LOCATION_URL = $LocationUrl
        $env:LOCATION_RAW = $LocationOut
        Invoke-PythonCode @'
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
print("location dataset ready: " + target)
'@
    }
}
else {
    Write-Host "Unknown dataset: $Dataset (supported: STL10 / location)"
    exit 1
}

Write-Banner "Step 4/5  Preparing VLM checkpoint ($VlmRepo)"
if ($SkipVlm) {
    Write-Host "VLM download skipped"
}
else {
    $env:VLM_DIR = $VlmDir
    $env:VLM_SOURCE = $VlmSource
    $env:VLM_REPO = $VlmRepo
    Invoke-PythonCode @'
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
    print("VLM checkpoint already available: " + target)
    sys.exit(0)

os.makedirs(target, exist_ok=True)
downloaded = False

if source in ("auto", "modelscope"):
    try:
        from modelscope import snapshot_download
        print("Downloading " + repo + " from ModelScope ...")
        snapshot_download(repo, local_dir=target)
        downloaded = True
        print("ModelScope download finished")
    except Exception as exc:
        print("[warning] ModelScope download failed: " + str(exc))

if not downloaded and source in ("auto", "hf", "huggingface"):
    try:
        from huggingface_hub import snapshot_download
        print("Downloading " + repo + " from HuggingFace ...")
        snapshot_download(repo, local_dir=target)
        downloaded = True
        print("HuggingFace download finished")
    except Exception as exc:
        print("[warning] HuggingFace download failed: " + str(exc))

if not downloaded:
    print("[error] Failed to download the VLM checkpoint, please download it manually into " + target)
    sys.exit(1)
'@
}

Write-Banner "Step 5/5  Running the pipeline ($Dataset + $Model)"
Write-Host "rounds: $Rounds, clients: $Clients, participants per round: $Participant, local epochs: $Epochs"
Write-Host "VLM: qwen3_2b @ $VlmDir"
Write-Host ""

$MainArgs = @(
    'main.py'
    '--dataset', $Dataset
    '--model', $Model
    '--client_num', "$Clients"
    '--participant', "$Participant"
    '--training_round', "$Rounds"
    '--epochs', "$Epochs"
    '--lr', "$Lr"
    '--batch_size', "$BatchSize"
    '--random_seed', "$Seed"
    '--method', 'ours'
    '--vlm_type', 'qwen3_2b'
    '--vlm_path', "$VlmDir"
)

if (-not $NoDataProcess) {
    $MainArgs += @('--data_process_flag', 'True')
    Write-Host "stage: data pre-processing  ON"
}
else {
    Write-Host "stage: data pre-processing  SKIPPED (reuse existing datas/$Dataset/full.npz)"
}

if (-not $NoTrain) {
    $MainArgs += @('--train_model', 'True')
    Write-Host "stage: federated training   ON"
}
else {
    Write-Host "stage: federated training   SKIPPED (reuse existing models_main/$Model/$Dataset/)"
}

if (-not $NoPlots) {
    $MainArgs += @('--regenerate_plots')
    Write-Host "stage: loss curve rendering ON"
}
else {
    Write-Host "stage: loss curve rendering SKIPPED (reuse existing plot/vlm_data/$Model/$Dataset/)"
}

Write-Host "stage: VLM attack           ON (always)"
Write-Host ""

& $Python @MainArgs
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Banner "Finished"
Write-Host "Attack report directory: ./reports_lira_lite"
Write-Host "  - audit_details.json             per-sample final score and physics features"
Write-Host "Training logs: ./log_file/$Model/$Dataset/"
Write-Host "Model checkpoints: ./models_main/$Model/$Dataset/"
Write-Host "Loss curve images: ./plot/vlm_data/$Model/$Dataset/"
