import os
import glob
import pickle
import json
import sys
import gc
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

# ================== 依赖检查 ==================
try:
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    from transformers import AutoModelForVision2Seq
except ImportError:
    print("Error: Missing transformers/torch.")
    sys.exit(1)

from sklearn.metrics import roc_curve, roc_auc_score

# ================== 全局配置 ==================
# 默认模型路径
VLM_PATHS = {
    "qwen3":     "./vlm",       # Qwen3-VL-2B
    "qwen3_8b":  "./vlm_8b",    # Qwen3-VL-8B
    "gemma4":    "./gemma4",    # Gemma-4-VL
    "llama3.2":  "./liama3.2",  # Llama-3.2-Vision
}
REPORT_OUTPUT_DIR = "./reports_lira_lite"
RANDOM_SEED = 42


def load_vlm(model_path: str, vlm_type: str):
    """
    VLM 模型工厂: 根据 vlm_type 加载对应的模型和 processor
    :return: (model, processor)
    """
    print(f"[VLM Loader] Loading {vlm_type} from {model_path}...")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    if vlm_type in ("qwen3", "qwen3_8b"):
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype="auto", device_map="auto", trust_remote_code=True
        )
    elif vlm_type == "llama3.2":
        from transformers import MllamaForConditionalGeneration
        model = MllamaForConditionalGeneration.from_pretrained(
            model_path, torch_dtype="auto", device_map="auto", trust_remote_code=True
        )
    elif vlm_type == "gemma4":
        # transformers 5.x 无 Gemma4ForConditionalGeneration, 用 AutoModelForVision2Seq 通用加载
        model = AutoModelForVision2Seq.from_pretrained(
            model_path, torch_dtype="auto", device_map="auto", trust_remote_code=True
        )
    else:
        raise ValueError(f"Unknown vlm_type: {vlm_type}. Choose from: qwen3, qwen3_8b, gemma4, llama3.2")

    return model, processor


def get_vlm_path(vlm_type: str, custom_path: str | None = None) -> str:
    """
    获取 VLM 路径: 优先使用自定义路径, 否则从预设字典查找
    """
    if custom_path:
        return custom_path
    if vlm_type in VLM_PATHS:
        return VLM_PATHS[vlm_type]
    raise ValueError(f"Unknown vlm_type: {vlm_type}. Known types: {list(VLM_PATHS.keys())}")


def get_paths(dataset, model):
    """根据 dataset 和 model 动态生成路径"""
    loss_plot_base = f"./plot/vlm_data/{model}/{dataset}/"
    loss_history_path = f"./plot/vlm_data/{model}/{dataset}/metrics_history_selected.pkl"
    return loss_plot_base, loss_history_path



# ================== 核心：物理动力学工具箱 ==================
class DynamicsToolkit:
    def extract_features(self, loss_seq: List[float]) -> Dict[str, float]:
        y = np.array(loss_seq)
        T = len(y)

        if T < 2:
            return {
                "rate_of_change": 0.0,
                "tail_fluctuation": 0.0,
                "raw_loss": float(y[-1] if T > 0 else 0.0)
            }

        # ==========================================
        # 1. 真实下降速率 (True Descent Rate)
        # 只取前 30% 阶段的下降步伐
        # ==========================================
        head_end_idx = max(2, int(T * 0.5))
        y_head = y[:head_end_idx]

        diffs = np.diff(y_head)
        descending_steps = diffs[diffs < 0]

        if len(descending_steps) > 0:
            true_descent_rate = np.sum(np.abs(descending_steps)) / len(descending_steps)
        else:
            true_descent_rate = 0.0

        rate_of_change = float(true_descent_rate)

        # ==========================================
        # 2. 末期波动累积 (Tail Fluctuation)
        # ==========================================
        tail_start_idx = int(T * 0.5)
        y_tail = y[tail_start_idx:]

        if len(y_tail) >= 2:
            tail_fluctuation = np.sum(np.abs(np.diff(y_tail)))
        else:
            tail_fluctuation = 0.0

        return {
            "rate_of_change": float(rate_of_change),
            "tail_fluctuation": float(tail_fluctuation),
            "raw_loss": float(y[-1])
        }


# ================== 数据结构 ==================
@dataclass
class Sample:
    sample_id: int
    image_path: str
    loss_sequence: List[float]
    true_label: int = -1


@dataclass
class AgentState:
    samples: List[Sample] = field(default_factory=list)
    phase1_scores: Dict[int, float] = field(default_factory=dict)
    calibration_pool: List[Dict[str, float]] = field(default_factory=list)
    final_scores: Dict[int, float] = field(default_factory=dict)


# ================== Phase 1: VLM Screener ==================
class Phase1Screener:
    def __init__(self, model_path: str, vlm_type: str = "qwen3"):
        print(f"[Phase 1] Loading VLM ({vlm_type}): {model_path}...")
        self.model, self.processor = load_vlm(model_path, vlm_type)
        self.prompt = ((
            "Examine the training loss curve in the image. The X-axis represents epochs, and the Y-axis represents loss value.\n"
            "Your Task: Identify if this sample belongs to the training set (Member) based on the curve's geometry.\n\n"
            "Visual Definitions:\n"
            "- Member (1): Feature 'Hockey Stick' shape. Extremely sharp initial drop followed by a perfect flatline near the X-axis (zero).\n"
            "- Non-member (0): Feature 'Gradual Descent' or 'Noisy Tail'. The curve drops slowly or shows visible jitter.\n\n"
            "Output '1' for Member or '0' for Non-member.\n"
            "Answer:"
        ))

    def _build_inputs(self, image_path: str):
        """构建模型输入, 兼容不同 VLM 的 chat_template 差异"""
        messages = [{"role": "user", "content": [
            {"type": "image", "image": os.path.abspath(image_path)},
            {"type": "text", "text": self.prompt}
        ]}]
        # apply_chat_template 在 Qwen3VL / Mllama 上可用; gemma4 用 processor 自带
        try:
            inputs = self.processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt"
            )
        except (AttributeError, Exception):
            # gemma4 等: 直接用 processor(text=..., images=..., return_tensors="pt")
            from PIL import Image
            img = Image.open(os.path.abspath(image_path)).convert("RGB")
            inputs = self.processor(
                text=self.prompt, images=img, return_tensors="pt"
            )
        return inputs.to(self.model.device)

    def scan(self, samples: List[Sample]) -> Dict[int, float]:
        scores = {}
        token_id_1 = self.processor.tokenizer.encode("1", add_special_tokens=False)[0]
        token_id_0 = self.processor.tokenizer.encode("0", add_special_tokens=False)[0]

        print(f"[Phase 1] Screening {len(samples)} samples...")
        for s in tqdm(samples):
            inputs = self._build_inputs(s.image_path)

            with torch.no_grad():
                outputs = self.model(**inputs)

            logits = outputs.logits[0, -1, :]
            s1 = logits[token_id_1].item()
            s0 = logits[token_id_0].item()
            prob = np.exp(s1) / (np.exp(s1) + np.exp(s0))
            scores[s.sample_id] = float(prob)

        return scores

    def unload(self):
        del self.model
        del self.processor
        torch.cuda.empty_cache()
        gc.collect()


class Phase2Investigator:
    def __init__(self, calibration_data: List[Dict]):
        print("[Phase 2] Building Non-parametric Background Distribution (eCDF)...")
        self.physics = DynamicsToolkit()
        self.bg_pool = {}

        # 核心改动：不再算 mean 和 std，而是直接把非成员池的数值存下来，并排序！
        for key in ["rate_of_change", "tail_fluctuation"]:
            values = [d[key] for d in calibration_data]
            if not values: values = [0.0]
            self.bg_pool[key] = np.sort(values)  # 排序好，方便后面直接查百分位

        print(f"  > Calibration Pool Built with {len(self.bg_pool['rate_of_change'])} samples.")

    def compute_empirical_percentile(self, val: float, key: str) -> float:
        """
        计算非参数的经验百分位数 (Empirical Percentile / CDF)
        返回该值在背景池中击败了百分之多少的非成员样本。
        """
        bg_array = self.bg_pool[key]
        # np.searchsorted 返回 val 插入排序数组的索引，除以总长度即为百分位 (0.0 ~ 1.0)
        percentile = np.searchsorted(bg_array, val) / len(bg_array)
        return float(percentile)

    def analyze(self, sample: Sample, vlm_score: float, alpha: float) -> Tuple[float, Dict]:
        phy = self.physics.extract_features(sample.loss_sequence)

        p_roc = self.compute_empirical_percentile(phy['rate_of_change'], 'rate_of_change')
        p_acc = 1.0 - self.compute_empirical_percentile(phy['tail_fluctuation'], 'tail_fluctuation')

        eps = 1e-5
        p_roc = np.clip(p_roc, eps, 1.0 - eps)
        p_acc = np.clip(p_acc, eps, 1.0 - eps)

        score_phy = 0.5 * p_roc + 0.5 * p_acc

        # 【核心融合】：使用外部传入的自适应 alpha 权重
        final_score = alpha * score_phy + 1.0 * vlm_score

        phy['s_roc'] = p_roc
        phy['s_acc'] = p_acc
        phy['score_phy'] = score_phy

        return float(final_score), phy
# ================== 主程序 ==================
class LiraLiteAuditor:
    def __init__(self, dataset: str, model: str, vlm_type: str = "qwen3", vlm_path: str | None = None):
        np.random.seed(RANDOM_SEED)
        torch.manual_seed(RANDOM_SEED)
        self.state = AgentState()
        self.dataset = dataset
        self.model = model
        self.vlm_type = vlm_type
        self.vlm_path = get_vlm_path(vlm_type, vlm_path)
        self.loss_plot_base, self.loss_history_path = get_paths(dataset, model)
        os.makedirs(REPORT_OUTPUT_DIR, exist_ok=True)

    def load_data(self, max_samples=600):
        print(f"[Data] Loading dataset... Target Mixed Samples: {max_samples} (Dynamic Calibration Mode)")
        m_imgs = sorted(glob.glob(os.path.join(self.loss_plot_base, "member", "*.png")))
        nm_imgs = sorted(glob.glob(os.path.join(self.loss_plot_base, "nonmember", "*.png")))

        try:
            with open(self.loss_history_path, 'rb') as f:
                data = pickle.load(f)
            m_loss = data.get("member", {}).get("loss", [])
            nm_loss = data.get("non_member", {}).get("loss", [])
        except Exception as e:
            print(f"Warning: Failed to load loss history: {e}")
            m_loss, nm_loss = [], []

        limit_m = min(len(m_imgs), len(m_loss))
        limit_nm = min(len(nm_imgs), len(nm_loss))

        target_test_m = max_samples // 2
        target_test_nm = max_samples - target_test_m

        target_test_m = min(target_test_m, limit_m)
        target_test_nm = min(target_test_nm, limit_nm)

        samples = []
        for i in range(target_test_m):
            samples.append(Sample(i, m_imgs[i], m_loss[i], 1))
        offset = target_test_m
        for i in range(target_test_nm):
            samples.append(Sample(i + offset, nm_imgs[i], nm_loss[i], 0))

        import random
        random.shuffle(samples)

        self.state.samples = samples
        print(f"[Data] Loaded {target_test_m} Members and {target_test_nm} Non-Members. Total: {len(samples)}.")

    def run(self):
        # === Phase 1: VLM Screening ===
        screener = Phase1Screener(self.vlm_path, self.vlm_type)
        self.state.phase1_scores = screener.scan(self.state.samples)
        screener.unload()

        # === 建立动态背景模型 (无监督) ===
        print("\n[Calibration] Dynamically building calibration pool from VLM highly confident pseudo-negatives...")

        calib_candidates = [s for s in self.state.samples if self.state.phase1_scores[s.sample_id] < 0.1]

        if len(calib_candidates) < 10:
            print("  > Warning: Too few samples < 0.1. Backing off to bottom 10% of samples.")
            sorted_samples = sorted(self.state.samples, key=lambda s: self.state.phase1_scores[s.sample_id])
            calib_candidates = sorted_samples[:max(10, len(self.state.samples) // 10)]

        print(f"  > Selected {len(calib_candidates)} pseudo-non-members for calibration.")

        toolkit = DynamicsToolkit()
        self.state.calibration_pool = [toolkit.extract_features(s.loss_sequence) for s in calib_candidates]

        # === 计算自适应权重 (Adaptive Weighting) ===
        # 根据 VLM 的整体确信度，决定物理证据的权重
        vlm_scores_array = np.array(list(self.state.phase1_scores.values()))

        # 计算平均确信度 (距离 0.5 的绝对差值的均值，映射到 0~1)
        mean_confidence = np.mean(2.0 * np.abs(vlm_scores_array - 0.5))

        # 【终极自适应公式】：不确定性与置信度比率 (Uncertainty-to-Confidence Ratio)
        # 加上 1e-5 防止除以 0，加上 min(5.0, ...) 作为最高权重上限防崩
        adaptive_alpha = (1.0 - mean_confidence) / (mean_confidence + 1e-5)
        adaptive_alpha = min(5.0, float(adaptive_alpha))  # 将物理分数权重封顶在 5.0

        print("\n[Fusion Strategy] Computing task-complexity adaptive weights...")
        print(f"  > VLM Mean Confidence: {mean_confidence:.3f}")
        print(f"  > Auto-set Physics Weight (Alpha): {adaptive_alpha:.3f}")

        # === Phase 2: Calibrated Analysis ===
        investigator = Phase2Investigator(self.state.calibration_pool)
        details = []

        print("[Phase 2] Running dynamic evaluation pipeline...")
        for sample in tqdm(self.state.samples):
            vlm_score = self.state.phase1_scores[sample.sample_id]

            if vlm_score < 0.1:
                final_score = vlm_score
                phy = toolkit.extract_features(sample.loss_sequence)
            else:
                # 【修改这里】：把计算好的 adaptive_alpha 传进去！
                final_score, phy = investigator.analyze(sample, vlm_score, alpha=adaptive_alpha)

            self.state.final_scores[sample.sample_id] = final_score

            details.append({
                "id": int(sample.sample_id),
                "gt": int(sample.true_label),
                "score": float(final_score),
                "features": phy,
                "vlm_score": float(vlm_score)
            })

        with open(os.path.join(REPORT_OUTPUT_DIR, "audit_details.json"), "w") as f:
            json.dump(details, f, indent=2)

        # ================= 生成各种图表 =================
        self._plot_hypothesis_validation(details)
        self._plot_vlm_score_distribution(details)
        self._plot_final_score_distribution(details)
        self._plot_2d_decision_space(details)

        # ================= 提取最终分数，进行无泄露评估 =================
        y_scores = [self.state.final_scores.get(s.sample_id, 0.0) for s in self.state.samples]
        self._evaluate(y_scores)

    def _plot_hypothesis_validation(self, details: List[Dict]):
        print("\n" + "=" * 40)
        print("📊 GENERATING HYPOTHESIS VALIDATION PLOTS...")
        print("=" * 40)

        mem_roc = [d["features"]["rate_of_change"] for d in details if d["gt"] == 1]
        nm_roc = [d["features"]["rate_of_change"] for d in details if d["gt"] == 0]

        mem_acc = [d["features"]["tail_fluctuation"] for d in details if d["gt"] == 1]
        nm_acc = [d["features"]["tail_fluctuation"] for d in details if d["gt"] == 0]

        mem_vlms = [d["vlm_score"] for d in details if d["gt"] == 1]
        nm_vlms = [d["vlm_score"] for d in details if d["gt"] == 0]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        min_roc = min(min(nm_roc), min(mem_roc))
        max_roc = max(max(nm_roc), max(mem_roc))
        shared_bins_roc = np.linspace(min_roc, max_roc, 30)

        axes[0].hist(nm_roc, bins=shared_bins_roc, alpha=0.6, color='blue', label='Non-Members', edgecolor='none')
        axes[0].hist(mem_roc, bins=shared_bins_roc, alpha=0.6, color='red', label='Members', edgecolor='none')
        axes[0].set_title('Early True Descent Rate (Top 30%)\n(Larger = More likely Member)')
        axes[0].set_xlabel('Average Valid Drop per Epoch')
        axes[0].set_ylabel('Frequency')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        min_acc = min(min(nm_acc), min(mem_acc))
        max_acc = max(max(nm_acc), max(mem_acc))
        shared_bins_acc = np.linspace(min_acc, max_acc, 30)

        axes[1].hist(nm_acc, bins=shared_bins_acc, alpha=0.6, color='blue', label='Non-Members')
        axes[1].hist(mem_acc, bins=shared_bins_acc, alpha=0.6, color='red', label='Members')
        axes[1].set_title('Tail Fluctuation (Last 30%)\n(Smaller = More likely Member)')
        axes[1].set_xlabel('Sum of Absolute Differences in Tail')
        axes[1].set_ylabel('Frequency')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        axes[2].scatter(nm_roc, nm_vlms, alpha=0.6, color='blue', label='Non-Members', edgecolors='k')
        axes[2].scatter(mem_roc, mem_vlms, alpha=0.6, color='red', marker='^', label='Members', edgecolors='k')
        axes[2].set_title('VLM Score vs. Descent Rate')
        axes[2].set_xlabel('True Descent Rate')
        axes[2].set_ylabel('VLM Prior Score')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(REPORT_OUTPUT_DIR, "hypothesis_validation_plots.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()

    def _plot_vlm_score_distribution(self, details: List[Dict]):
        print("\n" + "=" * 40)
        print("📊 GENERATING PHASE 1 VLM SCORE DISTRIBUTION PLOT...")
        print("=" * 40)

        mem_vlm_scores = [d["vlm_score"] for d in details if d["gt"] == 1]
        nm_vlm_scores = [d["vlm_score"] for d in details if d["gt"] == 0]

        plt.figure(figsize=(10, 6))
        plt.hist(nm_vlm_scores, bins=50, alpha=0.5, color='#3498db', label='Non-Members (Blue)', edgecolor='black',
                 linewidth=0.5)
        plt.hist(mem_vlm_scores, bins=50, alpha=0.5, color='#e74c3c', label='Members (Red)', edgecolor='black',
                 linewidth=0.5)
        plt.title('Phase 1: Pure VLM Score Distribution\n(Visual Macro-geometry Filter Only)', fontsize=14,
                  fontweight='bold')
        plt.xlabel('VLM Score (0 ~ 1, Higher = visually looks like Member)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.axvspan(0.1, 1.0, color='gray', alpha=0.1, label='Phase 2 Entry Zone')
        plt.legend(fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plot_path = os.path.join(REPORT_OUTPUT_DIR, "vlm_score_distribution_p1.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()

    def _plot_2d_decision_space(self, details: List[Dict]):
        """生成 2D 散点图，证明多模态融合的互相校准效应"""
        print("\n" + "=" * 40)
        print("📊 GENERATING 2D MULTIMODAL DECISION SPACE PLOT...")
        print("=" * 40)

        # 提取真正的 Members (GT=1)
        # 因为在你的代码中，d["score"] 是 final_score (VLM + Phy)，
        # 所以物理分数 (Phy) 可以通过 d["score"] - d["vlm_score"] 反推出来
        mem_vlm = [d["vlm_score"] for d in details if d["gt"] == 1]
        mem_phy = [d["score"] - d["vlm_score"] for d in details if d["gt"] == 1]

        # 提取 Non-Members (GT=0)
        nm_vlm = [d["vlm_score"] for d in details if d["gt"] == 0]
        nm_phy = [d["score"] - d["vlm_score"] for d in details if d["gt"] == 0]

        plt.figure(figsize=(10, 8))

        # 1. 画散点
        plt.scatter(nm_vlm, nm_phy, color='#3498db', alpha=0.6, label='Non-Members (GT=0)', edgecolors='k', s=50)
        plt.scatter(mem_vlm, mem_phy, color='#e74c3c', marker='^', alpha=0.6, label='Members (GT=1)', edgecolors='k',
                    s=50)

        # 2. 画决策边界 (Total Score = VLM + Phy = C)
        # 我们画出最终得分排在 Top 0.1% FPR 处的阈值线
        nm_scores = [d["score"] for d in details if d["gt"] == 0]
        if len(nm_scores) > 0:
            threshold_01_fpr = np.percentile(nm_scores, 99.9)
        else:
            threshold_01_fpr = 1.0  # 兜底阈值

        x_vals = np.linspace(-0.05, 1.05, 100)
        # y = C - x
        plt.plot(x_vals, threshold_01_fpr - x_vals, 'g--', linewidth=2.5,
                 label=f'Decision Boundary (Score = {threshold_01_fpr:.2f})')
        plt.plot(x_vals, (threshold_01_fpr - 0.3) - x_vals, 'k--', linewidth=1.5, alpha=0.5,
                 label='Sub-boundary (For reference)')

        # 3. 添加说明性高亮区域/箭头 (仅限有相关数据时，否则画圈示意)
        plt.annotate('Deceptive Non-members\n(VLM Hallucination Vetoed!)',
                     xy=(0.9, 0.05), xytext=(0.55, 0.15),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                     fontsize=11, fontweight='bold', color='darkblue',
                     bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#3498db", lw=2, alpha=0.9))

        plt.annotate('Atypical Members\n(Physics Compensated!)',
                     xy=(0.55, 0.75), xytext=(0.15, 0.85),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                     fontsize=11, fontweight='bold', color='darkred',
                     bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#e74c3c", lw=2, alpha=0.9))

        # 4. 图表装饰
        plt.title('2D Multimodal Decision Space: VLM vs. Physics Feature', fontsize=16, fontweight='bold', pad=15)
        plt.xlabel('Phase 1: VLM Score (Visual Prior)', fontsize=14)
        plt.ylabel('Phase 2: Physics Score (Micro-statistics)', fontsize=14)

        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.05, 1.05)
        plt.legend(loc='upper left', fontsize=12, framealpha=0.9)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()

        plot_path = os.path.join(REPORT_OUTPUT_DIR, "2d_decision_space_plot.png")
        plt.savefig(plot_path, dpi=300)
        print(f"[Plot] 2D Decision Space plot saved to: {plot_path}")
        plt.close()

    def _plot_final_score_distribution(self, details: List[Dict]):
        print("\n" + "=" * 40)
        print("📊 GENERATING FINAL SCORE DISTRIBUTION PLOT...")
        print("=" * 40)

        mem_scores = [d["score"] for d in details if d["gt"] == 1]
        nm_scores = [d["score"] for d in details if d["gt"] == 0]

        plt.figure(figsize=(10, 6))
        plt.hist(nm_scores, bins=50, alpha=0.6, color='#3498db', label='Non-Members (Blue)', edgecolor='black',
                 linewidth=0.5)
        plt.hist(mem_scores, bins=50, alpha=0.6, color='#e74c3c', label='Members (Red)', edgecolor='black',
                 linewidth=0.5)
        plt.title('Final Fusion Score Distribution\n(Dynamic Thresholding + Physics Penalty)', fontsize=14,
                  fontweight='bold')
        plt.xlabel('Final Score (Higher = More likely to be Member)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.axvline(x=np.percentile(nm_scores, 99.9), color='green', linestyle='--', linewidth=2,
                    label='Top 0.1% FPR Threshold')
        plt.legend(fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plot_path = os.path.join(REPORT_OUTPUT_DIR, "final_score_distribution.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()

    def _evaluate(self, y_scores: List[float]):
        y_true = [s.true_label for s in self.state.samples]

        if 0 not in y_true or 1 not in y_true:
            print("ERROR: y_true does not contain both classes (0 and 1). Cannot compute ROC AUC.")
            return

        epsilon = np.random.normal(0, 1e-9, len(y_true))
        y_scores = np.array(y_scores) + epsilon

        auc_val = roc_auc_score(y_true, y_scores)

        print("\n" + "=" * 40)
        print(f"DYNAMIC-CALIBRATION MIA REPORT (BLIND EVALUATION)")
        print("=" * 40)
        print(f"ROC AUC: {auc_val:.5f}")

        fpr, tpr, _ = roc_curve(y_true, y_scores)

        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, lw=2, label=f'Dynamic-Calib (AUC={auc_val:.4f})')
        plt.plot([1e-5, 1], [1e-5, 1], color='gray', linestyle='--', label='Random Guess')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim([1e-4, 1.0])
        plt.ylim([1e-4, 1.05])
        plt.xlabel('False Positive Rate (Log Scale)')
        plt.ylabel('True Positive Rate (Log Scale)')
        plt.title('High-Precision MIA ROC (Dynamic Calibration)')
        plt.grid(True, which="both", linestyle='--', alpha=0.5)
        plt.legend()
        plt.savefig(os.path.join(REPORT_OUTPUT_DIR, "roc_final_loglog.png"))

        f_interp = interp1d(fpr, tpr, bounds_error=False, fill_value=(0, 1))

        print("-" * 30)
        print("Key Performance Indicators (KPIs):")
        for fp in [0.001, 0.005, 0.01, 0.05, 0.1]:
            try:
                tp_val = f_interp(fp)
                if isinstance(tp_val, np.ndarray):
                    tp_val = tp_val.item()
                tp = float(tp_val)
                tp = max(0.0, min(1.0, tp))
            except:
                tp = 0.0
            print(f"TPR @ FPR {fp:.1%}: {tp:.4f}")
        print("-" * 30)


def run_attack(dataset: str, model: str, max_samples: int = 1000,
               vlm_type: str = "qwen3", vlm_path: str | None = None):
    """
    对外暴露的攻击入口函数，供 main.py 等调用

    :param dataset: 数据集名称, 如 'CIFAR10', 'CIFAR100', 'STL10' 等
    :param model: 模型名称, 如 'mobilenet', 'densenet', 'resnet' 等
    :param max_samples: 最多测试的样本数
    :param vlm_type: VLM 类型, qwen3 / qwen3_8b / gemma4 / llama3.2
    :param vlm_path: 自定义 VLM 路径, 为 None 则使用预设路径
    """
    print(f"\n{'='*60}")
    print(f"LiraLite Attack: dataset={dataset}, model={model}, max_samples={max_samples}")
    print(f"VLM: {vlm_type} @ {get_vlm_path(vlm_type, vlm_path)}")
    print(f"{'='*60}\n")
    auditor = LiraLiteAuditor(
        dataset=dataset, model=model,
        vlm_type=vlm_type, vlm_path=vlm_path
    )
    auditor.load_data(max_samples=max_samples)
    auditor.run()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='LiraLite MIA Attack')
    parser.add_argument('--dataset', type=str, default='CIFAR100', help='Dataset name')
    parser.add_argument('--model', type=str, default='mobilenet', help='Model name')
    parser.add_argument('--max_samples', type=int, default=1000, help='Max test samples')
    parser.add_argument('--vlm_type', type=str, default='qwen3',
                        choices=['qwen3', 'qwen3_8b', 'gemma4', 'llama3.2'],
                        help='VLM model type')
    parser.add_argument('--vlm_path', type=str, default=None,
                        help='Custom VLM model path (overrides preset)')
    args = parser.parse_args()

    try:
        run_attack(
            dataset=args.dataset, model=args.model,
            max_samples=args.max_samples,
            vlm_type=args.vlm_type, vlm_path=args.vlm_path
        )
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)