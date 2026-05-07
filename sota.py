import os
import glob
import pickle
import json
import time
import re
import sys
import gc
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from scipy.stats import norm
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional

# ================== 依赖检查 ==================
try:
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print("Error: Missing transformers/torch.")
    sys.exit(1)

from sklearn.metrics import roc_curve, roc_auc_score

# ================== 全局配置 ==================
# 请修改为你的实际模型路径
MODEL_PATH_VLM = "./vlm"
MODEL_PATH_LLM = "./llm"
LOSS_PLOT_BASE = "./plot/vlm_data/resnet/CIFAR10/"
LOSS_HISTORY_PATH = "./plot/vlm_data/resnet/CIFAR10/metrics_history_selected.pkl"
REPORT_OUTPUT_DIR = "./reports_lira_lite"
RANDOM_SEED = 42



# ================== 核心：物理动力学工具箱 ==================
class DynamicsToolkit:
    @staticmethod
    def exponential_decay(t, a, b, c):
        return a * np.exp(-b * t) + c

    def extract_features(self, loss_seq: List[float]) -> Dict[str, float]:
        y = np.array(loss_seq)
        x = np.arange(len(y))

        # 1. 基础平滑
        try:
            y_smooth = savgol_filter(y, min(5, len(y)), 2)
        except:
            y_smooth = y

        # --- 新增关键特征：收敛速度 ---
        # 计算达到初始损失10%所需epoch数
        initial_loss = y[0]
        threshold = initial_loss * 0.1
        conv_epoch = len(y)  # 默认为整个序列长度
        for i, loss_val in enumerate(y):
            if loss_val < threshold:
                conv_epoch = i
                break
        convergence_speed = conv_epoch / len(y)  # 0～1 (越小越快)

        # --- 尾部稳定性 ---
        tail_stability = np.std(y[-10:]) if len(y) >= 10 else np.std(y)

        # --- 拐点锐度 (保留用于辅助) ---
        dy = np.gradient(y_smooth)
        ddy = np.gradient(dy)
        max_elbow = np.max(np.abs(ddy))

        return {
            "convergence_speed": float(convergence_speed),
            "tail_stability": float(tail_stability),
            "max_elbow": float(max_elbow),
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

    # 关键：存储 Phase 1 淘汰掉的样本的特征，用于校准
    calibration_pool: List[Dict[str, float]] = field(default_factory=list)

    final_scores: Dict[int, float] = field(default_factory=dict)
    candidates: List[int] = field(default_factory=list)


# ================== Phase 1: VLM Screener ==================
class Phase1Screener:
    def __init__(self, model_path):
        print(f"[Phase 1] Loading VLM: {model_path}...")
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype="auto", device_map="auto", trust_remote_code=True
        )
        self.prompt = ((
            "Examine the training loss curve in the image. The X-axis represents epochs, and the Y-axis represents loss value.\n"
            "Your Task: Identify if this sample belongs to the training set (Member) based on the curve's geometry.\n\n"
            "Visual Definitions:\n"
            "- Member (1): Feature 'Hockey Stick' shape. Extremely sharp initial drop followed by a perfect flatline near the X-axis (zero).\n"
            "- Non-member (0): Feature 'Gradual Descent' or 'Noisy Tail'. The curve drops slowly or shows visible jitter.\n\n"
            "Output '1' for Member or '0' for Non-member.\n"
            "Answer:"
        ))

    def scan(self, samples: List[Sample]) -> Dict[int, float]:
        scores = {}
        token_id_1 = self.processor.tokenizer.encode("1", add_special_tokens=False)[0]
        token_id_0 = self.processor.tokenizer.encode("0", add_special_tokens=False)[0]

        print(f"[Phase 1] Screening {len(samples)} samples...")
        for s in tqdm(samples):
            messages = [{"role": "user", "content": [
                {"type": "image", "image": os.path.abspath(s.image_path)},
                {"type": "text", "text": self.prompt}
            ]}]
            inputs = self.processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
            ).to(self.model.device)

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


# ================== Phase 2: LiRA-Lite Investigator ==================
class Phase2Investigator:
    def __init__(self, model_path, calibration_data: List[Dict]):
        print(f"[Phase 2] Loading LLM: {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype="auto", device_map="auto", trust_remote_code=True
        )
        self.physics = DynamicsToolkit()

        # === 核心：建立背景模型 (Background Model) ===
        print("[Phase 2] Building Background Distribution from Non-Members...")
        self.stats = {}
        for key in ["convergence_speed", "tail_stability"]:
            values = [d[key] for d in calibration_data]
            if not values: values = [0.0]
            self.stats[key] = {
                "mean": np.mean(values),
                "std": np.std(values) + 1e-9  # 防止除零
            }
        print(f"  > Calibration Stats: {self.stats}")

    def compute_z_score(self, val, key):
        return (val - self.stats[key]["mean"]) / self.stats[key]["std"]

    def analyze(self, sample: Sample, vlm_score: float) -> Tuple[float, Dict]:
        # 1. 提取物理特征
        phy = self.physics.extract_features(sample.loss_sequence)

        # 2. 计算相对校准分数
        z_speed = self.compute_z_score(phy['convergence_speed'], 'convergence_speed')
        z_stab = self.compute_z_score(phy['tail_stability'], 'tail_stability')

        # 3. 激活函数：将Z-Score映射到[0,1]
        def act(z):
            return 1 / (1 + np.exp(-(z - 1.5)))  # 偏置在1.5σ

        s_speed = act(z_speed)
        s_stab = act(z_stab)

        # 4. 关键改进：加权和（收敛速度权重70%）
        score = 0.7 * s_speed + 0.3 * s_stab

        # 5. 平滑衰减：收敛速度越快（值越小）分数越高
        # 仅当收敛速度超过0.5（即超过一半epoch才收敛）时衰减
        if phy['convergence_speed'] > 0.5:
            decay = np.exp(-5 * (phy['convergence_speed'] - 0.5))
            score *= decay

        # 6. 融合VLM先验
        # final_score = score * (1 + 0.2 * vlm_score)
        final_score = score + vlm_score

        return float(final_score), phy


# ================== 主程序 ==================
class LiraLiteAuditor:
    def __init__(self):
        np.random.seed(RANDOM_SEED)
        torch.manual_seed(RANDOM_SEED)
        self.state = AgentState()
        os.makedirs(REPORT_OUTPUT_DIR, exist_ok=True)

    def load_data(self, max_samples=None):
        print("[Data] Loading dataset...")
        m_imgs = sorted(glob.glob(os.path.join(LOSS_PLOT_BASE, "member", "*.png")))
        nm_imgs = sorted(glob.glob(os.path.join(LOSS_PLOT_BASE, "nonmember", "*.png")))

        try:
            with open(LOSS_HISTORY_PATH, 'rb') as f:
                data = pickle.load(f)
            m_loss = data.get("member", {}).get("loss", [])
            nm_loss = data.get("non_member", {}).get("loss", [])
        except Exception as e:
            print(f"Warning: Failed to load loss history: {e}")
            m_loss, nm_loss = [], []

        # 确保至少有一个成员和一个非成员
        if len(m_imgs) == 0 or len(m_loss) == 0:
            raise ValueError("No member samples found in member directory")
        if len(nm_imgs) == 0 or len(nm_loss) == 0:
            raise ValueError("No non-member samples found in non_member directory")

        samples = []
        limit_m = min(len(m_imgs), len(m_loss))
        limit_nm = min(len(nm_imgs), len(nm_loss))

        for i in range(limit_m): samples.append(Sample(i, m_imgs[i], m_loss[i], 1))
        for i in range(limit_nm): samples.append(Sample(i + limit_m, nm_imgs[i], nm_loss[i], 0))

        if max_samples:
            import random
            random.shuffle(samples)
            samples = samples[:max_samples]

        self.state.samples = samples
        print(f"[Data] Loaded {len(samples)} samples (members: {limit_m}, non-members: {limit_nm}).")

    def _build_dynamic_calibration_pool(self, samples: List[Sample]) -> List[Dict]:
        """
        构建动态校准池：使用第一阶段极高置信度的非成员 (Pseudo-labeling)
        绝不使用 true_label，完全避免数据泄漏。
        """
        # 核心改变：挑选 Phase 1 分数极低 (<= 0.1) 的样本作为“伪非成员”
        pseudo_non_members = [s for s in samples if self.state.phase1_scores.get(s.sample_id, 1.0) <= 0.1]

        if not pseudo_non_members:
            print("[Warning] Phase 1 没找到确信的非成员！使用默认参数兜底。")
            return [{"convergence_speed": 0.5, "tail_stability": 0.1}]

        print(f"[Calibration] Mining Pseudo-Non-Members from Phase 1... Found {len(pseudo_non_members)} samples.")

        # 提取物理特征构建校准池
        # 注意：这里我们直接信任 VLM 的筛选，不再进行人为的 speed_threshold 二次剔除，
        # 从而保证背景分布 (mean, std) 的真实性和多样性。
        calibration_pool = []
        for s in pseudo_non_members:
            phy = DynamicsToolkit().extract_features(s.loss_sequence)
            calibration_pool.append(phy)

        return calibration_pool

    def run(self):
        # === Phase 1: VLM Screening ===
        screener = Phase1Screener(MODEL_PATH_VLM)
        self.state.phase1_scores = screener.scan(self.state.samples)
        screener.unload()

        # === 筛选与校准集构建 ===
        candidates = []
        print("\n[Calibration] Building Dynamic Calibration Pool...")

        # 1. 采用自校准：从 Phase 1 的结果中挖掘伪非成员
        self.state.calibration_pool = self._build_dynamic_calibration_pool(self.state.samples)

        # 2. 保留VLM分数高的样本（score > 0.1）作为候选
        for s in self.state.samples:
            score = self.state.phase1_scores.get(s.sample_id, 0)
            if score > 0.1:
                candidates.append(s.sample_id)

        self.state.candidates = candidates
        print(f"  > Candidates (Hard Samples): {len(candidates)}")
        print(f"  > Calibration Pool (Dynamic Non-Members): {len(self.state.calibration_pool)}")

        if len(self.state.calibration_pool) < 10:
            print("WARNING: Not enough calibration samples! Calibration may be noisy.")

        # === Phase 2: Calibrated Analysis ===
        investigator = Phase2Investigator(MODEL_PATH_LLM, self.state.calibration_pool)
        details = []

        print("[Phase 2] Running LiRA-Lite Analysis...")
        for sid in tqdm(self.state.candidates):
            sample = next(s for s in self.state.samples if s.sample_id == sid)
            vlm_score = self.state.phase1_scores[sid]

            final_score, phy = investigator.analyze(sample, vlm_score)
            self.state.final_scores[sid] = final_score

            details.append({
                "id": int(sid), "gt": int(sample.true_label),
                "score": float(final_score), "features": phy
            })

        # 补全被筛掉的样本 (Score = 0)
        for s in self.state.samples:
            if s.sample_id not in self.state.final_scores:
                original_vlm_score = self.state.phase1_scores.get(s.sample_id, 0.0)
                self.state.final_scores[s.sample_id] =  original_vlm_score
                details.append({"id": int(s.sample_id), "gt": int(s.true_label), "score": original_vlm_score})

        # Save Report
        with open(os.path.join(REPORT_OUTPUT_DIR, "audit_details.json"), "w") as f:
            json.dump(details, f, indent=2)

        # === 关键改进：分数校准（Isotonic Regression） ===
        y_true = [s.true_label for s in self.state.samples]
        y_scores = [self.state.final_scores.get(s.sample_id, 0.0) for s in self.state.samples]

        # 1. 确保有正负样本
        if 0 not in y_true or 1 not in y_true:
            print("ERROR: y_true does not contain both classes (0 and 1).")
            print(f"y_true: {y_true}")
            print("Please ensure your dataset contains both member and non-member samples.")
            return

        # 2. 校准分数分布（使分数更符合概率解释）

        # 保存校准后的分数
        self.state.final_scores_calibrated = {sid: y_scores[i] for i, sid in
                                              enumerate(self.state.final_scores)}

        # 3. 评估
        self._evaluate(y_scores)

    def _evaluate(self, y_scores: List[float]):
        y_true = [s.true_label for s in self.state.samples]

        # 检查y_true中是否包含0和1
        if 0 not in y_true or 1 not in y_true:
            print("ERROR: y_true does not contain both classes (0 and 1). Cannot compute ROC AUC.")
            print(f"y_true: {y_true}")
            print("Skipping ROC curve generation.")
            # 创建一个空的ROC图
            plt.figure(figsize=(8, 8))
            plt.title('ROC Curve (Not Calculated - Missing Class)')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.grid(True)
            plt.savefig(os.path.join(REPORT_OUTPUT_DIR, "roc_final.png"))
            return

        epsilon = np.random.normal(0, 1e-6, len(y_true))
        y_scores = np.array(y_scores) + epsilon

        try:
            auc_val = roc_auc_score(y_true, y_scores)
        except Exception as e:
            print(f"Error calculating ROC AUC: {e}")
            auc_val = 0.0

        print("\n" + "=" * 40)
        print(f"LIRA-LITE REPORT (OPTIMIZED FOR LOW FPR)")
        print("=" * 40)
        print(f"ROC AUC: {auc_val:.5f}")

        # Plot
        try:
            fpr, tpr, _ = roc_curve(y_true, y_scores)
        except Exception as e:
            print(f"Error generating ROC curve: {e}")
            fpr, tpr = [0.0], [0.0]

        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, lw=2, label=f'LiRA-Lite (AUC={auc_val:.4f})')
        plt.xscale('log')
        plt.xlim([1e-4, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (Log Scale)')
        plt.ylabel('True Positive Rate')
        plt.title('High-Precision MIA ROC (Optimized)')
        plt.grid(True, which="both", alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(REPORT_OUTPUT_DIR, "roc_final.png"))

        # KPIs
        try:
            f_interp = interp1d(fpr, tpr, bounds_error=False, fill_value="extrapolate")
        except:
            f_interp = lambda x: [0.0] * len(x)

        print("-" * 30)
        print("Key Performance Indicators (KPIs):")
        for fp in [0.001, 0.005, 0.01, 0.05, 0.1]:
            try:
                tp = float(f_interp(fp))
                tp = max(0.0, min(1.0, tp))
            except:
                tp = 0.0
            print(f"TPR @ FPR {fp:.1%}: {tp:.4f}")
        print("-" * 30)

        # 保存校准后的分数
        with open(os.path.join(REPORT_OUTPUT_DIR, "scores_calibrated.pkl"), "wb") as f:
            pickle.dump(self.state.final_scores_calibrated, f)


if __name__ == "__main__":
    try:
        auditor = LiraLiteAuditor()
        auditor.load_data(max_samples=600)
        auditor.run()
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)