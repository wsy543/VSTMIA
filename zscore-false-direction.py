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
except ImportError:
    print("Error: Missing transformers/torch.")
    sys.exit(1)

from sklearn.metrics import roc_curve, roc_auc_score

# ================== 全局配置 ==================
# 请修改为你的实际模型路径



# ================== 核心：物理动力学工具箱 ==================
class DynamicsToolkit:
    def extract_features(self, loss_seq: List[float]) -> Dict[str, float]:
        y = np.array(loss_seq)

        # 1. 基础平滑
        try:
            y_smooth = savgol_filter(y, min(5, len(y)), 2)
        except:
            y_smooth = y

        # --- 收敛速度 ---
        # 计算达到初始损失10%所需epoch数
        initial_loss = y[0]
        threshold = initial_loss * 0.1
        conv_epoch = len(y)  # 默认为整个序列长度
        for i, loss_val in enumerate(y):
            if loss_val < threshold:
                conv_epoch = i
                break
        convergence_speed = conv_epoch / len(y)  # 0～1 (越小代表收敛越快)

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
    calib_samples: List[Sample] = field(default_factory=list)  # 【新增】：存放专门用于校准的非成员
    phase1_scores: Dict[int, float] = field(default_factory=dict)
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


# ================== Phase 2: Statistical Investigator ==================
class Phase2Investigator:
    def __init__(self, calibration_data: List[Dict]):
        print("[Phase 2] Building Background Distribution from Hold-out Non-Members...")
        self.physics = DynamicsToolkit()
        self.stats = {}

        for key in ["convergence_speed", "tail_stability"]:
            values = [d[key] for d in calibration_data]
            if not values: values = [0.0]
            self.stats[key] = {
                "mean": np.mean(values),
                "std": np.std(values) + 1e-9
            }
        print(f"  > Calibration Stats: {self.stats}")

    def compute_z_score(self, val, key):
        # 【核心黑科技】：倒置的 Z-Score (val - mean)
        # 针对极简样本（Easy Sample）的微观惩罚机制
        return (val - self.stats[key]["mean"]) / self.stats[key]["std"]
        # return (self.stats[key]["mean"]-val) / self.stats[key]["std"]

    def analyze(self, sample: Sample, vlm_score: float) -> Tuple[float, Dict]:
        phy = self.physics.extract_features(sample.loss_sequence)

        z_speed = self.compute_z_score(phy['convergence_speed'], 'convergence_speed')
        z_stab = self.compute_z_score(phy['tail_stability'], 'tail_stability')

        # 激活函数
        def act(z):
            # return z
            return 1 / (1 + np.exp(-(z - 1.5)))

        s_speed = act(z_speed)
        s_stab = act(z_stab)

        score = 0.7 * s_speed + 0.3 * s_stab

        # 宏观惩罚：收敛太慢的直接指数级衰减
        if phy['convergence_speed'] > 0.5:
            decay = np.exp(-5 * (phy['convergence_speed'] - 0.5))
            score *= decay

        # 神级融合：统计微观偏置 + VLM 宏观直觉
        final_score = score + vlm_score

        return float(final_score), phy


# ================== 主程序 ==================
class LiraLiteAuditor:
    def __init__(self):
        np.random.seed(RANDOM_SEED)
        torch.manual_seed(RANDOM_SEED)
        self.state = AgentState()
        os.makedirs(REPORT_OUTPUT_DIR, exist_ok=True)

    def load_data(self, max_samples=None, num_calib_nm=100):
        # 【修改点】：增加 num_calib_nm 参数，指定切分出多少非成员作校准
        print("[Data] Loading dataset with physical split for calibration...")
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

        if len(m_imgs) == 0 or len(m_loss) == 0:
            raise ValueError("No member samples found in member directory")
        if len(nm_imgs) == 0 or len(nm_loss) == 0:
            raise ValueError("No non-member samples found in non_member directory")

        limit_m = min(len(m_imgs), len(m_loss))
        limit_nm = min(len(nm_imgs), len(nm_loss))
        
        if limit_nm <= num_calib_nm:
            raise ValueError(f"Not enough non-members for calibration. Total available: {limit_nm}, Requested: {num_calib_nm}")

        # --- 核心切分逻辑 ---
        # 1. 专门切出一部分给校准集 (0 ~ num_calib_nm)
        calib_nm_imgs = nm_imgs[:num_calib_nm]
        calib_nm_loss = nm_loss[:num_calib_nm]
        
        # 2. 剩下的用于测试 (num_calib_nm ~ 结束)
        test_nm_imgs = nm_imgs[num_calib_nm:limit_nm]
        test_nm_loss = nm_loss[num_calib_nm:limit_nm]

        # 保存隔离的校准样本
        for i in range(num_calib_nm):
            self.state.calib_samples.append(
                Sample(sample_id=-1, image_path=calib_nm_imgs[i], loss_sequence=calib_nm_loss[i], true_label=0)
            )

        # 构建测试集（所有成员 + 剩下的非成员）
        samples = []
        for i in range(limit_m): 
            samples.append(Sample(i, m_imgs[i], m_loss[i], 1))
        for i in range(len(test_nm_imgs)): 
            samples.append(Sample(i + limit_m, test_nm_imgs[i], test_nm_loss[i], 0))

        if max_samples:
            import random
            random.shuffle(samples)
            samples = samples[:max_samples]

        self.state.samples = samples
        print(f"[Data] Loaded {len(samples)} test samples (Mixed). Isolated {num_calib_nm} true non-members for calibration.")

    def _build_static_calibration_pool(self) -> List[Dict]:
        # 【修改点】：直接使用之前物理切分好的 calib_samples 提取特征
        print(f"[Calibration] Building Static Calibration Pool from {len(self.state.calib_samples)} non-overlapping non-members...")
        calibration_pool = []
        toolkit = DynamicsToolkit()
        for s in self.state.calib_samples:
            phy = toolkit.extract_features(s.loss_sequence)
            calibration_pool.append(phy)
        return calibration_pool

    def run(self):
        # === Phase 1: VLM Screening ===
        screener = Phase1Screener(MODEL_PATH_VLM)
        self.state.phase1_scores = screener.scan(self.state.samples)
        screener.unload()

        # === 筛选与校准集构建 ===
        candidates = []
        
        # 【修改点】：直接调用新的静态构建方法，确保独立分布
        self.state.calibration_pool = self._build_static_calibration_pool()

        for s in self.state.samples:
            score = self.state.phase1_scores.get(s.sample_id, 0)
            if score > 0.1:
                candidates.append(s.sample_id)

        self.state.candidates = candidates
        print(f"  > Phase 2 Candidates (Hard Samples): {len(candidates)}")

        # === Phase 2: Calibrated Analysis ===
        investigator = Phase2Investigator(self.state.calibration_pool)
        details = []

        print("[Phase 2] Running Statistical Analysis...")
        for sid in tqdm(self.state.candidates):
            sample = next(s for s in self.state.samples if s.sample_id == sid)
            vlm_score = self.state.phase1_scores[sid]

            final_score, phy = investigator.analyze(sample, vlm_score)
            self.state.final_scores[sid] = final_score

            details.append({
                "id": int(sid), "gt": int(sample.true_label),
                "score": float(final_score), "features": phy,
                "vlm_score": float(vlm_score)  # 用于画图的先验分数
            })

        # 补全被筛掉的伪非成员样本
        for s in self.state.samples:
            if s.sample_id not in self.state.final_scores:
                original_vlm_score = self.state.phase1_scores.get(s.sample_id, 0.0)
                self.state.final_scores[s.sample_id] = original_vlm_score
                details.append({
                    "id": int(s.sample_id), "gt": int(s.true_label),
                    "score": original_vlm_score, "vlm_score": original_vlm_score
                })

        # 保存明细
        with open(os.path.join(REPORT_OUTPUT_DIR, "audit_details.json"), "w") as f:
            json.dump(details, f, indent=2)

        # ================= 生成假设验证图表 =================
        self._plot_hypothesis_validation(details)

        # ================= 提取最终分数，进行无泄露评估 =================
        y_scores = [self.state.final_scores.get(s.sample_id, 0.0) for s in self.state.samples]
        self._evaluate(y_scores)

    def _plot_hypothesis_validation(self, details: List[Dict]):
        """验证 'Easy Sample Trap' 猜想的统计画图函数"""
        print("\n" + "=" * 40)
        print("📊 GENERATING HYPOTHESIS VALIDATION PLOTS...")
        print("=" * 40)

        valid_details = [d for d in details if "features" in d]

        mem_speeds = [d["features"]["convergence_speed"] for d in valid_details if d["gt"] == 1]
        nm_speeds = [d["features"]["convergence_speed"] for d in valid_details if d["gt"] == 0]

        mem_stabs = [d["features"]["tail_stability"] for d in valid_details if d["gt"] == 1]
        nm_stabs = [d["features"]["tail_stability"] for d in valid_details if d["gt"] == 0]

        mem_vlms = [d["vlm_score"] for d in valid_details if d["gt"] == 1]
        nm_vlms = [d["vlm_score"] for d in valid_details if d["gt"] == 0]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 图 1：收敛速度
        axes[0].hist(nm_speeds, bins=30, alpha=0.6, color='blue', label='Non-Members', density=True)
        axes[0].hist(mem_speeds, bins=30, alpha=0.6, color='red', label='Members', density=True)
        axes[0].set_title('Convergence Speed Distribution\n(Smaller = Faster)')
        axes[0].set_xlabel('Convergence Speed (Ratio of Epochs)')
        axes[0].set_ylabel('Density')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].axvspan(0, 0.1, color='orange', alpha=0.2, label='Easy Sample Trap')

        # 图 2：尾部稳定性
        axes[1].hist(nm_stabs, bins=np.logspace(np.log10(1e-6), np.log10(1.0), 30), alpha=0.6, color='blue',
                     label='Non-Members', density=True)
        axes[1].hist(mem_stabs, bins=np.logspace(np.log10(1e-6), np.log10(1.0), 30), alpha=0.6, color='red',
                     label='Members', density=True)
        axes[1].set_xscale('log')
        axes[1].set_title('Tail Stability Distribution\n(Smaller = More Stable)')
        axes[1].set_xlabel('Tail Variance (Log Scale)')
        axes[1].set_ylabel('Density')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # 图 3：散点图 (寻找内鬼)
        axes[2].scatter(nm_speeds, nm_vlms, alpha=0.6, color='blue', label='Non-Members', edgecolors='k')
        axes[2].scatter(mem_speeds, mem_vlms, alpha=0.6, color='red', marker='^', label='Members', edgecolors='k')
        axes[2].set_title('VLM Score vs. Convergence Speed\n(The Illusion of Memorization)')
        axes[2].set_xlabel('Convergence Speed (Smaller = Faster)')
        axes[2].set_ylabel('VLM Prior Score')

        rect = plt.Rectangle((-0.02, 0.8), 0.15, 0.22, fill=False, edgecolor='green', linewidth=2, linestyle='--')
        axes[2].add_patch(rect)
        axes[2].text(0.15, 0.9, 'False Positives Zone', color='green', fontsize=10, fontweight='bold')

        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(REPORT_OUTPUT_DIR, "hypothesis_validation_plots.png")
        plt.savefig(plot_path, dpi=300)
        print(f"[Plot] Hypothesis validation plots saved to: {plot_path}")
        plt.close()

    def _evaluate(self, y_scores: List[float]):
        y_true = [s.true_label for s in self.state.samples]

        if 0 not in y_true or 1 not in y_true:
            print("ERROR: y_true does not contain both classes (0 and 1). Cannot compute ROC AUC.")
            return

        # 加入极小扰动打破评分平局（Tie-breaking），对 Low FPR 评估极其重要
        epsilon = np.random.normal(0, 1e-9, len(y_true))
        y_scores = np.array(y_scores) + epsilon

        auc_val = roc_auc_score(y_true, y_scores)

        print("\n" + "=" * 40)
        print(f"LIRA-LITE REPORT (BLIND EVALUATION - NO LEAKAGE)")
        print("=" * 40)
        print(f"ROC AUC: {auc_val:.5f}")

        fpr, tpr, _ = roc_curve(y_true, y_scores)

        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, lw=2, label=f'LiRA-Lite (AUC={auc_val:.4f})')
        plt.plot([1e-5, 1], [1e-5, 1], color='gray', linestyle='--', label='Random Guess')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim([1e-4, 1.0])
        plt.ylim([1e-4, 1.05])
        plt.xlabel('False Positive Rate (Log Scale)')
        plt.ylabel('True Positive Rate (Log Scale)')
        plt.title('High-Precision MIA ROC (Blind Evaluation)')
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


if __name__ == "__main__":
    try:
        auditor = LiraLiteAuditor()
        # 您可以在这里通过 num_calib_nm 控制拿多少非成员去作独立校准
        auditor.load_data(max_samples=600, num_calib_nm=100)
        auditor.run()
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)