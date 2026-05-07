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
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional

# ================== 依赖检查 ==================
try:
    # Phase 1: VLM
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    # Phase 2: LLM (纯文本)
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print("错误: 缺少必要的依赖库。")
    print("请运行: pip install git+https://github.com/huggingface/transformers torch matplotlib scikit-learn scipy")
    sys.exit(1)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, roc_auc_score, \
    f1_score, precision_score, recall_score

# ================== 全局配置 ==================
# 模型路径 (请修改为你本地的实际路径)
MODEL_PATH = "E:\VLM-MIA\model\Qwen3-VL-4B-Instruct"  # Phase 1
MODEN_LLM_PATH = "E:\VLM-MIA\model\Qwen3-4B"  # Phase 2

# 输出路径配置
LOSS_PLOT_BASE = "./loss_plots_separate_1000"
REPORT_OUTPUT_DIR = "./reports_react_qwen3_final"
LOSS_HISTORY_PATH = "./metrics_history_selected.pkl"

# Phase 1 参数
SCREENER_CONF_THRESHOLD = 0  # 如果Phase 1非常确信，就不进Phase 2 (可选)
SOFTMAX_TEMPERATURE = 1
RANDOM_SEED = 42

# Phase 2 参数
REACT_MAX_STEPS = 3
# Qwen3-8B 生成参数
GEN_CONFIG_LLM = {
    "top_p": 0.8,
    "temperature": 0.7,
    "repetition_penalty": 1.05,
    "max_new_tokens": 512
}

# 标签定义
CLASS_LABELS = ["0", "1"]
CLASS_TO_ID = {label: int(label) for label in CLASS_LABELS}
ID_TO_CLASS = {int(label): label for label in CLASS_LABELS}


# ================== 数据结构 ==================
@dataclass
class Sample:
    sample_id: int
    image_path: str
    loss_sequence: List[float]
    true_label: int = -1  # 0=non-member, 1=member


@dataclass
class ScreenerResult:
    sample_id: int
    predicted_label: int
    predicted_label_str: str
    posterior_probs: Dict[str, float]
    member_prob: float
    max_posterior_prob: float
    raw_response: str
    prob_calculation_details: dict = field(default_factory=dict)


@dataclass
class ReActDecision:
    sample_id: int
    final_label: int
    inherited_member_prob: float
    trace_log: str


@dataclass
class AgentState:
    samples: List[Sample] = field(default_factory=list)
    screener_results: Dict[int, ScreenerResult] = field(default_factory=dict)
    easy_samples: List[int] = field(default_factory=list)
    hard_samples: List[int] = field(default_factory=list)
    react_decisions: Dict[int, ReActDecision] = field(default_factory=dict)

    # 最终结果存储
    final_predictions: Dict[int, int] = field(default_factory=dict)
    final_probabilities: Dict[int, float] = field(default_factory=dict)
    phase1_scores: Dict[int, float] = field(default_factory=dict)


# ================== Phase 2 工具箱 ==================
class MIAToolkit:
    """Agent调用的数值分析工具"""

    def get_basic_stats(self, loss_seq: List[float]) -> str:
        arr = np.array(loss_seq)
        if len(arr) == 0: return "Error: Empty sequence"
        return json.dumps({
            "min": round(float(np.min(arr)), 6),
            "max": round(float(np.max(arr)), 6),
            "final": round(float(arr[-1]), 6),
            "mean": round(float(np.mean(arr)), 6),
            "len": len(arr)
        })

    def analyze_stability(self, loss_seq: List[float]) -> str:
        arr = np.array(loss_seq)
        if len(arr) < 5: return "Error: Sequence too short"
        # 分析后 20% 的数据
        cutoff = max(1, int(len(arr) * 0.8))
        tail = arr[cutoff:]
        var_val = float(np.var(tail))
        mean_val = float(np.mean(tail))

        status = "Very Stable" if var_val < 1e-4 else ("Stable" if var_val < 1e-2 else "Unstable")
        return f"Tail Analysis (Last {len(tail)} pts): Mean={mean_val:.6f}, Variance={var_val:.8f}, Status={status}"

    def check_early_drop(self, loss_seq: List[float]) -> str:
        if len(loss_seq) < 5: return "Error: Sequence too short"
        # 计算前5个点的平均下降斜率
        slope = (loss_seq[0] - loss_seq[4]) / 4.0
        return f"Early Drop Rate (0->4): {slope:.4f} (Higher means faster convergence)"


# ================== Phase 1: 分类式后验概率计算 (VLM) ==================
class Phase1Screener:
    def __init__(self, model_path):
        print(f"[Phase 1] Loading VLM from {model_path}...")
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype="auto", device_map="auto", trust_remote_code=True
        )

        self.label_token_ids = {}
        print("[Init] Calculating Token IDs for Screener...")
        tokenizer = self.processor.tokenizer

        for label in CLASS_LABELS:
            candidates = set()
            prefixes = ["", " ", "\n", "Ġ"]
            for p in prefixes:
                txt = p + label
                ids = tokenizer.encode(txt, add_special_tokens=False)
                if ids: candidates.add(ids[-1])
            self.label_token_ids[label] = list(candidates)

        self.prompt = (
            "Examine the training loss curve in the image. The X-axis represents epochs, and the Y-axis represents loss value.\n"
            "Your Task: Identify if this sample belongs to the training set (Member) based on the curve's geometry.\n\n"
            "Visual Definitions:\n"
            "- Member (1): Feature 'Hockey Stick' shape. Extremely sharp initial drop followed by a perfect flatline near the X-axis (zero).\n"
            "- Non-member (0): Feature 'Gradual Descent' or 'Noisy Tail'. The curve drops slowly or shows visible jitter.\n\n"
            "Output '1' for Member or '0' for Non-member.\n"
            "Answer:"
        )

    def _get_classification_posterior(self, messages):
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_logits = outputs.logits[:, -1, :]
        label_logits = {}
        vocab_size = target_logits.shape[-1]

        for label in CLASS_LABELS:
            cands = [tid for tid in self.label_token_ids[label] if tid < vocab_size]
            if not cands:
                label_logits[label] = -float('inf')
            else:
                vals = target_logits[0, cands]
                label_logits[label] = torch.max(vals).item()

        logits_tensor = torch.tensor([label_logits[label] for label in CLASS_LABELS], device=self.model.device)
        scaled_logits = logits_tensor / SOFTMAX_TEMPERATURE
        posterior_probs = F.softmax(scaled_logits, dim=0).cpu().numpy()
        posterior_dict = {CLASS_LABELS[i]: float(posterior_probs[i]) for i in range(len(CLASS_LABELS))}

        max_prob_label = max(posterior_dict, key=posterior_dict.get)
        max_prob = posterior_dict[max_prob_label]
        pred_label = CLASS_TO_ID[max_prob_label]
        member_prob = posterior_dict.get("1", 0.0)

        return pred_label, posterior_dict, member_prob, max_prob, label_logits

    def scan(self, samples: List[Sample]) -> Dict[int, ScreenerResult]:
        results = {}
        for s in tqdm(samples, desc="Phase 1 (Screener)"):
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": f"{os.path.abspath(s.image_path)}"},
                    {"type": "text", "text": self.prompt}
                ]
            }]

            pred, probs, mem_prob, max_prob, raw_logits = self._get_classification_posterior(messages)

            results[s.sample_id] = ScreenerResult(
                sample_id=s.sample_id,
                predicted_label=pred,
                predicted_label_str=ID_TO_CLASS[pred],
                posterior_probs=probs,
                member_prob=mem_prob,
                max_posterior_prob=max_prob,
                raw_response="[Hidden]",
                prob_calculation_details={"logits": str(raw_logits)}
            )
        return results

    def unload(self):
        print("[Phase 1] Unloading VLM...")
        del self.model
        del self.processor
        torch.cuda.empty_cache()
        gc.collect()


# ================== Phase 2: ReAct Agent (LLM) ==================
class Phase2ReActAgent:
    def __init__(self, model_path):
        print(f"[Phase 2] Loading LLM from {model_path}...")
        # 使用 AutoClass 加载纯文本模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True
        )
        self.toolkit = MIAToolkit()
        self.tools_map = {
            "get_basic_stats": self.toolkit.get_basic_stats,
            "analyze_stability": self.toolkit.analyze_stability,
            "check_early_drop": self.toolkit.check_early_drop,
        }

    def _build_system_prompt(self):
        tool_desc = "\n".join([f"- {k}: {v.__doc__}" for k, v in self.tools_map.items()])
        return f"""You are an expert Membership Inference Attack (MIA) Investigator. 
The visual check was ambiguous. You must analyze the NUMERICAL loss data.

AVAILABLE TOOLS:
{tool_desc}

PROTOCOL:
1. Thought: Reason about the data or previous observations.
2. Action: Tool_Name (e.g., get_basic_stats) OR None.
3. Observation: [Wait for tool output]
... (Repeat until confident) ...
4. Final Answer: 0 (Non-Member) OR 1 (Member)

CRITERIA:
- MEMBER (1): Converges to < 0.1, very low variance in tail.
- NON-MEMBER (0): High loss (> 0.3), or unstable tail (spikes).
"""

    def investigate(self, sample: Sample, inherited_prob: float) -> ReActDecision:
        # 初始 Prompt
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user",
             "content": f"Loss Sequence: {sample.loss_sequence}\nBegin investigation."}
        ]

        full_trace = []
        final_label = 0  # 默认保守估计

        for step in range(REACT_MAX_STEPS):
            # 使用 tokenizer.apply_chat_template 构造输入
            text_input = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False  # 不需要 think 模式
            )

            inputs = self.tokenizer([text_input], return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    **GEN_CONFIG_LLM
                )

            # 解析输出：截取新生成的 token
            generated_ids_trimmed = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
            ]
            response_text = self.tokenizer.decode(generated_ids_trimmed[0], skip_special_tokens=True).strip()

            full_trace.append(f"Step {step + 1}: {response_text}")
            messages.append({"role": "assistant", "content": response_text})

            # === ReAct 逻辑解析 ===
            # 1. 检查 Final Answer
            if "Final Answer" in response_text:
                match = re.search(r"Final Answer:?\s*(?:is\s*)?([01])", response_text, re.IGNORECASE)
                if match:
                    final_label = int(match.group(1))
                    break

            # 2. 检查 Action
            action_match = re.search(r"Action:\s*[`']?(\w+)[`']?", response_text, re.IGNORECASE)

            if action_match:
                tool_name = action_match.group(1)
                if tool_name in self.tools_map:
                    try:
                        obs = self.tools_map[tool_name](sample.loss_sequence)
                    except Exception as e:
                        obs = f"Error: {str(e)}"
                else:
                    obs = f"Error: Tool '{tool_name}' unknown."

                obs_text = f"Observation: {obs}"
                full_trace.append(obs_text)
                messages.append({"role": "user", "content": obs_text})

            elif step == REACT_MAX_STEPS - 1:
                full_trace.append("Max steps reached. Forcing stop.")
            else:
                messages.append({"role": "user", "content": "Please continue to Action or Final Answer."})

        return ReActDecision(
            sample_id=sample.sample_id,
            final_label=final_label,
            inherited_member_prob=inherited_prob,
            trace_log="\n".join(full_trace)
        )


# ================== 绘图与评估 (保持不变) ==================
def ROC_AUC_Result_logshow(label_values, predict_values, output_dir):
    if len(np.unique(label_values)) < 2:
        print("[Plot Warning] Only one class present in data. Skipping ROC plot.")
        return

    plt.figure(figsize=(8, 6))

    # 1. 计算 AUC 和 ROC 数据
    auc_score = roc_auc_score(label_values, predict_values)
    print(f'[Metrics] ROC AUC Score: {auc_score:.4f}')

    fpr, tpr, thresholds = roc_curve(label_values, predict_values, pos_label=1)

    # 2. 绘制基础曲线 (Log-Log)
    plt.title(f'ROC Curve (AUC={auc_score:.4f})')
    plt.loglog(fpr, tpr, 'b', label='AUC=%0.4f' % auc_score)
    plt.legend(loc='lower right')

    plt.plot([0.001, 1], [0.001, 1], 'r--', alpha=0.5)
    plt.xlim([0.001, 1.0])
    plt.ylim([0.001, 1.0])
    plt.ylabel('TPR (True Positive Rate)')
    plt.xlabel('FPR (False Positive Rate)')
    plt.grid(True, which="both", ls="-", alpha=0.2)

    # 3. 插值计算
    ax = plt.gca()
    line = ax.lines[0]
    xdata = line.get_xdata()
    ydata = line.get_ydata()

    try:
        sort_idx = np.argsort(xdata)
        xdata_sorted = xdata[sort_idx]
        ydata_sorted = ydata[sort_idx]

        f = interp1d(xdata_sorted, ydata_sorted, kind='linear', bounds_error=False, fill_value="extrapolate")

        target_fprs = [0.001, 0.005, 0.01,0.05]
        colors = ['ro', 'go', 'mo']

        print("-" * 30)
        for i, fpr_val in enumerate(target_fprs):
            tpr_val = f(fpr_val)
            tpr_val = np.clip(tpr_val, 0.0, 1.0)
            print(f'TPR at {fpr_val:.3f} FPR is {tpr_val:.6f}')
            plt.plot([fpr_val], [tpr_val], colors[i])
            plt.text(fpr_val * 1.1, tpr_val, f"TPR={tpr_val:.3f}", fontsize=9)
        print("-" * 30)

    except Exception as e:
        print(f"[Plot Warning] Interpolation failed: {e}")

    save_path = os.path.join(output_dir, "roc_curve_log_refined.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Plot] Saved refined ROC to {save_path}")


# ================== 主管理器 ==================
class MIAAuditorAgent:
    def __init__(self):
        import random
        random.seed(RANDOM_SEED)
        torch.manual_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

        # 初始化时不加载模型，改为按需加载
        self.state = AgentState()

    def load_data(self, max_samples=None):
        print("\n[Data] Searching for samples...")
        m_imgs = sorted(glob.glob(os.path.join(LOSS_PLOT_BASE, "member", "loss", "*.png")))
        nm_imgs = sorted(glob.glob(os.path.join(LOSS_PLOT_BASE, "non_member", "loss", "*.png")))

        samples = []

        if m_imgs and nm_imgs:
            print(f"[Data] Found real data. Member: {len(m_imgs)}, Non-Member: {len(nm_imgs)}")
            m_loss_data = []
            nm_loss_data = []
            if os.path.exists(LOSS_HISTORY_PATH):
                try:
                    with open(LOSS_HISTORY_PATH, 'rb') as f:
                        data = pickle.load(f)
                    m_loss_data = data.get("member", {}).get("loss", [])
                    nm_loss_data = data.get("non_member", {}).get("loss", [])
                except Exception as e:
                    print(f"[Warning] Loss file error: {e}")

            for i in range(len(m_imgs)):
                loss = m_loss_data[i] if i < len(m_loss_data) else [0.05] * 20
                samples.append(Sample(len(samples), m_imgs[i], loss, true_label=1))

            for i in range(len(nm_imgs)):
                loss = nm_loss_data[i] if i < len(nm_loss_data) else [0.8] * 20
                samples.append(Sample(len(samples), nm_imgs[i], loss, true_label=0))
        else:
            print("[Data] ⚠️ No images found. Generating MOCK DATA for demonstration...")
            mock_dir = "./mock_images"
            os.makedirs(mock_dir, exist_ok=True)

            for i in range(50):
                is_member = 1 if i < 25 else 0
                fpath = os.path.join(mock_dir, f"mock_{i}_{is_member}.png")

                if is_member:
                    base = np.linspace(1.0, 0.05, 20) ** 2
                    noise = np.random.normal(0, 0.02, 20)
                else:
                    base = np.linspace(1.0, 0.4, 20) ** 0.8
                    noise = np.random.normal(0, 0.1, 20)

                loss_seq = list(np.clip(base + noise, 0, None))

                if not os.path.exists(fpath):
                    plt.figure(figsize=(4, 3))
                    plt.plot(loss_seq)
                    plt.title(f"Loss (GT={is_member})")
                    plt.savefig(fpath)
                    plt.close()

                samples.append(Sample(i, fpath, loss_seq, is_member))

        if max_samples:
            import random
            samples = random.sample(samples, min(len(samples), max_samples))

        self.state.samples = samples
        print(f"[Data] Loaded {len(samples)} samples.")

    def run(self):
        os.makedirs(REPORT_OUTPUT_DIR, exist_ok=True)
        if not self.state.samples: return

        # === Phase 1: VLM Screener ===
        # 1. 临时加载 Phase 1 模型
        screener = Phase1Screener(MODEL_PATH)
        print(f"\n=== Phase 1: Screener  ===")

        scr_results = screener.scan(self.state.samples)
        self.state.screener_results = scr_results

        for sid, res in scr_results.items():
            self.state.phase1_scores[sid] = res.member_prob

            # if res.max_posterior_prob >= SCREENER_CONF_THRESHOLD:
            if res.predicted_label == 0:
                self.state.easy_samples.append(sid)
                self.state.final_predictions[sid] = res.predicted_label
                self.state.final_probabilities[sid] = res.member_prob
            else:
                self.state.hard_samples.append(sid)

        # 2. 卸载 Phase 1 模型，释放显存
        screener.unload()
        del screener

        # === Phase 2: LLM Investigator ===
        if not self.state.hard_samples:
            print("No hard samples for Phase 2. Done.")
        else:
            print(f"\n=== Phase 2: Investigator ({len(self.state.hard_samples)} Hard Samples) ===")

            # 3. 临时加载 Phase 2 模型
            investigator = Phase2ReActAgent(MODEN_LLM_PATH)

            for sid in tqdm(self.state.hard_samples, desc="ReAct Reasoning"):
                sample = next(s for s in self.state.samples if s.sample_id == sid)
                p1_score = scr_results[sid].member_prob

                decision = investigator.investigate(sample, p1_score)
                self.state.react_decisions[sid] = decision

                final_pred_label = decision.final_label
                self.state.final_predictions[sid] = final_pred_label

                # 区间映射校准法
                if final_pred_label == 1:
                    calibrated_prob = p1_score
                    # calibrated_prob = min(0.999, calibrated_prob)
                else:
                    calibrated_prob = 1 - p1_score
                    # calibrated_prob = max(0.001, calibrated_prob)

                self.state.final_probabilities[sid] = calibrated_prob

        self._generate_report()

    def _generate_report(self):
        y_true = [s.true_label for s in self.state.samples]
        y_pred = [self.state.final_predictions.get(s.sample_id, 0) for s in self.state.samples]
        y_prob = [self.state.final_probabilities.get(s.sample_id, 0.0) for s in self.state.samples]

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, pos_label=1)
        prec = precision_score(y_true, y_pred, pos_label=1)
        rec = recall_score(y_true, y_pred, pos_label=1)

        print("\n" + "=" * 40)
        print("FINAL AUDIT REPORT")
        print("=" * 40)
        print(f"Total Samples: {len(y_true)}")
        print(f"Phase 2 Refined Samples: {len(self.state.hard_samples)}")
        print(f"Accuracy: {acc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")

        ROC_AUC_Result_logshow(y_true, y_prob, REPORT_OUTPUT_DIR)

        detailed = []
        for s in self.state.samples:
            res = {
                "id": s.sample_id,
                "gt": s.true_label,
                "pred": self.state.final_predictions.get(s.sample_id),
                "prob": self.state.final_probabilities.get(s.sample_id),
                "phase": "Phase2" if s.sample_id in self.state.hard_samples else "Phase1"
            }
            if s.sample_id in self.state.react_decisions:
                res["trace"] = self.state.react_decisions[s.sample_id].trace_log
            detailed.append(res)

        json_path = os.path.join(REPORT_OUTPUT_DIR, "audit_details.json")
        with open(json_path, "w") as f:
            json.dump(detailed, f, indent=2)
        print(f"[Done] Detailed report saved to {json_path}")


if __name__ == "__main__":
    agent = MIAAuditorAgent()
    agent.load_data(max_samples=600)
    agent.run()