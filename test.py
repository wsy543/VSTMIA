import os
import glob
import pickle
import json
import sys
import gc
import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

try:
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    _HAS_TRANSFORMERS = True
except ImportError:
    Qwen3VLForConditionalGeneration = None
    AutoProcessor = None
    _HAS_TRANSFORMERS = False

from sklearn.metrics import roc_curve, roc_auc_score

VLM_PATHS = {
    "qwen3_2b":       "./vlm_2b",
}
REPORT_OUTPUT_DIR = "./reports_lira_lite"
RANDOM_SEED = 42

def load_vlm(model_path: str, vlm_type: str):
    if not _HAS_TRANSFORMERS:
        raise ImportError(
            "transformers not found. Please install: pip install transformers"
        )
    print(f"[VLM Loader] Loading {vlm_type} from {model_path}...")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype="auto", device_map="auto", trust_remote_code=True
    )

    return model, processor


def get_vlm_path(vlm_type: str, custom_path: str | None = None) -> str:
    if custom_path:
        return custom_path
    if vlm_type in VLM_PATHS:
        return VLM_PATHS[vlm_type]
    raise ValueError(f"Unknown vlm_type: {vlm_type}. Known types: {list(VLM_PATHS.keys())}")


def get_paths(dataset, model):
    loss_plot_base = f"./plot/vlm_data/{model}/{dataset}/"
    loss_history_path = f"./plot/vlm_data/{model}/{dataset}/metrics_history_selected.pkl"
    return loss_plot_base, loss_history_path



class DynamicsToolkit:
    def __init__(self, head_ratio: float = 0.5, tail_ratio: float = 0.5):
        self.head_ratio = head_ratio
        self.tail_ratio = tail_ratio

    def extract_features(self, loss_seq: List[float]) -> Dict[str, float]:
        y = np.array(loss_seq)
        T = len(y)

        if T < 2:
            return {
                "rate_of_change": 0.0,
                "tail_fluctuation": 0.0,
                "raw_loss": float(y[-1] if T > 0 else 0.0)
            }

        head_end_idx = max(2, int(T * self.head_ratio))
        y_head = y[:head_end_idx]

        diffs = np.diff(y_head)
        descending_steps = diffs[diffs < 0]

        if len(descending_steps) > 0:
            true_descent_rate = np.sum(np.abs(descending_steps))
        else:
            true_descent_rate = 0.0

        rate_of_change = float(true_descent_rate)

        tail_start_idx = int(T * self.tail_ratio)
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


class Phase1Screener:
    def __init__(self, model_path: str, vlm_type: str = "qwen3_2b"):
        print(f"[Phase 1] Loading VLM ({vlm_type}): {model_path}...")
        self.model, self.processor = load_vlm(model_path, vlm_type)
        self.vlm_type = vlm_type
        self.prompt = ((
            "Examine the training loss curve in the image. "
            "The X-axis represents training rounds (model checkpoints over time), "
            "and the Y-axis represents loss value.\n"
            "Your Task: Identify if this sample belongs to the training set (Member) based on the curve's geometry.\n\n"
            "Visual Definitions:\n"
            "- Member (1): Feature 'Hockey Stick' shape. Extremely sharp initial drop followed by a perfect flatline near the X-axis (zero).\n"
            "- Non-member (0): Feature 'Gradual Descent' or 'Noisy Tail'. The curve drops slowly or shows visible jitter.\n\n"
            "Output '1' for Member or '0' for Non-member.\n"
            "Answer:"
        ))

    def _build_inputs(self, image_path: str):
        messages = [{"role": "user", "content": [
            {"type": "image", "image": os.path.abspath(image_path)},
            {"type": "text", "text": self.prompt}
        ]}]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt"
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
    def __init__(self, calibration_data: List[Dict], head_ratio: float = 0.5, tail_ratio: float = 0.5):
        print("[Phase 2] Building Non-parametric Background Distribution (eCDF)...")
        self.physics = DynamicsToolkit(head_ratio=head_ratio, tail_ratio=tail_ratio)
        self.bg_pool = {}

        for key in ["rate_of_change", "tail_fluctuation"]:
            values = [d[key] for d in calibration_data]
            if not values: values = [0.0]
            self.bg_pool[key] = np.sort(values)

        print(f"  > Calibration Pool Built with {len(self.bg_pool['rate_of_change'])} samples.")

    def compute_empirical_percentile(self, val: float, key: str) -> float:
        bg_array = self.bg_pool[key]
        percentile = np.searchsorted(bg_array, val) / len(bg_array)
        return float(percentile)

    def analyze(self, sample: Sample, vlm_score: float, alpha: float) -> Tuple[float, Dict]:
        phy = self.physics.extract_features(sample.loss_sequence)

        p_roc = 1.0 - self.compute_empirical_percentile(phy['rate_of_change'], 'rate_of_change')
        p_acc = 1.0 - self.compute_empirical_percentile(phy['tail_fluctuation'], 'tail_fluctuation')

        eps = 1e-5
        p_roc = np.clip(p_roc, eps, 1.0 - eps)
        p_acc = np.clip(p_acc, eps, 1.0 - eps)

        score_phy = 0.5 * p_roc + 0.5 * p_acc

        final_score = alpha * score_phy + 1.0 * vlm_score

        phy['s_roc'] = p_roc
        phy['s_acc'] = p_acc
        phy['score_phy'] = score_phy

        return float(final_score), phy
class LiraLiteAuditor:
    def __init__(self, dataset: str, model: str, vlm_type: str = "qwen3_2b", vlm_path: str | None = None,
                 enable_phase2: bool = True,
                 alpha_cap: float = 5.0, calib_threshold: float = 0.1,
                 head_ratio: float = 0.5, tail_ratio: float = 0.5):
        np.random.seed(RANDOM_SEED)
        torch.manual_seed(RANDOM_SEED)
        self.state = AgentState()
        self.dataset = dataset
        self.model = model
        self.vlm_type = vlm_type
        self.vlm_path = get_vlm_path(vlm_type, vlm_path)
        self.loss_plot_base, self.loss_history_path = get_paths(dataset, model)
        self.enable_phase2 = enable_phase2
        self.alpha_cap = alpha_cap
        self.calib_threshold = calib_threshold
        self.head_ratio = head_ratio
        self.tail_ratio = tail_ratio
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
        screener = Phase1Screener(self.vlm_path, self.vlm_type)
        self.state.phase1_scores = screener.scan(self.state.samples)
        screener.unload()

        print("\n[Calibration] Dynamically building calibration pool from VLM highly confident pseudo-negatives...")

        calib_candidates = [s for s in self.state.samples if self.state.phase1_scores[s.sample_id] < self.calib_threshold]

        if len(calib_candidates) < 10:
            print(f"  > Warning: Too few samples < {self.calib_threshold}. Backing off to bottom 10% of samples.")
            sorted_samples = sorted(self.state.samples, key=lambda s: self.state.phase1_scores[s.sample_id])
            calib_candidates = sorted_samples[:max(10, len(self.state.samples) // 10)]

        print(f"  > Selected {len(calib_candidates)} pseudo-non-members for calibration.")

        toolkit = DynamicsToolkit(head_ratio=self.head_ratio, tail_ratio=self.tail_ratio)
        self.state.calibration_pool = [toolkit.extract_features(s.loss_sequence) for s in calib_candidates]

        vlm_scores_array = np.array(list(self.state.phase1_scores.values()))

        mean_confidence = np.mean(2.0 * np.abs(vlm_scores_array - 0.5))

        adaptive_alpha = (1.0 - mean_confidence) / (mean_confidence + 1e-5)
        adaptive_alpha = min(self.alpha_cap, float(adaptive_alpha))

        print("\n[Fusion Strategy] Computing task-complexity adaptive weights...")
        print(f"  > VLM Mean Confidence: {mean_confidence:.3f}")
        print(f"  > Auto-set Physics Weight (Alpha): {adaptive_alpha:.3f}")

        details = []

        if self.enable_phase2:
            investigator = Phase2Investigator(self.state.calibration_pool,
                                              head_ratio=self.head_ratio, tail_ratio=self.tail_ratio)

            print("[Phase 2] Running dynamic evaluation pipeline...")
            for sample in tqdm(self.state.samples):
                vlm_score = self.state.phase1_scores[sample.sample_id]

                if vlm_score < self.calib_threshold:
                    final_score = vlm_score
                    phy = toolkit.extract_features(sample.loss_sequence)
                else:
                    final_score, phy = investigator.analyze(sample, vlm_score, alpha=adaptive_alpha)

                self.state.final_scores[sample.sample_id] = final_score

                details.append({
                    "id": int(sample.sample_id),
                    "gt": int(sample.true_label),
                    "score": float(final_score),
                    "features": phy,
                    "vlm_score": float(vlm_score)
                })
        else:
            print("[Phase 2] SKIPPED — using Phase 1 scores directly as final scores")
            for sample in tqdm(self.state.samples):
                vlm_score = self.state.phase1_scores[sample.sample_id]
                phy = toolkit.extract_features(sample.loss_sequence)
                self.state.final_scores[sample.sample_id] = vlm_score
                details.append({
                    "id": int(sample.sample_id),
                    "gt": int(sample.true_label),
                    "score": float(vlm_score),
                    "features": phy,
                    "vlm_score": float(vlm_score)
                })

        with open(os.path.join(REPORT_OUTPUT_DIR, "audit_details.json"), "w") as f:
            json.dump(details, f, indent=2)

        y_scores = [self.state.final_scores.get(s.sample_id, 0.0) for s in self.state.samples]
        self._evaluate(y_scores)


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
               vlm_type: str = "qwen3_2b", vlm_path: str | None = None,
               enable_phase2: bool = True,
               alpha_cap: float = 5.0, calib_threshold: float = 0.1,
               head_ratio: float = 0.5, tail_ratio: float = 0.5):
    print(f"\n{'='*60}")
    print(f"LiraLite Attack: dataset={dataset}, model={model}, max_samples={max_samples}")
    print(f"VLM: {vlm_type} @ {get_vlm_path(vlm_type, vlm_path)}")
    print(f"Phase2={enable_phase2}")
    print(f"alpha_cap={alpha_cap}, calib_threshold={calib_threshold}, head_ratio={head_ratio}, tail_ratio={tail_ratio}")
    print(f"{'='*60}\n")
    auditor = LiraLiteAuditor(
        dataset=dataset, model=model,
        vlm_type=vlm_type, vlm_path=vlm_path,
        enable_phase2=enable_phase2,
        alpha_cap=alpha_cap, calib_threshold=calib_threshold,
        head_ratio=head_ratio, tail_ratio=tail_ratio
    )
    auditor.load_data(max_samples=max_samples)
    auditor.run()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='LiraLite MIA Attack')
    parser.add_argument('--dataset', type=str, default='STL10', choices=['STL10', 'location'],
                        help='Dataset name')
    parser.add_argument('--model', type=str, default='resnet', choices=['resnet', 'nn'],
                        help='Model name')
    parser.add_argument('--max_samples', type=int, default=1000, help='Max test samples')
    parser.add_argument('--vlm_type', type=str, default='qwen3_2b', choices=['qwen3_2b'],
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