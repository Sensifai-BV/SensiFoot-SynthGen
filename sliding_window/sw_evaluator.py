"""
sw_evaluator.py
===============
Sliding-window inference engine for all 19 ablation cells.

Inference protocol (matches the original LOSO fine-tuning script)
──────────────────────────────────────────────────────────────────
1. Slice one CSV into overlapping windows (WINDOW_SIZE=60, STEP_SIZE=5).
2. Run a forward pass on every window → softmax probability vector.
3. Average the probability vectors across all windows of the same video.
4. argmax of the averaged vector → video-level prediction.

This is identical to the evaluate_video_level() logic in the original
lstm_attention_slidingwindow.py, so results are directly comparable.

CSV filename convention (unchanged):
    <subject>-<view>-<class>-<trial>_features.csv
    view: 'f' = front, 's' = side

Usage (standalone):
    python sw_evaluator.py \\
        --arch TCN --phase PHASE2 \\
        --model ./outputs_sw/best_model_TCN_PHASE2.pth \\
        --test_dir /path/to/all_csv \\
        --output_dir ./outputs_sw

Usage (via sw_run_ablation.py):
    Called programmatically via run_inference().
"""

import os
import glob
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

from models_zoo import build_model, ARCH_REGISTRY
from feature_extractors import PHASE_REGISTRY

# ── Constants ─────────────────────────────────────────────────────────────────
NUM_CLASSES  = 8
CLASS_LABELS = list(range(1, NUM_CLASSES + 1))   # [1, 2, …, 8]

# Default window params — overridden by whatever is passed from the orchestrator
_DEFAULT_WINDOW_SIZE = 60
_DEFAULT_STEP_SIZE   = 5
_DEFAULT_HIDDEN_DIM  = 32


# ── Video-level predictor ─────────────────────────────────────────────────────

class WindowPredictor:
    """
    Loads one (arch, phase) model and predicts at video level:
    slices the full sequence into windows, averages softmax scores,
    returns the top-1 and top-2 predicted class labels (1-indexed).
    """

    def __init__(self, arch: str, phase_name: str, model_path: str,
                 hidden_dim: int = _DEFAULT_HIDDEN_DIM,
                 window_size: int = _DEFAULT_WINDOW_SIZE,
                 step_size:   int = _DEFAULT_STEP_SIZE):

        self.device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        phase_cfg        = PHASE_REGISTRY[phase_name]
        self.extractor   = phase_cfg['fn']
        input_dim        = phase_cfg['input_dim']
        self.window_size = window_size
        self.step_size   = step_size

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Weights not found: {model_path}")

        self.model = build_model(arch, input_dim, hidden_dim, NUM_CLASSES)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.to(self.device)
        self.model.eval()

    def _windows(self, arr: np.ndarray):
        """Yield (W, D) windows from a (T, D) array."""
        T, D = arr.shape
        W, S = self.window_size, self.step_size
        if T < W:
            arr = np.vstack([arr, np.zeros((W - T, D), dtype=np.float32)])
            T   = W
        starts = list(range(0, T - W + 1, S)) or [0]
        for s in starts:
            yield arr[s:s + W]

    def predict_top2(self, csv_path: str):
        """
        Returns (top1_label, top2_label) as 1-indexed ints,
        or ('Unknown', 'Unknown') on error.
        """
        try:
            df       = pd.read_csv(csv_path, header=0)
            features = self.extractor(df).astype(np.float32)
        except Exception as e:
            print(f"  [ERROR] reading {csv_path}: {e}")
            return 'Unknown', 'Unknown'

        win_probs = []
        with torch.no_grad():
            for win in self._windows(features):
                x = torch.tensor(win).unsqueeze(0).to(self.device)
                logits = self.model(x, lengths=None)
                win_probs.append(F.softmax(logits, dim=1)[0].cpu().numpy())

        if not win_probs:
            return 'Unknown', 'Unknown'

        avg_probs = np.mean(win_probs, axis=0)          # (num_classes,)
        top2_idx  = np.argsort(avg_probs)[::-1][:2]
        return CLASS_LABELS[top2_idx[0]], CLASS_LABELS[top2_idx[1]]


# ── Report helpers ────────────────────────────────────────────────────────────

def _format_cm(cm: np.ndarray, labels: list) -> str:
    """Pretty-print a confusion matrix with row/column headers."""
    col_w   = max(4, max(len(str(v)) for row in cm for v in row) + 1)
    lbl_w   = max(len("True\\Pred"), max(len(str(l)) for l in labels)) + 1
    header  = f"{'True\\Pred':<{lbl_w}}" + "".join(f"{l:>{col_w}}" for l in labels)
    divider = "-" * len(header)
    rows    = [header, divider]
    for i, row in enumerate(cm):
        rows.append(f"{labels[i]:<{lbl_w}}" + "".join(f"{v:>{col_w}}" for v in row))
    return "\n".join(rows)


def _section(title: str, subset: list) -> str:
    if not subset:
        return f"=== {title} ===\nNo data found.\n\n"
    y_true  = [r['true'] for r in subset]
    y_pred  = [r['pred'] for r in subset]
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    acc     = accuracy_score(y_true, y_pred)
    cm      = confusion_matrix(y_true, y_pred, labels=CLASS_LABELS)
    txt  = f"=== {title} ===\n"
    txt += f"Accuracy: {acc*100:.2f}%  ({correct} / {len(subset)} correct)\n"
    txt += "Confusion Matrix (rows=True, cols=Predicted):\n"
    txt += _format_cm(cm, CLASS_LABELS) + "\n\n"
    return txt


# ── Core inference function ───────────────────────────────────────────────────

def run_inference(
    arch:         str,
    phase_name:   str,
    model_path:   str,
    test_csv_dir: str,
    output_dir:   str,
    hidden_dim:   int = _DEFAULT_HIDDEN_DIM,
    window_size:  int = _DEFAULT_WINDOW_SIZE,
    step_size:    int = _DEFAULT_STEP_SIZE,
    log_file=None,
):
    """
    Run sliding-window video-level inference for one (arch, phase) cell.

    Returns
    -------
    dict  {run_key, arch, phase, combined_acc, front_acc, side_acc}
    """
    run_key = f"{arch}_{phase_name}"

    def log(msg):
        print(msg)
        if log_file:
            log_file.write(msg + '\n')
            log_file.flush()

    log(f"\n{'='*64}")
    log(f"  INFERENCE  arch={arch}  phase={phase_name}  key={run_key}")
    log(f"  Model   : {model_path}")
    log(f"  Test    : {test_csv_dir}")
    log(f"  Window  : size={window_size}  step={step_size}")
    log(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"{'='*64}")

    null_result = {
        'run_key': run_key, 'arch': arch, 'phase': phase_name,
        'combined_acc': 0.0, 'front_acc': 0.0, 'side_acc': 0.0,
    }

    if not os.path.exists(model_path):
        log("  [SKIP] Weights not found.")
        return null_result

    try:
        predictor = WindowPredictor(arch, phase_name, model_path,
                                    hidden_dim=hidden_dim,
                                    window_size=window_size,
                                    step_size=step_size)
    except Exception as e:
        log(f"  [ERROR] Could not load model: {e}")
        return null_result

    csv_files = glob.glob(os.path.join(test_csv_dir, '*.csv'))
    if not csv_files:
        log(f"  [SKIP] No CSV files in {test_csv_dir}")
        return null_result

    log(f"  {len(csv_files)} files found. Running inference …")
    results = []

    for idx, fp in enumerate(csv_files, 1):
        fname = os.path.basename(fp)
        parts = fname.replace('_features.csv', '').replace('.csv', '').split('-')
        try:
            view       = parts[1].lower()
            true_class = int(parts[2])
        except (IndexError, ValueError):
            log(f"  ⚠️  Skipping {fname}: cannot parse filename.")
            continue

        p1, p2 = predictor.predict_top2(fp)
        if p1 == 'Unknown':
            continue

        results.append({'filename': fname, 'view': view,
                        'true': true_class, 'pred': p1})
        if idx % 50 == 0:
            log(f"  … {idx}/{len(csv_files)}")

    if not results:
        log("  [WARN] No valid predictions.")
        return null_result

    front = [r for r in results if r['view'] == 'f']
    side  = [r for r in results if r['view'] == 's']

    def _acc(s):
        if not s:
            return 0.0
        return accuracy_score([r['true'] for r in s],
                               [r['pred'] for r in s]) * 100

    combined_acc = _acc(results)
    front_acc    = _acc(front)
    side_acc     = _acc(side)

    # ── per-cell report ───────────────────────────────────────────────────────
    report  = (f"Inference Report\n"
               f"arch={arch}   phase={phase_name}   "
               f"window={window_size}   step={step_size}\n"
               + "=" * 56 + "\n\n")
    report += _section("ALL VIEWS (Combined)", results)
    report += _section("FRONT VIEWS ONLY ('f')", front)
    report += _section("SIDE VIEWS ONLY ('s')", side)
    report += "=== INDIVIDUAL GESTURE ACCURACY ===\n"
    for cid in CLASS_LABELS:
        sub = [r for r in results if r['true'] == cid]
        if sub:
            a    = accuracy_score([r['true'] for r in sub],
                                  [r['pred'] for r in sub])
            n_ok = sum(r['true'] == r['pred'] for r in sub)
            report += (f"  Gesture {cid}: {a*100:.2f}%"
                       f"  ({n_ok} / {len(sub)})\n")

    report_path = os.path.join(output_dir, f'inference_log_{run_key}.txt')
    with open(report_path, 'w') as f:
        f.write(report)

    log(f"\n  Combined : {combined_acc:.2f}%  "
        f"Front : {front_acc:.2f}%  Side : {side_acc:.2f}%")

    # ── confusion matrices → master log / console ─────────────────────────────
    for _title, _subset in [
        ("ALL VIEWS (Combined)", results),
        ("FRONT VIEWS ONLY",     front),
        ("SIDE VIEWS ONLY",      side),
    ]:
        if not _subset:
            continue
        _yt  = [r['true'] for r in _subset]
        _yp  = [r['pred'] for r in _subset]
        _cm  = confusion_matrix(_yt, _yp, labels=CLASS_LABELS)
        _acc = accuracy_score(_yt, _yp)
        _ok  = sum(t == p for t, p in zip(_yt, _yp))
        log(f"\n  ── {_title}  ({_acc*100:.2f}%,  {_ok}/{len(_subset)}) ──")
        log("  Confusion Matrix (rows=True, cols=Predicted):")
        for _line in _format_cm(_cm, CLASS_LABELS).splitlines():
            log(f"    {_line}")

    log(f"\n  Report → {report_path}\n")

    return {
        'run_key':      run_key,
        'arch':         arch,
        'phase':        phase_name,
        'combined_acc': combined_acc,
        'front_acc':    front_acc,
        'side_acc':     side_acc,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Sliding-window inference for one (arch, phase) cell.')
    parser.add_argument('--arch',        required=True,
                        choices=list(ARCH_REGISTRY.keys()))
    parser.add_argument('--phase',       required=True,
                        choices=list(PHASE_REGISTRY.keys()))
    parser.add_argument('--model',       required=True)
    parser.add_argument('--test_dir',    required=True)
    parser.add_argument('--output_dir',  default='./outputs_sw')
    parser.add_argument('--hidden_dim',  type=int, default=_DEFAULT_HIDDEN_DIM)
    parser.add_argument('--window_size', type=int, default=_DEFAULT_WINDOW_SIZE)
    parser.add_argument('--step_size',   type=int, default=_DEFAULT_STEP_SIZE)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    run_key  = f"{args.arch}_{args.phase}"
    log_path = os.path.join(args.output_dir, f'inference_log_{run_key}.txt')

    with open(log_path, 'w') as lf:
        result = run_inference(
            arch=args.arch, phase_name=args.phase,
            model_path=args.model, test_csv_dir=args.test_dir,
            output_dir=args.output_dir,
            hidden_dim=args.hidden_dim,
            window_size=args.window_size, step_size=args.step_size,
            log_file=lf,
        )

    print(f"\nCombined : {result['combined_acc']:.2f}%")
    print(f"Front    : {result['front_acc']:.2f}%")
    print(f"Side     : {result['side_acc']:.2f}%")
