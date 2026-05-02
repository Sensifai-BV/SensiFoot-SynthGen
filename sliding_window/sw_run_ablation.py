"""
sw_run_ablation.py
==================
Master orchestration script for the 19-cell sliding-window ablation study.

What changed vs run_ablation_expanded.py
─────────────────────────────────────────
• Imports sw_trainer / sw_evaluator instead of trainer_expanded / evaluator_expanded
• hidden_dim  : 32   (was 64)
• epochs      : 30   (was 15)
• window_size : 60   (new — was full-sequence)
• step_size   : 5    (new — was full-sequence)
• GPU         : auto-selected via torch.cuda.is_available() in trainer + evaluator
• Output dir  : ./outputs_sw  (keeps old outputs_expanded untouched)
• Summary note updated: "Sliding-window W=60 S=5"

Everything else is identical:
• Same 19 cells  (3 archs × 6 phases + STGCN)
• Same file-save names  → best_model_<ARCH>_<PHASE>.pth
• Same log names        → train_log_*.txt / inference_log_*.txt
• Same ABLATION_SUMMARY.txt format
• Same --inference_only / --train_only / --cells / --epochs CLI flags

Cell matrix (PHASE2 and PHASE3 excluded)
───────────────────────────────────────────────────────────
          BASELINE PHASE1 PHASE4 PHASE5
CNN_LSTM     1       2      3      4
TCN          5       6      7      8
BiGRU        9      10     11     12
───────────────────────────────────────────────────────────
STGCN (standalone, STGCN feature set only)           13
───────────────────────────────────────────────────────────

Output layout
─────────────
    outputs_sw/
        best_model_CNN_LSTM_BASELINE.pth
        best_model_TCN_PHASE2.pth
        ...                              (19 weight files)
        train_log_CNN_LSTM_BASELINE.txt
        ...                              (19 training logs)
        inference_log_CNN_LSTM_BASELINE.txt
        ...                              (19 inference logs)
        ablation_master_log.txt
        ABLATION_SUMMARY.txt             ← final table

Usage
─────
# Full run (train + evaluate all 19 cells):
    python sw_run_ablation.py \\
        --data_path /your/train/data \\
        --test_dir  /your/test/all_csv

# Inference only (weights already exist):
    python sw_run_ablation.py \\
        --data_path ... --test_dir ... --inference_only

# Train only:
    python sw_run_ablation.py \\
        --data_path ... --test_dir ... --train_only

# Run a specific subset:
    python sw_run_ablation.py \\
        --data_path ... --test_dir ... \\
        --cells TCN:PHASE2 BiGRU:PHASE5 STGCN:STGCN

# Quick smoke-test (2 epochs):
    python sw_run_ablation.py \\
        --data_path ... --test_dir ... --epochs 2
"""

import os
import argparse
import time
from datetime import datetime, timedelta

from feature_extractors import PHASE_REGISTRY
from models_zoo import ARCH_REGISTRY
from sw_trainer   import train_one_cell, DEFAULTS
from sw_evaluator import run_inference

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION  ── edit paths here, or pass everything via CLI flags
# ══════════════════════════════════════════════════════════════════════════════
CONFIG = {
    'data_path':   '/PATH/TO/SYNTHETIC/CSV/FILES',
    'test_dir':    '/PATH/TO/TEST/CSV/FILES',
    'output_dir':  './outputs_sw',

    # ── changed defaults ──────────────────────────────────────────────────────
    'hidden_dim':  DEFAULTS['hidden_dim'],    # 32
    'epochs':      DEFAULTS['epochs'],        # 30
    'window_size': DEFAULTS['window_size'],   # 60
    'step_size':   DEFAULTS['step_size'],     # 5

    # ── unchanged ─────────────────────────────────────────────────────────────
    'lr':          DEFAULTS['lr'],            # 0.0007
    'batch_size':  DEFAULTS['batch_size'],    # 32
    'patience':    DEFAULTS['patience'],      # 6
}
# ══════════════════════════════════════════════════════════════════════════════

_SEQ_ARCHS  = ['CNN_LSTM', 'TCN', 'BiGRU'] #add stgcn too
_SEQ_PHASES = ['BASELINE', 'PHASE1', 'PHASE4', 'PHASE5'] # add phase 2 and phase 3 too!

ALL_CELLS = (
    [(a, p) for a in _SEQ_ARCHS for p in _SEQ_PHASES]
    + [('STGCN', 'STGCN')]
)


# ── Summary table (same format as run_ablation_expanded.py) ───────────────────

def _summary_table(inference_results: list) -> str:
    hdr = (
        "\n################ SUMMARY TABLE ################\n"
        f"{'#':<5}{'Architecture':<14}{'Feature Phase':<16}"
        f"{'Combined Acc (%)':<20}{'Front Acc (%)':<17}{'Side Acc (%)'}\n"
        + "─" * 72 + "\n"
    )
    rows         = ""
    current_arch = None
    num          = 0
    for r in inference_results:
        num += 1
        if r['arch'] != current_arch:
            if current_arch is not None:
                rows += "─" * 72 + "\n"
            current_arch = r['arch']
        rows += (f"{num:<5}{r['arch']:<14}{r['phase']:<16}"
                 f"{r['combined_acc']:<20.2f}{r['front_acc']:<17.2f}"
                 f"{r['side_acc']:.2f}\n")
    rows += "─" * 72 + "\n"
    return hdr + rows


def _parse_cells(cell_strs: list):
    cells = []
    for s in cell_strs:
        parts = s.split(':')
        if len(parts) != 2:
            raise ValueError(f"Cell must be ARCH:PHASE, got '{s}'")
        arch, phase = parts[0].strip(), parts[1].strip()
        if arch not in ARCH_REGISTRY:
            raise ValueError(f"Unknown arch '{arch}'. "
                             f"Choose from {list(ARCH_REGISTRY.keys())}")
        if phase not in PHASE_REGISTRY:
            raise ValueError(f"Unknown phase '{phase}'. "
                             f"Choose from {list(PHASE_REGISTRY.keys())}")
        if arch == 'STGCN' and phase != 'STGCN':
            raise ValueError("STGCN architecture must use phase='STGCN'.")
        if phase == 'STGCN' and arch != 'STGCN':
            raise ValueError("STGCN phase is only valid with arch='STGCN'.")
        cells.append((arch, phase))
    return cells


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Sliding-window 13-cell ablation study (W=60, S=5, no PHASE2/PHASE3).')
    parser.add_argument('--data_path',      default=None)
    parser.add_argument('--test_dir',       default=None)
    parser.add_argument('--output_dir',     default=None)
    parser.add_argument('--epochs',         type=int,   default=None)
    parser.add_argument('--lr',             type=float, default=None)
    parser.add_argument('--batch_size',     type=int,   default=None)
    parser.add_argument('--hidden_dim',     type=int,   default=None)
    parser.add_argument('--window_size',    type=int,   default=None)
    parser.add_argument('--step_size',      type=int,   default=None)
    parser.add_argument('--cells',          nargs='+',  default=None,
                        metavar='ARCH:PHASE',
                        help='Subset of cells, e.g. TCN:PHASE2 STGCN:STGCN')
    parser.add_argument('--inference_only', action='store_true')
    parser.add_argument('--train_only',     action='store_true')
    args = parser.parse_args()

    cfg = dict(CONFIG)
    if args.data_path:   cfg['data_path']   = args.data_path
    if args.test_dir:    cfg['test_dir']     = args.test_dir
    if args.output_dir:  cfg['output_dir']   = args.output_dir
    if args.epochs:      cfg['epochs']       = args.epochs
    if args.lr:          cfg['lr']           = args.lr
    if args.batch_size:  cfg['batch_size']   = args.batch_size
    if args.hidden_dim:  cfg['hidden_dim']   = args.hidden_dim
    if args.window_size: cfg['window_size']  = args.window_size
    if args.step_size:   cfg['step_size']    = args.step_size

    cells_to_run = _parse_cells(args.cells) if args.cells else ALL_CELLS

    os.makedirs(cfg['output_dir'], exist_ok=True)

    master_log_path = os.path.join(cfg['output_dir'], 'ablation_master_log.txt')
    master_log      = open(master_log_path, 'w')

    def log(msg):
        print(msg)
        master_log.write(msg + '\n')
        master_log.flush()

    log("=" * 72)
    log("  SLIDING-WINDOW ABLATION STUDY  –  13 cells (PHASE2 & PHASE3 excluded)")
    log(f"  Started     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  Cells       : {len(cells_to_run)}")
    log(f"  Window      : size={cfg['window_size']}  step={cfg['step_size']}")
    log(f"  hidden_dim  : {cfg['hidden_dim']}")
    log(f"  epochs      : {cfg['epochs']}  (early stopping patience={cfg['patience']})")
    log(f"  Output      : {cfg['output_dir']}")
    log("=" * 72)
    for i, (a, p) in enumerate(cells_to_run, 1):
        log(f"    {i:02d}.  {a:<12} {p}")
    log("")

    # Keys forwarded to train_one_cell
    train_cfg = {k: cfg[k] for k in (
        'epochs', 'lr', 'batch_size', 'patience',
        'hidden_dim', 'window_size', 'step_size')}

    train_results = {}

    # ── STEP 1: Training ──────────────────────────────────────────────────────
    if not args.inference_only:
        log("─" * 72)
        log(f"  STEP 1 / 2 : TRAINING  ({len(cells_to_run)} cells)")
        log("─" * 72)

        for arch, phase_name in cells_to_run:
            run_key  = f"{arch}_{phase_name}"
            t0       = time.time()
            log_path = os.path.join(cfg['output_dir'], f'train_log_{run_key}.txt')

            with open(log_path, 'w') as tlf:
                result = train_one_cell(
                    arch=arch, phase_name=phase_name,
                    data_path=cfg['data_path'],
                    output_dir=cfg['output_dir'],
                    cfg=train_cfg, log_file=tlf,
                )

            elapsed = timedelta(seconds=int(time.time() - t0))
            if result:
                train_results[run_key] = result
                log(f"  ✅ {run_key:<22}  val={result['best_val_acc']:.2f}%  "
                    f"time={elapsed}")
            else:
                log(f"  ❌ {run_key:<22}  FAILED  time={elapsed}")

        log("\n  Training complete.\n")

    else:
        log("\n  [--inference_only] Skipping training.\n")
        for arch, phase_name in cells_to_run:
            run_key = f"{arch}_{phase_name}"
            train_results[run_key] = {
                'arch': arch, 'phase': phase_name, 'run_key': run_key,
                'best_val_acc': None,
                'model_path':   os.path.join(
                    cfg['output_dir'], f'best_model_{run_key}.pth'),
                'input_dim':    PHASE_REGISTRY[phase_name]['input_dim'],
            }

    # ── STEP 2: Inference ─────────────────────────────────────────────────────
    inference_results = []

    if not args.train_only:
        log("─" * 72)
        log(f"  STEP 2 / 2 : INFERENCE  ({len(cells_to_run)} cells)")
        log("─" * 72)

        for arch, phase_name in cells_to_run:
            run_key = f"{arch}_{phase_name}"
            if run_key not in train_results:
                log(f"  ⚠️  {run_key}: no training result, skipping.")
                continue

            model_path   = train_results[run_key]['model_path']
            inf_log_path = os.path.join(
                cfg['output_dir'], f'inference_log_{run_key}.txt')

            with open(inf_log_path, 'w') as ilf:
                res = run_inference(
                    arch=arch, phase_name=phase_name,
                    model_path=model_path,
                    test_csv_dir=cfg['test_dir'],
                    output_dir=cfg['output_dir'],
                    hidden_dim=cfg['hidden_dim'],
                    window_size=cfg['window_size'],
                    step_size=cfg['step_size'],
                    log_file=ilf,
                )
            inference_results.append(res)
            log(f"  ✅ {run_key:<22}  "
                f"combined={res['combined_acc']:.2f}%  "
                f"front={res['front_acc']:.2f}%  "
                f"side={res['side_acc']:.2f}%")

        # ── summary table ──────────────────────────────────────────────────────
        table = _summary_table(inference_results)
        log(table)

        summary_path = os.path.join(cfg['output_dir'], 'ABLATION_SUMMARY.txt')
        with open(summary_path, 'w') as sf:
            sf.write("Expanded Gesture Recognition Ablation Study – Summary\n")
            sf.write(f"(Sliding-window  W={cfg['window_size']}  S={cfg['step_size']}  "
                     f"hidden_dim={cfg['hidden_dim']})\n")
            sf.write(f"Generated   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            sf.write(f"Test dir    : {cfg['test_dir']}\n\n")

            if not args.inference_only:
                sf.write("=== Training Best Validation Accuracy ===\n")
                for arch, phase_name in cells_to_run:
                    rk = f"{arch}_{phase_name}"
                    if (rk in train_results
                            and train_results[rk]['best_val_acc'] is not None):
                        sf.write(f"  {rk:<22}: "
                                 f"{train_results[rk]['best_val_acc']:.2f}%\n")
                sf.write("\n")

            sf.write(table)

        log(f"\n  📄 Summary saved → {summary_path}")

    else:
        log("\n  [--train_only] Skipping inference.\n")

    log("\n" + "=" * 72)
    log(f"  COMPLETE – {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 72)
    master_log.close()
    print(f"\nMaster log → {master_log_path}")


if __name__ == '__main__':
    main()
