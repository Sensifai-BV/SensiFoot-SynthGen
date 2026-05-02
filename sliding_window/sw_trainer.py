"""
sw_trainer.py
=============
Unified sliding-window training loop for all 19 ablation cells.

Changes vs trainer_expanded.py
───────────────────────────────
• Dataset    : SlidingWindowDataset (WINDOW_SIZE=60, STEP_SIZE=5)
• hidden_dim : 32  (was 64)
• epochs     : 30  (was 15; early stopping gives real headroom)
• GPU        : torch.device auto-selects CUDA if available
• No other behavioural changes: same architectures, same file layout,
  same loso_split (val = 'mohi_'), same SupCon logic, same scheduler,
  same weight filenames → best_model_<ARCH>_<PHASE>.pth

Saved weight filenames (unchanged):
    best_model_<ARCH>_<PHASE>.pth

Per-cell training logs (unchanged):
    train_log_<ARCH>_<PHASE>.txt

Usage (standalone):
    python sw_trainer.py --arch TCN --phase PHASE2 \\
        --data_path /your/data --output_dir ./outputs_sw

Usage (via sw_run_ablation.py):
    Called programmatically via train_one_cell().
"""

import os
import glob
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models_zoo import build_model, SupConLoss, ARCH_REGISTRY
from feature_extractors import PHASE_REGISTRY
from sw_dataset import SlidingWindowDataset, collate_train, collate_eval
# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULTS = {
    'hidden_dim':    32,      # ← 32 (was 64)
    'lr':            0.0007,
    'epochs':        30,      # ← 30 (was 15)
    'batch_size':    32,
    'patience':      6,       # proportionally larger for 30-epoch budget
    'weight_decay':  1e-3,
    'num_classes':   8,
    'classes':       [str(i) for i in range(1, 9)],
    'supcon_temp':   0.07,
    'supcon_weight': 0.5,
    'window_size':   60,      # ← sliding window length
    'step_size':     5,       # ← stride
}


# ── Utilities ─────────────────────────────────────────────────────────────────

def build_file_list(data_path: str, classes: list):
    """Glob every class sub-folder and return [(filepath, class_name), ...]."""
    files = []
    for cls in classes:
        for fp in glob.glob(os.path.join(data_path, cls, '*.csv')):
            files.append((fp, cls))
    return files


def loso_split(all_files: list, val_subject: str = 'mohi_'):
    """
    Leave-one-subject-out split.
    Files whose basename contains val_subject go to val; rest to train.
    """
    train, val = [], []
    for fp, cls in all_files:
        bucket = val if val_subject in os.path.basename(fp).lower() else train
        bucket.append((fp, cls))
    return train, val


# ── Core training function ────────────────────────────────────────────────────

def train_one_cell(
    arch:        str,
    phase_name:  str,
    data_path:   str,
    output_dir:  str,
    cfg:         dict = None,
    log_file=None,
):
    """
    Train one (arch, phase) cell with sliding windows and save the best weights.

    Returns
    -------
    dict  {arch, phase, run_key, best_val_acc, model_path, input_dim}
    or None on unrecoverable error.
    """
    cfg     = {**DEFAULTS, **(cfg or {})}
    run_key = f"{arch}_{phase_name}"

    phase_cfg    = PHASE_REGISTRY[phase_name]
    extractor_fn = phase_cfg['fn']
    input_dim    = phase_cfg['input_dim']
    use_supcon   = phase_cfg['use_supcon']

    classes      = cfg['classes']
    class_to_idx = {c: i for i, c in enumerate(classes)}

    # ── logging helper ────────────────────────────────────────────────────────
    def log(msg):
        print(msg)
        if log_file:
            log_file.write(msg + '\n')
            log_file.flush()

    log(f"\n{'='*64}")
    log(f"  ARCH={arch:<10}  PHASE={phase_name:<8}  "
        f"input_dim={input_dim}  SupCon={use_supcon}")
    log(f"  Run key    : {run_key}")
    log(f"  Window     : size={cfg['window_size']}  step={cfg['step_size']}")
    log(f"  hidden_dim : {cfg['hidden_dim']}  epochs : {cfg['epochs']}")
    log(f"  Start      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"{'='*64}")

    # ── data ──────────────────────────────────────────────────────────────────
    all_files = build_file_list(data_path, classes)
    if not all_files:
        log(f"  [ERROR] No CSV files found under {data_path}")
        return None

    train_files, val_files = loso_split(all_files)
    log(f"  Train files : {len(train_files)}  |  Val files : {len(val_files)}")

    train_ds = SlidingWindowDataset(
        train_files, class_to_idx, extractor_fn,
        window_size=cfg['window_size'], step_size=cfg['step_size'],
        augment=True)
    val_ds = SlidingWindowDataset(
        val_files, class_to_idx, extractor_fn,
        window_size=cfg['window_size'], step_size=cfg['step_size'],
        augment=False)

    log(f"  Train windows : {len(train_ds)}  |  Val windows : {len(val_ds)}")

    if len(train_ds) == 0:
        log("  [ERROR] Empty training set after windowing.")
        return None

    train_loader = DataLoader(
        train_ds, batch_size=cfg['batch_size'], shuffle=True,
        collate_fn=collate_train, num_workers=0, pin_memory=True)
    val_loader = DataLoader(
        val_ds, batch_size=cfg['batch_size'], shuffle=False,
        collate_fn=collate_eval, num_workers=0, pin_memory=True)

    # ── model ─────────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"  Device : {device}"
        + (f"  [{torch.cuda.get_device_name(0)}]"
           if device.type == 'cuda' else ""))

    model        = build_model(arch, input_dim, cfg['hidden_dim'],
                               cfg['num_classes']).to(device)
    criterion_ce = nn.CrossEntropyLoss()
    
    criterion_sc = SupConLoss(cfg['supcon_temp']) if use_supcon else None
    optimizer    = optim.Adam(model.parameters(), lr=cfg['lr'],
                              weight_decay=cfg['weight_decay'])
    scheduler    = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3)

    model_path   = os.path.join(output_dir, f'best_model_{run_key}.pth')
    best_val_acc = 0.0
    no_improve   = 0

    # ── epoch loop ────────────────────────────────────────────────────────────
    for epoch in range(cfg['epochs']):
        # ── train ─────────────────────────────────────────────────────────────
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            # Sliding-window models receive fixed-length windows;
            # pass lengths=None so the model uses the full window.
            optimizer.zero_grad()
            logits  = model(x_batch, lengths=None)
            loss_ce = criterion_ce(logits, y_batch)

            if use_supcon and criterion_sc is not None:
                with torch.no_grad():
                    emb = model.get_embedding(x_batch, lengths=None)
                loss = loss_ce + cfg['supcon_weight'] * criterion_sc(
                    emb, y_batch)
            else:
                loss = loss_ce

            loss.backward()
            optimizer.step()

            t_loss    += loss_ce.item()
            _, preds   = torch.max(logits, 1)
            t_total   += y_batch.size(0)
            t_correct += (preds == y_batch).sum().item()

        train_acc  = 100.0 * t_correct / t_total
        avg_t_loss = t_loss / len(train_loader)

        # ── validate (window-level; good enough for model selection) ──────────
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0

        with torch.no_grad():
            for x_batch, y_batch, _ in val_loader:
                x_batch = x_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)
                logits  = model(x_batch, lengths=None)
                v_loss += criterion_ce(logits, y_batch).item()
                _, preds  = torch.max(logits, 1)
                v_total  += y_batch.size(0)
                v_correct += (preds == y_batch).sum().item()

        val_acc    = 100.0 * v_correct / v_total
        avg_v_loss = v_loss / len(val_loader)
        scheduler.step(val_acc)

        log(f"  Epoch {epoch+1:02d}/{cfg['epochs']:02d} | "
            f"Train Loss: {avg_t_loss:.4f} | Train Acc: {train_acc:.1f}% | "
            f"Val Loss: {avg_v_loss:.4f} | Val Acc: {val_acc:.1f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve   = 0
            torch.save(model.state_dict(), model_path)
            log(f"    💾 Saved best  val_acc={best_val_acc:.2f}%")
        else:
            no_improve += 1

        if no_improve >= cfg['patience']:
            log(f"    🛑 Early stop at epoch {epoch+1}  "
                f"(no improvement for {cfg['patience']} epochs)")
            break

    log(f"\n  ✅ {run_key} done.  Best Val Acc: {best_val_acc:.2f}%")
    log(f"     Weights → {model_path}\n")

    return {
        'arch':         arch,
        'phase':        phase_name,
        'run_key':      run_key,
        'best_val_acc': best_val_acc,
        'model_path':   model_path,
        'input_dim':    input_dim,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train one (arch, phase) cell with sliding windows.')
    parser.add_argument('--arch',        required=True,
                        choices=list(ARCH_REGISTRY.keys()))
    parser.add_argument('--phase',       required=True,
                        choices=list(PHASE_REGISTRY.keys()))
    parser.add_argument('--data_path',   required=True)
    parser.add_argument('--output_dir',  default='./outputs_sw')
    parser.add_argument('--epochs',      type=int,   default=DEFAULTS['epochs'])
    parser.add_argument('--lr',          type=float, default=DEFAULTS['lr'])
    parser.add_argument('--batch_size',  type=int,   default=DEFAULTS['batch_size'])
    parser.add_argument('--hidden_dim',  type=int,   default=DEFAULTS['hidden_dim'])
    parser.add_argument('--window_size', type=int,   default=DEFAULTS['window_size'])
    parser.add_argument('--step_size',   type=int,   default=DEFAULTS['step_size'])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    run_key  = f"{args.arch}_{args.phase}"
    log_path = os.path.join(args.output_dir, f'train_log_{run_key}.txt')

    with open(log_path, 'w') as lf:
        result = train_one_cell(
            arch=args.arch, phase_name=args.phase,
            data_path=args.data_path, output_dir=args.output_dir,
            cfg={
                'epochs':      args.epochs,
                'lr':          args.lr,
                'batch_size':  args.batch_size,
                'hidden_dim':  args.hidden_dim,
                'window_size': args.window_size,
                'step_size':   args.step_size,
            },
            log_file=lf,
        )

    if result:
        print(f"\nDone.  Best val acc : {result['best_val_acc']:.2f}%")
        print(f"Log               → {log_path}")
