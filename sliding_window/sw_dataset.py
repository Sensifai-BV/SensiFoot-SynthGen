"""
sw_dataset.py
=============
Sliding-window dataset and video-level collate function.
Shared by sw_trainer.py and sw_evaluator.py.

Data representation
───────────────────
Training  : one CSV → N overlapping windows of fixed length W.
            Each window is an independent training sample.
            Label is the same for all windows from the same file.

Inference : one CSV → N windows → average softmax probabilities
            → single video-level prediction.
            (Video ID is tracked so the evaluator can group windows.)

Window parameters (set once in CONFIG, passed through):
    WINDOW_SIZE : int  – number of frames per window  (default 60)
    STEP_SIZE   : int  – stride between windows        (default 5)

If a sequence is shorter than WINDOW_SIZE it is zero-padded on the right.

Augmentation (training only):
    • Gaussian jitter  (p=0.7) on the first 30 feature columns
    • Random frame zeroing (p=0.3, 1-3 frames) on the first 30 columns
"""

import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SlidingWindowDataset(Dataset):
    """
    Windowed dataset.

    Parameters
    ----------
    file_list    : list of (filepath: str, class_name: str)
    class_to_idx : dict  class_name → int label (0-indexed)
    extractor_fn : callable(pd.DataFrame) → np.ndarray (T, D)
    window_size  : int   frames per window
    step_size    : int   stride between window starts
    augment      : bool  apply jitter + frame dropout
    """

    def __init__(self, file_list, class_to_idx: dict,
                 extractor_fn, window_size: int, step_size: int,
                 augment: bool = False):
        self.window_size = window_size
        self.step_size   = step_size
        self.augment     = augment

        # Each entry: (np.ndarray window (W,D), int label, int video_id)
        self.windows   = []
        self.labels    = []
        self.video_ids = []

        skipped = 0
        for vid_id, (fp, cls_name) in enumerate(file_list):
            arr = self._load(fp, extractor_fn)
            if arr is None:
                skipped += 1
                continue
            label = class_to_idx[cls_name]
            for w in self._slice(arr):
                self.windows.append(w)
                self.labels.append(label)
                self.video_ids.append(vid_id)

        if skipped:
            print(f"  [Dataset] Skipped {skipped} unreadable files.")

    # ── internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _load(fp, extractor_fn):
        try:
            df  = pd.read_csv(fp, header=0)
            arr = extractor_fn(df)
            if arr is None or len(arr) == 0:
                return None
            return arr.astype(np.float32)
        except Exception as e:
            print(f"  [WARN] {fp}: {e}")
            return None

    def _slice(self, arr: np.ndarray):
        """Yield fixed-length windows; pad short sequences once."""
        T, D = arr.shape
        W    = self.window_size
        S    = self.step_size

        if T < W:
            pad = np.zeros((W - T, D), dtype=np.float32)
            arr = np.vstack([arr, pad])
            T   = W

        starts = range(0, T - W + 1, S)
        if not list(starts):          # edge case: exactly W frames
            starts = [0]
        for start in starts:
            yield arr[start:start + W].copy()

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        data  = self.windows[idx].copy()
        label = self.labels[idx]
        vid   = self.video_ids[idx]

        if self.augment:
            n = min(30, data.shape[1])
            if random.random() < 0.7:
                data[:, :n] += np.random.normal(
                    0, 0.015, (data.shape[0], n)).astype(np.float32)
            if random.random() < 0.3:
                for _ in range(random.randint(1, 3)):
                    data[random.randint(0, len(data) - 1), :n] = 0.0

        return (torch.tensor(data, dtype=torch.float32),
                torch.tensor(label, dtype=torch.long),
                vid)


def collate_train(batch):
    """
    Collate for training: windows are all the same length, so no padding needed.
    Returns (x: B×W×D,  y: B).
    video_id is dropped — not needed during training.
    """
    xs, ys, _ = zip(*batch)
    return torch.stack(xs), torch.stack(ys)


def collate_eval(batch):
    """
    Collate for evaluation: keeps video_id so the evaluator can group windows.
    Returns (x: B×W×D,  y: B,  video_ids: list[int]).
    """
    xs, ys, vids = zip(*batch)
    return torch.stack(xs), torch.stack(ys), list(vids)
