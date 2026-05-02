"""
models_zoo.py
=============
All four model architectures used in the expanded ablation study.

Every model exposes an identical interface so the trainer and evaluator
never need to branch on architecture type:

    logits            = model(x, lengths)
    embedding         = model.get_embedding(x, lengths)

where
    x       : (B, T, F)   zero-padded batch of full sequences
    lengths : (B,)  int64  true frame counts before padding
    logits  : (B, num_classes)
    embedding : (B, hidden_dim)  – pre-classifier feature vector for SupCon

─────────────────────────────────────────────────────────────────────────────
Models
─────────────────────────────────────────────────────────────────────────────
1. CNN_LSTM   – Causal Conv1D → LSTM → Temporal Attention
               (exact replica of GestureModelFullSeq; kept here so all
                architectures live in one file)

2. TCN        – Pure Temporal Convolutional Network
               Stacked dilated causal residual blocks → global attention pool
               Receptive field grows as 2^(layer) so deep context is captured
               without any recurrence.

3. BiGRU      – Bidirectional GRU → Temporal Attention
               No CNN front-end; the bidirectional hidden states at each frame
               are attention-pooled into a single context vector.
               Bidirectional is safe here because the full sequence is always
               available at inference time (offline classification).

4. STGCN      – Spatial-Temporal Graph Convolutional Network
               Strictly consumes raw 3D joint coordinates (no feature
               engineering). Used as a standalone 19th benchmark to test
               whether graph topology adds value over engineered features.

               Skeleton graph: 12 joints
                   0 L_Hip    1 R_Hip    2 L_Knee   3 R_Knee
                   4 L_Ankle  5 R_Ankle  6 L_Heel   7 R_Heel
                   8 L_Toe    9 R_Toe   10 L_Shoulder 11 R_Shoulder

               Input: (B, T, 12, 3)  – 12 joints × 3 coordinates
               The STGCN feature extractor (extract_stgcn) is in
               feature_extractors.py and returns shape (T, 36), which is
               then reshaped inside the model to (T, 12, 3).

─────────────────────────────────────────────────────────────────────────────
Factory
─────────────────────────────────────────────────────────────────────────────
    from models_zoo import build_model
    model = build_model(arch='TCN', input_dim=34, hidden_dim=64, num_classes=8)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═════════════════════════════════════════════════════════════════════════════
#  Shared utilities
# ═════════════════════════════════════════════════════════════════════════════

class _MaskedAttention(nn.Module):
    """
    Additive attention that masks zero-padded time steps.
    Input  : (B, T, H)
    Output : context (B, H), weights (B, T, 1)
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None):
        scores = self.attn(x)                              # (B, T, 1)
        if lengths is not None:
            B, T, _ = scores.shape
            mask = (torch.arange(T, device=x.device).unsqueeze(0)
                    >= lengths.unsqueeze(1))
            scores = scores.masked_fill(mask.unsqueeze(2), float('-inf'))
        weights = torch.softmax(scores, dim=1)             # (B, T, 1)
        context = (weights * x).sum(dim=1)                 # (B, H)
        return context, weights


def _pack_rnn(rnn, x, lengths):
    """Pack → RNN → unpack helper. Returns output (B, T, H)."""
    if lengths is not None:
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
    else:
        out, _ = rnn(x)
    return out


# ═════════════════════════════════════════════════════════════════════════════
#  1. CNN-LSTM  (Causal Conv1D + LSTM + Temporal Attention)
# ═════════════════════════════════════════════════════════════════════════════

class CNN_LSTM(nn.Module):
    """
    Causal CNN-LSTM with masked Temporal Attention.
    Exact replica of the original GestureModelFullSeq.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_classes: int = 8):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.ConstantPad1d((2, 0), 0),
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.rnn  = nn.LSTM(64, hidden_dim, num_layers=1,
                            batch_first=True, bidirectional=False)
        self.attn = _MaskedAttention(hidden_dim)
        self.drop = nn.Dropout(0.3)
        self.fc   = nn.Linear(hidden_dim, num_classes)

    def _encode(self, x, lengths):
        x = self.cnn(x.transpose(1, 2)).transpose(1, 2)   # (B, T, 64)
        x = _pack_rnn(self.rnn, x, lengths)                # (B, T, H)
        ctx, _ = self.attn(x, lengths)                     # (B, H)
        return ctx

    def forward(self, x, lengths=None):
        return self.fc(self.drop(self._encode(x, lengths)))

    def get_embedding(self, x, lengths=None):
        return self._encode(x, lengths)


# ═════════════════════════════════════════════════════════════════════════════
#  2. TCN  (Pure Temporal Convolutional Network)
# ═════════════════════════════════════════════════════════════════════════════

class _TCNBlock(nn.Module):
    """
    Single dilated causal residual block.
    dilation doubles at each layer so the receptive field grows as 1,2,4,8…
    Causal padding: pad (kernel-1)*dilation on the left only.
    """
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, dilation: int = 1,
                 dropout: float = 0.2):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.net = nn.Sequential(
            nn.ConstantPad1d((pad, 0), 0),
            nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation, padding=0),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.ConstantPad1d((pad, 0), 0),
            nn.Conv1d(out_ch, out_ch, kernel, dilation=dilation, padding=0),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = (nn.Conv1d(in_ch, out_ch, 1)
                          if in_ch != out_ch else nn.Identity())
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.net(x) + self.downsample(x))


class TCN(nn.Module):
    """
    Stacked dilated causal TCN with masked attention pooling.
    hidden_dim is the channel width throughout the TCN stack.
    4 blocks with dilations [1, 2, 4, 8] give receptive field of 57 frames
    at kernel=3, more than enough for sequences of 100-300 frames.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_classes: int = 8):
        super().__init__()
        dilations = [1, 2, 4, 8]
        layers = []
        in_ch = input_dim
        for d in dilations:
            layers.append(_TCNBlock(in_ch, hidden_dim, kernel=3,
                                    dilation=d, dropout=0.2))
            in_ch = hidden_dim
        self.tcn  = nn.Sequential(*layers)
        self.attn = _MaskedAttention(hidden_dim)
        self.drop = nn.Dropout(0.3)
        self.fc   = nn.Linear(hidden_dim, num_classes)

    def _encode(self, x, lengths):
        # x: (B, T, F) → (B, F, T) for Conv1d → (B, T, H)
        out = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        ctx, _ = self.attn(out, lengths)
        return ctx

    def forward(self, x, lengths=None):
        return self.fc(self.drop(self._encode(x, lengths)))

    def get_embedding(self, x, lengths=None):
        return self._encode(x, lengths)


# ═════════════════════════════════════════════════════════════════════════════
#  3. BiGRU  (Bidirectional GRU + Temporal Attention)
# ═════════════════════════════════════════════════════════════════════════════

class BiGRU(nn.Module):
    """
    Bidirectional GRU with masked Temporal Attention pooling.
    No CNN front-end.
    hidden_dim is the size of each direction; concatenated output is 2*hidden_dim.
    A linear projection reduces it back to hidden_dim before the classifier
    so the embedding dimension is consistent with the other models.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_classes: int = 8):
        super().__init__()
        self.rnn  = nn.GRU(input_dim, hidden_dim, num_layers=2,
                           batch_first=True, bidirectional=True,
                           dropout=0.2)
        self.proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.attn = _MaskedAttention(hidden_dim)
        self.drop = nn.Dropout(0.3)
        self.fc   = nn.Linear(hidden_dim, num_classes)

    def _encode(self, x, lengths):
        out = _pack_rnn(self.rnn, x, lengths)          # (B, T, 2H)
        out = torch.relu(self.proj(out))               # (B, T, H)
        ctx, _ = self.attn(out, lengths)               # (B, H)
        return ctx

    def forward(self, x, lengths=None):
        return self.fc(self.drop(self._encode(x, lengths)))

    def get_embedding(self, x, lengths=None):
        return self._encode(x, lengths)


# ═════════════════════════════════════════════════════════════════════════════
#  4. ST-GCN  (Spatial-Temporal Graph Convolutional Network)
# ═════════════════════════════════════════════════════════════════════════════

# ── Skeleton adjacency ────────────────────────────────────────────────────────
#
#  12 joints, 0-indexed:
#   0=L_Hip  1=R_Hip  2=L_Knee  3=R_Knee  4=L_Ankle  5=R_Ankle
#   6=L_Heel 7=R_Heel 8=L_Toe   9=R_Toe  10=L_Shoulder 11=R_Shoulder
#
#  Edges: anatomical connections + across-body hip/shoulder links
_STGCN_EDGES = [
    (0, 1),   # hip cross
    (0, 2),   # L_Hip  → L_Knee
    (1, 3),   # R_Hip  → R_Knee
    (2, 4),   # L_Knee → L_Ankle
    (3, 5),   # R_Knee → R_Ankle
    (4, 6),   # L_Ankle→ L_Heel
    (5, 7),   # R_Ankle→ R_Heel
    (4, 8),   # L_Ankle→ L_Toe
    (5, 9),   # R_Ankle→ R_Toe
    (0, 10),  # L_Hip  → L_Shoulder
    (1, 11),  # R_Hip  → R_Shoulder
    (10, 11), # shoulder cross
]
_N_JOINTS = 12


def _build_adj(n: int, edges: list, device='cpu') -> torch.Tensor:
    """Normalised symmetric adjacency matrix (n, n) with self-loops."""
    A = torch.zeros(n, n)
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0
    A = A + torch.eye(n)                    # self-loops
    d = A.sum(dim=1, keepdim=True).clamp(min=1e-6)
    A = A / d                               # row-normalise
    return A.to(device)


class _STGCNBlock(nn.Module):
    """
    One ST-GCN layer.
    Spatial step  : graph convolution over joint dimension
    Temporal step : depthwise Conv1d with causal padding over the time axis
    """
    def __init__(self, in_ch: int, out_ch: int, n_joints: int,
                 t_kernel: int = 9, dropout: float = 0.2):
        super().__init__()
        # Spatial graph conv: (B, C, T, V) → apply weight per joint → (B, C', T, V)
        self.gcn_weight = nn.Parameter(torch.empty(in_ch, out_ch, 1))
        nn.init.xavier_uniform_(self.gcn_weight)

        # Temporal conv: causal, operates per-joint along time
        t_pad = t_kernel - 1
        self.tcn = nn.Sequential(
            nn.ConstantPad2d((0, 0, t_pad, 0), 0),   # pad time dim only
            nn.Conv2d(out_ch, out_ch, (t_kernel, 1),
                      padding=0, groups=out_ch),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Dropout2d(dropout),
        )
        self.bn_gcn = nn.BatchNorm2d(out_ch)
        self.relu   = nn.ReLU()
        self.res    = (nn.Conv2d(in_ch, out_ch, 1)
                       if in_ch != out_ch else nn.Identity())

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        x : (B, C, T, V)
        A : (V, V)  normalised adjacency
        Returns (B, C', T, V)
        """
        B, C, T, V = x.shape
        # Spatial: x @ A then linear mix over channels
        # Reshape for matmul: (B*C, T, V) @ (V, V) → (B*C, T, V)
        xs = x.view(B * C, T, V)
        xs = torch.bmm(xs, A.unsqueeze(0).expand(B * C, -1, -1))
        xs = xs.view(B, C, T, V)
        # Channel mix: einsum over in_ch → out_ch
        xs = torch.einsum('bctv,cov->botv', xs,
                          self.gcn_weight.expand(-1, -1, 1)).contiguous()
        xs = self.bn_gcn(xs)
        xs = self.relu(xs)
        # Temporal
        out = self.tcn(xs)
        return self.relu(out + self.res(x))


class STGCN(nn.Module):
    """
    Spatial-Temporal Graph CNN for skeleton gesture recognition.

    Input shape  : (B, T, 36)  where the 36 = 12 joints × 3 coords (X,Y,Z).
                   The feature extractor `extract_stgcn` in feature_extractors.py
                   produces this shape.
    Architecture : 3 ST-GCN blocks, global avg-pool over (T, V), FC classifier.
    """
    def __init__(self, input_dim: int = 36, hidden_dim: int = 64,
                 num_classes: int = 8):
        super().__init__()
        # input_dim must be 36 (12 joints × 3 coords)
        assert input_dim == 36, (
            f"STGCN expects input_dim=36 (12 joints × 3 coords), got {input_dim}."
            " Use phase='STGCN' and the extract_stgcn feature extractor.")

        self.n_joints = _N_JOINTS
        # Adjacency is fixed; register as buffer so it moves with .to(device)
        self.register_buffer('A', _build_adj(_N_JOINTS, _STGCN_EDGES))

        # Data batch-norm on raw coordinates (3 channels, one per XYZ)
        self.data_bn = nn.BatchNorm1d(_N_JOINTS * 3)

        in_ch = 3   # XYZ per joint
        self.blocks = nn.ModuleList([
            _STGCNBlock(in_ch,       64, _N_JOINTS, t_kernel=9, dropout=0.2),
            _STGCNBlock(64,          64, _N_JOINTS, t_kernel=9, dropout=0.2),
            _STGCNBlock(64, hidden_dim, _N_JOINTS, t_kernel=9, dropout=0.2),
        ])

        self.drop = nn.Dropout(0.3)
        self.fc   = nn.Linear(hidden_dim, num_classes)

    def _encode(self, x: torch.Tensor, lengths=None) -> torch.Tensor:
        """
        x       : (B, T, 36)
        lengths : ignored for STGCN (global avg pool handles variable lengths
                  implicitly; masked avg-pool is used when lengths are provided)
        Returns : (B, hidden_dim)
        """
        B, T, _ = x.shape
        V = self.n_joints

        # BN on flattened joints
        x_bn = self.data_bn(x.reshape(B * T, _))
        x_bn = x_bn.reshape(B, T, V, 3)

        # Rearrange to (B, C, T, V) for ST-GCN blocks
        x_out = x_bn.permute(0, 3, 1, 2).contiguous()   # (B, 3, T, V)

        for block in self.blocks:
            x_out = block(x_out, self.A)                 # (B, C', T, V)

        # Global average pool over (T, V)
        if lengths is not None:
            # Masked mean: zero out padding frames first
            mask = (torch.arange(T, device=x.device)
                    .unsqueeze(0) < lengths.unsqueeze(1))  # (B, T)
            mask = mask.float().unsqueeze(1).unsqueeze(-1)  # (B,1,T,1)
            x_out = (x_out * mask).sum(dim=(2, 3)) / (
                lengths.float().unsqueeze(1) * V + 1e-6)    # (B, C')
        else:
            x_out = x_out.mean(dim=(2, 3))               # (B, C')

        return x_out                                     # (B, hidden_dim)

    def forward(self, x: torch.Tensor, lengths=None) -> torch.Tensor:
        return self.fc(self.drop(self._encode(x, lengths)))

    def get_embedding(self, x: torch.Tensor, lengths=None) -> torch.Tensor:
        return self._encode(x, lengths)

# ═════════════════════════════════════════════════════════════════════════════
#  SupCon loss (shared across all architectures)
# ═════════════════════════════════════════════════════════════════════════════

class SupConLoss(nn.Module):
    """Supervised Contrastive Loss – Khosla et al., NeurIPS 2020."""
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device   = features.device
        B        = features.shape[0]
        features = F.normalize(features, dim=1)
        sim      = torch.matmul(features, features.T) / self.temperature
        self_mask = torch.eye(B, dtype=torch.bool, device=device)
        sim.masked_fill_(self_mask, float('-inf'))
        labels   = labels.view(-1, 1)
        pos_mask = (labels == labels.T) & ~self_mask
        log_prob = F.log_softmax(sim, dim=1)
        pos_cnt  = pos_mask.sum(dim=1).float().clamp(min=1)
        loss     = -(log_prob * pos_mask.float()).sum(dim=1) / pos_cnt
        return loss.mean()


# ═════════════════════════════════════════════════════════════════════════════
#  Factory
# ═════════════════════════════════════════════════════════════════════════════

ARCH_REGISTRY = {
    'CNN_LSTM': CNN_LSTM,
    'TCN':      TCN,
    'BiGRU':    BiGRU,
    'STGCN':    STGCN,
}


def build_model(arch: str, input_dim: int, hidden_dim: int = 64,
                num_classes: int = 8) -> nn.Module:
    """
    Instantiate a model by architecture name.
    All models share the same (input_dim, hidden_dim, num_classes) signature.
    """
    if arch not in ARCH_REGISTRY:
        raise ValueError(f"Unknown architecture '{arch}'. "
                         f"Choose from: {list(ARCH_REGISTRY.keys())}")
    return ARCH_REGISTRY[arch](input_dim, hidden_dim, num_classes)
