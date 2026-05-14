"""Small 1D-CNN for primary-class Raman classification.

Per plan/03_architecture.md sec E (small variant, default).

Channel doubling decision (2026-05-14, see plan/10_decision_log.md):
The doc lists channels (16, 32, 48, 64) AND a '~110K params' target
in the same sentence. Those two facts are internally inconsistent --
(16, 32, 48, 64) with kernels 15-7-7-5 arithmetic-out to ~33K, not
110K. Empirically the 33K version cannot fit even the training set
(train_acc plateaus at 0.74 after 120 epochs on Protocol A fold 0;
val_macro_f1 ceiling 0.56 vs classical PLS-DA 0.951). To preserve the
spec's intent (small = ~110K, medium = ~450K) we DOUBLE the channels
to (32, 64, 96, 128) here. Medium remains differentiated by its extra
dilation (2 and 4), GAP+GMP concat, and 2-layer MLP head.

Final architecture:
    Input  : (B, 1, ~987). InstanceNorm1d at input.
    Stages : 4 convs, channels 32 -> 64 -> 96 -> 128,
             kernels 15-7-7-5, GELU + BatchNorm,
             MaxPool/2 after stages 1, 2, 3 (none after stage 4),
             dilation 2 in stage 3.
    Head   : GAP -> Linear(128 -> 32) -> GELU -> Linear(32 -> 4).

`SmallCNN1D.encode()` returns the 32-dim penultimate activations --
used by the memprobe v2 to score the file-id leakage of the *trained
class encoder* (vs the from-scratch tiny network used in v1).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallCNN1D(nn.Module):
    def __init__(self, n_bins: int = 987, n_classes: int = 4):
        super().__init__()
        self.n_bins = n_bins
        self.n_classes = n_classes

        # Per-bin standardization, fit on the OUTER-train of each fold and
        # baked in via `set_input_stats`. Saved/restored as buffers in
        # state_dict so the memprobe v2 sees the SAME input pipeline as the
        # class-trained encoder. Defaults to (0, 1) which is a no-op.
        self.register_buffer("input_mu", torch.zeros(n_bins))
        self.register_buffer("input_sd", torch.ones(n_bins))

        # InstanceNorm1d at the front is a belt-and-suspenders pass for any
        # residual within-spectrum drift; mostly a no-op after per-bin
        # standardize and SNV but cheap to keep.
        self.input_norm = nn.InstanceNorm1d(1, affine=False)

        # Stage 1: 1 -> 32, k=15, padding=7 (same length)
        self.conv1 = nn.Conv1d(1, 32, kernel_size=15, padding=7)
        self.bn1 = nn.BatchNorm1d(32)

        # Stage 2: 32 -> 64, k=7, padding=3
        self.conv2 = nn.Conv1d(32, 64, kernel_size=7, padding=3)
        self.bn2 = nn.BatchNorm1d(64)

        # Stage 3: 64 -> 96, k=7, dilation=2 (effective rf 13), padding=6 (same)
        self.conv3 = nn.Conv1d(64, 96, kernel_size=7, padding=6, dilation=2)
        self.bn3 = nn.BatchNorm1d(96)

        # Stage 4: 96 -> 128, k=5, padding=2
        self.conv4 = nn.Conv1d(96, 128, kernel_size=5, padding=2)
        self.bn4 = nn.BatchNorm1d(128)

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, n_classes)

    def set_input_stats(self, mu: torch.Tensor | np.ndarray, sd: torch.Tensor | np.ndarray) -> None:
        """Bake per-bin (mu, sd) into the model. Call once per fold after
        fitting on outer-train; values are saved with state_dict.
        """
        if not isinstance(mu, torch.Tensor):
            mu = torch.as_tensor(mu, dtype=torch.float32)
        if not isinstance(sd, torch.Tensor):
            sd = torch.as_tensor(sd, dtype=torch.float32)
        self.input_mu.data = mu.to(self.input_mu.device).float()
        self.input_sd.data = sd.to(self.input_sd.device).float()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the 32-dim penultimate activations (post-fc1, post-GELU).

        x: (B, 1, n_bins). Per-bin standardize is applied first using the
        baked-in (input_mu, input_sd) buffers.
        """
        x = (x - self.input_mu.view(1, 1, -1)) / self.input_sd.view(1, 1, -1)
        x = self.input_norm(x)
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, 2)
        x = F.gelu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, 2)
        x = F.gelu(self.bn3(self.conv3(x)))
        x = F.max_pool1d(x, 2)
        x = F.gelu(self.bn4(self.conv4(x)))
        x = self.gap(x).squeeze(-1)  # (B, 64)
        x = F.gelu(self.fc1(x))       # (B, 32)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.encode(x))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def select_device(prefer_mps: bool = True) -> torch.device:
    """Auto-detect best PyTorch device. MPS on Apple Silicon by default."""
    if prefer_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
