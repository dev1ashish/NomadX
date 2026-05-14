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


# ----------------------------------------------------------------------------
# DANN: Gradient Reversal Layer + domain-adversarial CNN
# ----------------------------------------------------------------------------
#
# Why two flavors live in the same file:
#   - The DANN variant IS the SmallCNN1D encoder plus a parallel domain head.
#     Subclassing keeps `set_input_stats` / `encode` / `forward` (used by
#     memprobe v2) identical, so the same memprobe code works on either.
#   - The domain head is registered AFTER `super().__init__()` so the encoder
#     and class-head parameters consume the SAME RNG draws as vanilla
#     SmallCNN1D. With lambda_max=0 the DANN encoder trains identically to
#     vanilla, up to MPS numerical noise (verified in sanity check 1).


class GradReverse(torch.autograd.Function):
    """Identity in the forward pass; negates and scales by `lambda_grl` in
    the backward pass.

    Standard Ganin & Lempitsky 2015 formulation: encoder sees
    `-lambda_grl * dL_domain/dfeat`, while the domain head sees the unscaled
    gradient on its own weights. lambda_grl is a Python float passed at
    forward time so the schedule (warmup) can change every step without
    rebuilding the graph.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_grl: float) -> torch.Tensor:
        ctx.lambda_grl = float(lambda_grl)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambda_grl * grad_output, None


def grad_reverse(x: torch.Tensor, lambda_grl: float) -> torch.Tensor:
    return GradReverse.apply(x, lambda_grl)


class DANNCNN1D(SmallCNN1D):
    """SmallCNN1D + a parallel domain classifier head fed by GRL.

    The encoder and class head are inherited unchanged; init order is
    identical to SmallCNN1D, so given the same fold_seed the encoder's
    initial weights match bit-for-bit. Domain head MLP is appended last so
    its init draws don't perturb anything upstream.

    Args:
        n_bins:    spectrum length (default 987).
        n_classes: number of primary classes (default 4).
        n_domains: number of unique file_ids in the outer-train -- this
                   varies per fold, so the factory in scripts/run_dann.py
                   sets it per fold.
        domain_hidden: hidden width of the domain MLP. Default 64 -- has
                   enough capacity to actually fight the encoder; if too
                   shallow the discriminator loses trivially and the
                   adversarial signal evaporates.
    """

    def __init__(
        self,
        n_bins: int = 987,
        n_classes: int = 4,
        n_domains: int = 87,
        domain_hidden: int = 64,
    ):
        super().__init__(n_bins=n_bins, n_classes=n_classes)
        self.n_domains = n_domains
        # MUST be created AFTER super().__init__() -- see module docstring.
        self.domain_head = nn.Sequential(
            nn.Linear(32, domain_hidden),
            nn.GELU(),
            nn.Linear(domain_hidden, n_domains),
        )

    def forward_with_domain(
        self, x: torch.Tensor, lambda_grl: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (class_logits, domain_logits).

        Encoder gets `-lambda_grl * dL_domain/dfeat` via the GRL. Domain head
        weights get the unscaled +dL_domain/dparams.
        """
        feat = self.encode(x)                  # (B, 32)
        class_logits = self.fc2(feat)          # (B, n_classes)
        reversed_feat = grad_reverse(feat, lambda_grl)
        domain_logits = self.domain_head(reversed_feat)  # (B, n_domains)
        return class_logits, domain_logits

    # forward() is inherited unchanged: SmallCNN1D.forward returns class
    # logits only. This is what memprobe v2 / encode_dataset rely on.
