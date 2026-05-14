"""Small 1D-Transformer for primary-class Raman classification.

Per plan/03_architecture.md sec F (~200K params target):
    Patch-embed (~20-bin patches) -> learned positional encoding ->
    [CLS] token -> 4 encoder blocks -> classifier head.

Two carry-overs from the CNN session (see plan/10_decision_log.md
§cnn-architectural-fixes-mid-session):

1. **Per-bin standardize at input.** SNV makes per-row mean=0/std=1 but
   per-bin mean ranges -0.46 to +3.84 across the 987 bins. Without
   per-bin standardize the CNN couldn't fit train data; the Transformer's
   patch-embed Conv1d does NOT absorb this (it learns weights against a
   non-stationary input). Carry the same `register_buffer` design.

2. **Sanity-check param count vs spec.** The CNN spec listed (16,32,48,64)
   and ~110K params in the same paragraph — the channel widths
   arithmetic-out to 33K. Here d_model=80, nhead=4, dim_feedforward=160,
   num_layers=4 lands near 210K, matching the spec's ~200K target. Counts
   logged at first instantiation.

Architecture details:
    Input        : (B, 1, ~987) -> per-bin standardize via buffer.
    Patch-embed  : Conv1d(1, d_model, k=patch_size, s=patch_size).
                   Patch=20 -> floor((987-20)/20)+1 = 49 tokens.
    [CLS] token  : learnable (1, 1, d_model) prepended.
    Pos embed    : learned (1, n_tokens+1, d_model).
    Encoder      : nn.TransformerEncoder, 4 layers, GELU activation,
                   pre-LN (norm_first=True) — stabler on small datasets.
    Head         : LayerNorm -> Linear(d_model, 32) -> GELU
                   -> Linear(32, n_classes). The 32-dim activation pre-
                   logits is the `encode()` output, mirroring SmallCNN1D.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallTransformer1D(nn.Module):
    def __init__(
        self,
        n_bins: int = 987,
        n_classes: int = 4,
        patch_size: int = 20,
        d_model: int = 80,
        nhead: int = 4,
        dim_feedforward: int = 160,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_bins = n_bins
        self.n_classes = n_classes
        self.patch_size = patch_size
        self.d_model = d_model

        # Per-bin standardize, same contract as SmallCNN1D.
        self.register_buffer("input_mu", torch.zeros(n_bins))
        self.register_buffer("input_sd", torch.ones(n_bins))

        # Patch-embed via strided Conv1d.
        # n_patches = floor((n_bins - patch_size) / patch_size) + 1.
        n_patches = (n_bins - patch_size) // patch_size + 1
        self.n_patches = n_patches
        self.patch_embed = nn.Conv1d(
            1, d_model, kernel_size=patch_size, stride=patch_size
        )

        # Learnable [CLS] token and positional embedding.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Encoder stack. norm_first=True (pre-LN) is the more stable
        # configuration on small data per modern Transformer best practice.
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # Head: LN over the CLS embedding, then 32-dim bottleneck so
        # encode() shape matches SmallCNN1D for downstream memprobe v2.
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, 32)
        self.fc2 = nn.Linear(32, n_classes)

    def set_input_stats(self, mu: torch.Tensor | np.ndarray, sd: torch.Tensor | np.ndarray) -> None:
        """Bake per-bin (mu, sd) into model buffers."""
        if not isinstance(mu, torch.Tensor):
            mu = torch.as_tensor(mu, dtype=torch.float32)
        if not isinstance(sd, torch.Tensor):
            sd = torch.as_tensor(sd, dtype=torch.float32)
        self.input_mu.data = mu.to(self.input_mu.device).float()
        self.input_sd.data = sd.to(self.input_sd.device).float()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the 32-dim penultimate activations (post-fc1, post-GELU).

        x: (B, 1, n_bins). Same shape contract as SmallCNN1D.encode.
        """
        x = (x - self.input_mu.view(1, 1, -1)) / self.input_sd.view(1, 1, -1)
        # Patch-embed: (B, 1, L) -> (B, d_model, n_patches) -> (B, n_patches, d_model)
        x = self.patch_embed(x).transpose(1, 2)
        B = x.size(0)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)            # (B, n_patches+1, d_model)
        x = x + self.pos_embed
        x = self.encoder(x)                        # (B, n_patches+1, d_model)
        cls_out = self.norm(x[:, 0])               # (B, d_model)
        feat = F.gelu(self.fc1(cls_out))           # (B, 32)
        return feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.encode(x))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
