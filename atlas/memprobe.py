"""Memorization probe: tiny 1D-CNN that predicts file_id from a single spectrum.

The question: how much per-file acquisition signature is left in the preprocessed
spectra after our cleaning pipeline? If a CNN can distinguish 87 individual files
from one spectrum each, that signature exists and our class-level CNN will be
tempted to use it (= leak across the bacterial-class boundary via calibration_date
batch effect).

Design:
    - Input:  (B, 1, 987) preprocessed SNV spectra.
    - Output: 87-way softmax (one class per file_id).
    - Split:  WITHIN each file, 80% train / 20% test. So every file_id is in both.
              That way we can measure "does the spectrum carry a file fingerprint?"
              -- NOT "do held-out files share traits with training files" (which is
              what our LOSO already tests).
    - Chance: 1/87 = 1.15%.
    - Threshold per plan/02_decisions.md: > 10% accuracy => enable DANN for CNN.

Model: 3-conv-block tiny network ~20K params. Tiny on purpose -- if even this
overfits to file fingerprint, file signature is a strong leakage risk.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

# Pin BLAS threads BEFORE numpy import.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


REPO = Path(__file__).resolve().parent.parent


def select_device(prefer_mps: bool = True) -> torch.device:
    if prefer_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class TinyCNN(nn.Module):
    """Small 1D-CNN for file_id classification (~20K params)."""

    def __init__(self, n_classes: int, n_bins: int = 987):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 8, kernel_size=15, padding=7)
        self.bn1 = nn.BatchNorm1d(8)
        self.conv2 = nn.Conv1d(8, 16, kernel_size=7, padding=3)
        self.bn2 = nn.BatchNorm1d(16)
        self.conv3 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(32)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(32, n_classes)

    def forward(self, x):  # x: (B, 1, n_bins)
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, 2)
        x = F.gelu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, 2)
        x = F.gelu(self.bn3(self.conv3(x)))
        x = self.gap(x).squeeze(-1)
        return self.head(x)


def build_within_file_split(spec_df: pd.DataFrame, qc_mask: np.ndarray, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Within each file, 80% of QC-passing pixels go to train, 20% to test.

    Returns (train_idx, test_idx, file_id_int_per_row, file_id_list).
    Indices are into the 7,999-row preprocessed array.
    """
    rng = np.random.default_rng(seed)
    file_id_arr = spec_df["file_id"].to_numpy()
    qc_indices = np.where(qc_mask)[0]
    unique_files = sorted(set(file_id_arr[qc_indices]))
    file_to_int = {f: i for i, f in enumerate(unique_files)}

    train_idx = []
    test_idx = []
    for f in unique_files:
        rows = qc_indices[file_id_arr[qc_indices] == f]
        rng.shuffle(rows)
        n_test = max(1, int(0.2 * len(rows)))
        test_idx.extend(rows[:n_test].tolist())
        train_idx.extend(rows[n_test:].tolist())

    train_idx = np.array(sorted(train_idx), dtype=np.int64)
    test_idx = np.array(sorted(test_idx), dtype=np.int64)
    file_id_int_per_row = np.array([file_to_int[f] for f in file_id_arr], dtype=np.int64)
    return train_idx, test_idx, file_id_int_per_row, unique_files


def run_memprobe(
    *,
    cache_dir: Path = REPO / "data_cache",
    seed: int = 42,
    n_epochs: int = 15,
    batch_size: int = 256,
    lr: float = 3e-4,
    log_fn=print,
) -> dict:
    log_fn(f"[memprobe] loading data from {cache_dir}")
    spec_df = pd.read_parquet(cache_dir / "spectra.parquet")
    X = np.load(cache_dir / "spectra_array_preprocessed.npy")
    qc_mask = np.load(cache_dir / "qc_mask.npy")

    train_idx, test_idx, file_id_int, unique_files = build_within_file_split(spec_df, qc_mask, seed)
    n_classes = len(unique_files)
    log_fn(f"[memprobe] n_files={n_classes}  n_train_pixels={train_idx.size}  n_test_pixels={test_idx.size}  chance={1/n_classes:.4f}")

    device = select_device()
    log_fn(f"[memprobe] device={device}")

    torch.manual_seed(seed)
    if device.type == "mps":
        torch.mps.manual_seed(seed) if hasattr(torch.mps, "manual_seed") else None

    # Build tensors
    X_train_t = torch.from_numpy(X[train_idx]).float().unsqueeze(1)  # (N_tr, 1, 987)
    y_train_t = torch.from_numpy(file_id_int[train_idx])
    X_test_t = torch.from_numpy(X[test_idx]).float().unsqueeze(1)
    y_test_t = torch.from_numpy(file_id_int[test_idx])

    train_ds = TensorDataset(X_train_t, y_train_t)
    test_ds = TensorDataset(X_test_t, y_test_t)
    gen = torch.Generator()
    gen.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=gen)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = TinyCNN(n_classes=n_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log_fn(f"[memprobe] model params: {n_params:,}")

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    history = []
    t0 = time.perf_counter()
    for epoch in range(n_epochs):
        model.train()
        loss_sum = 0.0
        n_train_seen = 0
        n_train_correct = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optim.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            optim.step()
            loss_sum += float(loss) * xb.size(0)
            n_train_seen += xb.size(0)
            n_train_correct += int((logits.argmax(dim=1) == yb).sum().item())

        # Eval
        model.eval()
        n_test_correct = 0
        n_test_seen = 0
        top5_correct = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                pred = logits.argmax(dim=1)
                n_test_correct += int((pred == yb).sum().item())
                n_test_seen += xb.size(0)
                top5 = logits.topk(5, dim=1).indices
                top5_correct += int((top5 == yb.unsqueeze(1)).any(dim=1).sum().item())

        train_acc = n_train_correct / n_train_seen
        test_acc = n_test_correct / n_test_seen
        top5_acc = top5_correct / n_test_seen
        history.append({"epoch": epoch, "train_loss": loss_sum / n_train_seen,
                        "train_acc": train_acc, "test_acc": test_acc, "top5_acc": top5_acc})
        log_fn(f"  epoch {epoch+1:>2}: train_loss={loss_sum/n_train_seen:.3f}  train_acc={train_acc:.3f}  test_acc={test_acc:.3f}  top5_acc={top5_acc:.3f}")

    duration = time.perf_counter() - t0
    log_fn(f"[memprobe] done in {duration:.1f}s  final test_acc={history[-1]['test_acc']:.3f}  top5={history[-1]['top5_acc']:.3f}")

    return {
        "n_files": n_classes,
        "chance_acc": 1.0 / n_classes,
        "final_test_acc": history[-1]["test_acc"],
        "final_top5_acc": history[-1]["top5_acc"],
        "history": history,
        "duration_s": duration,
        "device": str(device),
        "n_params": n_params,
    }


if __name__ == "__main__":
    import json
    res = run_memprobe()
    out_path = REPO / "outputs" / "memprobe_result.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n=== Memorization probe verdict ===")
    print(f"  chance accuracy:   {res['chance_acc']:.4f}  (1/{res['n_files']})")
    print(f"  observed top-1:    {res['final_test_acc']:.4f}")
    print(f"  observed top-5:    {res['final_top5_acc']:.4f}")
    print(f"  threshold (DANN):  0.10")
    if res['final_test_acc'] > 0.10:
        print(f"  -> ABOVE THRESHOLD. File-id signature is leaking. Enable DANN for CNN session.")
    else:
        print(f"  -> Below threshold. File-id signature is weak; DANN may not be needed.")
