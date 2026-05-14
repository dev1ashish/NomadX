"""Memorization probe v2 — file-id leakage of the trained CNN encoder.

The question:
    Does the 1D-CNN's penultimate-layer feature space ALSO encode file_id,
    even though we trained the encoder to predict primary_class (4-way)?
    If yes, the encoder is implicitly memorizing acquisition signature
    and the LOSO crater could be partly batch-effect leakage.

Difference from `atlas.memprobe`:
    v1 trained a *from-scratch* 6.6K-param tiny CNN to predict file_id
    directly. Result: 4.1% top-1 (3.5x chance) — below the 10% DANN
    threshold but non-zero.

    v2 takes the trained CLASS encoder (loaded from
    outputs/<run>/encoder_fold_*.pt), runs ALL QC-passing spectra through
    `encode()` to get (N, 32) features, then fits a multinomial Logistic
    Regression (87-way file_id) on a within-file 80/20 pixel split.

    The two probes ask different questions:
        v1 = "is file_id learnable from a single spectrum if I train a
              network FOR THAT?"
        v2 = "is file_id leaking into the features the CLASS-trained
              encoder learned?"
    v2 is the more honest leakage check for the CNN we ship.

Usage:
    python -m atlas.memprobe_v2 \\
        --encoder outputs/<run_id>/encoder_fold_<id>.pt \\
        --out outputs/memprobe_v2_<run_id>.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression


REPO = Path(__file__).resolve().parent.parent


def load_encoder(encoder_path: Path, n_bins: int, device: torch.device):
    from atlas.models_cnn import SmallCNN1D
    model = SmallCNN1D(n_bins=n_bins, n_classes=4)
    state = torch.load(encoder_path, map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def within_file_pixel_split(
    spec_df: pd.DataFrame, qc_mask: np.ndarray, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Within each QC-passing file, 80% pixels -> train, 20% -> test.

    Returns (train_idx, test_idx, file_id_int, unique_files) where indices
    are into the 7,999-row preprocessed array.
    """
    rng = np.random.default_rng(seed)
    file_id_arr = spec_df["file_id"].to_numpy()
    qc_indices = np.where(qc_mask)[0]
    unique_files = sorted(set(file_id_arr[qc_indices]))
    file_to_int = {f: i for i, f in enumerate(unique_files)}

    train_idx, test_idx = [], []
    for f in unique_files:
        rows = qc_indices[file_id_arr[qc_indices] == f].copy()
        rng.shuffle(rows)
        n_test = max(1, int(0.2 * len(rows)))
        test_idx.extend(rows[:n_test].tolist())
        train_idx.extend(rows[n_test:].tolist())

    train_idx = np.array(sorted(train_idx), dtype=np.int64)
    test_idx = np.array(sorted(test_idx), dtype=np.int64)
    file_id_int = np.array([file_to_int[f] for f in file_id_arr], dtype=np.int64)
    return train_idx, test_idx, file_id_int, unique_files


def run_memprobe_v2(
    *,
    encoder_path: Path,
    cache_dir: Path = REPO / "data_cache",
    seed: int = 42,
    log_fn=print,
) -> dict:
    log_fn(f"[memprobe-v2] encoder={encoder_path}  cache_dir={cache_dir}")

    spec_df = pd.read_parquet(cache_dir / "spectra.parquet")
    X = np.load(cache_dir / "spectra_array_preprocessed.npy")
    qc_mask = np.load(cache_dir / "qc_mask.npy")
    n_bins = X.shape[1]

    train_idx, test_idx, file_id_int, unique_files = within_file_pixel_split(
        spec_df, qc_mask, seed=seed
    )
    n_classes = len(unique_files)
    chance = 1.0 / n_classes
    log_fn(
        f"[memprobe-v2] n_files={n_classes}  chance={chance:.4f}  "
        f"n_train_pixels={train_idx.size}  n_test_pixels={test_idx.size}"
    )

    from atlas.models_cnn import select_device
    device = select_device()
    log_fn(f"[memprobe-v2] device={device}")

    t0 = time.perf_counter()
    model = load_encoder(encoder_path, n_bins=n_bins, device=device)
    n_params = sum(p.numel() for p in model.parameters())
    log_fn(f"[memprobe-v2] encoder params: {n_params:,}")

    from atlas.train import encode_dataset
    feats_train = encode_dataset(model=model, X=X[train_idx], device=device, batch_size=256)
    feats_test = encode_dataset(model=model, X=X[test_idx], device=device, batch_size=256)
    log_fn(
        f"[memprobe-v2] features train={feats_train.shape} test={feats_test.shape}  "
        f"encode_time={time.perf_counter()-t0:.1f}s"
    )

    y_train = file_id_int[train_idx]
    y_test = file_id_int[test_idx]

    # 87-way logistic regression. C=1 default; balanced class weights so the
    # smaller-pixel-count files (low-density grids) aren't drowned out by the
    # 200-pixel-cap big files.
    t1 = time.perf_counter()
    clf = LogisticRegression(
        max_iter=2000,
        multi_class="multinomial",
        solver="lbfgs",
        class_weight="balanced",
        n_jobs=1,
        random_state=seed,
    )
    clf.fit(feats_train, y_train)
    fit_time = time.perf_counter() - t1
    log_fn(f"[memprobe-v2] logreg fit_time={fit_time:.1f}s")

    proba_test = clf.predict_proba(feats_test)
    # If a class is missing from y_train (shouldn't happen since every file is
    # in both splits), proba may not span all 87. clf.classes_ tells us which.
    top1 = (proba_test.argmax(axis=1) == np.searchsorted(clf.classes_, y_test)).mean()
    # Top-5 over the predicted classes-vector
    top5_indices = np.argsort(-proba_test, axis=1)[:, :5]   # (N, 5) into clf.classes_
    top5_classes = clf.classes_[top5_indices]                # (N, 5) actual file_id_int
    top5_correct = (top5_classes == y_test.reshape(-1, 1)).any(axis=1).mean()

    duration = time.perf_counter() - t0

    result = {
        "encoder_path": str(encoder_path),
        "n_files": n_classes,
        "chance_acc": chance,
        "top1_acc": float(top1),
        "top5_acc": float(top5_correct),
        "n_train_pixels": int(train_idx.size),
        "n_test_pixels": int(test_idx.size),
        "n_features": int(feats_train.shape[1]),
        "encoder_params": int(n_params),
        "device": str(device),
        "logreg_fit_time_s": float(fit_time),
        "total_time_s": float(duration),
        "dann_threshold": 0.10,
        "fires": bool(float(top1) > 0.10),
    }
    log_fn(
        f"[memprobe-v2] top-1={result['top1_acc']:.4f}  "
        f"({result['top1_acc']/chance:.1f}x chance)  "
        f"top-5={result['top5_acc']:.4f}  "
        f"duration={duration:.1f}s"
    )
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True, type=Path,
                    help="Path to encoder_fold_*.pt produced by scripts/run_cnn.py")
    ap.add_argument("--cache-dir", default=str(REPO / "data_cache"), type=Path)
    ap.add_argument("--out", default=None,
                    help="Optional output JSON. Default: outputs/memprobe_v2_<encoder-stem>.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    result = run_memprobe_v2(
        encoder_path=args.encoder,
        cache_dir=args.cache_dir,
        seed=args.seed,
    )

    out_path = (
        Path(args.out) if args.out
        else REPO / "outputs" / f"memprobe_v2_{args.encoder.stem}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n=== Memorization probe v2 verdict ===")
    print(f"  chance accuracy:   {result['chance_acc']:.4f}  (1/{result['n_files']})")
    print(f"  observed top-1:    {result['top1_acc']:.4f}  ({result['top1_acc']/result['chance_acc']:.1f}x chance)")
    print(f"  observed top-5:    {result['top5_acc']:.4f}")
    print(f"  DANN threshold:    {result['dann_threshold']:.2f}")
    if result["fires"]:
        print(f"  -> ABOVE THRESHOLD. CNN encoder's penultimate features leak file_id.")
        print(f"     Enable DANN before claiming LOSO results are clean.")
    else:
        print(f"  -> Below threshold. Class objective discarded most file-id signal.")
        print(f"     LOSO crater is biology, not leakage (compare against probe v1 4.1%).")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
