"""Unified PyTorch trainer for the 1D-CNN (and later Transformer).

Per plan/03_architecture.md sec E:
    Loss      : CrossEntropy with per-fold balanced class weights,
                label smoothing 0.05.
    Optimizer : AdamW lr=3e-4, weight_decay=1e-4.
    Schedule  : 3-epoch linear warmup, then cosine annealing to 0,
                60 epochs total. Resched on per-batch step.
    Augment   : Gaussian noise (p=0.5, sigma uniform in
                [0.005, 0.03] * per-spectrum std),
                multiplicative scale (p=0.4, [0.9, 1.1]),
                wavenumber shift (p=0.4, +/- 3 bins),
                sinusoidal baseline (p=0.3, amp in [0, 0.05]*max,
                period uniform in [200, 800] bins),
                mixup alpha=0.2 (p=0.3).
    Early stop: best val macro-F1, patience 10, restore best weights.
    Grad clip : 1.0.
    Seeds     : per-fold seed = (master_seed*31337 + fold_idx) % 2**31,
                propagated to torch, numpy, DataLoader generator.

Inner train/val split for early stopping: fold 0 of
StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=fold_seed) on
the outer-train file_ids -- identical recipe to atlas/models_classical.py
so CNN and classical results live on the same inner-validation set
geometry.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from atlas.evaluate import PRIMARY_CLASSES


# ----------------------------------------------------------------------------
# Augmentation
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class AugConfig:
    p_noise: float = 0.5
    noise_sigma_lo: float = 0.005
    noise_sigma_hi: float = 0.03

    p_scale: float = 0.4
    scale_lo: float = 0.9
    scale_hi: float = 1.1

    p_shift: float = 0.4
    shift_max_bins: int = 3

    p_baseline: float = 0.3
    baseline_amp_max: float = 0.05      # fraction of per-spectrum max
    baseline_period_lo: int = 200
    baseline_period_hi: int = 800

    p_mixup: float = 0.3
    mixup_alpha: float = 0.2


def _augment_batch(
    x: torch.Tensor,          # (B, 1, L)
    y: torch.Tensor,          # (B,) int
    cfg: AugConfig,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (x_aug, y_a, y_b, lam) where the loss is

        lam * CE(logits, y_a) + (1 - lam) * CE(logits, y_b)

    so mixup composes cleanly with class weighting + label smoothing.
    Non-mixup samples get y_a == y_b and lam == 1.

    Per-spectrum augmentations are sampled independently per sample,
    not per batch -- means a batch typically contains a mix of clean
    and augmented spectra, the standard practice.
    """
    B, _, L = x.shape
    device = x.device
    x = x.clone()

    # 1. Gaussian noise (per-sample sigma proportional to per-sample std)
    if cfg.p_noise > 0:
        mask = torch.from_numpy(rng.random(B) < cfg.p_noise).to(device)
        if mask.any():
            sigma_scale = torch.from_numpy(
                rng.uniform(cfg.noise_sigma_lo, cfg.noise_sigma_hi, size=B)
            ).float().to(device)
            sigma_scale = sigma_scale * mask.float()
            spec_std = x.std(dim=-1, keepdim=True)         # (B, 1, 1)
            noise = torch.randn_like(x) * (
                sigma_scale.view(B, 1, 1) * spec_std
            )
            x = x + noise

    # 2. Multiplicative intensity scale
    if cfg.p_scale > 0:
        mask = rng.random(B) < cfg.p_scale
        scales = np.where(
            mask,
            rng.uniform(cfg.scale_lo, cfg.scale_hi, size=B),
            1.0,
        )
        scale_t = torch.from_numpy(scales).float().view(B, 1, 1).to(device)
        x = x * scale_t

    # 3. Wavenumber shift (roll along the spectral axis per-sample)
    if cfg.p_shift > 0 and cfg.shift_max_bins > 0:
        # torch.roll only takes a single int, so loop over samples that get shifted.
        # This is O(B) python overhead but for batch 128 it's <1ms.
        mask = rng.random(B) < cfg.p_shift
        if mask.any():
            shifts = rng.integers(
                -cfg.shift_max_bins, cfg.shift_max_bins + 1, size=B
            )
            for i in np.where(mask)[0]:
                if shifts[i] != 0:
                    x[i] = torch.roll(x[i], shifts=int(shifts[i]), dims=-1)

    # 4. Sinusoidal baseline drift
    if cfg.p_baseline > 0:
        mask = rng.random(B) < cfg.p_baseline
        if mask.any():
            t = torch.arange(L, device=device, dtype=torch.float32)
            amps = rng.uniform(0, cfg.baseline_amp_max, size=B).astype(np.float32)
            periods = rng.uniform(
                cfg.baseline_period_lo, cfg.baseline_period_hi, size=B
            ).astype(np.float32)
            phases = rng.uniform(0, 2 * np.pi, size=B).astype(np.float32)
            spec_max = x.abs().amax(dim=-1, keepdim=True)  # (B, 1, 1)
            for i in np.where(mask)[0]:
                wave = torch.sin(
                    2 * torch.pi * t / float(periods[i]) + float(phases[i])
                )
                x[i] = x[i] + float(amps[i]) * spec_max[i] * wave

    # 5. Mixup (whole-batch decision: with prob p_mixup, apply to entire batch)
    if cfg.p_mixup > 0 and rng.random() < cfg.p_mixup:
        lam = float(rng.beta(cfg.mixup_alpha, cfg.mixup_alpha))
        perm = torch.from_numpy(rng.permutation(B)).long().to(device)
        x = lam * x + (1 - lam) * x[perm]
        y_b = y[perm]
        lam_t = torch.tensor(lam, dtype=torch.float32, device=device)
        return x, y, y_b, lam_t

    lam_t = torch.tensor(1.0, dtype=torch.float32, device=device)
    return x, y, y, lam_t


# ----------------------------------------------------------------------------
# LR schedule: 3-epoch linear warmup -> cosine annealing
# ----------------------------------------------------------------------------


def make_warmup_cosine_lr(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup [0, lr] over `warmup_steps`, then cosine [lr, 0] over the rest."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + float(np.cos(np.pi * min(1.0, progress))))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ----------------------------------------------------------------------------
# Inner train/val split for early stopping
# ----------------------------------------------------------------------------


def inner_train_val_split(
    y: np.ndarray,
    groups: np.ndarray,
    seed: int,
    n_inner_folds: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Fold 0 of StratifiedGroupKFold(n_inner_folds) on the outer-train set.

    Identical recipe to atlas/models_classical.py so val geometry matches
    across CNN/classical for fair comparison.
    """
    inner = StratifiedGroupKFold(n_splits=n_inner_folds, shuffle=True, random_state=seed)
    for tr_idx, val_idx in inner.split(np.zeros(len(y)), y, groups=groups):
        return tr_idx, val_idx
    raise RuntimeError("inner split produced no folds")


# ----------------------------------------------------------------------------
# Training config + main loop
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainConfig:
    n_epochs: int = 60
    batch_size: int = 128             # MPS / GPU
    cpu_batch_size: int = 32
    warmup_epochs: int = 3
    lr: float = 3e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    grad_clip: float = 1.0
    patience: int = 10
    num_workers: int = 0
    aug: AugConfig = field(default_factory=AugConfig)
    log_every: int = 5  # epochs between log lines
    use_tqdm: bool = True  # show per-epoch progress bar inside each fold
    tqdm_desc: str = ""    # optional prefix for the tqdm bar (e.g. fold id)


def _set_seeds(seed: int) -> None:
    """Propagate seed across python random, numpy, torch (CPU + MPS + CUDA)."""
    import random as _r
    _r.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available() and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(seed)


def _epoch_train(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    class_weights: torch.Tensor,
    cfg: TrainConfig,
    rng: np.random.Generator,
    device: torch.device,
) -> dict:
    model.train()
    n_seen = 0
    loss_sum = 0.0
    n_correct = 0
    for xb, yb in loader:
        # `non_blocking=True` here causes intermittent NaN on MPS (verified
        # locally: torch 2.3.1 on macOS 24.6 with float32 tensors from a
        # CPU-side TensorDataset). MPS doesn't expose pinned-memory transfer
        # the way CUDA does, and the async copy can race with the kernel
        # that consumes it. Synchronous transfer fixes it.
        xb = xb.to(device)
        yb = yb.to(device)

        xb_aug, y_a, y_b, lam = _augment_batch(xb, yb, cfg.aug, rng)

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb_aug)
        # Class weights + label smoothing live inside cross_entropy;
        # mixup is handled by the convex sum over (y_a, y_b).
        if float(lam) >= 1.0 - 1e-6:
            loss = F.cross_entropy(
                logits, y_a, weight=class_weights, label_smoothing=cfg.label_smoothing
            )
        else:
            loss_a = F.cross_entropy(
                logits, y_a, weight=class_weights, label_smoothing=cfg.label_smoothing
            )
            loss_b = F.cross_entropy(
                logits, y_b, weight=class_weights, label_smoothing=cfg.label_smoothing
            )
            loss = lam * loss_a + (1 - lam) * loss_b

        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()

        loss_sum += float(loss) * xb.size(0)
        n_seen += xb.size(0)
        # Training accuracy against the dominant mixup target only.
        n_correct += int((logits.argmax(dim=1) == yb).sum().item())

    return {
        "loss": loss_sum / max(1, n_seen),
        "acc": n_correct / max(1, n_seen),
        "n_seen": n_seen,
    }


@torch.no_grad()
def _epoch_eval(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (proba, y_true_idx) over the eval set, no augmentation, model.eval()."""
    model.eval()
    probas = []
    ys = []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        proba = F.softmax(logits, dim=1)
        probas.append(proba.detach().cpu().numpy())
        ys.append(yb.numpy())
    return np.concatenate(probas, axis=0), np.concatenate(ys, axis=0)


def _val_macro_f1(proba: np.ndarray, y_true_idx: np.ndarray) -> float:
    """Spectrum-level macro-F1 over the canonical PRIMARY_CLASSES order."""
    y_pred = proba.argmax(axis=1)
    return float(
        f1_score(y_true_idx, y_pred, labels=list(range(len(PRIMARY_CLASSES))),
                 average="macro", zero_division=0)
    )


def train_cnn_fold(
    *,
    model_factory: Callable[[int], nn.Module],
    X_train: np.ndarray,        # (N, L) float32 preprocessed spectra
    y_train: np.ndarray,        # (N,) string labels OR int labels
    groups_train: np.ndarray,   # (N,) file_id for inner-split grouping
    X_test: np.ndarray,
    fold_seed: int,
    device: torch.device | None = None,
    n_bins: int = 987,
    n_classes: int = 4,
    cfg: TrainConfig | None = None,
    log_fn: Callable[[str], None] = print,
) -> tuple[np.ndarray, dict, float, nn.Module]:
    """Train one outer fold; return (proba_test, info, training_time_s, trained_model).

    `y_train` may be strings (e.g. 'STEC') or already-encoded ints; we
    normalise to ints against PRIMARY_CLASSES internally.

    The returned `trained_model` has the best-val-F1 weights loaded and lives
    on `device`. Caller is responsible for moving to CPU + saving if needed.
    """
    cfg = cfg or TrainConfig()
    _set_seeds(fold_seed)

    if device is None:
        from atlas.models_cnn import select_device
        device = select_device()

    # Encode labels to canonical int order [H2O, Non-STEC, STEC, Salmonella]
    if y_train.dtype.kind in ("U", "O"):
        lookup = {c: i for i, c in enumerate(PRIMARY_CLASSES)}
        y_train_int = np.array([lookup[c] for c in y_train], dtype=np.int64)
    else:
        y_train_int = y_train.astype(np.int64)

    # Per-bin (mu, sd) fit on OUTER-train; baked into the model as buffers
    # via `set_input_stats` below. The model's encode() applies the same
    # transform automatically at inference (incl. memprobe v2 loading the
    # encoder later). The SNV-preprocessed spectra have per-row mean=0
    # std=1 but per-BIN mean ranges -0.46 to +3.84 and per-BIN std 0.05 to
    # 0.39 — classical models pipe through StandardScaler before PCA+LogReg.
    # Skipping the equivalent step makes the CNN unable to fit train data
    # (verified Protocol A fold 0 2026-05-14: 120 epochs no-aug, train_acc
    # plateau 0.74; WITH standardize train_acc reaches 0.88 in 60 epochs).
    mu = X_train.mean(axis=0, dtype=np.float64).astype(np.float32)
    sd = (X_train.std(axis=0, dtype=np.float64) + 1e-6).astype(np.float32)

    # Inner split for early-stopping val (file-grouped)
    inner_tr_idx, inner_val_idx = inner_train_val_split(
        y_train_int, groups_train, seed=fold_seed, n_inner_folds=4
    )
    X_inner_tr = X_train[inner_tr_idx]
    y_inner_tr = y_train_int[inner_tr_idx]
    X_inner_val = X_train[inner_val_idx]
    y_inner_val = y_train_int[inner_val_idx]

    # Per-fold balanced class weights from the OUTER-train (not inner-train);
    # we want the loss surface to reflect the deployment-time prior, not the
    # accidentally-skewed inner split. Some classes may be absent from a LOSO
    # outer-train (one strain held out can leave its parent class understocked
    # but never absent given our 4-class layout); the loop below still fills
    # in 1.0 as a safe default for any missing class.
    present = np.unique(y_train_int)
    cw_present = compute_class_weight("balanced", classes=present, y=y_train_int)
    cw = np.ones(n_classes, dtype=np.float32)
    for k, cls in enumerate(present):
        cw[cls] = float(cw_present[k])
    class_weights = torch.from_numpy(cw).to(device)

    # Tensor datasets
    Xt = torch.from_numpy(X_inner_tr.astype(np.float32)).unsqueeze(1)
    Yt = torch.from_numpy(y_inner_tr.astype(np.int64))
    Xv = torch.from_numpy(X_inner_val.astype(np.float32)).unsqueeze(1)
    Yv = torch.from_numpy(y_inner_val.astype(np.int64))
    Xte = torch.from_numpy(X_test.astype(np.float32)).unsqueeze(1)
    Yte_dummy = torch.zeros(X_test.shape[0], dtype=torch.int64)

    bs = cfg.batch_size if device.type != "cpu" else cfg.cpu_batch_size

    gen = torch.Generator(); gen.manual_seed(fold_seed)
    train_loader = DataLoader(
        TensorDataset(Xt, Yt), batch_size=bs, shuffle=True,
        num_workers=cfg.num_workers, drop_last=False, generator=gen,
    )
    val_loader = DataLoader(
        TensorDataset(Xv, Yv), batch_size=bs, shuffle=False,
        num_workers=cfg.num_workers,
    )
    test_loader = DataLoader(
        TensorDataset(Xte, Yte_dummy), batch_size=bs, shuffle=False,
        num_workers=cfg.num_workers,
    )

    model = model_factory(fold_seed).to(device)
    # Bake per-bin (mu, sd) into model so state_dict captures them; encode()
    # will apply the transform on every forward pass, including memprobe v2.
    if hasattr(model, "set_input_stats"):
        model.set_input_stats(mu, sd)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * cfg.n_epochs
    warmup_steps = steps_per_epoch * cfg.warmup_epochs
    scheduler = make_warmup_cosine_lr(optimizer, warmup_steps, total_steps)

    rng = np.random.default_rng(fold_seed)

    # Training loop with early stopping on inner-val macro-F1.
    # Initialize best to -inf so the first epoch always wins; this defends
    # against a degenerate fold where epoch 0 val_f1 happens to be 0.0 or
    # negative and the strict-greater check would skip the snapshot. (Per
    # mlops-engineer review 2026-05-14.)
    best_val_f1 = -float("inf")
    best_epoch = -1
    best_state: dict | None = None
    history: list[dict] = []
    bad_epochs = 0
    t0 = time.perf_counter()

    epoch_iter: range | tqdm = range(cfg.n_epochs)
    pbar: tqdm | None = None
    if cfg.use_tqdm:
        desc = f"{cfg.tqdm_desc} epochs" if cfg.tqdm_desc else "epochs"
        pbar = tqdm(epoch_iter, desc=desc, total=cfg.n_epochs, leave=False, ncols=100)
        epoch_iter = pbar

    for epoch in epoch_iter:
        tr_stats = _epoch_train(
            model, train_loader, optimizer, scheduler,
            class_weights, cfg, rng, device,
        )
        val_proba, val_y = _epoch_eval(model, val_loader, device)
        val_f1 = _val_macro_f1(val_proba, val_y)
        cur_lr = float(optimizer.param_groups[0]["lr"])
        history.append({
            "epoch": epoch,
            "lr": cur_lr,
            "train_loss": tr_stats["loss"],
            "train_acc": tr_stats["acc"],
            "val_macro_f1": val_f1,
        })

        if pbar is not None:
            pbar.set_postfix(
                loss=f"{tr_stats['loss']:.3f}",
                tr_acc=f"{tr_stats['acc']:.3f}",
                val_f1=f"{val_f1:.3f}",
                best=f"{max(best_val_f1, val_f1):.3f}",
                lr=f"{cur_lr:.1e}",
                refresh=False,
            )

        if (epoch + 1) % cfg.log_every == 0 or epoch == 0 or epoch == cfg.n_epochs - 1:
            log_fn(
                f"  epoch {epoch+1:>2d}/{cfg.n_epochs}  "
                f"loss={tr_stats['loss']:.3f}  train_acc={tr_stats['acc']:.3f}  "
                f"val_f1={val_f1:.3f}  lr={cur_lr:.2e}"
            )

        if val_f1 > best_val_f1 + 1e-6:
            best_val_f1 = val_f1
            best_epoch = epoch
            # MPS tensors need cpu() before deepcopy is reliable
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                if pbar is not None:
                    pbar.close()
                log_fn(f"  early stop at epoch {epoch+1} (best val_f1={best_val_f1:.3f} @ epoch {best_epoch+1})")
                break

    if pbar is not None:
        pbar.close()

    # Restore best weights.
    assert best_state is not None, "best_state never set -- training loop ran 0 epochs"
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # Predict on test
    test_proba, _ = _epoch_eval(model, test_loader, device)
    dt = time.perf_counter() - t0
    info = {
        "best_val_macro_f1": best_val_f1,
        "best_epoch": best_epoch + 1,
        "n_epochs_run": len(history),
        "n_params": sum(p.numel() for p in model.parameters()),
        "device": str(device),
        "class_weights": cw.tolist(),
        "history": history,
        "n_inner_train": int(len(inner_tr_idx)),
        "n_inner_val": int(len(inner_val_idx)),
    }
    return test_proba, info, dt, model


def encode_dataset(
    *,
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """Run model.encode() over X; return (N, feature_dim) penultimate features.

    Used by memprobe v2.
    """
    model.eval()
    Xt = torch.from_numpy(X.astype(np.float32)).unsqueeze(1)
    loader = DataLoader(TensorDataset(Xt), batch_size=batch_size, shuffle=False)
    feats = []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            feats.append(model.encode(xb).detach().cpu().numpy())
    return np.concatenate(feats, axis=0)


# ----------------------------------------------------------------------------
# DANN training mode (additive — does NOT touch the vanilla code path above)
# ----------------------------------------------------------------------------
#
# Why a separate function instead of a flag on train_cnn_fold:
#   Threading DANN through train_cnn_fold requires:
#     - extra inputs (per-row domain_int = file_id encoded to 0..K-1),
#     - extra forward path (forward_with_domain),
#     - extra loss + history fields,
#     - lambda schedule that updates every epoch.
#   That's enough net new logic that keeping it in a dedicated function makes
#   the vanilla code easier to read and removes any risk of accidentally
#   changing vanilla CNN behavior. The two functions deliberately share the
#   inner-split / class-weight / per-bin standardize / LR-schedule code.


@dataclass(frozen=True)
class DANNConfig:
    """Hyperparameters specific to the DANN training mode.

    lambda_max:
        Final coefficient on the gradient-reversed domain term. Per-Raman
        DANN literature the useful range is 0.05-0.5; we default to 0.1 as
        the conservative starting point.
    warmup_epochs_dann:
        Linear warmup: lambda(epoch) = lambda_max * min(1, epoch / warmup).
        DANN paper default is "ramp over the first ~10% of training" or so.
        With 60-epoch budget we use 10 epochs.
    domain_hidden:
        Width of the hidden Linear inside the domain head. Passed through to
        DANNCNN1D's factory; tracked here so it lands in config.resolved.
    """

    lambda_max: float = 0.1
    warmup_epochs_dann: int = 10
    domain_hidden: int = 64


def _lambda_at_epoch(epoch: int, dann_cfg: DANNConfig) -> float:
    """Linear warmup from 0 to lambda_max over `warmup_epochs_dann`, then hold."""
    if dann_cfg.warmup_epochs_dann <= 0:
        return float(dann_cfg.lambda_max)
    frac = min(1.0, epoch / float(dann_cfg.warmup_epochs_dann))
    return float(dann_cfg.lambda_max * frac)


def _epoch_train_dann(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    class_weights: torch.Tensor,
    cfg: TrainConfig,
    rng: np.random.Generator,
    device: torch.device,
    lambda_grl: float,
) -> dict:
    """One epoch of joint class + domain training.

    Joint loss = L_class + L_domain. The GRL inside the model multiplies the
    gradient through the feature path by -lambda_grl, so the encoder pulls
    *toward* domain-invariant features while the domain head still trains on
    +grad. lambda_grl is constant for the whole epoch (epoch-level warmup).
    """
    model.train()
    n_seen = 0
    class_loss_sum = 0.0
    domain_loss_sum = 0.0
    n_class_correct = 0
    n_domain_correct = 0
    for batch in loader:
        # Loader yields (x, y, domain). Domain is the in-train file_id index.
        xb, yb, db = batch
        xb = xb.to(device)
        yb = yb.to(device)
        db = db.to(device)

        xb_aug, y_a, y_b, lam = _augment_batch(xb, yb, cfg.aug, rng)
        # Mixup permutes samples; the matching permutation must be applied to
        # the domain labels too, otherwise the domain target no longer matches
        # the (mixed) spectrum. Since _augment_batch returns y_a == y_b (and
        # lam == 1) when no mixup happened, we detect "is this a mixup batch"
        # by lam < 1.
        is_mixup = float(lam) < 1.0 - 1e-6
        if is_mixup:
            # _augment_batch built y_b via `y_b = y[perm]` on the same RNG
            # stream — but it didn't expose `perm`. The cleanest fix is to NOT
            # mix domains: train the domain head on the un-mixed view, which
            # we recover by running the encoder on `xb` (pre-aug) separately.
            # That doubles forward-pass cost on mixup batches (p_mixup=0.3 by
            # default), which is acceptable. The cleaner alternative (return
            # `perm` from _augment_batch) would touch the vanilla path; we
            # explicitly want to avoid that.
            class_logits, _ = model.forward_with_domain(xb_aug, lambda_grl)
            _, domain_logits = model.forward_with_domain(xb, lambda_grl)
        else:
            class_logits, domain_logits = model.forward_with_domain(
                xb_aug, lambda_grl
            )

        # Class loss with class weights + label smoothing + mixup target mix
        if not is_mixup:
            loss_class = F.cross_entropy(
                class_logits, y_a,
                weight=class_weights, label_smoothing=cfg.label_smoothing,
            )
        else:
            loss_a = F.cross_entropy(
                class_logits, y_a,
                weight=class_weights, label_smoothing=cfg.label_smoothing,
            )
            loss_b = F.cross_entropy(
                class_logits, y_b,
                weight=class_weights, label_smoothing=cfg.label_smoothing,
            )
            loss_class = lam * loss_a + (1 - lam) * loss_b

        # Domain loss: uniform cross-entropy over K file_ids. No class weights
        # — files are roughly equally represented per the 200-px-cap.
        loss_domain = F.cross_entropy(domain_logits, db)

        loss = loss_class + loss_domain

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()

        bs = xb.size(0)
        class_loss_sum += float(loss_class) * bs
        domain_loss_sum += float(loss_domain) * bs
        n_seen += bs
        # Training class-acc against the dominant mixup target only.
        n_class_correct += int((class_logits.argmax(dim=1) == yb).sum().item())
        n_domain_correct += int((domain_logits.argmax(dim=1) == db).sum().item())

    return {
        "class_loss": class_loss_sum / max(1, n_seen),
        "domain_loss": domain_loss_sum / max(1, n_seen),
        "class_acc": n_class_correct / max(1, n_seen),
        "domain_acc": n_domain_correct / max(1, n_seen),
        "n_seen": n_seen,
    }


def train_dann_fold(
    *,
    model_factory: Callable[[int, int], nn.Module],  # (fold_seed, n_domains) -> DANNCNN1D
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,   # (N,) file_id strings, ALWAYS used for inner-split grouping
    X_test: np.ndarray,
    fold_seed: int,
    domain_labels: np.ndarray | None = None,  # (N,) explicit domain target; default = groups_train
    device: torch.device | None = None,
    n_bins: int = 987,
    n_classes: int = 4,
    cfg: TrainConfig | None = None,
    dann_cfg: DANNConfig | None = None,
    log_fn: Callable[[str], None] = print,
) -> tuple[np.ndarray, dict, float, nn.Module]:
    """Train one outer fold with the DANN objective.

    Returns (proba_test, info, training_time_s, trained_model). `info` includes
    a `history` list whose rows carry class_loss, domain_loss, domain_acc,
    val_macro_f1, lambda_grl per epoch — that's what plan/07's verdict tables
    will read off of.

    The model returned has best-val-class-F1 weights loaded. Early stop is on
    *class* macro-F1, not domain accuracy: we're shipping a class model.

    Args:
        domain_labels: optional per-row domain target (strings or ints). If
            None, defaults to groups_train (file_id, 87-way under Protocol A).
            Passing e.g. subclass strings collapses the GRL target to a
            coarser grouping — see grouped-domain DANN finding in plan/07.
            `groups_train` is ALWAYS used for inner-split file-grouped CV;
            decoupling these two is the whole point of this parameter.
    """
    cfg = cfg or TrainConfig()
    dann_cfg = dann_cfg or DANNConfig()
    _set_seeds(fold_seed)

    if device is None:
        from atlas.models_cnn import select_device
        device = select_device()

    # Encode class labels to canonical int order
    if y_train.dtype.kind in ("U", "O"):
        lookup = {c: i for i, c in enumerate(PRIMARY_CLASSES)}
        y_train_int = np.array([lookup[c] for c in y_train], dtype=np.int64)
    else:
        y_train_int = y_train.astype(np.int64)

    # Domain target source. Defaults to file_id (groups_train); caller can
    # override with subclass / cal_date / etc. for grouped-domain DANN.
    domain_source = domain_labels if domain_labels is not None else groups_train
    if not isinstance(domain_source, np.ndarray):
        domain_source = np.asarray(domain_source)

    # Encode the chosen domain target to ints over the OUTER-train unique values.
    # K varies per fold (file_id: ~70 Protocol A / ~78 LOSO; subclass: ~9; cal_date: ~10-12).
    unique_domains = sorted(set(domain_source.tolist()))
    domain_to_int = {d: i for i, d in enumerate(unique_domains)}
    domains_train_int = np.array(
        [domain_to_int[d] for d in domain_source], dtype=np.int64
    )
    n_domains = len(unique_domains)
    # Backwards-compat alias for the rest of this function and downstream
    # `info["unique_files"]` reads (the field name is historical; we keep it
    # but the content is whatever domain grouping was used).
    unique_files = unique_domains

    # Per-bin standardize, same as train_cnn_fold
    mu = X_train.mean(axis=0, dtype=np.float64).astype(np.float32)
    sd = (X_train.std(axis=0, dtype=np.float64) + 1e-6).astype(np.float32)

    # Inner train/val split for early stopping (class objective)
    inner_tr_idx, inner_val_idx = inner_train_val_split(
        y_train_int, groups_train, seed=fold_seed, n_inner_folds=4
    )
    X_inner_tr = X_train[inner_tr_idx]
    y_inner_tr = y_train_int[inner_tr_idx]
    d_inner_tr = domains_train_int[inner_tr_idx]
    X_inner_val = X_train[inner_val_idx]
    y_inner_val = y_train_int[inner_val_idx]

    # Per-fold balanced class weights from OUTER-train (same logic as vanilla)
    present = np.unique(y_train_int)
    cw_present = compute_class_weight("balanced", classes=present, y=y_train_int)
    cw = np.ones(n_classes, dtype=np.float32)
    for k, cls in enumerate(present):
        cw[cls] = float(cw_present[k])
    class_weights = torch.from_numpy(cw).to(device)

    # Tensor datasets — train loader now ALSO yields the per-sample domain int
    Xt = torch.from_numpy(X_inner_tr.astype(np.float32)).unsqueeze(1)
    Yt = torch.from_numpy(y_inner_tr.astype(np.int64))
    Dt = torch.from_numpy(d_inner_tr.astype(np.int64))
    Xv = torch.from_numpy(X_inner_val.astype(np.float32)).unsqueeze(1)
    Yv = torch.from_numpy(y_inner_val.astype(np.int64))
    Xte = torch.from_numpy(X_test.astype(np.float32)).unsqueeze(1)
    Yte_dummy = torch.zeros(X_test.shape[0], dtype=torch.int64)

    bs = cfg.batch_size if device.type != "cpu" else cfg.cpu_batch_size

    gen = torch.Generator(); gen.manual_seed(fold_seed)
    train_loader = DataLoader(
        TensorDataset(Xt, Yt, Dt), batch_size=bs, shuffle=True,
        num_workers=cfg.num_workers, drop_last=False, generator=gen,
    )
    val_loader = DataLoader(
        TensorDataset(Xv, Yv), batch_size=bs, shuffle=False,
        num_workers=cfg.num_workers,
    )
    test_loader = DataLoader(
        TensorDataset(Xte, Yte_dummy), batch_size=bs, shuffle=False,
        num_workers=cfg.num_workers,
    )

    model = model_factory(fold_seed, n_domains).to(device)
    if hasattr(model, "set_input_stats"):
        model.set_input_stats(mu, sd)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * cfg.n_epochs
    warmup_steps = steps_per_epoch * cfg.warmup_epochs
    scheduler = make_warmup_cosine_lr(optimizer, warmup_steps, total_steps)

    rng = np.random.default_rng(fold_seed)

    best_val_f1 = -float("inf")
    best_epoch = -1
    best_state: dict | None = None
    history: list[dict] = []
    bad_epochs = 0
    t0 = time.perf_counter()

    epoch_iter: range | tqdm = range(cfg.n_epochs)
    pbar: tqdm | None = None
    if cfg.use_tqdm:
        desc = f"{cfg.tqdm_desc} epochs" if cfg.tqdm_desc else "epochs"
        pbar = tqdm(epoch_iter, desc=desc, total=cfg.n_epochs, leave=False, ncols=120)
        epoch_iter = pbar

    for epoch in epoch_iter:
        lam_grl = _lambda_at_epoch(epoch, dann_cfg)
        tr_stats = _epoch_train_dann(
            model, train_loader, optimizer, scheduler,
            class_weights, cfg, rng, device, lambda_grl=lam_grl,
        )
        val_proba, val_y = _epoch_eval(model, val_loader, device)
        val_f1 = _val_macro_f1(val_proba, val_y)
        cur_lr = float(optimizer.param_groups[0]["lr"])
        history.append({
            "epoch": epoch,
            "lr": cur_lr,
            "lambda_grl": lam_grl,
            "class_loss": tr_stats["class_loss"],
            "domain_loss": tr_stats["domain_loss"],
            "train_class_acc": tr_stats["class_acc"],
            "train_domain_acc": tr_stats["domain_acc"],
            "val_macro_f1": val_f1,
        })

        if pbar is not None:
            pbar.set_postfix(
                cls=f"{tr_stats['class_loss']:.2f}",
                dom=f"{tr_stats['domain_loss']:.2f}",
                tr_acc=f"{tr_stats['class_acc']:.2f}",
                dom_acc=f"{tr_stats['domain_acc']:.2f}",
                val=f"{val_f1:.3f}",
                best=f"{max(best_val_f1, val_f1):.3f}",
                lam=f"{lam_grl:.3f}",
                refresh=False,
            )

        if (epoch + 1) % cfg.log_every == 0 or epoch == 0 or epoch == cfg.n_epochs - 1:
            log_fn(
                f"  ep {epoch+1:>2d}/{cfg.n_epochs}  "
                f"cls={tr_stats['class_loss']:.3f}  "
                f"dom={tr_stats['domain_loss']:.3f}  "
                f"tr_acc={tr_stats['class_acc']:.3f}  "
                f"dom_acc={tr_stats['domain_acc']:.3f}  "
                f"val_f1={val_f1:.3f}  lam={lam_grl:.3f}  lr={cur_lr:.2e}"
            )

        if val_f1 > best_val_f1 + 1e-6:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                if pbar is not None:
                    pbar.close()
                log_fn(f"  early stop at epoch {epoch+1} (best val_f1={best_val_f1:.3f} @ epoch {best_epoch+1})")
                break

    if pbar is not None:
        pbar.close()

    assert best_state is not None, "best_state never set -- training loop ran 0 epochs"
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # Predict on test (class head only)
    test_proba, _ = _epoch_eval(model, test_loader, device)
    dt = time.perf_counter() - t0
    info = {
        "best_val_macro_f1": best_val_f1,
        "best_epoch": best_epoch + 1,
        "n_epochs_run": len(history),
        "n_params": sum(p.numel() for p in model.parameters()),
        "device": str(device),
        "class_weights": cw.tolist(),
        "history": history,
        "n_inner_train": int(len(inner_tr_idx)),
        "n_inner_val": int(len(inner_val_idx)),
        "n_domains": int(n_domains),
        "lambda_max": float(dann_cfg.lambda_max),
        "warmup_epochs_dann": int(dann_cfg.warmup_epochs_dann),
        "domain_hidden": int(dann_cfg.domain_hidden),
        "unique_files": unique_files,
    }
    return test_proba, info, dt, model
