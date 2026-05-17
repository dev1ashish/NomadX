"""MCR-ALS spectral unmixing features for the Atlas Raman dataset.

Stage 15C of plan/15 — pre-registered in
`plan/experiments/2026-05-18_stage15c_mcr_als_unmixing.md`. Highest single-feature
EV experiment in the project per plan/13 §2.3.

Decomposes the (N_pixels, B) data matrix `D` into `C · S^T` under non-negativity
constraints so that pure-component spectra (S^T) map onto biology + substrate
components and per-pixel concentrations (C) become file-level features.

Public surface
--------------
- ``simplisma_init(X, n_components, offset_pct=5.0)``
    Windig 1991 SIMPLISMA initial pure-spectrum guess.
- ``class MCRALSWrapper``
    Train/transform interface wrapping ``pymcr.mcr.McrAR``.
- ``mcr_concentration_summary(C, file_ids, k_prefix='mcr_C')``
    Aggregates (mean, std, max, p90) per concentration column per file.
- ``feature_frame_unmix(X, wn, spec_df, n_components=8, ...)``
    One-shot file-level DataFrame builder (used for caching; LOSO classifier
    must do per-fold fits via ``MCRALSWrapper`` directly).

Non-negativity caveat
---------------------
The preprocessed cache is arPLS + SG + SNV. SNV outputs can be negative, but
MCR's non-negativity constraint requires D ≥ 0. The caller is responsible for
shifting the input (e.g. ``X - X.min()``) before passing to ``MCRALSWrapper.fit``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# SIMPLISMA initialization (Windig 1991)
# ---------------------------------------------------------------------------

def simplisma_init(
    X: np.ndarray,
    n_components: int,
    offset_pct: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Pick K 'purest variables' and return initial S^T plus their wn-indices.

    Parameters
    ----------
    X : (N, B) ndarray
        Input data. Treated as N samples × B variables (= wavenumber bins).
        Should be ≥ 0 for the purity metric to be physically interpretable.
    n_components : int
        Number of components K to extract.
    offset_pct : float
        Stabilizer α in % of max-column-mean. Prevents purity blow-up at
        low-mean variables. Windig recommends 1–5%.

    Returns
    -------
    ST_init : (K, B) ndarray
        Initial pure-spectrum guess for MCR-ALS. Each row is the spectrum of the
        pixel that has the maximum intensity at the k-th purest wavenumber,
        L2-normalized.
    pure_var_idx : (K,) ndarray of int
        The K selected wavenumber-bin indices, in selection order.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (N, B); got shape {X.shape}")
    N, B = X.shape
    if n_components < 1 or n_components > B:
        raise ValueError(f"n_components must be in [1, {B}]; got {n_components}")

    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0)
    alpha = (offset_pct / 100.0) * float(mu.max() if mu.size else 1.0)

    # Length-normalized "purity-base" vectors for each variable.
    # base[j] = X[:, j] / sqrt(mu_j^2 + sd_j^2 + alpha^2) — Windig normalization.
    norm_j = np.sqrt(mu ** 2 + sd ** 2 + alpha ** 2)
    norm_j = np.where(norm_j < 1e-12, 1.0, norm_j)
    base = X / norm_j[None, :]                            # (N, B)

    # Iterative selection: pick the variable that is most independent from
    # already-selected variables, as measured by the determinant of the
    # cross-correlation matrix of selected normalized columns.
    selected: list[int] = []
    pure_spectra_init = np.zeros((n_components, B), dtype=np.float64)

    # Standard SIMPLISMA "first variable" criterion: pure_j = sd_j / (mu_j + alpha)
    purity = sd / (mu + alpha)
    j0 = int(np.argmax(purity))
    selected.append(j0)

    # For subsequent components, weight purity by the determinant of the
    # correlation matrix built from already-selected columns + the candidate.
    # Equivalent to a Gram-Schmidt orthogonalization in the normalized space.
    Y = base[:, [j0]]                                     # (N, 1)
    for k in range(1, n_components):
        # Build (k+1)×(k+1) correlation determinant for each candidate j.
        # corr matrix elements c_{ij} = (1/N) * Y[:,i] . base[:,j_cand]
        # We compute det of YtY block plus the candidate row/col — but simpler:
        # weight_j = det(C_{k+1}(j)) / det(C_k); maximize.
        Ct = (Y.T @ Y) / N                                # (k, k)
        # Cross-correlation of each candidate with selected:
        c_cross = (Y.T @ base) / N                        # (k, B)
        # Self-correlation of each candidate with itself:
        c_self = (base * base).sum(axis=0) / N            # (B,)

        # det of [[Ct, c_cross[:, j]], [c_cross[:, j].T, c_self[j]]]
        # = c_self[j] * det(Ct) - c_cross[:, j].T @ adj(Ct) @ c_cross[:, j]
        # = det(Ct) * (c_self[j] - c_cross[:, j].T @ inv(Ct) @ c_cross[:, j])
        # (Schur-complement determinant identity.)
        try:
            inv_Ct = np.linalg.inv(Ct)
        except np.linalg.LinAlgError:
            inv_Ct = np.linalg.pinv(Ct)
        residual = c_self - np.einsum("ij,ij->j", c_cross, inv_Ct @ c_cross)
        residual = np.maximum(residual, 0.0)              # numeric safety
        weights = residual * purity
        # Exclude already-selected variables.
        weights[selected] = -np.inf
        j_next = int(np.argmax(weights))
        selected.append(j_next)
        Y = base[:, selected]                             # (N, k+1)

    pure_var_idx = np.array(selected, dtype=int)

    # Build initial pure-spectrum guess: the pixel with maximum intensity at
    # each purest variable is the most "saturated" example of that component.
    for k, j_k in enumerate(pure_var_idx):
        i_max = int(np.argmax(X[:, j_k]))
        s = X[i_max].copy()
        s = np.maximum(s, 0.0)
        n = float(np.linalg.norm(s))
        if n > 0:
            s = s / n
        pure_spectra_init[k] = s

    return pure_spectra_init, pure_var_idx


# ---------------------------------------------------------------------------
# MCR-ALS wrapper
# ---------------------------------------------------------------------------

@dataclass
class MCRALSResult:
    """Outputs of a successful MCR-ALS fit."""
    pure_spectra: np.ndarray         # (K, B)  S^T
    concentrations: np.ndarray       # (N, K)  C
    n_iter: int
    err_history: list[float] = field(default_factory=list)
    pure_var_idx: np.ndarray | None = None


class MCRALSWrapper:
    """Thin wrapper around ``pymcr.mcr.McrAR`` with SIMPLISMA init.

    Train/transform contract
    ------------------------
    - ``fit(X_train)``      → fits both S^T and C on X_train.
    - ``transform(X)``      → projects X onto the frozen S^T via NNLS, returns
                              concentrations C of shape (N_new, K).
    - ``pure_spectra``      → fitted (K, B) array.
    - ``concentrations``    → fit-time C of shape (N_train, K).

    The wrapper is the only object the Stage 15F LOSO classifier should use
    for per-fold fitting (avoids leakage; see plan/15 §7 R2).
    """

    def __init__(
        self,
        n_components: int = 8,
        max_iter: int = 100,
        tol_err_change: float | None = 1e-7,
        offset_pct: float = 5.0,
        normalize_spectra: bool = True,
        random_state: int = 42,
    ) -> None:
        self.n_components = int(n_components)
        self.max_iter = int(max_iter)
        self.tol_err_change = tol_err_change
        self.offset_pct = float(offset_pct)
        self.normalize_spectra = bool(normalize_spectra)
        self.random_state = int(random_state)
        self._result: MCRALSResult | None = None

    # ----- private helpers -----

    def _build_mcrar(self):
        from pymcr.mcr import McrAR
        from pymcr.regressors import NNLS
        from pymcr.constraints import ConstraintNonneg, ConstraintNorm
        st_constraints = [ConstraintNonneg()]
        if self.normalize_spectra:
            st_constraints.append(ConstraintNorm())
        return McrAR(
            c_regr=NNLS(),
            st_regr=NNLS(),
            c_constraints=[ConstraintNonneg()],
            st_constraints=st_constraints,
            max_iter=self.max_iter,
            tol_err_change=self.tol_err_change,
            # pyMCR's tol_increase fires on transient ST-step error bumps even
            # when the algorithm would converge — empirically it exits at iter 1
            # on well-posed synthetic data. Disable; tol_err_change handles
            # termination via absolute convergence instead.
            tol_increase=None,
            tol_n_increase=None,
            tol_n_above_min=None,
        )

    # ----- public API -----

    def fit(self, X: np.ndarray) -> "MCRALSWrapper":
        """Fit pure spectra S^T and concentrations C on X.

        X must be non-negative. Caller is responsible for shifting (e.g.
        SNV'd data needs ``X - X.min()`` before passing here).
        """
        np.random.seed(self.random_state)
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D; got shape {X.shape}")
        if X.min() < -1e-9:
            raise ValueError(
                f"X has negative values (min={X.min():.3g}); shift to ≥0 before fitting."
            )

        ST_init, pure_idx = simplisma_init(X, self.n_components, offset_pct=self.offset_pct)
        mcr = self._build_mcrar()
        # c_first=False so the first regression step solves for C given ST_init
        # (which is the natural pairing — we initialized S^T, not C).
        mcr.fit(X, ST=ST_init, c_first=False, verbose=False)

        # pymcr stores final results in attributes
        ST_opt = np.asarray(mcr.ST_opt_, dtype=np.float64)
        C_opt = np.asarray(mcr.C_opt_, dtype=np.float64)
        n_iter = int(getattr(mcr, "n_iter", -1))
        err_history = [float(e) for e in getattr(mcr, "err", [])]

        self._result = MCRALSResult(
            pure_spectra=ST_opt,
            concentrations=C_opt,
            n_iter=n_iter,
            err_history=err_history,
            pure_var_idx=pure_idx,
        )
        self._mcr = mcr
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project X onto the frozen pure spectra S^T.

        Uses NNLS per row so concentrations stay non-negative.
        """
        if self._result is None:
            raise RuntimeError("fit must be called before transform")
        X = np.asarray(X, dtype=np.float64)
        from scipy.optimize import nnls
        ST = self._result.pure_spectra                       # (K, B)
        K = ST.shape[0]
        C_new = np.zeros((X.shape[0], K), dtype=np.float64)
        ST_T = ST.T                                          # (B, K)
        for i in range(X.shape[0]):
            c, _ = nnls(ST_T, X[i])
            C_new[i] = c
        return C_new

    @property
    def pure_spectra(self) -> np.ndarray:
        if self._result is None:
            raise RuntimeError("not fit")
        return self._result.pure_spectra

    @property
    def concentrations(self) -> np.ndarray:
        if self._result is None:
            raise RuntimeError("not fit")
        return self._result.concentrations

    @property
    def n_iter(self) -> int:
        if self._result is None:
            raise RuntimeError("not fit")
        return self._result.n_iter

    @property
    def pure_var_idx(self) -> np.ndarray | None:
        if self._result is None:
            return None
        return self._result.pure_var_idx


# ---------------------------------------------------------------------------
# Per-file concentration summary (DD1 feature builder)
# ---------------------------------------------------------------------------

def mcr_concentration_summary(
    C: np.ndarray,
    file_ids: np.ndarray | pd.Series,
    k_prefix: str = "mcr_C",
) -> pd.DataFrame:
    """Aggregate (mean, std, max, p90) per concentration column per file.

    Parameters
    ----------
    C : (N, K) ndarray
        Per-pixel concentrations.
    file_ids : (N,) array-like of str
        One file_id per pixel (row of C). Must align with C.
    k_prefix : str
        Prefix for output column names. Default ``mcr_C`` → columns named
        ``mcr_C{k}_mean``, ``mcr_C{k}_std``, etc. for k=1..K.

    Returns
    -------
    DataFrame indexed by file_id with 4·K columns.
    """
    C = np.asarray(C)
    if C.ndim != 2:
        raise ValueError(f"C must be 2D; got shape {C.shape}")
    file_ids = np.asarray(file_ids)
    if file_ids.shape[0] != C.shape[0]:
        raise ValueError(
            f"file_ids length {file_ids.shape[0]} != C rows {C.shape[0]}"
        )
    K = C.shape[1]
    df = pd.DataFrame(
        C,
        columns=[f"{k_prefix}{k+1}" for k in range(K)],
    )
    df["file_id"] = file_ids
    agg = df.groupby("file_id").agg(
        ["mean", "std", "max", lambda x: np.nanpercentile(x, 90)]
    )
    # Flatten MultiIndex columns: ('mcr_C1', 'mean') -> 'mcr_C1_mean'
    new_cols = []
    for col, stat in agg.columns:
        stat_name = "p90" if "<lambda" in str(stat) else stat
        new_cols.append(f"{col}_{stat_name}")
    agg.columns = new_cols
    return agg


# ---------------------------------------------------------------------------
# One-shot feature frame
# ---------------------------------------------------------------------------

def feature_frame_unmix(
    X: np.ndarray,
    wn: np.ndarray,
    spec_df: pd.DataFrame,
    n_components: int = 8,
    max_iter: int = 100,
    offset_pct: float = 5.0,
    random_state: int = 42,
    return_intermediates: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, MCRALSWrapper]:
    """Fit MCR-ALS globally and return file-level concentration summary.

    For caching only. Stage 15F LOSO classifier must fit per fold.
    """
    if "file_id" not in spec_df.columns:
        raise ValueError("spec_df must have a 'file_id' column")
    X_offset = X - X.min()
    wrapper = MCRALSWrapper(
        n_components=n_components,
        max_iter=max_iter,
        offset_pct=offset_pct,
        random_state=random_state,
    )
    wrapper.fit(X_offset)
    df = mcr_concentration_summary(wrapper.concentrations, spec_df["file_id"].values)
    if return_intermediates:
        return df, wrapper
    return df


# ---------------------------------------------------------------------------
# Smoke test (runs when this file is executed directly)
# ---------------------------------------------------------------------------

def _smoke_test() -> bool:
    """Synthetic 2-component recovery test.

    Generates two Gaussian "pure spectra" + 100 random non-negative
    concentration profiles, fits K=2 MCR-ALS, and verifies cosine similarity
    to ground truth ≥ 0.95 (after permutation matching).
    """
    rng = np.random.default_rng(0)
    B = 200
    wn = np.linspace(400, 1800, B)
    # Two Gaussian pure spectra at distinct centers
    def gauss(c, s):
        return np.exp(-0.5 * ((wn - c) / s) ** 2)
    S_true = np.vstack([
        gauss(800, 30) + 0.6 * gauss(1450, 25),
        gauss(1200, 30) + 0.4 * gauss(1650, 20),
    ])
    # Normalize ground truth
    S_true = S_true / np.linalg.norm(S_true, axis=1, keepdims=True)
    N = 100
    C_true = rng.uniform(0.1, 1.5, size=(N, 2))
    D = C_true @ S_true + 0.005 * rng.standard_normal((N, B))
    D = np.maximum(D, 0.0)
    w = MCRALSWrapper(n_components=2, max_iter=200, offset_pct=2.0, random_state=0)
    w.fit(D)
    S_est = w.pure_spectra
    S_est = S_est / np.linalg.norm(S_est, axis=1, keepdims=True)
    # Permutation-match: cosine of (i, j) pairs
    cos_matrix = S_est @ S_true.T
    # Pick best permutation
    if cos_matrix[0, 0] + cos_matrix[1, 1] >= cos_matrix[0, 1] + cos_matrix[1, 0]:
        pair = [(0, 0), (1, 1)]
    else:
        pair = [(0, 1), (1, 0)]
    cos_vals = [float(cos_matrix[i, j]) for i, j in pair]
    print(f"[smoke] cosine similarity to ground truth: {cos_vals}")
    print(f"[smoke] n_iter={w.n_iter}, err_history_len={len(w._result.err_history)}")
    ok = all(c >= 0.95 for c in cos_vals)
    print(f"[smoke] {'PASS' if ok else 'FAIL'}: {ok}")
    return ok


if __name__ == "__main__":
    ok = _smoke_test()
    raise SystemExit(0 if ok else 1)
