"""Classical model pipelines + per-fold trainer with inner-CV HPO.

Six models: LogReg, LinearSVM, RBF-SVM, RandomForest, XGBoost, PLS-DA.

Each model is a (factory, hp_sampler) pair. The factory builds a fresh sklearn
Pipeline given a hyperparameter dict; the sampler returns random configs from
the search space.

The fold trainer:
    1. Splits the outer-train set into inner_train + inner_val (one fold of
       StratifiedGroupKFold(4), fold 0 fixed -- budget compromise per plan).
    2. Random-searches the HP space on inner_train, scores macro-F1 on
       inner_val. Picks the best config.
    3. Refits the chosen config on the FULL outer-train set.
    4. Returns predict_proba on the outer-test set.

All scaling/PCA fit inside CV folds. No data leaks across folds.

Notes:
    - LinearSVC has no predict_proba -- we wrap it with CalibratedClassifierCV
      (cv='prefit' is wrong inside a Pipeline; we use cv=3 isotonic calibration
      fit on the inner-train set).
    - XGBoost uses sample_weight derived from sklearn.utils.class_weight
      because scale_pos_weight is binary-only.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.utils.class_weight import compute_sample_weight


# ----------------------------------------------------------------------------
# PLS-DA: wrap PLSRegression -> LogReg in a single sklearn estimator
# ----------------------------------------------------------------------------


import warnings
warnings.filterwarnings("ignore", message=".*'multi_class' was deprecated.*", category=FutureWarning)

from tqdm.auto import tqdm


class PLSDA(PLSRegression):
    """PLSRegression -> LogisticRegression on the latent variables.

    sklearn doesn't ship a PLS-DA out of the box. We compose:
        PLSRegression(n_components).fit(X, y_onehot)   -> transform -> LV scores
        LogisticRegression(multinomial, balanced)      -> classify

    Exposed as a single classifier with fit / predict_proba.
    """

    def __init__(self, n_components: int = 10, random_state: int | None = None):
        super().__init__(n_components=n_components, scale=False)
        self.random_state = random_state
        self.logreg_: LogisticRegression | None = None
        self.classes_: np.ndarray | None = None

    def fit(self, X, y):
        classes = np.unique(y)
        self.classes_ = classes
        # one-hot encode y for PLS regression
        Y = np.zeros((len(y), len(classes)))
        for i, c in enumerate(classes):
            Y[y == c, i] = 1.0
        super().fit(X, Y)
        scores = super().transform(X)
        self.logreg_ = LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=self.random_state,
        )
        self.logreg_.fit(scores, y)
        return self

    def predict_proba(self, X):
        scores = super().transform(X)
        return self.logreg_.predict_proba(scores)

    def predict(self, X):
        return self.logreg_.predict(super().transform(X))


# ----------------------------------------------------------------------------
# Model factories
# ----------------------------------------------------------------------------


def make_logreg(hp: dict, seed: int) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=hp["n_pca"], random_state=seed)),
        ("clf", LogisticRegression(
            C=hp["C"],
            class_weight="balanced",
            max_iter=2000,
            random_state=seed,
        )),
    ])


def make_linsvm(hp: dict, seed: int) -> Pipeline:
    # LinearSVC has no predict_proba; wrap with isotonic calibration
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=hp["n_pca"], random_state=seed)),
        ("clf", CalibratedClassifierCV(
            estimator=LinearSVC(
                C=hp["C"],
                class_weight="balanced",
                dual="auto",
                max_iter=4000,
                random_state=seed,
            ),
            method="isotonic",
            cv=3,
        )),
    ])


def make_rbfsvm(hp: dict, seed: int) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=hp["n_pca"], random_state=seed)),
        ("clf", SVC(
            C=hp["C"],
            gamma=hp["gamma"],
            kernel="rbf",
            class_weight="balanced",
            probability=True,
            random_state=seed,
        )),
    ])


def make_rf(hp: dict, seed: int) -> Pipeline:
    return Pipeline([
        ("clf", RandomForestClassifier(
            n_estimators=hp["n_estimators"],
            max_features=hp["max_features"],
            max_depth=hp["max_depth"],
            min_samples_leaf=hp["min_samples_leaf"],
            class_weight="balanced",
            n_jobs=1,  # explicit; OMP threads pinned to 1 in runner
            random_state=seed,
        )),
    ])


def make_xgb(hp: dict, seed: int) -> Pipeline:
    # Lazy import: xgboost requires libomp.dylib on macOS (brew install libomp).
    from xgboost import XGBClassifier
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", XGBClassifier(
            n_estimators=hp["n_estimators"],
            max_depth=hp["max_depth"],
            learning_rate=hp["learning_rate"],
            reg_lambda=hp["reg_lambda"],
            subsample=hp["subsample"],
            colsample_bytree=hp["colsample_bytree"],
            objective="multi:softprob",
            num_class=4,
            tree_method="hist",
            eval_metric="mlogloss",
            n_jobs=4,
            seed=seed,
            verbosity=0,
        )),
    ])


def make_plsda(hp: dict, seed: int) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", PLSDA(n_components=hp["n_components"], random_state=seed)),
    ])


# ----------------------------------------------------------------------------
# Hyperparameter samplers (random search)
# ----------------------------------------------------------------------------


def sample_logreg(rng: np.random.Generator) -> dict:
    return {
        "C": float(10 ** rng.uniform(-3, 1)),
        "n_pca": int(rng.choice([50, 100, 150])),
    }


def sample_linsvm(rng: np.random.Generator) -> dict:
    return {
        "C": float(10 ** rng.uniform(-3, 1)),
        "n_pca": int(rng.choice([50, 100, 150])),
    }


def sample_rbfsvm(rng: np.random.Generator) -> dict:
    return {
        "C": float(10 ** rng.uniform(0, 3)),
        "gamma": float(10 ** rng.uniform(-4, -1)),
        "n_pca": int(rng.choice([50, 100, 150])),
    }


def sample_rf(rng: np.random.Generator) -> dict:
    mf_options = ["sqrt", 0.1]
    md_options = [None, 20, 40]
    return {
        "n_estimators": int(rng.choice([300, 500, 800])),
        "max_features": mf_options[rng.integers(0, len(mf_options))],
        "max_depth": md_options[rng.integers(0, len(md_options))],
        "min_samples_leaf": int(rng.choice([1, 3, 5])),
    }


def sample_xgb(rng: np.random.Generator) -> dict:
    # Cheapened from original 200-1000 / 20 trials per (rf-loso-2hr post-mortem):
    # the LOSO crater is biology, not under-fitting, so big-n_estimators wins zero
    # information for the questions we're actually asking. See plan/10_decision_log.md.
    return {
        "n_estimators": int(rng.integers(100, 301)),
        "max_depth": int(rng.integers(3, 7)),
        "learning_rate": float(10 ** rng.uniform(-1.5, np.log10(0.3))),
        "reg_lambda": float(10 ** rng.uniform(-1, 1)),
        "subsample": float(rng.uniform(0.6, 1.0)),
        "colsample_bytree": float(rng.uniform(0.5, 0.9)),
    }


def sample_plsda(rng: np.random.Generator) -> dict:
    grid = [5, 8, 10, 12, 15, 20, 25, 30]
    return {"n_components": int(rng.choice(grid))}


# ----------------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    name: str
    factory: Callable[[dict, int], Pipeline]
    sampler: Callable[[np.random.Generator], dict]
    n_trials: int


MODEL_REGISTRY: list[ModelSpec] = [
    ModelSpec("logreg",  make_logreg,  sample_logreg,  12),
    ModelSpec("linsvm",  make_linsvm,  sample_linsvm,  12),
    ModelSpec("rbfsvm",  make_rbfsvm,  sample_rbfsvm,  20),
    ModelSpec("rf",      make_rf,      sample_rf,      12),
    ModelSpec("xgb",     make_xgb,     sample_xgb,     10),  # cheapened post-rf-postmortem
    ModelSpec("plsda",   make_plsda,   sample_plsda,   8),
]


def get_model_spec(name: str) -> ModelSpec:
    for m in MODEL_REGISTRY:
        if m.name == name:
            return m
    raise KeyError(f"unknown model {name!r}; options: {[m.name for m in MODEL_REGISTRY]}")


# ----------------------------------------------------------------------------
# Fold trainer
# ----------------------------------------------------------------------------


def _inner_train_val_split(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    seed: int,
    n_inner_folds: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (inner_train_idx, inner_val_idx) for fold 0 of inner StratifiedGroupKFold.

    Fold 0 fixed (not nested) -- budget compromise documented in plan §C.
    """
    inner = StratifiedGroupKFold(n_splits=n_inner_folds, shuffle=True, random_state=seed)
    for tr_idx, val_idx in inner.split(X_train, y_train, groups=groups_train):
        return tr_idx, val_idx
    raise RuntimeError("inner split produced no folds")


def train_fold(
    *,
    spec: ModelSpec,
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    X_test: np.ndarray,
    fold_seed: int,
    n_inner_folds: int = 4,
    log_fn: Callable[[str], None] = print,
) -> tuple[np.ndarray, dict, float]:
    """Train one outer fold with random-search HPO on inner fold 0.

    Returns (y_proba_test, best_hp, training_time_s).
    """
    rng = np.random.default_rng(fold_seed)

    # Inner split for HPO
    inner_tr_idx, inner_val_idx = _inner_train_val_split(
        X_train, y_train, groups_train, seed=fold_seed, n_inner_folds=n_inner_folds
    )
    X_inner_tr = X_train[inner_tr_idx]
    y_inner_tr = y_train[inner_tr_idx]
    X_inner_val = X_train[inner_val_idx]
    y_inner_val = y_train[inner_val_idx]

    # Random search
    best_score = -1.0
    best_hp: dict | None = None
    t0 = time.perf_counter()
    pbar = tqdm(range(spec.n_trials), desc=f"  {spec.name} HPO", leave=False, ncols=80)
    for trial in pbar:
        hp = spec.sampler(rng)
        pipe = spec.factory(hp, fold_seed)

        if spec.name == "xgb":
            # XGBoost wants integer-encoded labels
            classes = np.unique(y_inner_tr)
            class_to_int = {c: i for i, c in enumerate(classes)}
            y_inner_tr_int = np.array([class_to_int[c] for c in y_inner_tr], dtype=np.int32)
            sample_weight = compute_sample_weight("balanced", y_inner_tr)
            pipe.fit(X_inner_tr, y_inner_tr_int, clf__sample_weight=sample_weight)
            proba = pipe.predict_proba(X_inner_val)
            y_inner_val_int = np.array([class_to_int[c] for c in y_inner_val], dtype=np.int32)
            pred_int = np.argmax(proba, axis=1)
            pred = np.array([classes[i] for i in pred_int])
        else:
            pipe.fit(X_inner_tr, y_inner_tr)
            pred = pipe.predict(X_inner_val)

        score = f1_score(y_inner_val, pred, average="macro", zero_division=0)
        if score > best_score:
            best_score = score
            best_hp = hp
        pbar.set_postfix(best_f1=f"{best_score:.3f}", trial_f1=f"{score:.3f}")
    pbar.close()

    assert best_hp is not None

    # Refit on full outer-train with best HP
    final = spec.factory(best_hp, fold_seed)
    if spec.name == "xgb":
        classes = np.unique(y_train)
        class_to_int = {c: i for i, c in enumerate(classes)}
        y_train_int = np.array([class_to_int[c] for c in y_train], dtype=np.int32)
        sample_weight = compute_sample_weight("balanced", y_train)
        final.fit(X_train, y_train_int, clf__sample_weight=sample_weight)
        proba_test = final.predict_proba(X_test)
        # Reorder columns to canonical PRIMARY_CLASSES order [H2O, Non-STEC, STEC, Salmonella]
        from atlas.evaluate import PRIMARY_CLASSES
        col_map = [list(classes).index(c) for c in PRIMARY_CLASSES]
        proba_test = proba_test[:, col_map]
    else:
        final.fit(X_train, y_train)
        proba_test = final.predict_proba(X_test)
        # sklearn classifiers' classes_ may be in a different order; reorder
        from atlas.evaluate import PRIMARY_CLASSES
        classes_ = final.classes_ if hasattr(final, "classes_") else final.named_steps["clf"].classes_
        col_map = [list(classes_).index(c) for c in PRIMARY_CLASSES]
        proba_test = proba_test[:, col_map]

    dt = time.perf_counter() - t0
    log_fn(
        f"  [{spec.name}] best_inner_f1={best_score:.3f}  best_hp={best_hp}  "
        f"train_time={dt:.1f}s"
    )
    return proba_test, best_hp, dt
