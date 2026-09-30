"""Fitting, saving and scoring the ridge oracle.

The oracle is a ridge regression on whichever representation a backend
produces, so the fitting code is identical for METL and ESM-2.
"""
import csv
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

from .encoders import encode, fit_projection, parse_backend

# ══════════════════════════════════════════════════════════════════════════════
# Fit / score
# ══════════════════════════════════════════════════════════════════════════════

def read_split(csv_path: Path) -> tuple[list[str], np.ndarray]:
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return ([r["sequence"].strip().upper() for r in rows],
            np.array([float(r["score"]) for r in rows], dtype=np.float32))


def load_fasta(path: Path) -> list[str]:
    sequences, current = [], ""
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if current:
                sequences.append(current)
            current = ""
        else:
            current += line.strip()
    return sequences + ([current] if current else [])


def fit_oracle(sequences, scores, wt, backend, alpha=1.0, val_fraction=0.2, seed=7):
    """Fit ridge on the backend's representation; report held-out agreement."""
    features = encode(sequences, wt, backend)
    rng   = np.random.default_rng(seed)
    order = rng.permutation(len(sequences))
    n_val = int(len(order) * val_fraction)
    val, fit = order[:n_val], order[n_val:]
    model = Ridge(alpha=alpha).fit(features[fit], scores[fit])
    predicted = model.predict(features[val])
    rho = spearmanr(predicted, scores[val]).statistic
    mae = float(np.abs(predicted - scores[val]).mean())
    domain = {"score_min": float(scores.min()), "score_max": float(scores.max())}
    return model, float(rho), mae, domain, features.shape[1]


def fit_on_splits(train, val, test, wt, backend,
                  alphas=(0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0,
                          300.0, 1000.0, 3000.0)):
    """Fit on train, choose the ridge penalty on val, report once on test.

    A dense representation has no sparse support to worry about -- METL and ESM
    embed any sequence -- so unlike the indicator oracle there is no in-support
    subset to report. The splits still matter: the penalty has to be chosen
    somewhere that is not the test set, and the stratified partition keeps the
    mutation-count profile of all three slices matched to the assay.
    """
    tag, k = parse_backend(backend)
    projection = (None if k is None
                  else fit_projection(train[0], wt, tag, k))
    train_x = encode(train[0], wt, backend, projection=projection)
    val_x = encode(val[0], wt, backend, projection=projection)
    print(f"  selecting the ridge penalty on {len(val[0])} validation variants")
    tried = []
    for alpha in alphas:
        model = Ridge(alpha=alpha).fit(train_x, train[1])
        predicted = model.predict(val_x)
        tried.append((model, alpha,
                      {"rho": float(spearmanr(predicted, val[1]).statistic),
                       "mae": float(np.abs(predicted - val[1]).mean()),
                       "n": len(val[0])}))
    best = max(tried, key=lambda t: t[2]["rho"])
    for candidate in tried:
        _, alpha, stats = candidate
        print(f"    alpha {alpha:>5}: val Spearman {stats['rho']:.4f}  "
              f"MAE {stats['mae']:.4f}"
              + ("  <-- chosen" if candidate is best else ""))
    model, alpha, val_stats = best
    predicted = model.predict(encode(test[0], wt, backend, projection=projection))
    test_stats = {"rho": float(spearmanr(predicted, test[1]).statistic),
                  "mae": float(np.abs(predicted - test[1]).mean()),
                  "n": len(test[0])}
    domain = {"score_min": float(train[1].min()), "score_max": float(train[1].max())}
    return (model, alpha, {"val": val_stats, "test": test_stats}, domain,
            train_x.shape[1], projection)


def save_oracle(path: Path, model, wt, backend, rho, mae, domain, n_features,
                projection=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    extra = {}
    if projection is not None:
        # The projection is half the featurizer; an oracle saved without it can
        # load but cannot score.
        extra = {"pca_mean": projection[0], "pca_components": projection[1]}
    np.savez(path, coef=model.coef_.astype(np.float32),
             intercept=np.float32(model.intercept_), wt=wt, backend=backend,
             rho=rho, mae=mae, n_features=n_features,
             score_min=domain["score_min"], score_max=domain["score_max"], **extra)


def load_oracle(path: Path):
    """Return (coef, intercept, wt, backend, domain).

    A PCA-projected backend carries its projection in `domain["projection"]`, so
    the returned tuple stays the shape every caller already expects.
    """
    saved = np.load(path, allow_pickle=False)
    projection = None
    if "pca_components" in saved.files:
        projection = (saved["pca_mean"], saved["pca_components"])
    domain = {"score_min": float(saved["score_min"]),
              "score_max": float(saved["score_max"]),
              "projection": projection}
    return (saved["coef"], float(saved["intercept"]), str(saved["wt"]),
            str(saved["backend"]), domain)


def score_sequences(sequences: list[str], oracle, clip: bool = True) -> np.ndarray:
    """Predicted wild-type-centered brightness.

    Ridge stays linear in the representation, so a sequence far outside the
    fitted range still extrapolates without bound. Clipping holds predictions to
    the measured score range, which is the only range the label means anything in.
    """
    coef, intercept, wt, backend, domain = oracle
    features = encode(list(sequences), wt, backend,
                      projection=domain.get("projection"))
    predicted = features @ coef + intercept
    if clip:
        predicted = predicted.clip(domain["score_min"], domain["score_max"])
    return np.asarray(predicted, dtype=np.float64)

