#!/usr/bin/env python3
"""Embedding-based brightness oracles for the avGFP study.

The ridge oracle in gfp_oracle.py scores a variant from indicators of its exact
(position, residue) substitutions. That works well inside the measured library
and not at all outside it: the avGFP deep mutational scan contains only 1,810 of
4,503 possible single substitutions, so any substitution the assay never sampled
contributes a zero and the prediction collapses to the intercept. Generated
sequences are mostly made of such substitutions.

This module fits the same ridge regression on a learned representation instead,
which assigns a value to substitutions it has never seen.

Two backends:

  metl   METL-L-2M-3D-GFP, a 2.5M-parameter transformer pretrained to predict 55
         Rosetta biophysical attributes from sequence (Gelman et al., Nature
         Methods 2025). We read its 256-dimensional pooled representation. It
         shares no architecture, tokenizer or training corpus with the ESM-2
         encoder the generator samples in, so it is an independent judge.

  esm2_* Per-residue ESM-2 embeddings, taken as the difference from wild type and
         flattened, keeping positional resolution. Mean pooling over 237 residues
         destroys the signal from a handful of substitutions and is not used.
         Note this backend shares its representation with the generator, so a
         score it assigns to a generated sequence is not fully independent.

Usage:
  python embedding_oracle.py --fit --backend metl
  python embedding_oracle.py --fit --backend esm2_150m
  python embedding_oracle.py --score some.fasta --backend metl

Import:
  from embedding_oracle import load_oracle, score_sequences
"""
import argparse
import contextlib
import csv
import os
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

ROOT        = Path(__file__).resolve().parent
METL_ROOT   = ROOT.parent.parent / "metl"
METL_CKPT   = METL_ROOT / "pretrained_models" / "Hr4GNHws.pt"
METL_PDB    = "1gfl_cm.pdb"
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

ESM_HF = {
    "esm2_8m":   "facebook/esm2_t6_8M_UR50D",
    "esm2_35m":  "facebook/esm2_t12_35M_UR50D",
    "esm2_150m": "facebook/esm2_t30_150M_UR50D",
    "esm2_650m": "facebook/esm2_t33_650M_UR50D",
    "esm2_3b":   "facebook/esm2_t36_3B_UR50D",
    "esm2_15b":  "facebook/esm2_t48_15B_UR50D",
}


def device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ══════════════════════════════════════════════════════════════════════════════
# Sequence helpers
# ══════════════════════════════════════════════════════════════════════════════

def seq_to_variant(sequence: str, wt: str) -> str:
    """METL's mutation-string form, e.g. 'K1E,A108D'. '_wt' when identical."""
    muts = [f"{a}{i}{b}" for i, (a, b) in enumerate(zip(wt, sequence)) if a != b]
    return ",".join(muts) if muts else "_wt"


@contextlib.contextmanager
def chdir(path: Path):
    """METL resolves its PDB files relative to the repository root."""
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


# ══════════════════════════════════════════════════════════════════════════════
# Backends
# ══════════════════════════════════════════════════════════════════════════════

# metl/code/models.py and this directory's models.py share a name, and whichever
# landed in sys.modules first wins. METL's inference.py does a plain `import
# models`, so without isolation it picks up our ESM-2 registry and fails with
# "module 'models' has no attribute 'Model'".
_METL_MODULE_NAMES = (
    "models", "tasks", "encode", "utils", "structure", "relative_attention",
    "datasets", "datamodules", "training_utils", "inference", "constants",
    "metrics", "rosetta_data_utils", "pdb_sampler", "analysis_utils",
)
_METL_CACHE: dict = {}


@contextlib.contextmanager
def metl_imports():
    """Let METL import its own modules, then put ours back."""
    import sys
    saved_path = list(sys.path)
    saved_modules = {name: sys.modules.pop(name)
                     for name in _METL_MODULE_NAMES if name in sys.modules}
    sys.path.insert(0, str(METL_ROOT / "code"))
    try:
        yield
    finally:
        sys.path[:] = saved_path
        for name in _METL_MODULE_NAMES:
            sys.modules.pop(name, None)
        sys.modules.update(saved_modules)


def _load_metl():
    """Load METL once and keep it; the checkpoint costs seconds to rebuild."""
    if "model" in _METL_CACHE:
        return _METL_CACHE["model"], _METL_CACHE["encoder"]
    if not METL_CKPT.is_file():
        raise FileNotFoundError(
            f"METL checkpoint not found at {METL_CKPT}. It ships with the metl "
            f"repository under pretrained_models/."
        )
    with metl_imports(), chdir(METL_ROOT):
        from inference import load_pytorch_module
        import encode as metl_encode
        model = load_pytorch_module(str(METL_CKPT), pdb_fns=[METL_PDB]).eval().to(device())
        # Hold a reference to the function itself: the module is about to be
        # removed from sys.modules again.
        encoder = metl_encode.encode
    _METL_CACHE.update(model=model, encoder=encoder)
    return model, encoder


def encode_metl(sequences: list[str], wt: str, batch_size: int = 64) -> np.ndarray:
    """Pooled 256-dimensional METL representation, one row per sequence."""
    import torch
    model, encoder = _load_metl()
    captured = {}
    handle = model.model.fc1.register_forward_hook(
        lambda mod, inp, out: captured.__setitem__("h", out.detach().cpu())
    )
    out = []
    try:
        with chdir(METL_ROOT), torch.no_grad():
            for start in range(0, len(sequences), batch_size):
                chunk = sequences[start:start + batch_size]
                variants = [seq_to_variant(s, wt) for s in chunk]
                x = encoder(encoding="int_seqs", variants=variants,
                            wt_aa=wt, wt_offset=0)
                model(torch.tensor(x).to(device()), pdb_fn=METL_PDB)
                out.append(captured["h"].numpy().astype(np.float32))
    finally:
        handle.remove()
    return np.concatenate(out)


# ══════════════════════════════════════════════════════════════════════════════
# Rosetta biophysical attributes (METL pretraining head)
# ══════════════════════════════════════════════════════════════════════════════
#
# METL-Local's pretraining task is to predict 55 Rosetta biophysical attributes
# from sequence, and the shipped checkpoint keeps that head intact:
# `model.prediction.weight` is (55, 256). So the attributes come free with a
# forward pass -- no Rosetta run, which matters because Rosetta costs ~135-215 s
# per variant for a 237-residue protein like avGFP (Gelman et al., Nature
# Methods 2025). Twenty thousand generated sequences would be weeks of CPU.
#
# Outputs are in METL's standardized pretraining units (roughly mean 0, sd 1
# across its simulated variants), not raw Rosetta energies.
#
# Measured on 3,000 avgfp_train rows, Spearman against experimental brightness:
#
#     0   total_score           -0.6902    most stable <-> brightest
#    32   pack                  +0.5675
#     1   fa_atr                -0.5834
#    41   total_sasa            -0.2541
#    27   exposed_hydrophobics  -0.1495    near-orthogonal to brightness
#    43   unsat_hbond           -0.1310    near-orthogonal to brightness
#
# total_score is the best stability measure AND the most redundant with
# brightness, so it belongs in a constraint rather than in a scalarized
# objective: weighting it against brightness traces a line, not a Pareto front.
# exposed_hydrophobics and unsat_hbond carry real variance (sd 0.83 and 0.72)
# while being nearly independent of brightness, which is what a second objective
# has to be for the tradeoff to mean anything.

ATTRIBUTE_NAMES: tuple[str, ...] = ()


def attribute_names() -> tuple[str, ...]:
    """The 55 attribute names, in the order the prediction head emits them."""
    global ATTRIBUTE_NAMES
    if not ATTRIBUTE_NAMES:
        with metl_imports():
            import constants
            ATTRIBUTE_NAMES = tuple(constants.ROSETTA_ATTRIBUTES_TRAINING)
    return ATTRIBUTE_NAMES


def attribute_index(name: str) -> int:
    """Index of an attribute by name, so callers need not hardcode integers."""
    names = attribute_names()
    if name not in names:
        raise ValueError(f"Unknown Rosetta attribute '{name}'. "
                         f"Known: {', '.join(names)}")
    return names.index(name)


def metl_attributes(sequences: list[str], wt: str, batch_size: int = 64) -> np.ndarray:
    """Predicted Rosetta attributes, (len(sequences), 55) in standardized units.

    Reads the pretraining head rather than the pooled fc1 representation that
    encode_metl() returns. Same frozen checkpoint, same forward pass.
    """
    import torch
    model, encoder = _load_metl()
    out = []
    with chdir(METL_ROOT), torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            chunk = sequences[start:start + batch_size]
            variants = [seq_to_variant(s, wt) for s in chunk]
            x = encoder(encoding="int_seqs", variants=variants,
                        wt_aa=wt, wt_offset=0)
            y = model(torch.tensor(x).to(device()), pdb_fn=METL_PDB)
            out.append(y.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(out)


def metl_attributes_wt(wt: str) -> np.ndarray:
    """The same 55 attributes for the wild type, as a (55,) vector.

    The stability constraint is expressed relative to this: a design is allowed
    to be less stable than wild type only by a stated margin delta.
    """
    import torch
    model, encoder = _load_metl()
    with chdir(METL_ROOT), torch.no_grad():
        x = encoder(encoding="int_seqs", variants=["_wt"], wt_aa=wt, wt_offset=0)
        y = model(torch.tensor(x).to(device()), pdb_fn=METL_PDB)
    return y.detach().cpu().numpy().astype(np.float32)[0]


def encode_esm(sequences: list[str], wt: str, tag: str,
               cache_dir: Path | None = None, batch_size: int = 32) -> np.ndarray:
    """Per-residue ESM-2 embedding minus wild type, flattened to keep position."""
    import torch
    from transformers import AutoTokenizer, EsmForMaskedLM
    if tag not in ESM_HF:
        raise ValueError(f"Unknown ESM tag '{tag}'. Known: {', '.join(ESM_HF)}")
    cache_dir = cache_dir or ROOT / "cache"
    tokenizer = AutoTokenizer.from_pretrained(ESM_HF[tag], cache_dir=str(cache_dir))
    model = EsmForMaskedLM.from_pretrained(
        ESM_HF[tag], cache_dir=str(cache_dir), use_safetensors=True
    ).to(device()).eval().requires_grad_(False)

    def embed(batch):
        toks = {k: v.to(device()) for k, v in tokenizer(batch, return_tensors="pt").items()}
        return model.esm(**toks).last_hidden_state[:, 1:-1].cpu().numpy().astype(np.float32)

    with torch.no_grad():
        z_wt = embed([wt])[0]
        out = []
        for start in range(0, len(sequences), batch_size):
            h = embed(sequences[start:start + batch_size])
            out.append((h - z_wt).reshape(len(h), -1))
    return np.concatenate(out)


def parse_backend(backend: str):
    """Split a backend name into (tag, k). 'esm2_8m_pca64' -> ('esm2_8m', 64).

    A bare ESM tag keeps the legacy flattened per-residue delta, which is 237 x d
    features and needs an iterative solver. The _pca<k> form is the one measured
    to work: see esm_compare.py, where a 64-component per-position projection beat
    one-hot indicators on the held-out split by +0.0046 Spearman (paired bootstrap
    95% interval [+0.0015, +0.0078]), while mean pooling over positions scored
    0.448 because it discards which position mutated.
    """
    if "_pca" in backend:
        tag, _, k = backend.partition("_pca")
        return tag, int(k)
    return backend, None


def fit_projection(sequences, wt, tag, k, cache_dir=None, batch_size=32, pca_n=1500):
    """PCA basis for the per-residue delta, shared across positions.

    Returned as (mean, components) so it can be saved beside the ridge weights --
    the projection is part of the featurizer, and an oracle that cannot reproduce
    it cannot score anything.
    """
    from sklearn.decomposition import PCA
    rng = np.random.default_rng(0)
    pick = rng.choice(len(sequences), min(pca_n, len(sequences)), replace=False)
    sample = [sequences[i] for i in pick]
    pooled = np.concatenate([b.reshape(-1, b.shape[-1]) for b in
                             _stream_delta(sample, wt, tag, cache_dir, batch_size)])
    pca = PCA(n_components=k, svd_solver="randomized", random_state=0).fit(pooled)
    kept = float(np.sum(pca.explained_variance_ratio_))
    print(f"  projection: {k} components, {100*kept:.1f}% of the delta variance")
    return pca.mean_.astype(np.float32), pca.components_.astype(np.float32)


def _stream_delta(sequences, wt, tag, cache_dir=None, batch_size=32):
    """Yield (batch, length, d) per-residue ESM-2 embeddings minus wild type."""
    import torch
    from transformers import AutoTokenizer, EsmForMaskedLM
    if tag not in ESM_HF:
        raise ValueError(f"Unknown ESM tag '{tag}'. Known: {', '.join(ESM_HF)}")
    cache_dir = cache_dir or ROOT / "cache"
    tokenizer = AutoTokenizer.from_pretrained(ESM_HF[tag], cache_dir=str(cache_dir))
    model = EsmForMaskedLM.from_pretrained(
        ESM_HF[tag], cache_dir=str(cache_dir), use_safetensors=True
    ).to(device()).eval().requires_grad_(False)

    def embed(batch):
        toks = {k: v.to(device())
                for k, v in tokenizer(batch, return_tensors="pt").items()}
        return model.esm(**toks).last_hidden_state[:, 1:-1]

    with torch.no_grad():
        wt_latent = embed([wt])[0]
        for start in range(0, len(sequences), batch_size):
            yield (embed(sequences[start:start + batch_size])
                   - wt_latent).cpu().numpy().astype(np.float32)


def encode_esm_pca(sequences, wt, tag, projection, cache_dir=None, batch_size=32):
    """Per-position ESM-2 delta projected onto a saved basis: 237 x k features."""
    mean, components = projection
    out = []
    for block in _stream_delta(sequences, wt, tag, cache_dir, batch_size):
        n, length, _ = block.shape
        flat = block.reshape(-1, block.shape[-1]) - mean
        out.append((flat @ components.T).reshape(n, length * len(components)))
    return np.concatenate(out).astype(np.float32)


def encode(sequences: list[str], wt: str, backend: str, projection=None,
           **kwargs) -> np.ndarray:
    if backend == "metl":
        return encode_metl(sequences, wt, **kwargs)
    tag, k = parse_backend(backend)
    if k is not None:
        if projection is None:
            raise ValueError(f"backend '{backend}' needs its saved PCA projection")
        return encode_esm_pca(sequences, wt, tag, projection, **kwargs)
    return encode_esm(sequences, wt, backend, **kwargs)


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


# ══════════════════════════════════════════════════════════════════════════════
# Parse args
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", default="metl",
                        help=f"metl (default) or one of: {', '.join(ESM_HF)}")
    parser.add_argument("--train", type=Path, default=None,
                        help="Training CSV from split_gfp.py (default: "
                             "data/avgfp_train.csv when it exists). With --train the "
                             "penalty is chosen on --val and reported on --test.")
    parser.add_argument("--val", type=Path, default=None,
                        help="Validation CSV; selects the ridge penalty.")
    parser.add_argument("--test", type=Path, default=None,
                        help="Test CSV; read once, for the reported accuracy.")
    parser.add_argument("--split", type=Path, default=ROOT / "data" / "avgfp_oracle.csv",
                        help="Oracle-split CSV from prepare_gfp.py")
    parser.add_argument("--wt", type=Path, default=None,
                        help="Wild-type file (default: <split dir>/avgfp_wt.txt)")
    parser.add_argument("--oracle", type=Path, default=None,
                        help="Where to save/load (default: data/avgfp_<backend>_oracle.npz)")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None,
                        help="Use at most this many variants from the split")
    parser.add_argument("--fit", action="store_true")
    parser.add_argument("--score", type=Path, default=None, help="FASTA to score")
    return parser.parse_args()


def main():
    args = parse_args()
    out = args.oracle or ROOT / "data" / f"avgfp_{args.backend}_oracle.npz"
    default_train = ROOT / "data" / "avgfp_train.csv"
    if args.fit and args.train is None and default_train.is_file():
        args.train = default_train
        args.val = args.val or ROOT / "data" / "avgfp_val.csv"
        args.test = args.test or ROOT / "data" / "avgfp_test.csv"
    if args.fit and args.train is not None:
        for name, path in (("--val", args.val), ("--test", args.test)):
            if path is None or not path.is_file():
                raise SystemExit(f"{name} is required with --train (got {path}). "
                                 f"Build the splits with: python split_gfp.py")
        wt = (args.wt or args.train.parent / "avgfp_wt.txt").read_text().strip()
        train, val, test = (read_split(q) for q in (args.train, args.val, args.test))
        if args.limit:
            train = (train[0][:args.limit], train[1][:args.limit])
        print(f"\nFitting the {args.backend} oracle on {len(train[0])} "
              f"training variants...")
        model, alpha, stats, domain, n_features, projection = fit_on_splits(
            train, val, test, wt, args.backend)
        save_oracle(out, model, wt, args.backend, stats["val"]["rho"],
                    stats["val"]["mae"], domain, n_features, projection)
        print(f"\n  representation: {n_features} features")
        print(f"  chosen penalty alpha = {alpha}")
        for name in ("val", "test"):
            s = stats[name]
            print(f"  {name:<5} n={s['n']:<6} Spearman {s['rho']:.4f}  "
                  f"MAE {s['mae']:.4f}")
        print(f"  saved to {out}")
    elif args.fit:
        sequences, scores = read_split(args.split)
        if args.limit:
            sequences, scores = sequences[:args.limit], scores[:args.limit]
        wt = (args.wt or args.split.parent / "avgfp_wt.txt").read_text().strip()
        print(f"Fitting {args.backend} oracle on {len(sequences)} variants...")
        model, rho, mae, domain, n_features = fit_oracle(
            sequences, scores, wt, args.backend, args.alpha)
        save_oracle(out, model, wt, args.backend, rho, mae, domain, n_features)
        print(f"  representation: {n_features} features")
        print(f"  validation Spearman {rho:.3f} | MAE {mae:.3f} | saved to {out}")
    if args.score is not None:
        oracle = load_oracle(out)
        sequences = load_fasta(args.score)
        predicted = score_sequences(sequences, oracle)
        for sequence, value in zip(sequences, predicted):
            print(f"  {value:+.3f}  {sequence[:40]}...")
        print(f"mean predicted brightness {predicted.mean():+.3f}")
    if not args.fit and args.score is None:
        print("Nothing to do: pass --fit and/or --score")


if __name__ == "__main__":
    main()
