#!/usr/bin/env python3
"""Compare protein representations as brightness oracles, on the stratified splits.

Written because the earlier sweep did not give ESM-2 a fair test. It reported
only two featurizations, and both were broken for this task:

  meanpool   averages the per-residue difference from wild type over all 237
             positions, which destroys WHICH position mutated. A K->E at
             position 10 and at position 200 give nearly the same vector. It
             scored 0.645 and that number says nothing about ESM-2.
  posdelta   keeps position by flattening to 237 x d = 75,840 features, and then
             died with MemoryError because Ridge's dense solver forms a
             75,840 x 75,840 Gram matrix. It never fit at all.

So the featurizations here all keep position, and none of them needs a dense
solve that large:

  onehot     one indicator per (position, residue). The baseline. 4,740 features.
  meanpool   kept only to show what the earlier sweep was measuring.
  pca<k>     the per-residue delta projected to k dimensions per position, so
             237 x k features. Position is preserved and the count is comparable
             to onehot. This is the honest test of whether ESM-2's representation
             carries information beyond substitution identity.
  aaemb      ESM-2's learned amino-acid embedding for the substituted residue,
             placed in that position's block. Sparse, needs no forward pass, and
             unlike onehot it can generalize: an unmeasured substitution at a
             measured position is predicted from chemically similar residues that
             WERE measured there. This is the featurization that could lift the
             38.9% support ceiling.
  aug        onehot concatenated with pca<k>. Hsu et al. (2022) found that
             augmenting indicators with a pretrained signal beats either alone
             once assay labels are plentiful, which is the regime here.

Every representation is scored three ways, because a single Spearman hides the
thing that actually broke this study: accuracy on substitutions the training
split contains, accuracy on substitutions it does not, and what fraction of a
held-out set each covers.

Usage:
  python esm_compare.py --limit 4000                      # quick, one model
  python esm_compare.py --models onehot aaemb pca16 aug
  python esm_compare.py --esm esm2_150m --models pca16 aug
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, hstack as sparse_hstack
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

from gfp_oracle import AA_INDEX, AMINO_ACIDS, featurize, support_of

ROOT = Path(__file__).resolve().parent
ESM_HF = {"esm2_8m": "facebook/esm2_t6_8M_UR50D",
          "esm2_35m": "facebook/esm2_t12_35M_UR50D",
          "esm2_150m": "facebook/esm2_t30_150M_UR50D",
          "esm2_650m": "facebook/esm2_t33_650M_UR50D"}
PREDICTIONS = {}
ALPHAS = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0)


METL_TASKS = {
    "in-support":  "splits/standard/standard_tr0.8_tu0.1_te0.1_w1abc2f4e9a64_r3597",
    "unseen-subs": "splits/mutation/mutation_tr-muts0.8_tu0.1_r4419",
    "unseen-pos":  "splits/position/position_tr-pos0.8_tu0.1_r6822",
}


def read_split(path):
    rows = list(csv.DictReader(open(path, newline="")))
    return ([r["sequence"].strip().upper() for r in rows],
            np.array([float(r["score"]) for r in rows], dtype=np.float64))


def find_dms_root():
    """METL's avgfp directory, under either checkout name."""
    for candidate in (ROOT.parent.parent / "metl" / "data" / "dms_data" / "avgfp",
                      ROOT.parent.parent / "Data_GFP" / "data" / "dms_data" / "avgfp"):
        if (candidate / "avgfp.tsv").is_file():
            return candidate
    raise SystemExit("cannot find METL's dms_data/avgfp directory")


def read_metl_task(task, wt):
    """Load one of METL's split families, which hold row indices into avgfp.tsv.

    These exist because the natural split cannot answer the question that matters
    here. A random 5% test slice of this assay is 99.3% in-support -- 17 variants
    carry a substitution the training split never saw -- so it measures
    interpolation almost exclusively. METL's mutation split instead partitions on
    the substitutions themselves, so test variants are guaranteed to carry
    substitutions absent from train, and the position split withholds whole
    positions. That is where a pretrained representation can beat indicators on
    something other than a rounding error: an indicator for an unseen
    substitution has no coefficient at all.
    """
    from prepare_gfp import apply_variant
    dms = find_dms_root()
    rows = [line.rstrip().split("\t") for line in
            (dms / "avgfp.tsv").read_text().splitlines()[1:]]
    out = {}
    for name, filename in (("train", "train.txt"), ("val", "val.txt"),
                           ("test", "test.txt")):
        path = dms / METL_TASKS[task] / filename
        if not path.is_file():
            raise SystemExit(f"missing {path}")
        index = [int(x) for x in path.read_text().split()]
        out[name] = ([apply_variant(wt, rows[i][0]) for i in index],
                     np.array([float(rows[i][2]) for i in index], dtype=np.float64))
    return out


def device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def aa_embedding_features(sequences, wt, tag, cache_dir):
    """Substituted residue's ESM-2 input embedding, placed in its position block.

    No forward pass: this is ESM-2's amino-acid embedding table, which already
    encodes residue chemistry. The design matrix is sparse -- a variant with m
    mutations touches m blocks -- so the whole thing costs almost nothing and
    still gives the model a reason to believe that an unmeasured D->E behaves
    like a measured D->Q.
    """
    import torch
    from transformers import AutoTokenizer, EsmForMaskedLM
    tokenizer = AutoTokenizer.from_pretrained(ESM_HF[tag], cache_dir=str(cache_dir))
    model = EsmForMaskedLM.from_pretrained(ESM_HF[tag], cache_dir=str(cache_dir),
                                          use_safetensors=True)
    table = model.esm.embeddings.word_embeddings.weight.detach().numpy()
    ids = tokenizer.convert_tokens_to_ids(list(AMINO_ACIDS))
    vectors = table[ids].astype(np.float32)                   # 20 x d
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9
    del model
    dim = vectors.shape[1]
    indices, data, indptr = [], [], [0]
    for seq in sequences:
        for position, (ref, alt) in enumerate(zip(wt, seq)):
            if alt != ref and alt in AA_INDEX:
                base = position * dim
                indices.extend(range(base, base + dim))
                data.extend(vectors[AA_INDEX[alt]])
        indptr.append(len(indices))
    return csr_matrix((np.array(data, dtype=np.float32), indices, indptr),
                      shape=(len(sequences), len(wt) * dim))


def per_residue_delta(sequences, wt, tag, cache_dir, batch_size=32):
    """Stream ESM-2 per-residue embeddings minus wild type. Yields float32 blocks."""
    import torch
    from transformers import AutoTokenizer, EsmForMaskedLM
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
            chunk = sequences[start:start + batch_size]
            yield (embed(chunk) - wt_latent).cpu().numpy().astype(np.float32)


def projected_delta(splits, wt, tag, ks, cache_dir, batch_size=32, pca_n=1500):
    """Per-position ESM-2 delta projected to k dimensions per position, for every k.

    ESM-2 is run over the data exactly once. PCA components come out ordered by
    variance, so projecting to max(ks) and keeping the leading k columns of each
    position's block is identical to having projected to k directly -- every
    smaller k is free. Encoding once per k instead was costing six passes over
    51,714 sequences.

    Two passes total: one over a subsample to fit the projection, one over
    everything to apply it. The full per-residue tensor is never materialized.

    One projection is shared across positions. A per-position basis would be
    fitted on ~175 variants each and would mostly learn noise; a shared basis
    asks which directions of the residue representation matter anywhere.
    """
    k_max = max(ks)
    rng = np.random.default_rng(0)
    pool = splits["train"][0]
    sample = [pool[i] for i in rng.choice(len(pool), min(pca_n, len(pool)),
                                          replace=False)]
    print(f"    fitting a {k_max}-dimensional projection on {len(sample)} variants",
          flush=True)
    pooled = np.concatenate([b.reshape(-1, b.shape[-1])
                             for b in per_residue_delta(sample, wt, tag,
                                                        cache_dir, batch_size)])
    from sklearn.decomposition import PCA
    pca = PCA(n_components=k_max, svd_solver="randomized", random_state=0).fit(pooled)
    ratios = np.cumsum(pca.explained_variance_ratio_)
    print("    delta variance retained: "
          + ", ".join(f"k={k}: {100*ratios[k-1]:.1f}%" for k in sorted(ks)), flush=True)
    del pooled

    full = {}
    for name, (sequences, _) in splits.items():
        started, blocks = time.time(), []
        for block in per_residue_delta(sequences, wt, tag, cache_dir, batch_size):
            n, length, _ = block.shape
            blocks.append(pca.transform(block.reshape(-1, block.shape[-1]))
                          .astype(np.float32).reshape(n, length, k_max))
        full[name] = np.concatenate(blocks)
        print(f"    {name}: encoded {full[name].shape[0]} variants in "
              f"{time.time()-started:.0f}s", flush=True)
    return full


def slice_k(full, k, flatten=True):
    """The leading k components of every position block."""
    out = {}
    for name, arr in full.items():
        block = arr[:, :, :k]
        out[name] = block.reshape(len(block), -1).copy() if flatten else block
    return out


def fit_and_score(name, features, splits, support, wt, results):
    """Choose alpha on val, report on test, split out in- and off-support rows."""
    x = {k: features[k] for k in splits}
    y = {k: splits[k][1] for k in splits}
    sparse = not isinstance(x["train"], np.ndarray)
    solver = "sparse_cg" if sparse or x["train"].shape[1] > 20000 else "auto"
    best = None
    for alpha in ALPHAS:
        model = Ridge(alpha=alpha, solver=solver).fit(x["train"], y["train"])
        rho = float(spearmanr(model.predict(x["val"]), y["val"]).statistic)
        if best is None or rho > best[1]:
            best = (model, rho, alpha)
    model, val_rho, alpha = best
    predicted = model.predict(x["test"])
    inside = np.array([all(support[p, AA_INDEX[b]]
                           for p, (a, b) in enumerate(zip(wt, s))
                           if a != b and b in AA_INDEX)
                       for s in splits["test"][0]])
    row = {"model": name, "features": int(x["train"].shape[1]), "alpha": alpha,
           "val_rho": round(val_rho, 4),
           "test_rho": round(float(spearmanr(predicted, y["test"]).statistic), 4),
           "test_mae": round(float(np.abs(predicted - y["test"]).mean()), 4),
           "in_support_frac": round(float(inside.mean()), 4)}
    for label, mask in (("in", inside), ("off", ~inside)):
        if mask.sum() > 10:
            p, t = predicted[mask], y["test"][mask]
            constant = bool(np.ptp(p) < 1e-9)
            row[f"rho_{label}"] = None if constant else round(
                float(spearmanr(p, t).statistic), 4)
            row[f"n_{label}"] = int(mask.sum())
    if row["alpha"] in (min(ALPHAS), max(ALPHAS)):
        row["alpha_at_grid_edge"] = True
    PREDICTIONS[name] = predicted
    results.append(row)
    print(f"  {name:<20}{row['features']:>9}{row['test_rho']:>10.4f}"
          f"{str(row.get('rho_in')):>10}{str(row.get('rho_off')):>10}"
          f"{row['test_mae']:>8.4f}{row['alpha']:>7}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+",
                   default=["onehot", "meanpool", "aaemb", "pca16", "aug"],
                   help="onehot, meanpool, aaemb, pca<k> for any k, aug (onehot+pca<k>)")
    p.add_argument("--esm", default="esm2_150m", choices=list(ESM_HF))
    p.add_argument("--data", type=Path, default=ROOT / "data")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap each split, for a fast check before the real run")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--cache-dir", type=Path, default=ROOT / "cache")
    p.add_argument("--task", default="stratified",
                   choices=["stratified"] + list(METL_TASKS),
                   help="Which split to evaluate on (default: stratified, the "
                        "80/15/5 partition from split_gfp.py). The METL families "
                        "withhold substitutions or positions instead of rows, which "
                        "is the only way to measure extrapolation -- the stratified "
                        "test set is 99.3%% in-support.")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    wt = (args.data / "avgfp_wt.txt").read_text().strip()
    if args.task == "stratified":
        splits = {name: read_split(args.data / f"avgfp_{name}.csv")
                  for name in ("train", "val", "test")}
    else:
        splits = read_metl_task(args.task, wt)
    if args.limit:
        splits = {k: (s[:args.limit], y[:args.limit]) for k, (s, y) in splits.items()}
    support = support_of(splits["train"][0], wt)
    support[np.arange(len(wt)), [AA_INDEX[a] for a in wt]] = False
    print(f"\nwild type {len(wt)} residues | ESM-2 model {args.esm} "
          f"| split: {args.task}")
    for name, (s, _) in splits.items():
        print(f"  {name:<6}{len(s):>7} variants")
    print(f"  train support: {int(support.sum())} of {len(wt)*19} substitutions "
          f"({100*support.sum()/(len(wt)*19):.1f}%)")

    ks = sorted({int(m[3:]) for m in args.models if m.startswith("pca")})
    if "aug" in args.models or "meanpool" in args.models:
        ks = sorted(set(ks) | {16})
    full = None
    if ks:
        print(f"\n[ESM-2 per-position delta, k = {ks}]", flush=True)
        full = projected_delta(splits, wt, args.esm, ks,
                               args.cache_dir, args.batch_size)

    onehot = {n: featurize(s, wt) for n, (s, _) in splits.items()}
    results = []
    print(f"\n  {'model':<20}{'features':>9}{'test rho':>10}{'in-supp':>10}"
          f"{'off-supp':>10}{'MAE':>8}{'alpha':>7}")
    print("  " + "-" * 74)
    for model in args.models:
        if model == "onehot":
            fit_and_score("onehot", onehot, splits, support, wt, results)
        elif model == "meanpool":
            feats = {n: v[:, :, :16].mean(1) for n, v in full.items()}
            fit_and_score("meanpool(k=16)", feats, splits, support, wt, results)
        elif model == "aaemb":
            feats = {n: aa_embedding_features(s, wt, args.esm, args.cache_dir)
                     for n, (s, _) in splits.items()}
            fit_and_score("aaemb", feats, splits, support, wt, results)
            del feats
        elif model.startswith("pca"):
            k = int(model[3:])
            feats = slice_k(full, k)
            fit_and_score(f"pca{k}", feats, splits, support, wt, results)
            del feats
        elif model == "aug":
            k = max(ks)
            feats = {n: sparse_hstack([onehot[n], csr_matrix(v)], format="csr")
                     for n, v in slice_k(full, k).items()}
            fit_and_score(f"aug(onehot+pca{k})", feats, splits, support, wt, results)
            del feats
        import gc; gc.collect()
    if PREDICTIONS:
        y = splits["test"][1]
        rng = np.random.default_rng(0)
        draws = rng.integers(0, len(y), size=(2000, len(y)))
        base = PREDICTIONS.get("onehot")
        # A constant baseline has no rank correlation to difference against, which
        # is itself the finding on a withheld-substitution split: indicators have
        # no coefficient for any substitution in the test set, so ridge returns
        # its intercept. Report each model's own interval in that case.
        base_ok = base is not None and np.ptp(base) > 1e-9
        print(f"\n  bootstrap, 2000 resamples of the same {len(y)} test rows"
              + ("" if base_ok else "  (onehot is constant — absolute intervals)"))
        header = f"  {'model':<20}{'test rho':>10}{'95% interval':>20}"
        if base_ok:
            header += f"{'d vs onehot':>13}{'P(better)':>11}"
        print(header + "\n  " + "-" * (len(header) - 2))
        for name, predicted in PREDICTIONS.items():
            if np.ptp(predicted) < 1e-9:
                print(f"  {name:<20}{'CONSTANT':>10}")
                continue
            rhos = np.array([spearmanr(predicted[d], y[d]).statistic for d in draws])
            low, high = np.percentile(rhos, [2.5, 97.5])
            line = f"  {name:<20}{rhos.mean():>10.4f}{f'[{low:.4f}, {high:.4f}]':>20}"
            record = {"rho_ci95": [round(float(low), 4), round(float(high), 4)]}
            if base_ok and name != "onehot":
                deltas = rhos - np.array([spearmanr(base[d], y[d]).statistic
                                          for d in draws])
                line += f"{deltas.mean():>+13.4f}{100*(deltas > 0).mean():>10.1f}%"
                record["delta_rho_vs_onehot"] = round(float(deltas.mean()), 4)
                record["p_better"] = round(float((deltas > 0).mean()), 4)
            print(line)
            for row in results:
                if row["model"] == name:
                    row.update(record)
    out = args.out or args.data / f"esm_compare_{args.esm}_{args.task}.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nWrote {out}")
    print("in-supp / off-supp split the SAME test set by whether every substitution\n"
          "a variant carries appears in train. 'None' means the prediction was\n"
          "constant, so the representation knows nothing about those substitutions.")


if __name__ == "__main__":
    main()
