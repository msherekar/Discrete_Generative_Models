#!/usr/bin/env python3
"""Which ESM-2 featurization can rank substitutions the assay never measured?

Four featurizations, one ridge regression, two tasks:

  onehot         indicator per (position, residue) substitution
  esm_mean       ESM-2 embedding averaged over residues
  esm_delta      (embedding - wild type) averaged over residues
  esm_flatdelta  (embedding - wild type) flattened, keeping positional resolution

esm_mean and esm_delta score identically, which is not a bug: ridge with an
intercept is invariant to subtracting a constant vector, so averaging a delta
discards exactly what averaging the raw embedding discards. Only the flattened
form keeps WHERE the sequence changed, and only it generalizes.

oracle_sweep.py supersedes this for comparing encoders across model sizes. This
script is kept because it isolates the pooling question, which is the reason
oracle_sweep.py reports posdelta rather than meanpool.

Usage:
  python diagnostics/check_esm_features.py
  python diagnostics/check_esm_features.py --train-n 8000 --esm-model esm2_150m
"""
import argparse

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from transformers import AutoTokenizer, EsmForMaskedLM

from dgm.common.paths import REPO_ROOT, project_dir

ROOT = project_dir()
from ..gfp_oracle import featurize                            # noqa: E402
from ..embedding_oracle import ESM_HF                         # noqa: E402

DMS = REPO_ROOT / "Data_GFP" / "data" / "dms_data" / "avgfp"
TASKS = {
    "A in-support (standard)":
        "splits/standard/standard_tr0.8_tu0.1_te0.1_w1abc2f4e9a64_r3597",
    "B unseen substitutions (mutation)":
        "splits/mutation/mutation_tr-muts0.8_tu0.1_r4419",
}


def apply_variant(wt, variant):
    seq = list(wt)
    for m in variant.split(","):
        seq[int(m[1:-1])] = m[-1]
    return "".join(seq)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--esm-model", default="esm2_8m", choices=list(ESM_HF))
    parser.add_argument("--train-n", type=int, default=4000)
    args = parser.parse_args()

    rows = [l.rstrip().split("\t") for l in (DMS / "avgfp.tsv").read_text().splitlines()[1:]]
    wt = (ROOT / "data" / "avgfp_wt.txt").read_text().strip()

    def indices(path):
        return [int(x) for x in (DMS / path).read_text().split()]

    selection = {}
    for name, d in TASKS.items():
        train = indices(f"{d}/train.txt")[:args.train_n]
        test  = indices(f"{d}/test.txt")
        assert not set(train) & set(test), f"{name}: train and test overlap"
        selection[name] = (train, test)

    index = sorted({i for tr, te in selection.values() for i in tr + te})
    sequences = {i: apply_variant(wt, rows[i][0]) for i in index}
    print(f"encoding {len(index)} unique sequences with {args.esm_model} ...", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hf = ESM_HF[args.esm_model]
    tokenizer = AutoTokenizer.from_pretrained(hf, cache_dir=str(ROOT / "cache"))
    esm = EsmForMaskedLM.from_pretrained(
        hf, cache_dir=str(ROOT / "cache"), use_safetensors=True
    ).to(device).eval().requires_grad_(False)

    emb = {}
    with torch.no_grad():
        for start in range(0, len(index), 64):
            chunk = index[start:start + 64]
            toks = {k: v.to(device) for k, v in
                    tokenizer([sequences[i] for i in chunk], return_tensors="pt").items()}
            h = esm.esm(**toks).last_hidden_state[:, 1:-1].cpu().numpy().astype(np.float32)
            for j, i in enumerate(chunk):
                emb[i] = h[j]
        toks = {k: v.to(device) for k, v in tokenizer([wt], return_tensors="pt").items()}
        z_wt = esm.esm(**toks).last_hidden_state[:, 1:-1].cpu().numpy().astype(np.float32)[0]

    def features(ii, kind):
        if kind == "onehot":
            return featurize([sequences[i] for i in ii], wt)
        stack = np.stack([emb[i] for i in ii])
        if kind == "esm_mean":
            return stack.mean(1)
        if kind == "esm_delta":
            return (stack - z_wt).mean(1)
        return (stack - z_wt).reshape(len(ii), -1)

    print(f"\n{'task':<36}{'features':<16}{'Spearman':>10}{'MAE':>9}{'constant':>10}")
    for name, (train, test) in selection.items():
        y_train = np.array([float(rows[i][2]) for i in train])
        y_test  = np.array([float(rows[i][2]) for i in test])
        for kind in ("onehot", "esm_mean", "esm_delta", "esm_flatdelta"):
            model = Ridge(alpha=1.0).fit(features(train, kind), y_train)
            predicted = model.predict(features(test, kind))
            constant = bool(np.ptp(predicted) < 1e-9)
            rho = "n/a" if constant else f"{spearmanr(predicted, y_test).statistic:+.3f}"
            print(f"{name:<36}{kind:<16}{rho:>10}"
                  f"{np.abs(predicted - y_test).mean():>9.3f}"
                  f"{'YES' if constant else 'no':>10}")
        print()


if __name__ == "__main__":
    main()
