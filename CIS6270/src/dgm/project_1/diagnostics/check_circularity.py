#!/usr/bin/env python3
"""Does an oracle reward sequences merely for being ESM-plausible?

The generator samples in ESM-2 latent space and decodes with the ESM-2 language
model head, so its substitutions are the ones ESM-2 most wants to make. If the
oracle also reads ESM-2 features, the thing being optimized and the thing doing
the measuring share a representation, and a high score could reflect that
agreement rather than brightness.

This builds sequences the same way the pipeline does — noise the wild-type
latent, push it through the LM head, keep the top-`budget` substitutions — and
checks whether each oracle rates them above genuinely bright measured variants.
An oracle that does is not measuring brightness.

The result on the bundled data is the opposite of the naive worry: the ESM
oracle rates these sequences near the dark end, while the indicator oracle rates
them ABOVE real bright variants, because ~90% of their substitutions were never
measured and therefore contribute nothing.

Usage:
  python diagnostics/check_circularity.py
  python diagnostics/check_circularity.py --fit-n 8000 --budget 4 --noise 0.7
"""
import argparse

import numpy as np
import torch
from sklearn.linear_model import Ridge
from transformers import AutoTokenizer, EsmForMaskedLM

from dgm.common.paths import REPO_ROOT, project_dir

ROOT = project_dir()
from ..gfp_oracle import featurize                                   # noqa: E402

DMS         = REPO_ROOT / "Data_GFP" / "data" / "dms_data" / "avgfp"
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def apply_variant(wt, variant):
    seq = list(wt)
    for m in variant.split(","):
        seq[int(m[1:-1])] = m[-1]
    return "".join(seq)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--esm-model", default="facebook/esm2_t6_8M_UR50D")
    parser.add_argument("--fit-n", type=int, default=8000)
    parser.add_argument("--budget", type=int, default=4,
                        help="Substitutions per decoded sequence")
    parser.add_argument("--noise", type=float, default=0.7,
                        help="Noise added to the standardized wild-type latent")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = [l.rstrip().split("\t") for l in (DMS / "avgfp.tsv").read_text().splitlines()[1:]]
    wt = (ROOT / "data" / "avgfp_wt.txt").read_text().strip()
    scores = np.array([float(r[2]) for r in rows])

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(rows))
    fit_idx = order[:args.fit_n]
    held    = order[args.fit_n:]
    bright  = [i for i in held if scores[i] > -0.5][:400]
    dark    = [i for i in held if scores[i] < -1.5][:400]
    print(f"fit {len(fit_idx)} | held-out bright {len(bright)} | dark {len(dark)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.esm_model, cache_dir=str(ROOT / "cache"))
    esm = EsmForMaskedLM.from_pretrained(
        args.esm_model, cache_dir=str(ROOT / "cache"), use_safetensors=True
    ).to(device).eval().requires_grad_(False)

    def encode(seqs, batch=64):
        out = []
        with torch.no_grad():
            for s in range(0, len(seqs), batch):
                toks = {k: v.to(device)
                        for k, v in tokenizer(seqs[s:s + batch], return_tensors="pt").items()}
                out.append(esm.esm(**toks).last_hidden_state[:, 1:-1].cpu())
        return torch.cat(out)

    z_wt = encode([wt])
    fit_seqs = [apply_variant(wt, rows[i][0]) for i in fit_idx]
    z_fit = encode(fit_seqs)
    z_mean = z_fit.mean((0, 1), keepdim=True)
    z_std  = z_fit.std((0, 1), correction=0, keepdim=True).clamp_min(1e-4)

    # Decode exactly as run_experiment.decode_budget does: keep the substitutions
    # whose argmax logit beats the reference residue by the largest margin.
    torch.manual_seed(args.seed)
    aa_ids  = torch.tensor(tokenizer.convert_tokens_to_ids(list(AMINO_ACIDS)))
    ref_ids = torch.tensor([AMINO_ACIDS.index(a) for a in wt])
    decoded = []
    with torch.no_grad():
        for _ in range(8):
            z = ((z_wt - z_mean) / z_std).expand(50, -1, -1) \
                + args.noise * torch.randn(50, z_wt.shape[1], z_wt.shape[2])
            logits = esm.lm_head((z * z_std + z_mean).to(device))
            logits = logits.index_select(-1, aa_ids.to(device)).cpu()
            best, choice = logits.max(-1)
            ref = logits.gather(-1, ref_ids.expand(len(logits), -1)[..., None]).squeeze(-1)
            margin = (best - ref).masked_fill(choice == ref_ids, -float("inf"))
            keep = margin.topk(args.budget, dim=1).indices
            out = ref_ids.expand(len(logits), -1).clone()
            for j in range(len(logits)):
                positions = keep[j][margin[j, keep[j]] > 0]
                out[j, positions] = choice[j, positions]
            decoded += ["".join(AMINO_ACIDS[i] for i in r) for r in out.tolist()]
    decoded = list(dict.fromkeys(decoded))
    print(f"ESM-decoded: {len(decoded)} unique, mean Hamming "
          f"{np.mean([sum(a != b for a, b in zip(s, wt)) for s in decoded]):.1f}")

    support = {(int(m[1:-1]), m[-1]) for r in rows for m in r[0].split(",")}
    inside = [np.mean([(p, b) in support
                       for p, (a, b) in enumerate(zip(wt, s)) if a != b] or [0])
              for s in decoded]
    print(f"substitutions inside the measured DMS support: {np.mean(inside):.1%}")

    y = scores[fit_idx]
    esm_ridge = Ridge(alpha=1.0).fit((z_fit - z_wt).reshape(len(z_fit), -1).numpy(), y)
    oh_ridge  = Ridge(alpha=1.0).fit(featurize(fit_seqs, wt), y)

    groups = {
        "held-out BRIGHT (measured)":   [apply_variant(wt, rows[i][0]) for i in bright],
        "held-out DARK (measured)":     [apply_variant(wt, rows[i][0]) for i in dark],
        "ESM-decoded (expected dead)":  decoded,
    }
    print(f"\n{'group':<30}{'ESM oracle':>13}{'indicator':>13}{'true score':>13}")
    for name, seqs in groups.items():
        z = encode(seqs)
        e = esm_ridge.predict((z - z_wt).reshape(len(z), -1).numpy())
        o = oh_ridge.predict(featurize(seqs, wt))
        if "measured" in name:
            truth = f"{scores[bright if 'BRIGHT' in name else dark].mean():+.3f}"
        else:
            truth = "unknown"
        print(f"{name:<30}{e.mean():>+13.3f}{o.mean():>+13.3f}{truth:>13}")
    print("\nAn oracle scoring the last row near or above the first is not "
          "measuring brightness on generated sequences.")


if __name__ == "__main__":
    main()
