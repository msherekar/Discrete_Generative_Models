#!/usr/bin/env python3
"""Does the conditional field point along the property axis, or just somewhere?

A CFG weight sweep shows only that the field moves: the difference between the
conditional and unconditional fields is nonzero and scales with w. It cannot say
whether that difference points the way the label means. This measures the
direction against ground truth.

Ground truth is the axis separating the two classes in the data itself: encode
real sequences of each label and take the difference of their means. Lecture
3.4 derives CFG from

    conditional score - unconditional score = the classifier-gradient direction,

so the difference of the model's conditional fields is supposed to approximate
the difference of the true conditional distributions. Cosine near 1 means it
did; cosine near 0 means the model separated the labels along an axis that is
not the property.

Measured on esm2_8m_calib3 (60 epochs), against a true axis of norm 10.49:

    flow       cos -0.033 / -0.029 / -0.021   at w = 1 / 2 / 4
    diffusion  cos +0.314 / +0.317 / +0.322

Flow's conditional direction was orthogonal to brightness while its magnitude
was well calibrated (10.02 against a true 10.49 at w=2), so it moved the right
distance sideways. Diffusion was about a third aligned. Because this is measured
on latents, before any decoding, a low cosine here is a property of the trained
field and cannot be blamed on the decoder.

Run it on the output of resample_cfg.py --condition 0 1.

  python diagnostics/check_condition_axis.py outputs/esm2_8m_calib3_cfg
  python diagnostics/check_condition_axis.py outputs/esm2_8m_calib3_cfg --n 1000
"""
import argparse
import csv
import random
from pathlib import Path


import torch
import torch.nn.functional as F

from dgm.project1 import run_experiment as R

from dgm.common.paths import project_dir

ROOT = project_dir()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path,
                        help="A resample_cfg.py output holding c0*/c1* arms.")
    parser.add_argument("--dataset", type=Path,
                        default=ROOT / "data" / "avgfp_train_props.csv")
    parser.add_argument("--n", type=int, default=600,
                        help="Real sequences per class for the ground-truth axis.")
    parser.add_argument("--esm", default="facebook/esm2_t6_8M_UR50D")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def encode(sequences, esm_id, cache, stats, device, batch=64):
    from transformers import AutoTokenizer, EsmForMaskedLM
    tokenizer = AutoTokenizer.from_pretrained(esm_id, cache_dir=str(cache))
    esm = EsmForMaskedLM.from_pretrained(
        esm_id, cache_dir=str(cache), use_safetensors=True).to(device).eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(sequences), batch):
            toks = tokenizer(sequences[start:start + batch], return_tensors="pt")
            toks = {k: v.to(device) for k, v in toks.items()}
            out.append(esm.esm(**toks).last_hidden_state[:, 1:-1].cpu())
    return (torch.cat(out) - stats["z_mean"]) / stats["z_std"]


def pairs_in(latents):
    """(w, c1_arm, c0_arm) for every weight present under both labels."""
    found = {}
    for name in latents:
        if not name.startswith(("c0", "c1")):
            continue
        label, _, tail = name.partition("@")
        weight = float(tail) if tail else 1.0
        found.setdefault(weight, {})[label] = name
    return [(w, d["c1"], d["c0"]) for w, d in sorted(found.items())
            if "c0" in d and "c1" in d]


def main():
    args = parse_args()
    device = R.DEVICE

    rows = list(csv.DictReader(args.dataset.open(newline="")))
    random.seed(args.seed)
    bright = [r["sequence"] for r in random.sample([r for r in rows if r["c"] == "1"], args.n)]
    dark = [r["sequence"] for r in random.sample([r for r in rows if r["c"] == "0"], args.n)]

    methods = [m for m in ("flow", "diffusion")
               if (args.run_dir / m / "results.pt").is_file()]
    if not methods:
        raise SystemExit(f"No results.pt under {args.run_dir}/{{flow,diffusion}}/")

    print(f"\n  ground truth: {args.n} real sequences per label from "
          f"{args.dataset.name}")
    reported = False
    for method in methods:
        saved = torch.load(args.run_dir / method / "results.pt",
                           weights_only=False, map_location="cpu")
        stats, latents = saved["stats"], saved["standardized_latents"]
        pairs = pairs_in(latents)
        if not pairs:
            print(f"\n  {method}: no c0/c1 pair. Re-run resample_cfg.py with "
                  f"--condition 0 1.")
            continue

        print(f"\n  encoding for {method}...")
        truth = (encode(bright, args.esm, ROOT / "cache", stats, device).mean(0)
                 - encode(dark, args.esm, ROOT / "cache", stats, device).mean(0)).flatten()
        print(f"  === {method} ===   true bright-minus-dark axis: "
              f"|d| = {float(truth.norm()):.2f}")
        print(f"  {'w':>5}{'|d_model|':>12}{'cosine':>10}{'aligned part':>14}"
              f"{'magnitude vs true':>19}")
        for w, hi, lo in pairs:
            d = (latents[hi].mean(0) - latents[lo].mean(0)).flatten()
            cos = float(F.cosine_similarity(d[None], truth[None]))
            print(f"  {w:>5g}{float(d.norm()):>12.2f}{cos:>10.4f}"
                  f"{cos * float(d.norm()):>14.2f}"
                  f"{float(d.norm()) / float(truth.norm()):>18.1%}")
        reported = True

    if reported:
        print(f"\n  'aligned part' is the component that actually moves along the "
              f"property axis:")
        print(f"  cosine times magnitude. A large magnitude with a small cosine "
              f"means the field")
        print(f"  moves confidently in a direction the label does not describe. "
              f"Because this is")
        print(f"  measured on latents, a low cosine is a property of the trained "
              f"field, not of")
        print(f"  the decoder.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
