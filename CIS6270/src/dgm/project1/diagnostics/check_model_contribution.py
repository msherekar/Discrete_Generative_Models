#!/usr/bin/env python3
"""Does the trained generative model contribute anything the decoder does not?

The sampler produces a latent and the decoder turns it into a sequence. Both can
generate diversity, so a run that looks like it is designing sequences may only
be sampling from ESM's language-model head around a fixed point. This separates
the two by decoding from the reference latent directly, with the generative
model removed and every other setting held fixed.

If the model's samples are not significantly better than this control, the model
is not contributing: whatever the run produced, the decoder produced.

On the avGFP run at --hidden 128 the control was indistinguishable from every
arm (flow p=0.42, diffusion p=0.52, random mutagenesis p=0.59), which is what
motivated testing a wider network.

Usage:
  python diagnostics/check_model_contribution.py --run-dir outputs/esm2_8m_k3
  python diagnostics/check_model_contribution.py --run-dir outputs/... --budget 5
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import mannwhitneyu
from transformers import AutoTokenizer, EsmForMaskedLM

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import run_experiment as R              # noqa: E402
import embedding_oracle as embo         # noqa: E402

METHODS = ("flow", "diffusion")
MODES = ("cfg", "single", "multi")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=ROOT / "data" / "avgfp_wt.txt")
    parser.add_argument("--oracle", type=Path, default=ROOT / "data" / "avgfp_metl_oracle.npz")
    parser.add_argument("--esm-model", default="facebook/esm2_t6_8M_UR50D")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--budget", type=int, default=None,
                        help="Substitutions per sample (default: the run's own setting)")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Decode temperature (default: the run's own setting)")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    reference = args.reference.read_text().strip()
    saved = None
    for method in METHODS:
        path = args.run_dir / method / "results.pt"
        if path.is_file():
            saved = torch.load(path, weights_only=False, map_location="cpu")
            break
    if saved is None:
        parser.error(f"no results.pt under {args.run_dir}/{{flow,diffusion}}/")
    config = saved.get("config", {})
    budget = args.budget or config.get("mut_budget") or 3
    temperature = args.temperature if args.temperature is not None \
        else config.get("decode_temperature", 0.0)
    frozen = tuple(config.get("freeze_positions", ()))
    exact = bool(config.get("exact_mutations", False))
    print(f"run     : {args.run_dir.name}")
    print(f"decoder : budget={budget} T={temperature} frozen={list(frozen)} exact={exact}")
    print(f"hidden  : {config.get('hidden', 'unknown')}\n")

    device = R.DEVICE
    tokenizer = AutoTokenizer.from_pretrained(args.esm_model, cache_dir=str(ROOT / "cache"))
    esm = EsmForMaskedLM.from_pretrained(
        args.esm_model, cache_dir=str(ROOT / "cache"), use_safetensors=True
    ).to(device).eval().requires_grad_(False)

    # The control: the reference latent, standardized with the run's own statistics,
    # repeated and pushed through the same decoder. No generative model involved.
    stats = saved["stats"]
    with torch.no_grad():
        toks = {k: v.to(device) for k, v in tokenizer([reference], return_tensors="pt").items()}
        h = esm.esm(**toks).last_hidden_state[:, 1:-1].cpu()
    z = ((h - stats["z_mean"]) / stats["z_std"]).to(device)
    z = z.expand(args.samples, -1, -1).contiguous()
    torch.manual_seed(args.seed)
    control_seqs = R.decode_budget(z, esm, tokenizer, stats, reference, budget,
                                   temperature, frozen, exact)
    oracle = embo.load_oracle(args.oracle)
    control = embo.score_sequences(control_seqs, oracle)

    print(f"{'group':<26}{'n':>5}{'mean':>9}{'best':>9}{'>WT':>6}{'top-10':>9}{'p vs control':>14}")
    print("-" * 78)
    for method in METHODS:
        path = args.run_dir / method / "results.pt"
        if not path.is_file():
            continue
        data = torch.load(path, weights_only=False, map_location="cpu")
        for mode in MODES:
            scores = data.get("oracle_brightness", {}).get(mode)
            if scores is None:
                continue
            v = np.asarray(scores)
            p = mannwhitneyu(v, control, alternative="two-sided").pvalue
            mark = "  *" if p < 0.05 else ""
            print(f"{method + '/' + mode:<26}{len(v):>5}{v.mean():>9.3f}{v.max():>9.3f}"
                  f"{int((v > 0).sum()):>6}{np.sort(v)[-10:].mean():>9.3f}{p:>14.4f}{mark}")
    print(f"{'CONTROL (no model)':<26}{len(control):>5}{control.mean():>9.3f}"
          f"{control.max():>9.3f}{int((control > 0).sum()):>6}"
          f"{np.sort(control)[-10:].mean():>9.3f}{'--':>14}")
    print(f"\ncontrol distinct sequences: {len(set(control_seqs))}/{len(control_seqs)}")
    print("An arm that does not beat the control (* = p<0.05) is being produced by the\n"
          "decoder, not by the generative model.")


if __name__ == "__main__":
    main()
