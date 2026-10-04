#!/usr/bin/env python3
"""Do the guidance arms actually differ? A blocking check before any sweep.

The 2026-10-03 H100 run reported mean Hamming-to-reference 8.2-8.4 for EVERY
diffusion arm -- cfg 0/1/2/4, eta 10/30/60, all lambdas, all setpoints -- so
this separates "guidance does nothing" from "decoding hides what it does".
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from dgm.common.paths import data_dir, outputs_dir

# Where the divergence to watch for showed up: two spp*@60 diffusion samples
# reached latent norm ~1e5 despite --guidance-clip 2.0.
DIVERGENCE_NORM = 1e3

# A latent difference below this, relative to the latents' own scale, means the
# arms are the same point and no decoding choice can separate them.
SAME_LATENT = 1e-4
# Below this many samples per arm, complete overlap is uninformative rather
# than a finding. See the INCONCLUSIVE branch in verdict() for the measurement.
MIN_SAMPLES = 32
# Summed norm of the trunk's output projection below which the trunk has not
# escaped its zero init. See untrained_trunk() for the measurement.
TRUNK_FLOOR = 1e-2


def hamming(a: str, b: str) -> int:
    """Substitutions between two equal-length sequences."""
    return sum(x != y for x, y in zip(a, b))


def load(path: Path):
    """results.pt plus the reference sequence it was decoded against."""
    target = path / "results.pt" if path.is_dir() else path
    results = torch.load(target, map_location="cpu", weights_only=False)
    return results


def latent_table(results):
    """Per-arm latent norm, and the distance from the first arm.

    This is the layer the check has to start at. If two arms' LATENTS are
    identical then guidance genuinely did nothing and the decoder is innocent;
    if the latents differ but the sequences do not, the ceiling is in decoding
    -- --mut-budget, --restrict-support, or the frozen positions -- and no
    amount of eta will show an effect.
    """
    latents = results["standardized_latents"]
    names = list(latents)
    base = latents[names[0]].float()
    rows = []
    for name in names:
        z = latents[name].float()
        norm = z.flatten(1).norm(dim=1)
        delta = (z - base).flatten(1).norm(dim=1) / norm.clamp_min(1e-9)
        rows.append({"arm": name,
                     "norm_mean": float(norm.mean()),
                     "norm_max": float(norm.max()),
                     "rel_delta": float(delta.mean()),
                     "diverged": int((norm > DIVERGENCE_NORM).sum())})
    return rows


def sequence_table(results, reference):
    """Per-arm distinct count and Hamming spread, against a shared reference."""
    rows = []
    for name, entry in results["sequences"].items():
        sequences = list(entry.values())[0] if isinstance(entry, dict) else list(entry)
        if reference is None or len(reference) != len(sequences[0]):
            reference = sequences[0]
        distances = [hamming(s, reference) for s in sequences]
        rows.append({"arm": name,
                     "n": len(sequences),
                     "distinct": len(set(sequences)),
                     "ham_mean": float(np.mean(distances)),
                     "ham_sd": float(np.std(distances)),
                     "ham_min": int(np.min(distances)),
                     "ham_max": int(np.max(distances))})
    return rows


def cross_arm_overlap(results):
    """How many sequences each pair of arms shares.

    The decisive number. Two arms that return the same SET of sequences are
    the same experiment whatever their nominal guidance strength, and that is
    exactly what a uniform 8.2-8.4 Hamming across every arm would look like.
    """
    sets = {}
    for name, entry in results["sequences"].items():
        sequences = list(entry.values())[0] if isinstance(entry, dict) else list(entry)
        sets[name] = set(sequences)
    names = list(sets)
    rows = []
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            shared = len(sets[left] & sets[right])
            union = len(sets[left] | sets[right])
            rows.append({"pair": f"{left} vs {right}", "shared": shared,
                         "jaccard": shared / max(1, union)})
    return rows


def untrained_trunk(results, floor=TRUNK_FLOOR):
    """The generative trunk's output-projection norm, if it is still ~zero.

    Returns None when the trunk has trained, else the measured norm.

    The DiT trunk zero-initializes project_out (Peebles & Xie, arXiv:2212.09748)
    so the residual branch starts as the identity. Until gradient descent moves
    it off that init the trunk contributes nothing, the conditioning cannot
    reach the output, and EVERY cfg weight produces a bit-identical sample.
    Measured on a 2-epoch smoke run: project_out norm 1.5e-04 and cfg arms
    identical to rel_delta 0.000; the same configuration at 120 epochs gave
    rel_delta 0.000/0.219/0.300 for w = 0/2/4. Without this check the first run
    looks like broken classifier-free guidance rather than a model that has not
    trained yet.
    """
    state = results.get("model") or {}
    norms = [v.float().norm().item() for k, v in state.items()
             if "project_out" in k]
    if not norms:
        return None
    total = sum(norms)
    return total if total < floor else None


def verdict(latents, sequences, overlaps, results=None):
    """The one line this script exists to print."""
    moved = [r for r in latents[1:] if r["rel_delta"] > SAME_LATENT]
    identical = [r for r in overlaps if r["jaccard"] > 0.99]
    diverged = sum(r["diverged"] for r in latents)
    lines = []
    if not moved:
        lines.append("FAIL: arm latents are identical -- guidance is not being "
                     "applied at all. Check eta reached the sampler.")
    elif identical and min(r["n"] for r in sequences) < MIN_SAMPLES:
        # Jaccard over a handful of samples is not evidence of anything. Two
        # arms drawing 4 sequences each can easily overlap completely while
        # being genuinely different distributions, and the smoke test runs at
        # exactly that size. Measured on the same 1-epoch flow model: 4 samples
        # gave single-vs-multi Jaccard 1.000, 64 samples gave 0.076. Calling
        # the first a FAIL would send someone hunting a decoding ceiling that
        # is not there.
        lines.append(f"INCONCLUSIVE: {len(identical)} arm pair(s) overlap "
                     f"completely, but only "
                     f"{min(r['n'] for r in sequences)} samples per arm were "
                     f"drawn (< {MIN_SAMPLES}). Overlap is not measurable at "
                     f"this size; re-run with more --samples before reading "
                     f"anything into it.")
    elif identical:
        lines.append(f"FAIL: {len(identical)} arm pair(s) decode to the same "
                     f"sequence set despite differing latents. The ceiling is "
                     f"in DECODING, not in guidance: relax --mut-budget, drop "
                     f"--restrict-support, or unfreeze positions.")
    else:
        spread = max(r["ham_mean"] for r in sequences) - min(
            r["ham_mean"] for r in sequences)
        shared = max(r["jaccard"] for r in overlaps) if overlaps else 0.0
        lines.append(f"PASS: arms differ in sequence space (max pairwise "
                     f"Jaccard {shared:.2f}; Hamming means span "
                     f"{spread:.2f} substitutions).")
        # Hamming saturation is the trap this check was written to expose.
        # Under a binding --mut-budget every arm spends the whole budget, so
        # the Hamming mean equals the budget for all of them with zero
        # variance -- and reporting it would say "no arm differs" while the
        # sequence sets plainly do. This is the most likely reading of the
        # 2026-10-03 run's uniform 8.2-8.4: an artifact of the ceiling, not
        # evidence that guidance did nothing.
        if all(r["ham_sd"] < 1e-9 for r in sequences) and spread < 1e-9:
            lines.append(
                "NOTE: every arm's Hamming is identical with zero variance, so "
                "the mutation budget is BINDING. Hamming-to-reference is not a "
                "valid discriminator here -- report oracle brightness and "
                "pairwise overlap instead, and raise --mut-budget if "
                "mutational distance itself is meant to be the readout.")
    if diverged:
        lines.append(f"WARNING: {diverged} sample(s) exceeded latent norm "
                     f"{DIVERGENCE_NORM:g}. Lower --reward-eta or tighten "
                     f"--guidance-clip; a diverged latent decodes to noise.")
    return lines


def show(title, rows, columns):
    """A small aligned table, since this output is read by a human."""
    print(f"\n  {title}")
    print("    " + "".join(f"{c:>{w}}" for c, w in columns))
    for row in rows:
        print("    " + "".join(
            f"{row[c]:>{w}.{3 if isinstance(row[c], float) else 0}f}"
            if isinstance(row[c], (int, float)) else f"{row[c]:>{w}}"
            for c, w in columns))


# ══════════════════════════════════════════════════════════════════════════════
# Parse args
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, nargs="+",
                        help="Run directories or results.pt files to check.")
    parser.add_argument("--reference", type=Path, default=None,
                        help="Reference sequence file (default: data/avgfp_wt.txt "
                             "when its length matches, else the first sample).")
    return parser.parse_args()


def main():
    args = parse_args()
    reference = None
    wt_path = args.reference or data_dir() / "avgfp_wt.txt"
    if wt_path.is_file():
        reference = wt_path.read_text().strip()

    failures = 0
    for run in args.run:
        path = run if run.exists() else outputs_dir() / run
        results = load(path)
        print(f"\n{'=' * 72}\n{path}")
        latents = latent_table(results)
        sequences = sequence_table(results, reference)
        overlaps = cross_arm_overlap(results)
        show("latents (rel_delta is distance from the first arm)", latents,
             [("arm", 16), ("norm_mean", 11), ("norm_max", 11),
              ("rel_delta", 11), ("diverged", 10)])
        show("sequences", sequences,
             [("arm", 16), ("n", 5), ("distinct", 10), ("ham_mean", 10),
              ("ham_sd", 9), ("ham_min", 9), ("ham_max", 9)])
        show("pairwise sequence overlap", overlaps,
             [("pair", 34), ("shared", 9), ("jaccard", 9)])
        print()
        for line in verdict(latents, sequences, overlaps):
            print(f"  {line}")
            failures += line.startswith("FAIL")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
