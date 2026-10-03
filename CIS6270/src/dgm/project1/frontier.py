#!/usr/bin/env python3
"""Does the innovation make GUIDANCE work better? The gain/fidelity frontier.

analyze_sweep.py answers a different question -- whether a coupling or a path
makes UNGUIDED samples more faithful -- and reads only the three arms named
cfg/single/multi. This reads every arm a run wrote, including the single@ETA and
cfg@W arms that a multi-strength run produces, and asks the question Project 1
actually poses: guidance is being demonstrated, so does the innovation change
how well it works?

Two axes, and the rule that makes them meaningful:

  x  GAIN      how far guidance moved the property it was told to optimize,
               measured against this run's own unguided arm, so a seed's base
               model drops out of the comparison.
  y  FIDELITY  how much the samples stopped looking like real data, measured on
               something guidance was NOT optimizing.

That second clause is the whole design. Guiding on net charge and then scoring
the net-charge distribution gives one number twice with opposite signs: the
"fidelity cost" would be the gain, and every curve would be a straight line
through the origin. So:

  peptides  amino-acid composition over the SIXTEEN NON-CHARGED residues,
            renormalized. Charged residues are excluded because the objective is
            net charge and their frequencies are what it moves directly. What
            remains is sharp: measured on these runs, generated peptides are
            14-29% leucine where no residue exceeds ~6% in the training set, so
            this axis sees the actual failure mode while net charge is blind to
            it.
  images    FID, reusing compute_fid.py's feature network rather than a second
            copy. Mean pixel intensity is the objective, and FID responds to
            structure rather than to brightness alone, so a uniformly bright
            blob cannot score well by matching a summary statistic.

Each curve is one variant, traced out by guidance strength. A variant whose
curve sits further right at the same height bought more property for the same
loss of realism, which is the claim an innovation has to support.

Usage:
  dgm-frontier --prefix pep_guid --variants base ot c1 c2 d --seeds 11 12 13 14 15
  dgm-frontier --prefix img_coup --variants base ot --levels 1 5 20 50 --seeds 11 12 13
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dgm.common.paths import SHARED_DATA, project_dir

ROOT = project_dir()
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
CHARGED = set("KRDE")
UNCHARGED = [a for a in AMINO_ACIDS if a not in CHARGED]


def charge(seq):
    return (sum(a in "KR" for a in seq) - sum(a in "DE" for a in seq)) / len(seq)


def uncharged_composition(sequences):
    """Frequency of each non-charged residue, renormalized to sum to one."""
    counts = Counter(a for s in sequences for a in s if a not in CHARGED)
    total = sum(counts.values()) or 1
    return np.array([counts[a] / total for a in UNCHARGED])


def total_variation(p, q):
    """Half the L1 distance: the fraction of residues that would have to differ.

    Bounded in [0, 1] and directly readable, which a KL divergence is not: 0.25
    means a quarter of the non-charged residues sit on the wrong letter.
    """
    return 0.5 * float(np.abs(p - q).sum())


# ══════════════════════════════════════════════════════════════════════════════
# Arm names carry the guidance strength
# ══════════════════════════════════════════════════════════════════════════════

def parse_arm(name, config):
    """(kind, strength) for one arm, or None for arms this study ignores.

    arms.py names the first value of each swept list with the bare name and the
    rest with an @suffix, so the strength of `single` has to come from the saved
    config rather than from the name.
    """
    etas = config.get("reward_etas") or [config.get("reward_eta") or 0.0]
    weights = config.get("cfg_weight")
    if not isinstance(weights, (list, tuple)):
        weights = [weights if weights is not None else 0.0]
    base, _, suffix = name.partition("@")
    if base == "cfg":
        return "cfg", float(suffix) if suffix else float(weights[0])
    if base in ("single", "multi"):
        kind = "reward" if base == "single" else "reward_multi"
        return kind, float(suffix) if suffix else float(etas[0])
    # lam*/sp* arms belong to the objective study, not this one.
    return None


def load_arms(run_dir, method):
    """{arm: (property_mean, sequences_or_images)} plus the run's config."""
    protein = run_dir / method / "results.pt"
    if protein.is_file():
        saved = torch.load(protein, weights_only=False, map_location="cpu")
        seqs = saved.get("sequences") or {}
        return ({k: (float(np.mean([charge(s) for s in v])), v)
                 for k, v in seqs.items()}, saved.get("config", {}), "protein")
    image = run_dir / "results.pt"
    if image.is_file():
        saved = torch.load(image, weights_only=False, map_location="cpu")
        block = saved.get("results", {}).get(method) or {}
        out = {}
        for k, v in block.items():
            if not isinstance(v, dict) or "images" not in v:
                continue
            out[k] = (float(v["properties"][:, 0].mean()), v["images"])
        return out, saved.get("config", {}), "image"
    return None, None, None


# ══════════════════════════════════════════════════════════════════════════════
# Fidelity: a property guidance is not optimizing
# ══════════════════════════════════════════════════════════════════════════════

class Fidelity:
    """Distance from the real data, on an axis the objective does not control."""

    def __init__(self, modality, dataset, data_dir, features="mnist"):
        self.modality = modality
        if modality == "protein":
            path = Path(dataset) if dataset else None
            if path is None or not path.is_file():
                path = ROOT.parent / "lecture" / "lecture_3" / "esm2_example.csv"
            rows = list(csv.DictReader(path.open(newline="")))
            self.reference = uncharged_composition(
                [r["sequence"].strip().upper() for r in rows])
            self.label = "composition distance (non-charged residues)"
            self.source = path
        else:
            from .compute_fid import features_mnist, frechet, load_real, train_small_net
            self._fid = (features_mnist, frechet)
            real, _ = load_real(data_dir, 5000)
            self._net = train_small_net(data_dir) if features == "mnist" else None
            self._real_feat = features_mnist(self._net, real)
            self.label = f"FID ({features} features)"
            self.source = data_dir

    def __call__(self, sample):
        if self.modality == "protein":
            return total_variation(uncharged_composition(sample), self.reference)
        features_mnist, frechet = self._fid
        return float(frechet(features_mnist(self._net, sample), self._real_feat))


# ══════════════════════════════════════════════════════════════════════════════
# Collect
# ══════════════════════════════════════════════════════════════════════════════

def discover(outputs, prefix, variant, seed, levels):
    """Run directories for one variant/seed, over whichever levels exist."""
    if levels:
        return [outputs / f"{prefix}_{variant}_{lv}_s{seed}" for lv in levels]
    return sorted(outputs.glob(f"{prefix}_{variant}_*_s{seed}"))


def collect(args, fidelity_for):
    rows, modality, fidelity = [], None, None
    for method in args.methods:
        for variant in args.variants:
            for seed in args.seeds:
                for run in discover(args.outputs, args.prefix, variant, seed,
                                    args.levels):
                    arms, config, found = load_arms(run, method)
                    if not arms:
                        continue
                    modality = modality or found
                    if fidelity is None:
                        fidelity = fidelity_for(modality)
                    # The unconditional arm is this run's own control: cfg at
                    # weight 0 drops the conditional term entirely.
                    parsed = {k: parse_arm(k, config) for k in arms}
                    control = next(
                        (k for k, p in parsed.items()
                         if p and p[0] == "cfg" and p[1] == 0.0), None)
                    if control is None:
                        print(f"  [skip] {run.name}/{method}: no w=0 arm, so there "
                              f"is no unguided control to measure gain against")
                        continue
                    base_property = arms[control][0]
                    for arm, (value, sample) in arms.items():
                        p = parsed[arm]
                        if p is None:
                            continue
                        kind, strength = p
                        rows.append({
                            "method": method, "variant": variant, "seed": seed,
                            "run": run.name, "arm": arm, "kind": kind,
                            "strength": strength, "property_mean": value,
                            "gain": value - base_property,
                            "fidelity": fidelity(sample), "n": len(sample),
                        })
    return rows, modality, fidelity


def summarize(rows, kind):
    """Mean gain and fidelity per (method, variant, strength), over seeds."""
    out = {}
    for r in rows:
        if r["kind"] != kind:
            continue
        out.setdefault((r["method"], r["variant"], r["strength"]), []).append(r)
    return {k: (float(np.mean([r["gain"] for r in v])),
                float(np.mean([r["fidelity"] for r in v])), len(v))
            for k, v in out.items()}


def plot(rows, kinds, label, outdir, prefix, methods, variants):
    fig, axes = plt.subplots(len(kinds), len(methods), squeeze=False,
                             figsize=(6.4 * len(methods), 4.8 * len(kinds)))
    for i, kind in enumerate(kinds):
        table = summarize(rows, kind)
        for j, method in enumerate(methods):
            ax = axes[i][j]
            for variant in variants:
                pts = sorted((s, g, f) for (m, v, s), (g, f, _) in table.items()
                             if m == method and v == variant)
                if not pts:
                    continue
                ax.plot([p[1] for p in pts], [p[2] for p in pts],
                        marker="o", label=variant)
                for s, g, f in pts:
                    ax.annotate(f"{s:g}", (g, f), fontsize=6,
                                textcoords="offset points", xytext=(3, 3))
            ax.set_xlabel("guidance gain in the optimized property")
            ax.set_ylabel(label)
            ax.set_title(f"{method} / {kind} guidance\n"
                         f"right and low is better; labels are the strength")
            ax.legend(fontsize=8)
    fig.tight_layout()
    path = outdir / f"{prefix}_15_guidance_frontier.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prefix", required=True)
    p.add_argument("--outputs", type=Path, default=ROOT / "outputs")
    p.add_argument("--variants", nargs="+", required=True)
    p.add_argument("--levels", nargs="+", default=None,
                   help="Directory levels to read. Default: every level present, "
                        "which is what a one-run-per-variant guidance sweep wants.")
    p.add_argument("--seeds", type=int, nargs="+", default=[11, 12, 13, 14, 15])
    p.add_argument("--methods", nargs="+", default=["flow", "diffusion"])
    p.add_argument("--dataset", type=Path, default=None,
                   help="Training CSV for the peptide composition reference.")
    p.add_argument("--data-dir", type=Path, default=SHARED_DATA)
    p.add_argument("--features", default="mnist", choices=("mnist", "inception"))
    p.add_argument("--outdir", type=Path, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    outdir = args.outdir or ROOT / "plots" / f"{args.prefix}_frontier"
    outdir.mkdir(parents=True, exist_ok=True)

    rows, modality, fidelity = collect(
        args, lambda m: Fidelity(m, args.dataset, args.data_dir, args.features))
    if not rows:
        raise SystemExit(f"No arms found under {args.outputs} for prefix "
                         f"'{args.prefix}'. Checked e.g. "
                         f"{args.prefix}_{args.variants[0]}_*_s{args.seeds[0]}")
    print(f"\nGuidance frontier -- {modality}")
    print(f"  {len(rows)} arm-runs read   fidelity axis: {fidelity.label}")
    print(f"  reference: {fidelity.source}")

    kinds = [k for k in ("cfg", "reward", "reward_multi")
             if any(r["kind"] == k for r in rows)]
    for kind in kinds:
        table = summarize(rows, kind)
        strengths = sorted({s for (_, _, s) in table})
        print(f"\n=== {kind} guidance ===")
        head = (f"  {'method':<11}{'variant':<8}" +
                "".join(f"{f'@{s:g}':>22}" for s in strengths))
        print(head + "\n  " + "-" * (len(head) - 2))
        for method in args.methods:
            for variant in args.variants:
                cells = []
                for s in strengths:
                    hit = table.get((method, variant, s))
                    cells.append("         --           " if hit is None
                                 else f"{hit[0]:+.4f} / {hit[1]:.4f}  ")
                if any("--" not in c for c in cells):
                    print(f"  {method:<11}{variant:<8}" +
                          "".join(f"{c:>22}" for c in cells))
        print("  (cell = mean gain / mean fidelity cost; lower fidelity is better)")

    csv_path = outdir / f"{args.prefix}_guidance_frontier.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    fig = plot(rows, kinds, fidelity.label, outdir, args.prefix,
               args.methods, args.variants)
    print(f"\nWrote {csv_path}\nWrote {fig}")


if __name__ == "__main__":
    main()
