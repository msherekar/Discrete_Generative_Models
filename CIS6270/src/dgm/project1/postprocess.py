#!/usr/bin/env python3
"""Score and plot a finished run's results.pt without retraining anything.

Everything expensive or dependency-heavy that was deliberately kept out of the
training job: METL brightness scoring, METL predicted stability as the fidelity
axis, and the summary plots.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dgm.common.paths import data_dir, outputs_dir, plots_dir
from dgm.project1.oracles import metl_finetuned as mf
from dgm.project1.oracles.common import seq_to_variant

# Rosetta attribute used as the orthogonal fidelity axis. Spearman -0.69 with
# experimental brightness, the best stability measure the pretraining head has.
STABILITY_ATTRIBUTE = "total_score"

# Guidance arms saved by run_experiment, in reporting order.
ARM_ORDER = ("cfg", "single", "multi")

# ══════════════════════════════════════════════════════════════════════════════
# Loading
# ══════════════════════════════════════════════════════════════════════════════

def load_run(path: Path):
    """results.pt from a run directory, or the file itself."""
    target = path / "results.pt" if path.is_dir() else path
    if not target.is_file():
        raise FileNotFoundError(f"no results.pt at {target}")
    return torch.load(target, map_location="cpu", weights_only=False)


def sequences_from(results, budget=None):
    """{arm: [sequence]} for one mutation budget, defaulting to the primary.

    run_experiment stores several budgets per arm because --mut-budget accepts
    a list; which one is primary is recorded alongside them, and a caller that
    does not care should get that rather than an arbitrary key.
    """
    out = {}
    for arm, entry in results["sequences"].items():
        # save_results stores the primary budget's sequences directly when
        # --mut-budget named one value, and a {budget: [sequence]} mapping when
        # it named several. Accept both so a run is scorable either way.
        if isinstance(entry, dict):
            key = budget if budget in entry else results.get("primary_budget")
            key = key if key in entry else next(iter(entry))
            out[arm] = list(entry[key])
        else:
            out[arm] = list(entry)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Scoring
# ══════════════════════════════════════════════════════════════════════════════

def score_brightness(arms, wt, which="ft-1d", pdb_path=None):
    """METL predicted functional score per arm, plus the validity mask.

    Invalid decodes are excluded rather than scored as zero: METL's encoder
    assumes an aligned variant over the canonical alphabet, so a wrong-length
    sequence would raise, and counting it as zero would quietly flatter or
    penalize an arm depending on the sign convention.
    """
    table = {}
    for arm, sequences in arms.items():
        keep = mf.valid_sequences(sequences, wt)
        usable = [s for s, ok in zip(sequences, keep) if ok]
        scores = (mf.predict(usable, wt, which, pdb_path=pdb_path)
                  if usable else np.zeros(0, dtype=np.float32))
        table[arm] = {"scores": scores, "n_valid": int(keep.sum()),
                      "n_total": len(sequences)}
    return table


def score_stability(arms, wt):
    """METL predicted total_score per arm, the orthogonal fidelity axis.

    Uses the SOURCE model's pretraining head, so this needs the full metl repo
    and the PDB file. Returns None when that is unavailable, which is the
    common case on an OSG node and is why this is a post-processing step.
    """
    try:
        from dgm.project1.oracles.metl import attribute_index, metl_attributes
    except Exception:
        return None
    try:
        index = attribute_index(STABILITY_ATTRIBUTE)
    except Exception:
        return None
    table = {}
    for arm, sequences in arms.items():
        keep = mf.valid_sequences(sequences, wt)
        usable = [s for s, ok in zip(sequences, keep) if ok]
        if not usable:
            table[arm] = np.zeros(0, dtype=np.float32)
            continue
        table[arm] = metl_attributes(usable, wt)[:, index]
    return table


def mutation_counts(arms, wt):
    """Substitutions from the wild type per arm, the design-budget axis."""
    return {arm: np.array([len([p for p in seq_to_variant(s, wt).split(",") if p])
                           for s in sequences], dtype=np.int32)
            for arm, sequences in arms.items()}


def summarize(brightness, stability, mutations, wt_score):
    """One row per arm: brightness gain over wild type, stability, diversity."""
    rows = []
    for arm in sorted(brightness, key=lambda a: (a not in ARM_ORDER, a)):
        entry = brightness[arm]
        scores = entry["scores"]
        row = {
            "arm": arm,
            "n_valid": entry["n_valid"],
            "n_total": entry["n_total"],
            "brightness_mean": float(scores.mean()) if len(scores) else float("nan"),
            "brightness_sd": float(scores.std()) if len(scores) else float("nan"),
            "gain_vs_wt": (float(scores.mean() - wt_score)
                           if len(scores) and wt_score is not None else float("nan")),
            "frac_above_wt": (float((scores > wt_score).mean())
                              if len(scores) and wt_score is not None else float("nan")),
            "mut_mean": float(mutations[arm].mean()) if len(mutations[arm]) else 0.0,
            "distinct": len(set(map(tuple, [[s] for s in range(0)]))) or None,
        }
        if stability is not None and len(stability.get(arm, [])):
            row["stability_mean"] = float(stability[arm].mean())
        rows.append(row)
    return rows


def print_table(rows, stability_present):
    """The summary a report can quote directly."""
    header = f"  {'arm':<10}{'valid':>8}{'bright':>10}{'sd':>8}{'vs WT':>9}{'>WT':>7}{'muts':>7}"
    if stability_present:
        header += f"{'stability':>11}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for row in rows:
        line = (f"  {row['arm']:<10}{row['n_valid']:>4}/{row['n_total']:<3}"
                f"{row['brightness_mean']:>10.4f}{row['brightness_sd']:>8.4f}"
                f"{row['gain_vs_wt']:>+9.4f}{row['frac_above_wt']:>7.2f}"
                f"{row['mut_mean']:>7.1f}")
        if stability_present:
            line += f"{row.get('stability_mean', float('nan')):>11.3f}"
        print(line)


# ══════════════════════════════════════════════════════════════════════════════
# Plots
# ══════════════════════════════════════════════════════════════════════════════

def plot_summary(rows, brightness, stability, wt_score, out_path, title):
    """Brightness distribution per arm, and the brightness-stability frontier."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = 2 if stability is not None else 1
    figure, axes = plt.subplots(1, panels, figsize=(6 * panels, 4.5), squeeze=False)

    names = [row["arm"] for row in rows]
    data = [brightness[name]["scores"] for name in names]
    axis = axes[0][0]
    # Violin over the raw per-sample scores rather than a bar of the mean: the
    # question is whether guidance moved the distribution or only its tail, and
    # a bar chart cannot show the difference.
    if any(len(d) for d in data):
        axis.violinplot([d for d in data if len(d)], showmeans=True)
        axis.set_xticks(range(1, 1 + sum(1 for d in data if len(d))))
        axis.set_xticklabels([n for n, d in zip(names, data) if len(d)],
                             rotation=20)
    if wt_score is not None:
        axis.axhline(wt_score, color="crimson", linestyle="--", linewidth=1,
                     label=f"wild type ({wt_score:+.3f})")
        axis.legend(fontsize=8)
    axis.set_ylabel("METL predicted functional score")
    axis.set_title("Brightness by guidance arm")

    if stability is not None:
        axis = axes[0][1]
        for name in names:
            x = stability.get(name, np.zeros(0))
            y = brightness[name]["scores"]
            if len(x) and len(y):
                axis.scatter(x[:len(y)], y[:len(x)], s=14, alpha=0.6, label=name)
        axis.set_xlabel(f"METL predicted {STABILITY_ATTRIBUTE} (stability)")
        axis.set_ylabel("METL predicted functional score")
        axis.set_title("Does brightness guidance cost stability?")
        axis.legend(fontsize=8)

    figure.suptitle(title)
    figure.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
    return out_path


# ══════════════════════════════════════════════════════════════════════════════
# Parse args
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, nargs="+",
                        help="Run directories (or results.pt files) to score. "
                             "Relative paths resolve against outputs/.")
    parser.add_argument("--wt", type=Path, default=None,
                        help="Wild-type sequence file (default: data/avgfp_wt.txt).")
    parser.add_argument("--budget", type=int, default=None,
                        help="Mutation budget to score (default: the run's primary).")
    parser.add_argument("--no-stability", action="store_true",
                        help="Skip the METL stability axis, which needs the full "
                             "metl repo and the PDB file.")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--json", type=Path, default=None,
                        help="Also write the summary rows here as JSON.")
    mf.add_arguments(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    wt = (args.wt or data_dir() / "avgfp_wt.txt").read_text().strip()
    which = args.metl_target or "ft-1d"
    if not mf.available():
        raise SystemExit(
            "metl-pretrained is not installed. Install it with:\n"
            "  uv pip install 'metl-pretrained @ "
            "git+https://github.com/gitter-lab/metl-pretrained'\n"
            "It resolves with no new dependencies beyond torch.")

    # The wild type is the reference every gain is quoted against, scored once.
    wt_score = float(mf.predict([wt], wt, which)[0])
    print(f"Oracle: METL {which}   wild-type score {wt_score:+.4f}")
    print("NOTE: ft-1d/ft-3d trained on 80% of this same DMS; valid as a second "
          "opinion on generated designs, not as a held-out accuracy claim.\n")

    all_rows = {}
    for run in args.run:
        path = run if run.exists() else outputs_dir() / run
        results = load_run(path)
        arms = sequences_from(results, args.budget)
        print(f"{path}")
        brightness = score_brightness(arms, wt, which)
        stability = None if args.no_stability else score_stability(arms, wt)
        mutations = mutation_counts(arms, wt)
        rows = summarize(brightness, stability, mutations, wt_score)
        print_table(rows, stability is not None)
        all_rows[str(path)] = rows
        if not args.no_plot:
            name = f"{path.parent.name}_{path.name}".strip("_") or "run"
            out = plot_summary(rows, brightness, stability, wt_score,
                               plots_dir() / "postprocess" / f"{name}.png",
                               f"{name}  (METL {which})")
            print(f"  plot -> {out}")
        print()

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(all_rows, indent=2))
        print(f"Summary -> {args.json}")


if __name__ == "__main__":
    main()
