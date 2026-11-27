#!/usr/bin/env python3
"""Simulated annealing and random variants over the METL oracle.

The two non-generative baselines the study is read against: Gelman et al.'s own
GFP design method, and their random-variant null. If guided flow matching cannot
beat annealing on the oracle it is being guided by, that is the headline result.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from dgm.common.paths import data_dir, outputs_dir
from dgm.project1.oracles import metl_finetuned as mf
from dgm.project1.oracles.common import AMINO_ACIDS, seq_to_variant

# Gelman et al. synthesized designs at these two mutation budgets, hitting 5/5
# and 5/5 fluorescent in the Observed setting and 4/5 and 2/5 in Unobserved.
BUDGETS = (5, 10)

# Annealing schedule. Geometric cooling from T0 to T1 over `steps` proposals.
T0, T1 = 1.0, 0.01

# Positions never to mutate. The chromophore triad (Thr65-Tyr66-Gly67 in the
# 1-indexed mature protein) is what fluoresces at all; the wild type here omits
# the initiator methionine, so these are the 0-indexed equivalents.
CHROMOPHORE = (64, 65, 66)


def load_wt(path=None):
    """The avGFP wild-type sequence, 237 residues."""
    return (path or data_dir() / "avgfp_wt.txt").read_text().strip()


def observed_substitutions(csv_path):
    """{position: {residue}} actually seen in the training DMS.

    Gelman et al. split their designs into an "Observed" setting, drawing only
    substitutions present in the training data, and an "Unobserved" setting
    drawing the rest. The distinction matters a great deal to their wet-lab
    result -- 10/10 fluorescent against 6/10 -- and it maps directly onto the
    --restrict-support mask already in the pipeline.
    """
    import csv as csv_module
    seen: dict[int, set] = {}
    with Path(csv_path).open(newline="") as handle:
        for row in csv_module.DictReader(handle):
            variant = row.get("variant") or ""
            for token in variant.split(","):
                token = token.strip()
                if len(token) < 3:
                    continue
                # Tokens look like "E3K": wild-type residue, 0-indexed
                # position, replacement.
                position, new = token[1:-1], token[-1]
                if position.isdigit():
                    seen.setdefault(int(position), set()).add(new)
    return seen


def propose(sequence, wt, budget, observed, rng, frozen=CHROMOPHORE):
    """One neighbour: re-draw a single substitution, respecting the budget.

    The move set is what keeps the walk inside the same design space the
    generative sampler decodes into, so the comparison is of search strategies
    rather than of feasible sets.
    """
    out = list(sequence)
    positions = [i for i in range(len(wt)) if i not in frozen]
    current = [i for i in positions if out[i] != wt[i]]
    # At the budget, either revert a substitution or move one; below it, add.
    if len(current) >= budget and current:
        site = int(rng.choice(current))
        if rng.random() < 0.5:
            out[site] = wt[site]                      # revert, freeing budget
            return "".join(out)
    else:
        site = int(rng.choice(positions))
    choices = sorted(observed.get(site, set()) - {wt[site]}) if observed else None
    if not choices:
        choices = [a for a in AMINO_ACIDS if a != wt[site]]
    out[site] = str(rng.choice(choices))
    return "".join(out)


def anneal(wt, budget, observed, rng, steps, score_batch, n_designs):
    """Run `n_designs` independent annealing chains, returning the best of each.

    Independent chains rather than one long chain with restarts, because the
    paper clusters its designs for diversity and a single chain would return
    `n_designs` neighbours of one optimum. Each chain is scored in a batch so
    the oracle sees `n_designs` sequences per proposal round, which is what
    makes this affordable at all -- a per-sequence forward pass would be
    `steps * n_designs` oracle calls.
    """
    current = [wt] * n_designs
    current_score = score_batch(current)
    best, best_score = list(current), current_score.copy()
    for step in range(steps):
        temperature = T0 * (T1 / T0) ** (step / max(1, steps - 1))
        proposed = [propose(s, wt, budget, observed, rng) for s in current]
        proposed_score = score_batch(proposed)
        # Metropolis: always accept an improvement, accept a worsening with
        # probability exp(delta / T).
        delta = proposed_score - current_score
        accept = (delta > 0) | (rng.random(n_designs) < np.exp(
            np.clip(delta, -50, 0) / max(temperature, 1e-6)))
        for i in range(n_designs):
            if accept[i]:
                current[i], current_score[i] = proposed[i], proposed_score[i]
                if proposed_score[i] > best_score[i]:
                    best[i], best_score[i] = proposed[i], proposed_score[i]
    return best, best_score


def random_variants(wt, budget, observed, rng, n_designs, frozen=CHROMOPHORE):
    """The paper's null: variants at a matched mutation count, drawn at random.

    Their random controls at matched budgets were essentially all dark, which
    is what makes it the honest baseline -- any method must clear it before its
    oracle gain means anything.
    """
    designs = []
    positions = [i for i in range(len(wt)) if i not in frozen]
    for _ in range(n_designs):
        out = list(wt)
        for site in rng.choice(positions, size=budget, replace=False):
            choices = sorted(observed.get(int(site), set()) - {wt[site]}) if observed else None
            if not choices:
                choices = [a for a in AMINO_ACIDS if a != wt[site]]
            out[int(site)] = str(rng.choice(choices))
        designs.append("".join(out))
    return designs


# ══════════════════════════════════════════════════════════════════════════════
# Parse args
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budgets", type=int, nargs="+", default=list(BUDGETS),
                        help="Mutation budgets to design at (default: 5 10, "
                             "the counts Gelman et al. synthesized).")
    parser.add_argument("--designs", type=int, default=50,
                        help="Designs per setting (default: 50).")
    parser.add_argument("--steps", type=int, default=200,
                        help="Annealing proposals per chain (default: 200).")
    parser.add_argument("--train-csv", type=Path,
                        default=data_dir() / "avgfp_train_props.csv",
                        help="Source of the Observed substitution set.")
    parser.add_argument("--settings", nargs="+", default=["observed", "unobserved"],
                        choices=("observed", "unobserved"),
                        help="Gelman et al.'s two support settings.")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--out", type=Path,
                        default=outputs_dir() / "baselines" / "anneal.json")
    mf.add_arguments(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    if not mf.available():
        raise SystemExit("metl-pretrained is required; see postprocess.py.")
    wt = load_wt()
    which = args.metl_target or "ft-1d"
    observed = observed_substitutions(args.train_csv)
    covered = sum(len(v) for v in observed.values())
    wt_score = float(mf.predict([wt], wt, which)[0])
    print(f"Oracle: METL {which}   wild type {wt_score:+.4f}")
    print(f"Observed substitutions in training data: {covered} "
          f"across {len(observed)} positions\n")

    def score_batch(sequences):
        return mf.predict(sequences, wt, which).astype(np.float64)

    rows = []
    for setting in args.settings:
        # "unobserved" passes no support mask, so any residue may be drawn.
        support = observed if setting == "observed" else None
        for budget in args.budgets:
            rng = np.random.default_rng(args.seed + budget)
            nulls = random_variants(wt, budget, support, rng, args.designs)
            null_score = score_batch(nulls)
            best, best_score = anneal(wt, budget, support, rng, args.steps,
                                      score_batch, args.designs)
            for name, scores, designs in (("random", null_score, nulls),
                                          ("anneal", best_score, best)):
                rows.append({
                    "setting": setting, "budget": budget, "method": name,
                    "mean": float(scores.mean()), "sd": float(scores.std()),
                    "max": float(scores.max()),
                    "frac_above_wt": float((scores > wt_score).mean()),
                    "mut_mean": float(np.mean(
                        [len([p for p in seq_to_variant(s, wt).split(",") if p])
                         for s in designs])),
                })
                row = rows[-1]
                print(f"  {setting:<11}{budget:>3} mut  {name:<7} "
                      f"mean {row['mean']:+.4f}  max {row['max']:+.4f}  "
                      f">WT {row['frac_above_wt']:.2f}  "
                      f"muts {row['mut_mean']:.1f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"wt_score": wt_score, "oracle": which,
                                    "rows": rows}, indent=2))
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
