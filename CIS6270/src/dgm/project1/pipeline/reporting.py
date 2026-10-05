"""Everything a run prints about its own samples.

Kept out of the run loop so the tables can be read, and reformatted, without
scrolling past the training code -- and so a table added here cannot break
sampling.
"""
import numpy as np

from .config import POLAR_RESIDUES
from .data import composition_proxies


def _hamming(sequences, reference):
    """Per-sequence substitution count against the reference."""
    return [sum(a != b for a, b in zip(s, reference)) for s in sequences]


def report_arm(name, seqs, reference, scores=None, latent=None):
    """One arm's samples: diversity, divergence, and brightness if scored."""
    unique = len(set(seqs))
    print(f"    distinct sequences {unique}/{len(seqs)}"
          + ("   <- collapsed; raise --decode-temperature" if unique < len(seqs) // 5
             else ""))
    if latent is not None:
        norms = latent.reshape(len(latent), -1).norm(dim=1)
        diverged = int((norms > 3 * norms.median()).sum())
        if diverged:
            print(f"    {diverged} sample(s) numerically diverged "
                  f"(latent norm up to {norms.max():.0f} vs median "
                  f"{norms.median():.0f}); lower --cfg-weight/--reward-eta")
    if scores is not None:
        print(f"    oracle brightness  mean {scores.mean():+.3f}   "
              f"best {scores.max():+.3f}   worst {scores.min():+.3f}")
    if reference is None:
        proxies = composition_proxies(seqs)
        print(f"  {name}: {seqs[0][:48]}...  "
              f"polar={sum(a in POLAR_RESIDUES for a in seqs[0])}  "
              f"r1={proxies[:, 0].mean():.3f}  r2={proxies[:, 1].mean():.3f}")
    else:
        distances = _hamming(seqs, reference)
        print(f"  {name}: {seqs[0][:48]}...  "
              f"mean hamming to reference {sum(distances)/len(distances):.1f} "
              f"(min {min(distances)}, max {max(distances)})")


def _brightness_table(runs, reference):
    """Oracle brightness by method and guidance mode."""
    print("\nOracle brightness by method and guidance mode "
          "(wild-type centered, higher is better)")
    header = (f"  {'method':<12}{'mode':<9}{'mean':>9}{'best':>9}"
              f"{'hamming':>9}{'uniq':>7}")
    print(header + "\n  " + "-" * (len(header) - 2))
    for method, latents in runs:
        for name, latent in latents.items():
            scores = latent["oracle"]
            distance = (float("nan") if reference is None
                        else float(np.mean(_hamming(latent["sequences"], reference))))
            print(f"  {method:<12}{name:<9}{scores.mean():>9.3f}"
                  f"{scores.max():>9.3f}{distance:>9.1f}"
                  f"{len(set(latent['sequences'])):>4}/{len(latent['sequences'])}")


def _pareto_table(runs, reference, arms):
    """Each objective's achieved value, not the weight it was given.

    Parsimony is only a real objective when the decoder is free to vary the
    count, so the substitution column is what shows whether the trade-off
    exists at all -- if it is constant down a block, lambda is weighting
    something the decoder overrides.
    """
    print("\nObjective trade-off (lambda = weight on brightness; "
          "parsimony gets the rest)")
    head = (f"  {'method':<11}{'arm':<14}{'lambda':>7}{'mutations':>11}"
            f"{'brightness':>12}{'best':>9}{'uniq':>8}")
    print(head + "\n  " + "-" * (len(head) - 2))
    for method, latents in runs:
        for name, latent in sorted(latents.items()):
            if not name.startswith("lam"):
                continue
            weight = arms[name]["lambdas"][0]
            seqs = latent["sequences"]
            mutations = np.mean(_hamming(seqs, reference))
            s = latent["oracle"]
            mean = float("nan") if s is None else s.mean()
            best = float("nan") if s is None else s.max()
            print(f"  {method:<11}{name:<14}{weight:>7.2f}{mutations:>11.2f}"
                  f"{mean:>12.3f}{best:>9.3f}"
                  f"{len(set(seqs)):>5}/{len(seqs)}")
    print("  A front needs both columns to move: brightness rising as "
          "mutations rise means the objectives genuinely compete. A flat "
          "mutation column means parsimony had no room to act.")


def _budget_table(runs):
    """The same latents decoded at each mutation budget."""
    print("\nSame latents decoded at each mutation budget "
          "(brightness is wild-type centered)")
    head = (f"  {'method':<12}{'mode':<9}{'budget':>7}{'mean':>9}"
            f"{'best':>9}{'uniq':>8}")
    print(head + "\n  " + "-" * (len(head) - 2))
    for method, latents in runs:
        for name, latent in latents.items():
            for budget, block in latent.get("by_budget", {}).items():
                s = block["oracle"]
                mean = float("nan") if s is None else s.mean()
                best = float("nan") if s is None else s.max()
                print(f"  {method:<12}{name:<9}{budget:>7}{mean:>9.3f}"
                      f"{best:>9.3f}"
                      f"{len(set(block['sequences'])):>5}/"
                      f"{len(block['sequences'])}")
    print("  Brightness falling as the budget rises is the expected "
          "cost of mutating more; rising means guidance is finding "
          "something the extra positions allow.")


def _collapse_warning(runs):
    """Warn when guidance changed nothing at all."""
    identical = all(
        latents[m]["sequences"] == latents[list(latents)[0]]["sequences"]
        for _, latents in runs for m in latents)
    if identical:
        print("\n  Warning: every guidance mode decoded to the same sequences. "
              "Guidance is not\n  changing the output — raise the CFG weight w and "
              "the reward weight eta.\n  Both defaults were tuned on a 24x320 latent; "
              "this one is much larger.")


def oracle_summary(args, setup, arms, flow_latents, diff_latents,
                   oracle_path, out_root):
    """The full post-run comparison, printed only when an oracle scored samples."""
    # Drop whichever modality this job did not run. With one method per OSG
    # job (--only), half of these are None, and a None would otherwise reach
    # the table builders as if it were a set of latents.
    runs = tuple((name, latents)
                 for name, latents in (("flow", flow_latents),
                                       ("diffusion", diff_latents))
                 if latents is not None)
    if not runs:
        return
    reference = setup.reference
    _brightness_table(runs, reference)
    if args.reward_lambda and reference is not None:
        _pareto_table(runs, reference, arms)
    if len(setup.decode_fns) > 1:
        _budget_table(runs)
    _collapse_warning(runs)
    print("\n  For the random-variant control and the full metric table, run:")
    print(f"    dgm-gfp-metrics --run-dir {out_root} \\")
    print(f"        --embedding-oracle {oracle_path} --baseline-n 50")


if __name__ == "__main__":
    # report_arm prints to stdout; verify it runs without raising.
    wt = "ACDEFGHIKLMN"
    seqs = [wt, "ECDEFGHIKLMN", "ACDEFGHIKLMK", "WCDEFGHIKLMN"]
    report_arm("test", seqs, wt, scores=None)   # should print hamming stats

    # report_arm with no reference (composition proxy path).
    report_arm("noref", seqs, reference=None, scores=None)

    print("reporting.py OK")
