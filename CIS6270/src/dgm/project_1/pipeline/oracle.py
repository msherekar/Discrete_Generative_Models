"""The independent judge: loading a fitted oracle and scoring decoded samples.

The oracle never touches guidance. It is fitted by embedding_oracle.py on held
-out data and only ever reads finished sequences, so a high score is evidence
rather than a restatement of the objective.
"""
from pathlib import Path

from dgm.common.paths import data_dir

def decode_all_budgets(z, decode_fns, primary, oracle):
    """Decode one set of latents at every requested mutation budget.

    The budget is a property of decoding, not of the model: the same latent
    yields a 3-substitution and a 15-substitution variant depending only on how
    many positions the decoder is allowed to change. Sweeping it therefore costs
    one decode per budget instead of one training run per budget, and the
    comparison is exact rather than confounded, since every budget reads the same
    latents.
    """
    entry = {"latent": z}
    by_budget = {}
    for budget, fn in decode_fns.items():
        seqs = fn(z)
        by_budget[budget] = {"sequences": seqs,
                             "oracle": score_with_oracle(seqs, oracle)}
    entry["sequences"] = by_budget[primary]["sequences"]
    entry["oracle"] = by_budget[primary]["oracle"]
    if len(by_budget) > 1:
        entry["by_budget"] = by_budget
    return entry


def load_brightness_oracle(path: Path):
    """Load a fitted oracle from embedding_oracle.py, or return None.

    Kept optional and lazily imported: the METL backend pulls in the metl
    repository and its dependencies, which the toy peptide runs do not need.
    """
    if path is None or not Path(path).is_file():
        return None
    try:
        from .. import embedding_oracle
        oracle = embedding_oracle.load_oracle(Path(path))
    except Exception as exc:
        print(f"  [skip] could not load oracle {path}: {type(exc).__name__}: {exc}")
        return None
    print(f"  Oracle: {oracle[3]} backend from {Path(path).name}")
    return oracle


def score_with_oracle(sequences, oracle):
    """Predicted brightness for decoded sequences, or None if no oracle."""
    if oracle is None:
        return None
    from .. import embedding_oracle
    return embedding_oracle.score_sequences(list(sequences), oracle)


def resolve_oracle(args, length):
    """The oracle to judge this run with, and the path it came from.

    Returns (None, None) when scoring is off, unavailable, or would be
    meaningless: an oracle is fitted to one protein at one length, so silently
    scoring a different one would produce confident nonsense.
    """
    default_oracle = data_dir() / "avgfp_metl_oracle.npz"
    oracle_path = None if args.no_oracle else (
        args.oracle or (default_oracle if default_oracle.is_file() else None))
    oracle = load_brightness_oracle(oracle_path)
    if oracle is not None and len(oracle[2]) != length:
        print(f"  [skip] oracle was fitted on a {len(oracle[2])}-residue protein but "
              f"these sequences are {length}; not scoring")
        return None, None
    if oracle is None and not args.no_oracle:
        # Say so even when oracle_path is None. Previously that case returned
        # silently, so a job launched with scoring ON but no oracle file staged
        # looked exactly like a successful run and produced no brightness
        # column -- the failure only showed up much later, as a missing table.
        where = oracle_path or f"{default_oracle} (default)"
        print(f"  [warn] scoring was requested but no brightness oracle loaded "
              f"from {where}; samples will NOT be scored. Pass --oracle "
              f"<file.npz>, or fit one with: "
              f"dgm-embedding-oracle --fit --backend metl")
    return oracle, oracle_path


if __name__ == "__main__":
    import numpy as np, tempfile
    from pathlib import Path

    # load_brightness_oracle with missing path returns None gracefully.
    result = load_brightness_oracle(None)
    assert result is None
    print("  load_brightness_oracle(None): None (expected)")

    result = load_brightness_oracle(Path("/nonexistent/oracle.npz"))
    assert result is None
    print("  load_brightness_oracle(missing): None (expected)")

    # score_with_oracle with None oracle returns None.
    preds = score_with_oracle(["ACDEFG", "ACDEFH"], None)
    assert preds is None
    print("  score_with_oracle(oracle=None): None (expected)")

    # resolve_oracle with --no-oracle flag skips loading entirely.
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-oracle", action="store_true")
    parser.add_argument("--oracle", default=None)
    args = parser.parse_args(["--no-oracle"])
    oracle_obj, oracle_path = resolve_oracle(args, length=6)
    assert oracle_obj is None and oracle_path is None
    print("  resolve_oracle(--no-oracle): (None, None)")

    print("oracle.py OK")
