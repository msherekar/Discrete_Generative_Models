#!/usr/bin/env python3
"""Annotate a prepared GFP CSV with METL-predicted Rosetta attributes.

prepare_gfp.py writes r1 = brightness and r2 = -num_mutations. Mutation count
is not a property anyone wants to optimize, though: it is the experimental axis
(fix it at k, report brightness) or a parsimony cost, not a biophysical
objective. This script replaces r2 and adds r3 with quantities that are.

  r1  brightness              experimental, unchanged (higher is better)
  r2  exposed_hydrophobics    aggregation proxy       (LOWER is better)
  r3  total_score             Rosetta stability       (LOWER is better)

Both come from the 55-attribute pretraining head of METL-L-2M-3D-GFP, which the
shipped checkpoint retains. No Rosetta runs: the simulation that produced these
attributes is already amortized into the pretrained weights, which is the only
reason this is computable -- Rosetta itself costs ~135-215 s per variant at 237
residues.

Values are written RAW, in METL's standardized pretraining units. Direction is
applied in the reward, not here, because the stability constraint compares
r3 against the wild type's own total_score and an already-negated column would
invert that comparison. run_experiment.py standardizes every rN column against
the training set before training, so absolute scale here does not matter.

Wild-type attributes are written to a sidecar JSON; the soft stability
constraint is expressed relative to them.

Usage:
  dgm-add-properties --csv data/avgfp_train.csv --reference data/avgfp_wt.txt
  dgm-add-properties --csv data/avgfp_train.csv --reference data/avgfp_wt.txt \
      --objective unsat_hbond --out data/avgfp_train_unsat.csv
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from dgm.project1 import embedding_oracle as eo

from dgm.common.paths import project_dir

ROOT = project_dir()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, required=True,
                        help="Prepared CSV with a 'sequence' column.")
    parser.add_argument("--reference", type=Path, required=True,
                        help="File holding the wild-type sequence.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output CSV (default: overwrite --csv in place).")
    parser.add_argument("--objective", default="exposed_hydrophobics",
                        help="Attribute written to r2, the second objective. "
                             "Default exposed_hydrophobics: Spearman -0.15 with "
                             "brightness on avgfp_train, so the tradeoff against "
                             "brightness is real. unsat_hbond (-0.13) is an "
                             "equally good choice. Do NOT use total_score here: "
                             "at -0.69 it is the most brightness-redundant of "
                             "the 55 and the lambda sweep would trace a line "
                             "rather than a front.")
    parser.add_argument("--constraint", default="total_score",
                        help="Attribute written to r3, used by the soft "
                             "constraint (default: total_score).")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--report-only", action="store_true",
                        help="Print correlations and wild-type values, write nothing.")
    return parser.parse_args()


def main():
    args = parse_args()
    wt = args.reference.read_text().strip().upper()
    out_path = args.out or args.csv

    with args.csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or "sequence" not in rows[0]:
        raise SystemExit(f"{args.csv} has no 'sequence' column")
    sequences = [r["sequence"].strip().upper() for r in rows]
    if any(len(s) != len(wt) for s in sequences):
        raise SystemExit("Every sequence must have the wild-type length; "
                         "this head is METL-Local and is specific to this protein")

    j_obj = eo.attribute_index(args.objective)
    j_con = eo.attribute_index(args.constraint)
    print(f"  sequences   : {len(sequences)}")
    print(f"  objective r2: {args.objective} (index {j_obj}, lower is better)")
    print(f"  constraint r3: {args.constraint} (index {j_con}, lower is better)")

    print("  running the METL prediction head...")
    attrs = eo.metl_attributes(sequences, wt, batch_size=args.batch_size)
    wt_attrs = eo.metl_attributes_wt(wt)
    print(f"  attributes  : {attrs.shape}")

    obj, con = attrs[:, j_obj], attrs[:, j_con]
    print(f"  {args.objective:<22} mean={obj.mean():+.3f} sd={obj.std():.3f} "
          f"WT={wt_attrs[j_obj]:+.3f}")
    print(f"  {args.constraint:<22} mean={con.mean():+.3f} sd={con.std():.3f} "
          f"WT={wt_attrs[j_con]:+.3f}")

    if "r1" in rows[0]:
        from scipy.stats import spearmanr
        brightness = np.array([float(r["r1"]) for r in rows])
        rho_obj = spearmanr(obj, brightness).statistic
        rho_con = spearmanr(con, brightness).statistic
        print(f"  spearman(r2, brightness) = {rho_obj:+.4f}"
              f"   <- want this near zero for a real Pareto front")
        print(f"  spearman(r3, brightness) = {rho_con:+.4f}")
        if abs(rho_obj) > 0.4:
            print(f"  [warn] r2 is strongly correlated with brightness. The lambda "
                  f"sweep will trace a line, not a front. Pick a different "
                  f"--objective.")

    if args.report_only:
        print("  --report-only: nothing written")
        return

    fields = [f for f in rows[0].keys() if f not in ("r2", "r3")]
    at = fields.index("r1") + 1 if "r1" in fields else len(fields)
    fields[at:at] = ["r2", "r3"]
    for row, o, c in zip(rows, obj, con):
        row["r2"] = f"{float(o):.6f}"
        row["r3"] = f"{float(c):.6f}"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {out_path}  ({len(rows)} rows, fields: {', '.join(fields)})")

    sidecar = out_path.with_suffix(".wt_attributes.json")
    sidecar.write_text(json.dumps({
        "wild_type": wt,
        "attribute_names": list(eo.attribute_names()),
        "wild_type_attributes": [float(v) for v in wt_attrs],
        "objective": {"name": args.objective, "index": j_obj,
                      "wild_type": float(wt_attrs[j_obj]), "column": "r2"},
        "constraint": {"name": args.constraint, "index": j_con,
                       "wild_type": float(wt_attrs[j_con]), "column": "r3"},
    }, indent=2))
    print(f"  wrote {sidecar}")


if __name__ == "__main__":
    main()
