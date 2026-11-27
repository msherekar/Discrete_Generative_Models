#!/usr/bin/env python3
"""Weights & Biases for Project 1, without touching the experiment code.

Two subcommands:

  run    wrap an experiment: open a W&B run, launch dgm-run-experiment with the
         flags you give it, stream its output, then log the finished results.
  sync   log a run directory that already exists -- an OSG run recovered from
         OSDF, or any past run you want backfilled.

Both read results.pt, so loss curves arrive at full per-epoch resolution even
though run_experiment prints only four or five lines.

Usage:
  dgm-wandb run --name long250_20261001 -- --esm-model esm2_8m \\
      --dataset data/avgfp_train_props.csv --epochs 250 --samples 100
  dgm-wandb sync --run-dir results/long250_20261001
  dgm-wandb sync --run-dir results/long250 --tags osg l40 --group long250

Defaults: entity proterial, project CIS6270. Override with --entity/--project
or WANDB_ENTITY/WANDB_PROJECT.
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

from dgm.common.paths import outputs_dir
from dgm.common.tracking import (DEFAULT_ENTITY, DEFAULT_PROJECT, log_run,
                                 start_run)

# "  [flow]      epoch   62/250: loss 0.4668"
EPOCH_LINE = re.compile(
    r"\[(?P<method>flow|diffusion)\]\s+epoch\s+(?P<epoch>\d+)/(?P<total>\d+):"
    r"\s+loss\s+(?P<loss>[\d.]+)")


def _common(parser):
    parser.add_argument("--entity", default=None,
                        help=f"W&B entity (default: {DEFAULT_ENTITY})")
    parser.add_argument("--project", default=None,
                        help=f"W&B project (default: {DEFAULT_PROJECT})")
    parser.add_argument("--group", default=None,
                        help="Group several runs, e.g. one sweep's name.")
    parser.add_argument("--tags", nargs="*", default=[],
                        help="Tags, e.g. --tags osg l40 transformer")
    parser.add_argument("--notes", default=None, help="Free-text note.")
    parser.add_argument("--mode", default=None,
                        choices=("online", "offline", "disabled"),
                        help="W&B mode. 'offline' writes locally for a later "
                             "`wandb sync`, which is the route for a worker "
                             "node with no credentials.")
    parser.add_argument("--no-artifacts", action="store_true",
                        help="Skip uploading FASTA files and plots.")


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subs = parser.add_subparsers(dest="command", required=True)

    run = subs.add_parser("run", help="wrap an experiment and log it")
    _common(run)
    run.add_argument("--name", default=None,
                     help="W&B run name. Defaults to the output directory's.")
    run.add_argument("--outdir", type=Path, default=None,
                     help="Where the experiment writes. Defaults to "
                          "outputs/<name>. Passed through as --outdir.")
    run.add_argument("flags", nargs=argparse.REMAINDER,
                     help="Everything after -- goes to dgm-run-experiment.")

    sync = subs.add_parser("sync", help="log an existing run directory")
    _common(sync)
    sync.add_argument("--run-dir", type=Path, required=True,
                      help="A directory holding flow/ and diffusion/.")
    sync.add_argument("--name", default=None,
                      help="W&B run name. Defaults to the directory's name.")
    return parser


def _passthrough(flags):
    """The flags after the -- separator, which argparse leaves in place."""
    return [f for f in flags if f != "--"]


def _stream(command, run):
    """Run the experiment, echo its output, and report progress to W&B.

    The sparse epoch lines go up as a `progress/*` series purely so a long run
    visibly moves; the full curve is logged afterwards from results.pt against
    an `epoch` axis, so the two never contend for the same step.
    """
    process = subprocess.Popen(command, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True,
                               bufsize=1)
    for line in process.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        match = EPOCH_LINE.search(line)
        if match and run is not None:
            run.log({"progress/method": match["method"],
                     "progress/epoch": int(match["epoch"]),
                     "progress/loss": float(match["loss"])})
    return process.wait()


def main():
    args = build_parser().parse_args()

    if args.command == "sync":
        url = log_run(args.run_dir, name=args.name, group=args.group,
                      tags=args.tags, entity=args.entity,
                      project=args.project, mode=args.mode,
                      notes=args.notes, artifacts=not args.no_artifacts)
        print(f"\nlogged to {url}")
        return

    flags = _passthrough(args.flags)
    if not flags:
        raise SystemExit("nothing to run; put the experiment flags after --")

    name = args.name or "wandb_run"
    outdir = args.outdir
    if outdir is None:
        outdir = (Path(flags[flags.index("--outdir") + 1])
                  if "--outdir" in flags else outputs_dir(create=True) / name)
    if "--outdir" not in flags:
        flags += ["--outdir", str(outdir)]

    run = start_run(name, group=args.group, tags=args.tags,
                    entity=args.entity, project=args.project, mode=args.mode,
                    notes=args.notes)
    print(f"W&B run: {run.url}\n")

    command = [sys.executable, "-u", "-m", "dgm.project1.run_experiment", *flags]
    code = _stream(command, run)

    if code != 0:
        run.summary["exit_code"] = code
        run.finish(exit_code=code)
        raise SystemExit(f"run_experiment exited {code}; W&B run marked failed")

    log_run(outdir, name=name, group=args.group, tags=args.tags,
            artifacts=not args.no_artifacts, run=run)
    print(f"\nlogged to {run.url}")
    run.finish()


if __name__ == "__main__":
    main()
