"""Turning parsed arguments into everything a run needs before training.

This is the decision-making half of the old main(): which reference sequence,
which substitutions are allowed, which mutation budgets to decode at, and
whether sampling starts from noise or from a partially noised reference. All of
it is settled before a single weight is trained, so a bad combination fails
here rather than after an hour on the GPU.
"""
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .config import AMINO_ACIDS, DEVICE
from .data import consensus, encode_reference, load_data
from .decoding import decode, decode_budget, load_support_mask


@dataclass
class RunSetup:
    """Everything fixed before training, in one object instead of ten locals."""

    dataset: Any
    esm: Any
    tokenizer: Any
    stats: dict
    sequences: list
    length: int
    dim: int
    reference: Optional[str]
    source: Optional[str]
    support: Optional[np.ndarray]
    frozen: tuple
    budgets: Optional[list]
    decode_fns: dict
    primary: Any
    anchor_kwargs: dict = field(default_factory=dict)

    @property
    def prop_names(self):
        n = self.dataset.tensors[2].shape[1]
        return self.stats.get("r_names") or [f"r{i + 1}" for i in range(n)]

    @property
    def n_props(self) -> int:
        return self.dataset.tensors[2].shape[1]


def describe_run(args, model_info, dataset_tag, run_tag, out_root):
    """Print the banner that records what this run was asked to do."""
    print("\nProject 1 — Experiment Runner")
    print(f"  ESM-2 model : {args.esm_model}  ({model_info['hf_id']})")
    print(f"  Dataset     : {args.dataset}  (tag: {dataset_tag})")
    print(f"  Run tag     : {run_tag}")
    print(f"  Output dir  : {out_root}")
    print(f"  Device      : {DEVICE}")
    print(f"  Guidance    : cfg w={args.cfg_weight}  reward eta={args.reward_eta}"
          + (f"  clip={args.guidance_clip}" if args.guidance_clip else "")
          + ("  normalized" if args.normalize_guidance else "")
          + ("  endpoint" if args.endpoint_guidance else ""))
    print(f"  Batch size  : {args.batch_size}   hidden width: {args.hidden}")
    # The path itself is printed once resolved, in run_experiment: a data-arc
    # geometry measures its scale from the encoded data, which is not loaded yet.
    print(f"  Architecture: {args.arch}   coupling: {args.coupling}"
          + (f" beta={args.coupling_beta}" if args.coupling == "aux" else ""))
    print(f"  Seed        : {args.seed} (train)  {args.sample_seed} (sampling)")
    print(f"  Diffusion   : predict={args.predict}  K={args.diffusion_steps}"
          + (f"  ema={args.ema}" if args.ema else "")
          + ("  stratified" if args.stratified_timesteps else ""))


def _resolve_reference(args, parser, sequences, length):
    """The reference sequence and where it came from, or (None, None)."""
    if args.reference is None and args.mut_budget is None \
            and args.anchor_strength is None:
        return None, None
    reference = (args.reference.read_text().strip().upper() if args.reference
                 else consensus(sequences))
    if len(reference) != length:
        parser.error(f"reference has {len(reference)} residues, expected {length}")
    return reference, ("supplied" if args.reference else "consensus")


def _resolve_support(args, parser, reference, length):
    """The allowed-substitution mask for --restrict-support, or None."""
    if args.restrict_support is None:
        return None
    if args.mut_budget is None:
        parser.error("--restrict-support only applies to budgeted decoding; "
                     "pass --mut-budget as well")
    support = load_support_mask(args.restrict_support, reference,
                                args.support_level)
    per_position = support.sum(axis=1)
    total = length * (len(AMINO_ACIDS) - 1)
    print(f"  Support     : {args.restrict_support.name} "
          f"({args.support_level}) — {int(support.sum())} of "
          f"{total} substitutions "
          f"({100 * support.sum() / total:.1f}%), "
          f"{int((per_position > 0).sum())} of {length} positions usable, "
          f"median {int(np.median(per_position))} alternatives each")
    return support


def _build_decoders(args, parser, setup_bits) -> dict:
    """One decode function per requested mutation budget, keyed by budget.

    The budget is a decode-time choice: the trained model does not depend on it,
    so several budgets cost one extra decode each rather than a retraining
    apiece. Binding `b` per lambda keeps late binding from collapsing them all
    onto the last budget.
    """
    esm, tokenizer, stats, reference, source, support, frozen = setup_bits
    budgets = args.mut_budget
    if budgets is None:
        return {None: lambda z: decode(z, esm, tokenizer, stats, args.min_polar)}
    if len(set(budgets)) != len(budgets):
        parser.error("--mut-budget has duplicate values")
    rule = ("argmax" if args.decode_temperature <= 0
            else f"sampled at T={args.decode_temperature}")
    rule += ", exactly" if args.exact_mutations else ", at most"
    print(f"  Decoding {rule} {budgets} substitutions from the "
          f"{source} reference")
    if frozen:
        print(f"  Frozen positions (never substituted): {list(frozen)}")
    if len(budgets) > 1:
        print(f"  Budget {budgets[0]} drives the saved sequences and FASTA; "
              f"{budgets[1:]} are decoded from the same latents for comparison.")
    return {
        b: (lambda z, b=b: decode_budget(z, esm, tokenizer, stats, reference,
                                         b, args.decode_temperature,
                                         frozen, args.exact_mutations, support))
        for b in budgets}


def _resolve_anchor(args, parser, reference, source, esm, tokenizer, stats):
    """Sampling kwargs that start the trajectory at a noised reference, or {}."""
    if args.anchor_strength is None:
        return {}
    if not 0.0 < args.anchor_strength <= 1.0:
        parser.error("--anchor-strength must be in (0, 1]")
    anchor = encode_reference(reference, esm, tokenizer, stats)
    print(f"  Anchoring sampling at strength {args.anchor_strength} "
          f"from the {source} reference")
    return {"anchor": anchor, "strength": args.anchor_strength}


def prepare_run(args, parser, model_info, cache_dir) -> RunSetup:
    """Encode the dataset and settle every pre-training choice."""
    # Imported here rather than at module scope: wiring imports sampling,
    # which imports this module's siblings, and a top-level import would close
    # the cycle.
    from .wiring import latent_kwargs
    print("\nEncoding sequences with ESM-2...")
    dataset, esm, tokenizer, stats, sequences = load_data(
        args.dataset, model_info["hf_id"], cache_dir, args.max_length,
        encode_batch=getattr(args, "encode_batch", None) or args.batch_size,
        **latent_kwargs(args),
    )
    _, length, dim = dataset.tensors[0].shape

    if not 0 <= args.min_polar <= length:
        parser.error(f"--min-polar must be between 0 and {length}")

    reference, source = _resolve_reference(args, parser, sequences, length)
    frozen = tuple(int(x) for x in args.freeze_positions.replace(",", " ").split())
    support = _resolve_support(args, parser, reference, length)
    decode_fns = _build_decoders(
        args, parser,
        (esm, tokenizer, stats, reference, source, support, frozen))
    anchor_kwargs = _resolve_anchor(args, parser, reference, source,
                                    esm, tokenizer, stats)
    return RunSetup(
        dataset=dataset, esm=esm, tokenizer=tokenizer, stats=stats,
        sequences=sequences, length=length, dim=dim,
        reference=reference, source=source, support=support, frozen=frozen,
        budgets=args.mut_budget, decode_fns=decode_fns,
        primary=next(iter(decode_fns)), anchor_kwargs=anchor_kwargs,
    )
