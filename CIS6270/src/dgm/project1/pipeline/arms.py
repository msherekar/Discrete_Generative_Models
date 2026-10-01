"""Assembling the objective and the guidance arms a run will sample.

An "arm" is one sampling configuration: a conditioning weight, a reward
strength, a lambda mix and an objective. Every arm reads the same trained
field, so a sweep over arms costs one extra sampling pass each rather than a
retraining apiece -- which is why they are built here, after training is
already decided and before any sampling happens.
"""
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .objective import Objective


@dataclass
class ObjectiveSpec:
    """The objective a run optimizes, plus what is needed to describe it."""

    objective: Objective
    with_setpoint: Callable[[float], Objective]
    prop_names: list
    senses: list
    n_obj: int
    setpoint_raw: list
    constraint_index: Optional[int]
    constraint_threshold: Optional[float]


def _check_column(parser, value, flag, n_props, prop_names) -> int:
    """A 1-based property column as a 0-based index, or a parser error."""
    if not 1 <= value <= n_props:
        parser.error(f"{flag} must be between 1 and {n_props} "
                     f"(the CSV has {n_props} property columns: "
                     f"{', '.join(prop_names)})")
    return value - 1


def _resolve_setpoints(args, parser, raw_r1) -> list:
    """Requested setpoints in raw r1 units, converting percentiles if asked."""
    setpoint_raw = list(args.setpoint or [])
    if args.setpoint_percentile:
        if any(not 0.0 <= v <= 100.0 for v in setpoint_raw):
            parser.error("--setpoint-percentile expects values in [0, 100]")
        return [float(np.percentile(raw_r1.numpy(), v)) for v in setpoint_raw]
    if setpoint_raw:
        low, high = float(raw_r1.min()), float(raw_r1.max())
        outside = [v for v in setpoint_raw if not low <= v <= high]
        if outside:
            print(f"  [warn] setpoints outside the observed r1 range "
                  f"[{low:+.3f}, {high:+.3f}]: {outside}. These ask the reward "
                  f"model to extrapolate, which is the regime it is least "
                  f"reliable in.")
    return setpoint_raw


def _reference_constraint_value(raw_c, reference, prop_names, index) -> float:
    """The reference sequence's own value of the constrained property.

    Read from METL when a reference exists, so the constraint reads "no less
    stable than wild type"; otherwise the training median, which is the best
    available stand-in.
    """
    if reference is None:
        return float(raw_c.min())
    try:
        from .. import embedding_oracle as _eo
        name = prop_names[index]
        attr = _eo.metl_attributes_wt(reference)
        return float(attr[_eo.attribute_index(
            "total_score" if name == "r3" else name)])
    except Exception as exc:                            # noqa: BLE001
        fallback = float(np.median(raw_c.numpy()))
        print(f"  [warn] could not read the reference value for "
              f"{prop_names[index]} ({exc}); falling back to "
              f"the training median {fallback:+.3f}")
        return fallback


def _constraint_threshold(args, setup, constraint_index, prop_names):
    """The standardized ceiling for the constrained property, or None."""
    if constraint_index is None or args.constraint_rho <= 0:
        return None
    stats, dataset = setup.stats, setup.dataset
    raw_c = dataset.tensors[2][:, constraint_index].clone()
    raw_c = raw_c * float(stats["r_std"][constraint_index]) \
        + float(stats["r_mean"][constraint_index])
    reference_value = _reference_constraint_value(
        raw_c, setup.reference, prop_names, constraint_index)
    threshold = Objective.from_raw(
        reference_value + args.constraint_delta, constraint_index, stats)
    print(f"  Constraint reference {prop_names[constraint_index]}="
          f"{reference_value:+.3f}, slack {args.constraint_delta:+.3f}")
    return threshold


def build_objective(args, parser, setup) -> ObjectiveSpec:
    """The objective this run optimizes, validated against the CSV's columns."""
    n_props, prop_names, stats = setup.n_props, setup.prop_names, setup.stats

    def column(value, flag):
        return _check_column(parser, value, flag, n_props, prop_names)

    constraint_index = (None if args.constraint_property is None
                        else column(args.constraint_property,
                                    "--constraint-property"))
    n_obj = n_props if constraint_index is None else constraint_index
    if n_obj < 1:
        parser.error("--constraint-property must leave at least one objective column")

    senses = [1.0] * n_obj
    for name in (args.minimize or []):
        index = column(name, "--minimize")
        if index >= n_obj:
            parser.error(f"--minimize {name} names a constrained column, "
                         f"which has no direction on the objective simplex")
        senses[index] = -1.0

    # Setpoints are raw; the reward head predicts standardized values.
    raw_r1 = (setup.dataset.tensors[2][:, 0] * float(stats["r_std"][0])
              + float(stats["r_mean"][0]))
    setpoint_raw = _resolve_setpoints(args, parser, raw_r1)
    threshold = _constraint_threshold(args, setup, constraint_index, prop_names)

    objective = Objective(
        n_props=n_props,
        setpoint=None,
        senses=senses,
        constraint_index=constraint_index if args.constraint_rho > 0 else None,
        constraint_threshold=threshold,
        rho=args.constraint_rho,
    )
    print(objective.describe(stats, prop_names))

    def with_setpoint(value):
        """A copy of the objective targeting one raw brightness value."""
        return Objective(
            n_props=n_props, setpoint=Objective.from_raw(value, 0, stats),
            senses=senses,
            constraint_index=objective.constraint_index,
            constraint_threshold=threshold, rho=args.constraint_rho,
            setpoint_saturation=args.setpoint_saturation,
        )

    return ObjectiveSpec(
        objective=objective, with_setpoint=with_setpoint,
        prop_names=prop_names, senses=senses, n_obj=n_obj,
        setpoint_raw=setpoint_raw, constraint_index=constraint_index,
        constraint_threshold=threshold,
    )


def _eta_tag(value) -> str:
    return f"{value:g}"


def _reward_arms(args, parser, spec, eta, suffix, base_lambdas, mixed_lambdas):
    """Every reward-guided arm at one eta: single, multi, lambda and setpoint."""
    objective, n_obj = spec.objective, spec.n_obj
    arms = {f"single{suffix}": dict(c=1, w=0.0, eta=eta,
                                    lambdas=base_lambdas, objective=objective)}
    if n_obj >= 2:
        arms[f"multi{suffix}"] = dict(c=1, w=0.0, eta=eta,
                                      lambdas=mixed_lambdas, objective=objective)
    for value in (args.reward_lambda or []):
        if not 0.0 <= value <= 1.0:
            parser.error("--reward-lambda values must lie in [0, 1]")
        if n_obj < 2:
            parser.error("--reward-lambda needs at least two objective "
                         "columns; this run has one")
        arms[f"lam{value:g}{suffix}"] = dict(
            c=1, w=0.0, eta=eta,
            lambdas=(value, 1.0 - value) + (0.,) * (n_obj - 2),
            objective=objective)
    # Setpoint arms keep the same lambda mix and swap only the brightness term,
    # so a difference between them is the objective and not the weighting.
    for value, target in zip(args.setpoint or [], spec.setpoint_raw):
        tag = f"p{value:g}" if args.setpoint_percentile else f"{value:g}"
        arms[f"sp{tag}{suffix}"] = dict(
            c=1, w=0.0, eta=eta, lambdas=base_lambdas,
            objective=spec.with_setpoint(target))
    return arms


def _announce_sweeps(args, spec):
    """Say out loud what the arm set sweeps, and warn where it cannot."""
    if args.setpoint:
        pairs = ", ".join(f"{v:g}->{t:+.3f}" for v, t in
                          zip(args.setpoint, spec.setpoint_raw))
        kind = "percentile" if args.setpoint_percentile else "raw"
        print(f"  Setpoint arms ({kind}): {pairs}")
    if len(args.cfg_weight) > 1:
        named = ", ".join(f"{'cfg' if i == 0 else 'cfg@' + _eta_tag(w)}=w{w:g}"
                          for i, w in enumerate(args.cfg_weight))
        print(f"  Sweeping cfg weight {args.cfg_weight} from one trained model: {named}")
        if 0.0 in args.cfg_weight:
            which = "cfg" if args.cfg_weight[0] == 0.0 else f"cfg@{_eta_tag(0.0)}"
            print(f"    '{which}' has w=0 and is the UNCONDITIONAL control: it drops "
                  f"the conditional term entirely, so the gap between it and the w>0 "
                  f"arms is what the conditioning contributes.")
    if len(args.reward_eta) > 1:
        print(f"  Sweeping reward eta {args.reward_eta} from one trained model; "
              f"eta={args.reward_eta[0]} drives the bare 'single'/'multi' arms.")
    if args.reward_lambda:
        print(f"  Sweeping lambda {args.reward_lambda} "
              f"(brightness weight; parsimony gets the remainder)")
        if args.exact_mutations:
            print("  [warn] --exact-mutations fixes the substitution count, so the "
                  "parsimony objective cannot change it and the lambda arms will be "
                  "near-identical. Drop it, and make --mut-budget a ceiling, for a "
                  "real trade-off.")


def build_arms(args, parser, spec) -> dict:
    """Every guidance configuration to sample, keyed by arm name.

    Eta and the CFG weight are sampling-time knobs, so several values cost one
    extra sampling pass each. The first value of each keeps the bare name
    ('cfg', 'single', 'multi') so every downstream tool reads the run unchanged.
    """
    n_obj = spec.n_obj
    base_lambdas = (1.,) + (0.,) * (n_obj - 1)
    mixed_lambdas = ((0.7, 0.3) + (0.,) * (n_obj - 2)) if n_obj >= 2 else (1.,)
    arms = {
        f"cfg{'' if i == 0 else '@' + _eta_tag(w)}": dict(
            c=1, w=w, eta=0.0, lambdas=base_lambdas, objective=spec.objective)
        for i, w in enumerate(args.cfg_weight)
    }
    for position, eta in enumerate(args.reward_eta):
        suffix = "" if position == 0 else f"@{_eta_tag(eta)}"
        arms.update(_reward_arms(args, parser, spec, eta, suffix,
                                 base_lambdas, mixed_lambdas))
    _announce_sweeps(args, spec)
    return arms
