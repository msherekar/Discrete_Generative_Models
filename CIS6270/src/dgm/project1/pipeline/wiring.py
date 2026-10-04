"""Translating parsed CLI arguments into the keyword sets each stage takes.

Exists so run_experiment.py stays an orchestrator: every flag added in Stage
0.5 through Stage 4 is read exactly once, here, and a stage that gains an
option does not grow another branch in the run script.
"""
import torch

from .losses import censor_floor_from_stats
from .optim import scale_lr_for_batch
from .sampling import X0_LIMIT

# Batch size the default learning rate was tuned at; see optim.scale_lr_for_batch.
LR_REFERENCE = 128


def latent_kwargs(args):
    """Storage options for the encoded latents, for data.load_data."""
    return {"store_dtype": getattr(torch, args.latent_dtype),
            "store_device": args.latent_device}


def censor_floor(args, stats):
    """The assay floor in the standardized units the reward head sees.

    Returned as None when --censor-floor was not given, which restores plain
    MSE and is the Stage 0.1 ablation control.
    """
    if args.censor_floor is None:
        return None
    return censor_floor_from_stats(stats, args.censor_floor)


def train_kwargs(args, stats, val_dataset=None):
    """Everything both training loops take beyond the dataset and epochs.

    The learning rate is resolved here rather than in the loops so that
    --scale-lr applies identically to both, and so the resolved value lands in
    the saved run config where a later reader can see what actually ran.
    """
    lr = args.lr if args.lr is not None else None
    if lr is not None and args.scale_lr:
        lr = scale_lr_for_batch(lr, args.batch_size, LR_REFERENCE)
    kwargs = {
        "batch_size": args.batch_size,
        "hidden": args.hidden,
        "arch": args.arch,
        "ema_decay": args.ema,
        "reward_lr": args.reward_lr,
        "warmup": args.warmup,
        "amp": args.amp,
        "val_dataset": val_dataset,
        "censor_floor": censor_floor(args, stats),
        "conditioning": args.conditioning,
        "modulation": args.modulation,
        "rope": args.rope,
        "n_cond": args.n_cond,
        "seed": args.seed,
    }
    if lr is not None:
        kwargs["lr"] = lr
    return kwargs


def flow_train_kwargs(args, stats, val_dataset=None, path=None, columns=None):
    """Flow-specific additions: the path and the coupling."""
    kwargs = train_kwargs(args, stats, val_dataset)
    kwargs.update({"path": path, "coupling": args.coupling,
                   "coupling_beta": args.coupling_beta,
                   "coupling_columns": columns})
    return kwargs


def diffusion_train_kwargs(args, stats, val_dataset=None):
    """Diffusion-specific additions: parameterization, schedule, weighting."""
    kwargs = train_kwargs(args, stats, val_dataset)
    kwargs.update({"predict": args.predict,
                   "steps": args.diffusion_steps,
                   "stratified": args.stratified_timesteps,
                   "weighting": args.loss_weighting,
                   "beta_schedule": args.beta_schedule})
    return kwargs


def _x0_limit(args):
    """--x0-limit, where 0 means "off" and absent means the sampler default."""
    if args.x0_limit is None:
        return X0_LIMIT
    return None if args.x0_limit <= 0 else args.x0_limit


def sample_kwargs(args):
    """Guidance and conditioning options common to both samplers."""
    return {"clip": args.guidance_clip,
            "normalize": args.normalize_guidance,
            "endpoint_guidance": args.endpoint_guidance,
            "scaling": args.guidance_scaling,
            "conditioning": args.conditioning,
            "corrector_steps": args.corrector_steps,
            "corrector_snr": args.corrector_snr,
            "seed": args.sample_seed}


def flow_sample_kwargs(args, path, source=None):
    """Flow sampler options, including the solver and its step budget."""
    return {"path": path, "steps": args.steps, "source": source,
            "solver": args.flow_solver}


def diffusion_sample_kwargs(args):
    """Diffusion sampler options.

    --sample-steps defaults to --steps so that one flag sets the budget for
    both methods, which is what makes an NFE-matched grid a single sweep axis
    rather than two that have to be kept in step by hand.
    """
    return {"solver": args.diffusion_solver,
            "steps": args.sample_steps or args.steps,
            "stochasticity": args.ddim_stochasticity,
            "spacing": args.step_spacing,
            "churn_amount": args.churn,
            "x0_limit": _x0_limit(args)}


def describe_innovations(args):
    """One block naming every non-default choice, for the run log.

    Printed because a sweep of this size produces directories whose names
    cannot carry every flag, and the log is what a later reader has. Only
    departures from the baseline are listed, so an unchanged run prints
    nothing and a changed one is self-documenting.
    """
    defaults = {"conditioning": "binary", "modulation": "adaln",
                "beta_schedule": "linear", "loss_weighting": "none",
                "predict": "x0", "flow_solver": "euler",
                "diffusion_solver": "ddpm", "step_spacing": "linear",
                "guidance_scaling": "score", "churn": 0.0,
                "corrector_steps": 0, "ddim_stochasticity": 0.0,
                "censor_floor": None, "ema": 0.0, "warmup": 0,
                "endpoint_guidance": False, "normalize_guidance": False,
                "guidance_clip": 0.0, "coupling": "independent",
                "path_geometry": "segment", "time_schedule": "linear",
                "n_cond": 1, "stratified_timesteps": False}
    changed = [f"{name.replace('_', '-')}={getattr(args, name)}"
               for name, default in sorted(defaults.items())
               if getattr(args, name, default) != default]
    if not args.rope:
        changed.append("no-rope")
    return "  Innovations : " + (", ".join(changed) if changed else "none (baseline)")
