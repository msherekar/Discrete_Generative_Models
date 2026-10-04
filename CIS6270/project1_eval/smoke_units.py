"""Unit-level smoke checks for the Stage 0-2 pipeline rewrite.

Exercises every trunk, conditioning, prediction and schedule combination on
tiny random tensors, so a signature break is caught in seconds rather than on
an OSG node ten hours in.
"""
import argparse
from pathlib import Path

import torch
from torch.utils.data import TensorDataset

from dgm.project1.pipeline.schedules import BETA_SCHEDULES, describe
from dgm.project1.pipeline.training import train_diffusion, train_flow

# Deliberately tiny: the point is signature and shape coverage, not learning.
SHAPE = (64, 32, 16)
HIDDEN, BATCH = 64, 16


def fake_dataset(n, length, dim, n_props=3, device=None):
    """A dataset with the same structure load_data returns, on the real device.

    float16 latents on the GPU specifically, because that is the layout
    Batches takes its on-device fast path for; a float32 CPU tensor would
    exercise the DataLoader fallback instead and miss the path that matters.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device == "cuda" else torch.float32
    return TensorDataset(torch.randn(n, length, dim, device=device, dtype=dtype),
                         torch.randint(0, 2, (n,), device=device),
                         torch.randn(n, n_props, device=device))


def check_schedules():
    """Print where each beta schedule puts its noise; see schedules.describe."""
    print("Beta schedules")
    for kind in BETA_SCHEDULES:
        print(f"  {describe(1000, kind)}")


def check_flow(dataset, arch, conditioning, path_schedule="linear"):
    """One short flow run, returning the final training loss."""
    from dgm.project1.pipeline.paths import PathSpec
    model, reward, losses, _ = train_flow(
        dataset, epochs=2, batch_size=BATCH, hidden=HIDDEN, arch=arch,
        path=PathSpec(schedule=path_schedule), conditioning=conditioning,
        ema_decay=0.999, warmup=5, val_dataset=dataset, censor_floor=-1.5,
        n_cond=3)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  flow      {arch:<12} {conditioning:<10} {path_schedule:<10} "
          f"{params:5.2f}M  loss {losses[-1]:.4f}")
    return model, reward


def check_diffusion(dataset, arch, conditioning, predict, weighting="none",
                    beta="linear"):
    """One short diffusion run over a given parameterization and schedule."""
    out = train_diffusion(
        dataset, epochs=1, batch_size=BATCH, hidden=HIDDEN, arch=arch,
        predict=predict, steps=100, conditioning=conditioning, ema_decay=0.999,
        weighting=weighting, beta_schedule=beta, val_dataset=dataset,
        censor_floor=-1.5, n_cond=3)
    model, reward, losses = out[0], out[1], out[2]
    print(f"  diffusion {arch:<12} {conditioning:<10} {predict:<4} "
          f"{weighting:<8} {beta:<8} loss {losses[-1]:.4f}")
    return model, reward, out


def check_recovery(mean=1.0, spread=0.3, epochs=400):
    """Train on data concentrated at a known mean; check sampling finds it.

    The strongest correctness test available without an oracle, and the one
    that distinguishes "the sampler runs" from "the sampler samples". If the
    schedule, the parameterization, the eps/x0 conversion and the solver all
    agree, the samples must land near `mean`; if any one of them has a sign or
    a sqrt in the wrong place, they land near zero or diverge.

    `mean` and `spread` are kept inside the standardized range the real
    latents occupy, because the parameterizations are not equally robust
    outside it: at mean 3.0 and spread 0.1, eps-prediction fails even with the
    endpoint clipped, so a test there would report a pathology of the test
    rather than of the method.

    Expect x0 to recover the mean closely, v to recover it with an inflated
    spread, and eps NOT to recover it. That ordering is a measured property of
    the parameterizations rather than a build failure, and it is the Axis C
    result in miniature. Diagnosed by comparing each model's eps accuracy at
    matched noise levels on this same data:

        abar     x0-mode eps rmse   eps-mode eps rmse
        0.972    1.426              0.873
        0.494    0.293              0.407
        0.024    0.048              0.343

    Each parameterization is accurate where its own target is nearly
    determined by its input: at high noise x0-mode's eps estimate is
    essentially z/sqrt(1-abar) and comes almost free, while eps-mode has to
    learn the whole function. The sampler needs eps accuracy most at HIGH
    noise, because that is where the trajectory's mode is decided -- so the
    7x gap at abar=0.024 is the one that matters, and it points the same way
    as Lecture 3.2's AMP-Diffusion recipe.

    Zero is the specific failure to rule out here, because the trunk's output
    projection is zero-initialized, so an x0-prediction model that has learned
    nothing emits exactly the clean estimate 0 and DDIM faithfully returns it.
    An untrained model therefore produces norm 0.00 legitimately, which is
    indistinguishable from a broken conversion unless the model is trained
    enough to have an opinion.
    """
    from dgm.project1.pipeline.sampling import sample_diffusion, sample_flow
    n, length, dim = 256, 8, 8
    device = "cuda" if torch.cuda.is_available() else "cpu"
    target = torch.full((n, length, dim), mean, device=device)
    data = TensorDataset((target + spread * torch.randn_like(target)).half(),
                         torch.ones(n, dtype=torch.long, device=device),
                         torch.zeros(n, 1, device=device))
    print(f"Recovery of a point mass at {mean:+.1f} (spread {spread})")

    flow, reward, _, _ = train_flow(data, epochs, batch_size=64, hidden=64,
                                    arch="mlp", conditioning="none", lr=3e-3)
    z = sample_flow(flow, reward, 64, length, dim, steps=50, conditioning="none")
    print(f"  flow      euler      mean {z.mean():+.3f}  sd {z.std():.3f}")

    for predict in ("x0", "v", "eps"):
        out = train_diffusion(data, epochs, batch_size=64, hidden=64,
                              arch="mlp", predict=predict, steps=200,
                              conditioning="none", lr=3e-3,
                              beta_schedule="cosine")
        model, rw = out[0], out[1]
        for solver, steps in (("ddpm", None), ("ddim", 50)):
            z = sample_diffusion(model, rw, out[3], out[4], out[5], out[6],
                                 n=64, length=length, dim=dim, solver=solver,
                                 steps=steps, conditioning="none")
            print(f"  diffusion {predict:<4} {solver:<6} mean {z.mean():+.3f}  "
                  f"sd {z.std():.3f}")


def check_samplers(dataset, arch="transformer"):
    """Round-trip every sampler and scaling, reporting NFE and latent norms.

    The norm is the diagnostic that matters: two `spp*@60` diffusion samples in
    the 2026-10-03 run diverged to latent norm ~1e5 despite --guidance-clip
    2.0, so an arm that leaves the standardized scale is a bug and not a
    strong-guidance result.
    """
    from dgm.project1.pipeline.guidance import GUIDANCE_SCALINGS
    from dgm.project1.pipeline.sampling import sample_diffusion, sample_flow
    from dgm.project1.pipeline.solvers import (DIFFUSION_SOLVERS, FLOW_SOLVERS,
                                               expected_nfe)
    _, length, dim = dataset.tensors[0].shape

    flow, reward = check_flow(dataset, arch, "continuous")
    for solver in FLOW_SOLVERS:
        z, nfe = sample_flow(flow, reward, 4, length, dim, steps=10, c=1.0,
                             solver=solver, return_nfe=True)
        want = expected_nfe(10, solver)
        print(f"  flow   {solver:<9} {'':<10} nfe {nfe:>4} (want {want:>4})  "
              f"norm {z.flatten(1).norm(dim=1).mean():7.2f}")
    for scaling in GUIDANCE_SCALINGS:
        z, nfe = sample_flow(flow, reward, 4, length, dim, steps=10, c=1.0,
                             w=2.0, eta=10.0, clip=2.0, scaling=scaling,
                             return_nfe=True)
        print(f"  flow   guided    {scaling:<10} nfe {nfe:>4}            "
              f"norm {z.flatten(1).norm(dim=1).mean():7.2f}")
    z = sample_flow(flow, reward, 4, length, dim, steps=5, c=1.0, eta=5.0,
                    clip=2.0, corrector_steps=1)
    print(f"  flow   corrector {'':<10}                      "
          f"norm {z.flatten(1).norm(dim=1).mean():7.2f}")

    _, _, out = check_diffusion(dataset, arch, "continuous", "x0")
    model, reward = out[0], out[1]
    alpha_bars, betas, alphas, post_vars = out[3], out[4], out[5], out[6]
    common = dict(n=4, length=length, dim=dim, c=1.0)
    for solver in DIFFUSION_SOLVERS:
        z, nfe = sample_diffusion(model, reward, alpha_bars, betas, alphas,
                                  post_vars, solver=solver, steps=10,
                                  return_nfe=True, **common)
        print(f"  diff   {solver:<9} {'':<10} nfe {nfe:>4}            "
              f"norm {z.flatten(1).norm(dim=1).mean():7.2f}")
    for label, extra in (("churn", {"churn_amount": 0.05}),
                         ("stochastic", {"stochasticity": 1.0}),
                         ("quadratic", {"spacing": "quadratic"}),
                         ("corrector", {"corrector_steps": 1}),
                         ("guided", {"w": 2.0, "eta": 10.0, "clip": 2.0}),
                         ("endpoint", {"eta": 10.0, "clip": 2.0,
                                       "endpoint_guidance": True})):
        z, nfe = sample_diffusion(model, reward, alpha_bars, betas, alphas,
                                  post_vars, solver="ddim", steps=10,
                                  return_nfe=True, **common, **extra)
        print(f"  diff   ddim      {label:<10} nfe {nfe:>4}            "
              f"norm {z.flatten(1).norm(dim=1).mean():7.2f}")


# ══════════════════════════════════════════════════════════════════════════════
# Parse args
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="all",
                        choices=("all", "schedules", "train", "sample", "recovery"))
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(0)
    dataset = fake_dataset(*SHAPE)
    if args.stage in ("all", "schedules"):
        check_schedules()
    if args.stage in ("all", "train"):
        print("Training heads")
        for arch in ("mlp", "transformer"):
            for conditioning in ("continuous", "binary", "none"):
                check_flow(dataset, arch, conditioning)
            for predict in ("x0", "v", "eps"):
                check_diffusion(dataset, arch, "continuous", predict,
                                weighting="min-snr", beta="cosine")
        for schedule in ("quadratic", "cosine", "hermite", "smoothstep"):
            check_flow(dataset, "transformer", "continuous", schedule)
    if args.stage in ("all", "sample"):
        print("Samplers")
        check_samplers(dataset)
    if args.stage in ("all", "recovery"):
        check_recovery()
    print("OK")


if __name__ == "__main__":
    main()
