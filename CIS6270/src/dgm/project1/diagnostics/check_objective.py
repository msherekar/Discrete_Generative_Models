#!/usr/bin/env python3
"""Verify the Objective / reward_gradient contract.

The end-to-end smoke run exercises the plumbing but cannot show that guidance
does the right thing: on a short run every arm decodes to the same argmax
sequence, so identical output there proves nothing either way. These checks
test the gradient itself, which is where the objective semantics live.

  python diagnostics/check_objective.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

import run_experiment as R

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILURES.append(name)


def main() -> int:
    torch.manual_seed(0)
    B, L, D = 6, 8, 16
    reward = R.RewardModel(L, D, hidden=32, arch="mlp", n_props=3).to(R.DEVICE)
    z = torch.randn(B, L, D, device=R.DEVICE)
    t = torch.rand(B, device=R.DEVICE)
    with torch.no_grad():
        raw = reward(z, t)

    constrained = R.Objective(3, constraint_index=2)
    # Weight r2 only, so a change in that term is visible in isolation.
    lam_r2 = R.weight_vector((0.0, 1.0), constrained)
    lam_r1 = R.weight_vector((1.0, 0.0), constrained)

    print("sense")
    g_max = R.reward_gradient(reward, z, t, lam_r2,
                              objective=R.Objective(3, senses=(1., 1.), constraint_index=2))
    g_min = R.reward_gradient(reward, z, t, lam_r2,
                              objective=R.Objective(3, senses=(1., -1.), constraint_index=2))
    cos = float(F.cosine_similarity(g_max.flatten(), g_min.flatten(), dim=0))
    check("minimizing a property exactly reverses its gradient",
          abs(cos + 1.0) < 1e-5, f"cosine {cos:+.6f}")

    print("setpoint")
    target = float(raw[0, 0])
    at = R.reward_gradient(reward, z[:1], t[:1], lam_r1,
                           objective=R.Objective(3, setpoint=target, constraint_index=2))
    away = R.reward_gradient(reward, z[:1], t[:1], lam_r1,
                             objective=R.Objective(3, setpoint=target + 5.0,
                                                   constraint_index=2))
    check("gradient vanishes where the prediction already equals the target",
          float(at.norm()) < 1e-6, f"|g| {float(at.norm()):.2e}")
    check("gradient is non-trivial away from the target",
          float(away.norm()) > 1e-3, f"|g| {float(away.norm()):.4f}")

    print("setpoint under normalization")
    # The regression this file originally missed. Normalizing a composed
    # setpoint term divides out its error factor and leaves only the sign, so
    # every target above the current prediction collapses onto plain maximize
    # and a setpoint sweep returns one answer repeated. Caught in a calibration
    # run where spp50 and spp99 scored identically to the maximize arm.
    with torch.no_grad():
        here = float(reward(z, t)[:, 0].mean())
    maximize = R.Objective(3, constraint_index=2)
    g_max = R.reward_gradient(reward, z, t, lam_r1, normalize=True,
                              objective=maximize)
    sizes = {}
    for error in (0.1, 0.5, 3.0):
        o = R.Objective(3, setpoint=here + error, constraint_index=2,
                        setpoint_saturation=4.0)
        g = R.reward_gradient(reward, z, t, lam_r1, normalize=True, objective=o)
        sizes[error] = float(g.flatten(1).norm(dim=1).mean())
    check("normalized setpoint magnitude still grows with the error",
          sizes[0.1] < sizes[0.5] < sizes[3.0],
          "  ".join(f"err {e}: |g| {v:.3f}" for e, v in sizes.items()))

    on_target = R.Objective(3, setpoint=here, constraint_index=2)
    g_on = R.reward_gradient(reward, z, t, lam_r1, normalize=True,
                             objective=on_target)
    check("normalized setpoint gradient nearly vanishes on target",
          float(g_on.flatten(1).norm(dim=1).mean()) < 0.1,
          f"|g| {float(g_on.flatten(1).norm(dim=1).mean()):.4f}")

    below = R.Objective(3, setpoint=here - 3.0, constraint_index=2)
    g_below = R.reward_gradient(reward, z, t, lam_r1, normalize=True, objective=below)
    cos = float(F.cosine_similarity(g_below.flatten(), g_max.flatten(), dim=0))
    check("a target below the prediction reverses against maximize",
          cos < -0.99, f"cosine {cos:+.4f}")

    check("saturation bounds the normalized step",
          float(g_max.flatten(1).norm(dim=1).max()) <= 1.0 + 1e-5
          and sizes[3.0] <= 4.0 + 1e-5,
          f"maximize {float(g_max.flatten(1).norm(dim=1).max()):.3f}, "
          f"far setpoint {sizes[3.0]:.3f}")

    print("constraint")
    off = R.Objective(3, senses=(1., 1.), constraint_index=2)
    satisfied = R.Objective(3, senses=(1., 1.), constraint_index=2,
                            constraint_threshold=float(raw[:, 2].max()) + 10.0, rho=1.0)
    violated = R.Objective(3, senses=(1., 1.), constraint_index=2,
                           constraint_threshold=float(raw[:, 2].min()) - 10.0, rho=1.0)
    g_off = R.reward_gradient(reward, z, t, lam_r2, objective=off)
    g_sat = R.reward_gradient(reward, z, t, lam_r2, objective=satisfied)
    g_vio = R.reward_gradient(reward, z, t, lam_r2, objective=violated)
    check("a satisfied constraint contributes nothing",
          float((g_sat - g_off).abs().max()) == 0.0)
    check("a violated constraint changes the gradient",
          float((g_vio - g_off).abs().max()) > 1e-3,
          f"max delta {float((g_vio - g_off).abs().max()):.4f}")

    zz = z.clone().requires_grad_(True)
    g_property = torch.autograd.grad(reward(zz, t)[:, 2].sum(), zz)[0]
    delta = g_vio - g_off
    cos = float(F.cosine_similarity(delta.flatten(), (-g_property).flatten(), dim=0))
    check("the penalty descends the constrained property",
          cos > 0.99, f"cosine {cos:+.6f}")

    print("weights")
    check("the constrained column leaves the objective simplex",
          constrained.n_obj == 2 and tuple(constrained.terms(raw).shape) == (B, 2),
          f"n_obj {constrained.n_obj}, terms {tuple(constrained.terms(raw).shape)}")
    check("a short lambda list is zero-padded",
          R.weight_vector((1.0,), constrained).tolist() == [1.0, 0.0])
    check("weights are normalized onto the simplex",
          R.weight_vector((3.0, 1.0), constrained).tolist() == [0.75, 0.25])
    for bad, why in [((0.5, 0.3, 0.2), "too many lambdas"),
                     ((0.0, 0.0), "all-zero lambdas"),
                     ((-1.0, 2.0), "negative lambdas")]:
        try:
            R.weight_vector(bad, constrained)
            check(f"rejects {why}", False)
        except ValueError:
            check(f"rejects {why}", True)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed: {', '.join(FAILURES)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
