"""What the two heads regress onto: parameterizations, weighting, censoring.

Axis C of the innovation study. Kept apart from training.py so a change to the
objective is a change to one file, and so the diffusion parameterization and
its loss weight cannot drift out of step.
"""
import math

import torch
import torch.nn.functional as F

# Per-noise-level weightings for the diffusion loss.
WEIGHTINGS = ("none", "min-snr", "snr", "sigma")

# Diffusion output parameterizations.
PREDICTIONS = ("eps", "x0", "v")

# Default gamma for Min-SNR, the value Hang et al. report as best across
# resolutions and schedules.
MIN_SNR_GAMMA = 5.0

# ══════════════════════════════════════════════════════════════════════════════
# Diffusion target and weighting
# ══════════════════════════════════════════════════════════════════════════════

def velocity_target(x0, eps, alpha_bar):
    """The v-prediction target, v = sqrt(abar) eps - sqrt(1-abar) x0.

    Salimans & Ho, "Progressive Distillation for Fast Sampling of Diffusion
    Models" (ICLR 2022), appendix D. v interpolates between predicting noise
    and predicting signal as the noise level moves, so neither end of the
    schedule hands the network a free answer: at abar -> 0 it is -x0 and at
    abar -> 1 it is +eps. That is the structural fix for the pathology
    documented in nets.py, where "eps" collapses to the identity at high noise.
    """
    return alpha_bar.sqrt() * eps - (1 - alpha_bar).sqrt() * x0


def x0_from_velocity(z, v, alpha_bar):
    """Invert v-prediction for the clean sample, so samplers stay uniform."""
    return alpha_bar.sqrt() * z - (1 - alpha_bar).sqrt() * v


def eps_from_velocity(z, v, alpha_bar):
    """Invert v-prediction for the noise, which is what the DDPM update wants."""
    return (1 - alpha_bar).sqrt() * z + alpha_bar.sqrt() * v


def loss_weight(alpha_bar, weighting="none", gamma=MIN_SNR_GAMMA):
    """Per-sample weight on the denoising loss, shaped to broadcast.

    The signal-to-noise ratio of the forward process at step k is
    SNR = abar / (1 - abar), and the plain unweighted loss used before this
    existed is implicitly weighted by it. The consequence is that easy, nearly
    clean noise levels dominate the gradient while the hard, high-noise levels
    that actually determine sample quality are drowned out.

      none     the historical behaviour, weight 1 everywhere.
      min-snr  min(SNR, gamma) / SNR, which caps the contribution of the
               low-noise end without touching the high-noise end. Hang et al.,
               "Efficient Diffusion Training via Min-SNR Weighting Strategy"
               (ICCV 2023); they report gamma=5 and a 3.4x training speedup.
      snr      the uncapped SNR weight, kept as the control that shows the cap
               is what matters rather than the reweighting as such.
      sigma    weight by 1-abar, the score-matching weight w(t)=sigma^2 that
               Lecture 3.2 substitutes to turn score matching into plain
               noise prediction.
    """
    if weighting == "none":
        return torch.ones_like(alpha_bar)
    snr = (alpha_bar / (1 - alpha_bar).clamp_min(1e-8)).clamp(1e-8, 1e8)
    if weighting == "min-snr":
        return snr.clamp(max=gamma) / snr
    if weighting == "snr":
        return snr
    if weighting == "sigma":
        return 1 - alpha_bar
    raise ValueError(f"unknown weighting '{weighting}', expected {WEIGHTINGS}")


def generative_loss(prediction, x0, eps, z, alpha_bar, predict="x0",
                    weighting="none", gamma=MIN_SNR_GAMMA):
    """Weighted denoising loss for whichever parameterization is in use.

    The target changes with `predict`, the weight with `weighting`, and the two
    are independent: v-prediction with no weighting and eps-prediction with
    Min-SNR are both reachable, which is what makes the Axis C ablation a grid
    rather than a ladder.
    """
    if predict == "x0":
        target = x0
    elif predict == "v":
        target = velocity_target(x0, eps, alpha_bar)
    elif predict == "eps":
        target = eps
    else:
        raise ValueError(f"unknown predict '{predict}', expected {PREDICTIONS}")
    per_sample = (prediction - target).pow(2).flatten(1).mean(1)
    weight = loss_weight(alpha_bar, weighting, gamma).flatten()
    return (weight * per_sample).mean()


# ══════════════════════════════════════════════════════════════════════════════
# Reward head
# ══════════════════════════════════════════════════════════════════════════════

# Standard normal log-CDF, for the censored term. torch.special.log_ndtr is
# the numerically stable form; the manual fallback keeps older torch working.
def _log_ndtr(x):
    if hasattr(torch.special, "log_ndtr"):
        return torch.special.log_ndtr(x)
    return torch.log(0.5 * torch.erfc(-x / math.sqrt(2.0)).clamp_min(1e-30))


def reward_loss(prediction, target, censor_floor=None, sigma=1.0):
    """Squared error on the reward head, optionally censored from below.

    avGFP's brightness assay has a hard floor: 11.8% of the 51,714 measured
    variants sit at exactly -2.418182 and 39.3% below -2. That value is not a
    measurement of how dim a protein is, it is the sort-seq readout saying it
    saw nothing. Plain MSE treats it as a precise target and spends the head's
    capacity memorizing a point mass.

    The censored (Tobit) likelihood says what the data actually says -- this
    variant is AT MOST this bright -- by scoring floored rows with the Gaussian
    log-CDF of the gap instead of the squared error:

        uncensored:  (r_hat - r)^2
        at floor:    -log Phi((floor - r_hat) / sigma)

    so a prediction comfortably below the floor is free, and one above it pays.
    Tobin, Econometrica 26:24 (1958).

    `censor_floor` is in the same standardized units as `target`, and None
    restores plain MSE for the ablation. Only column 0 (brightness) is
    censored; the Rosetta attribute columns are simulation outputs with no
    detection limit.
    """
    if censor_floor is None:
        return F.mse_loss(prediction, target)
    floored = target[:, 0] <= censor_floor + 1e-6
    squared = (prediction - target).pow(2)
    # Every column keeps its squared error, then column 0 is overwritten for
    # the floored rows only.
    column = squared[:, 0].clone()
    gap = (censor_floor - prediction[:, 0]) / sigma
    column = torch.where(floored, -_log_ndtr(gap), column)
    return torch.cat([column[:, None], squared[:, 1:]], dim=1).mean()


def optional_float(text):
    """A float, or None for "none"/"off"/"".

    Needed because the Condor params files are positional columns: an ablation
    row has to be able to say "this feature is OFF" in the same slot where
    another row says "-2.418182". Spelling that as 0 would be wrong rather than
    merely ugly -- --censor-floor 0 means "censor everything at or below zero",
    which on standardized avGFP scores is most of the dataset.
    """
    if text is None or str(text).strip().lower() in ("", "none", "off"):
        return None
    return float(text)


def add_arguments(parser):
    """Axis C flags: what the heads regress onto and how it is weighted."""
    group = parser.add_argument_group("objective (Axis C)")
    group.add_argument("--loss-weighting", default="none", choices=WEIGHTINGS,
                       help="Per-noise-level weight on the denoising loss "
                            "(default: none, the historical behaviour). "
                            "'min-snr' caps the low-noise end's contribution "
                            "at gamma; Hang et al. (ICCV 2023) report a 3.4x "
                            "training speedup. 'snr' is the uncapped control "
                            "that shows the cap is what matters.")
    group.add_argument("--min-snr-gamma", type=float, default=MIN_SNR_GAMMA,
                       metavar="G",
                       help=f"Cap for --loss-weighting min-snr (default: "
                            f"{MIN_SNR_GAMMA:g}, the value Hang et al. report "
                            f"as best across resolutions and schedules).")
    group.add_argument("--censor-floor", type=optional_float, default=None,
                       metavar="V",
                       help="Treat reward targets at or below V (in the CSV's "
                            "own units, e.g. -2.418182 for avGFP) as censored "
                            "rather than exact, scoring them with the Gaussian "
                            "log-CDF instead of squared error. 11.8%% of avGFP "
                            "variants sit at exactly that value and it is the "
                            "sort-seq readout's detection floor, not a "
                            "measurement; plain MSE spends the head's capacity "
                            "memorizing a point mass. Tobin (1958).")
    return parser


def censor_floor_from_stats(stats, raw_floor, index=0):
    """Convert a raw assay floor into the standardized units the head sees.

    load_data standardizes every property column, so a floor quoted in the
    CSV's own units (-2.418182 for avGFP) has to be mapped through the same
    transform before reward_loss can compare it to a prediction.
    """
    mean = float(stats["r_mean"][index])
    std = float(stats["r_std"][index])
    return (float(raw_floor) - mean) / std


if __name__ == "__main__":
    import torch
    B, L, D = 4, 8, 16
    z0 = torch.randn(B, L, D)
    eps = torch.randn(B, L, D)
    abar = torch.rand(B, 1, 1) * 0.9 + 0.05
    zk = abar.sqrt() * z0 + (1 - abar).sqrt() * eps

    # v-prediction round-trip.
    v = velocity_target(z0, eps, abar)
    eps_back = eps_from_velocity(zk, v, abar)
    x0_back = x0_from_velocity(zk, v, abar)
    assert (eps_back - eps).abs().max() < 1e-5, "v->eps roundtrip failed"
    assert (x0_back - z0).abs().max() < 1e-5, "v->x0 roundtrip failed"
    print("  v-prediction round-trips: OK")

    # generative_loss returns finite scalar for all parameterizations.
    for pred in PREDICTIONS:
        target = eps if pred == "eps" else (z0 if pred == "x0" else v)
        loss = generative_loss(target, z0, eps, zk, abar, pred)
        assert loss.isfinite(), f"generative_loss({pred}) not finite"
        print(f"  generative_loss({pred}): {loss.item():.4f}")

    # All weightings return finite, positive weights.
    for w in WEIGHTINGS:
        wt = loss_weight(abar, w)
        assert (wt > 0).all() and wt.isfinite().all(), f"weight {w} not positive/finite"
        print(f"  loss_weight({w}): mean={wt.mean().item():.4f}")

    print("losses.py OK")
