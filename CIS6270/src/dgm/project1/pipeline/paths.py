"""Probability paths and the schedules that walk them.

The interpolants and the DDPM schedule live together because both answer the
same question -- where the state is between prior and data -- and the endpoint
estimators invert them.
"""
import torch

from .config import DEVICE
# Re-exported: the diffusion forward process moved to schedules.py when the
# cosine and sigmoid variants were added, but every call site imports it from
# here and saved run configs name it here.
from .schedules import (BETA_SCHEDULES, make_ddpm_schedule,  # noqa: F401
                        sample_timesteps)


# ══════════════════════════════════════════════════════════════════════════════
# Probability paths: a GEOMETRY walked on a SCHEDULE
# ══════════════════════════════════════════════════════════════════════════════
#
# These were one flag with three values -- linear, quadratic, trig -- and that
# conflated two independent choices. `quadratic` is not a third path: its
# coefficients sum to one exactly as `linear`'s do, so it traces the SAME
# straight segment, and quadratic(t=0.5) is bit-identical to linear(s=0.25).
# What it changes is the speed along that segment. Only `trig` has
# a^2 + b^2 = 1 and is a genuinely different, curved geometry.
#
# So the real design space is a product:
#
#   geometry   where the path goes      segment | arc | data-arc
#   schedule   how fast s(t) moves      linear | quadratic | cosine
#
# and the three legacy names are three of its nine cells:
#
#   linear = (segment, linear)   quadratic = (segment, quadratic)
#   trig   = (arc, linear)       (arc, quadratic) was unreachable
#
# Keeping them separate matters because they are measured differently. A
# reparameterization of a straight path CANNOT change the final marginal in the
# exact-ODE limit -- it only reweights which noise levels training sees, and
# where a fixed-step integrator places its steps. A geometry change moves the
# intermediate distributions p_t themselves. Fused into one flag, a difference
# between `linear` and `quadratic` cannot be attributed to either.

GEOMETRIES = ("segment", "arc", "data-arc")
SCHEDULES = ("linear", "quadratic", "cosine", "hermite", "smoothstep")

# Endpoint tangent for the "hermite" schedule, as a fraction of the chord.
# 0.5 halves the endpoint speed rather than stalling it, which keeps the
# endpoint estimator away from the ds/dt clamp that `smoothstep` hits.
HERMITE_TENSION = 0.5

# The names this module accepted before geometry and schedule were separated.
LEGACY_INTERPOLANTS = {
    "linear":    ("segment", "linear"),
    "quadratic": ("segment", "quadratic"),
    "trig":      ("arc", "linear"),
}
INTERPOLANTS = tuple(LEGACY_INTERPOLANTS)


def resolve_path(interpolant=None, geometry="segment", schedule="linear"):
    """(geometry, schedule) from either spelling.

    A legacy `interpolant` name wins when given, so every existing call site and
    every saved run config keeps meaning exactly what it meant.
    """
    if interpolant:
        if interpolant not in LEGACY_INTERPOLANTS:
            raise ValueError(f"unknown interpolant '{interpolant}'")
        return LEGACY_INTERPOLANTS[interpolant]
    if geometry not in GEOMETRIES:
        raise ValueError(f"unknown geometry '{geometry}', expected {GEOMETRIES}")
    if schedule not in SCHEDULES:
        raise ValueError(f"unknown schedule '{schedule}', expected {SCHEDULES}")
    return geometry, schedule


def data_scale(z1):
    """Global standard deviation of the data, for the 'data-arc' geometry."""
    return float(z1.std())


def _schedule(t, kind):
    """Path parameter s(t) in [0,1] and its derivative ds/dt."""
    if kind == "linear":
        return t, torch.ones_like(t)
    if kind == "quadratic":
        # Leaves the prior slowly and accelerates: a quarter of the way through
        # in time, only 6.25% of the path is covered.
        return t ** 2, 2 * t
    if kind == "cosine":
        # The mirror image: fast at first, decelerating onto the data.
        half_pi = torch.pi / 2
        return torch.sin(half_pi * t), half_pi * torch.cos(half_pi * t)
    if kind in ("hermite", "smoothstep"):
        return _hermite(t, HERMITE_TENSION if kind == "hermite" else 0.0)
    raise ValueError(f"unknown schedule '{kind}'")


def _hermite(t, tension):
    """Cubic-Hermite reparameterization of the path parameter.

    Lecture 2.2 presents CHIME's cubic-Hermite interpolant with the note
    "Worked better than base methods!", so it belongs in the Axis A grid. The
    two-point Hermite form with both endpoint tangents set to `tension` times
    the chord reduces, for any geometry whose coefficients sum to one, to a
    reparameterization of s:

        s(t) = tension * t + (1 - tension) * (3t^2 - 2t^3)

    which is worth seeing explicitly, because it means CHIME on a straight
    segment is a SCHEDULE change and not a new geometry -- the same conflation
    this module's header untangles for `quadratic`. It cannot move the terminal
    marginal in the exact-ODE limit; what it changes is where a fixed-step
    integrator spends its steps and which noise levels training samples.

      tension = 1   recovers the linear schedule exactly.
      tension = 0   the smoothstep, whose velocity vanishes at both ends, so
                    steps bunch in the middle where the velocity field is
                    hardest to learn (Lecture 2.2's crossing-path argument).

    ds/dt = tension at both endpoints, so tension also sets how singular the
    endpoint estimator gets; see endpoint_from_velocity's clamp.
    """
    smooth, d_smooth = 3 * t ** 2 - 2 * t ** 3, 6 * t - 6 * t ** 2
    s = tension * t + (1 - tension) * smooth
    ds = tension + (1 - tension) * d_smooth
    return s, ds


def _geometry(s, kind, scale=1.0):
    """Coefficients a(s), b(s) and their derivatives a'(s), b'(s).

    Every geometry satisfies a(0)=b(1)=1 and a(1)=b(0)=0, so the endpoints are
    the prior and the data whatever the shape in between.

      segment   a=1-s, b=s. The straight line; constant speed in s.
      arc       a=cos(pi s/2), b=sin(pi s/2). a^2+b^2=1, so for independent
                centered endpoints of unit variance the marginal variance is
                preserved along the whole path. The straight segment instead
                dips to 0.5 at the midpoint, where samples lie off both the
                prior and the data.
      data-arc  the segment rescaled so the marginal second moment moves
                linearly from the prior's to the data's:

                    a^2 + scale^2 b^2 = (1-s) + s scale^2

                `arc` preserves the variance of a distribution with unit
                covariance; real data rarely has one. Measured here, peptide ESM
                latents have global variance 1.000 -- so arc's premise holds and
                data-arc reduces to it -- while MNIST as trained has mean -0.741
                and variance 0.377, where arc preserves a variance the data does
                not have. This is the correction for that case, and comparing
                the two is the ablation that shows whether it was needed.
    """
    if kind == "segment":
        return 1 - s, s, -torch.ones_like(s), torch.ones_like(s)
    if kind == "arc":
        half_pi = torch.pi / 2
        a, b = torch.cos(half_pi * s), torch.sin(half_pi * s)
        return a, b, -half_pi * b, half_pi * a
    if kind == "data-arc":
        v2 = float(scale) ** 2
        raw_a, raw_b = 1 - s, s
        target = (1 - s) + s * v2                     # variance we want
        raw = raw_a ** 2 + v2 * raw_b ** 2            # variance of the segment
        k = (target / raw.clamp_min(1e-12)).sqrt()
        # d/ds log k = (target'/target - raw'/raw) / 2
        d_target = torch.full_like(s, v2 - 1.0)
        d_raw = -2 * raw_a + 2 * v2 * raw_b
        dk = 0.5 * k * (d_target / target.clamp_min(1e-12)
                        - d_raw / raw.clamp_min(1e-12))
        return (k * raw_a, k * raw_b,
                dk * raw_a - k, dk * raw_b + k)
    raise ValueError(f"unknown geometry '{kind}'")


def path_coefficients(t, geometry="segment", schedule="linear", scale=1.0):
    """a(t), b(t) and their time derivatives, for one batch of times.

    The chain rule is applied here and nowhere else: a geometry is written in
    terms of the path parameter s and a schedule only says how s moves, so
    neither has to know about the other.
    """
    s, ds = _schedule(t, schedule)
    a, b, da_ds, db_ds = _geometry(s, geometry, scale)
    return a, b, da_ds * ds, db_ds * ds


def interpolate(z0, z1, t, interpolant=None, geometry="segment",
                schedule="linear", scale=1.0):
    """Intermediate state X_t and its conditional velocity U_t.

    X_t = a(t) X0 + b(t) X1 and U_t = a'(t) X0 + b'(t) X1, which is the velocity
    the flow-matching loss regresses onto. Lecture 2.2 works the straight case
    and notes that changing the path "can help when velocities are difficult to
    learn or sampling requires many integration steps, because each choice
    changes the intermediate distributions and velocity targets."

    Accepts a legacy `interpolant` name or an explicit (geometry, schedule).
    """
    geometry, schedule = resolve_path(interpolant, geometry, schedule)
    shape = (-1,) + (1,) * (z1.dim() - 1)
    a, b, da, db = (c.view(shape) for c in
                    path_coefficients(t, geometry, schedule, scale))
    return a * z0 + b * z1, da * z0 + db * z1


def endpoint_from_velocity(z, t, v, interpolant=None, geometry="segment",
                           schedule="linear", scale=1.0):
    """Estimate the clean endpoint X1 from the state and predicted velocity.

    Eliminating X0 between X_t = a X_0 + b X_1 and U_t = a' X_0 + b' X_1 gives

        X1 = (a'(t) X_t - a(t) U_t) / (a'(t) b(t) - a(t) b'(t))

    which for the straight path with linear time collapses to the familiar
    X1 = X_t + (1-t) U_t. One expression now serves every geometry and schedule,
    where each used to need its own closed form.

    Written in terms of s rather than t on purpose. The ds/dt factors cancel in
    the ratio, and doing that cancellation by hand is what keeps the expression
    well conditioned: the denominator a'(s)b - a b'(s) is a constant -1 for the
    segment and -pi/2 for the arc, never zero, so the ONLY singular quantity is
    ds/dt. That is a property of the schedule -- it stalls at t=0 under the
    quadratic schedule and at t=1 under the cosine one -- and clamping it there
    leaves the geometry's algebra exact. Clamping the combined denominator
    instead silently drops the X_t term and gives the wrong sign at t=0.
    """
    geometry, schedule = resolve_path(interpolant, geometry, schedule)
    shape = (-1,) + (1,) * (z.dim() - 1)
    s, ds = (c.view(shape) for c in _schedule(t, schedule))
    a, b, da, db = (c.view(shape) for c in _geometry(s, geometry, scale))
    # v is dX/dt; the geometry's algebra wants dX/ds.
    v_s = v / ds.abs().clamp_min(1e-3)
    return (da * z - a * v_s) / (da * b - a * db)


def endpoint_from_noise(z, eps, alpha_bar):
    """Clean-sample estimate from a DDPM state and its predicted noise.

    Inverting z_k = sqrt(abar) x0 + sqrt(1-abar) eps for x0.
    """
    return (z - (1 - alpha_bar).sqrt() * eps) / alpha_bar.sqrt().clamp_min(1e-4)


def time_grid(steps, start, kind="linear", device=None):
    """Integration times and step sizes, as (times[steps+1], deltas[steps]).

    Distinct from the training `schedule`, and that separation is the point. The
    training schedule decides which noise levels the loss sees and how the
    velocity target is scaled; this one decides only where a fixed-step
    integrator puts its steps. Fused, a difference between two schedules cannot
    be attributed to either, because one flag moved both.

    'linear' reproduces what the sampler did before this existed, bit for bit:
    the times are accumulated in double precision and rounded once, as
    `start + step * dt` was, and the step size is the constant dt rather than a
    difference of rounded times. The defaults have to reproduce earlier runs
    exactly, for the same reason --sample-seed defaults to the value that used
    to be hardcoded.
    """
    dt = (1.0 - start) / steps
    if kind == "linear":
        times = start + torch.arange(steps + 1, dtype=torch.float64) * dt
        deltas = torch.full((steps,), dt, dtype=torch.float64)
    else:
        u = torch.linspace(0.0, 1.0, steps + 1, dtype=torch.float64)
        times = start + (1.0 - start) * _schedule(u, kind)[0]
        deltas = times[1:] - times[:-1]
    return (times.to(torch.float32).to(device),
            deltas.to(torch.float32).to(device))


class PathSpec:
    """One run's probability path, resolved once and passed around whole.

    Exists so training and sampling cannot disagree about the path: a field
    trained on one geometry and integrated along another is not a comparison of
    anything. `scale` is only read by the 'data-arc' geometry.
    """

    def __init__(self, geometry="segment", schedule="linear", scale=1.0,
                 sample_schedule=None):
        self.geometry, self.schedule = resolve_path(None, geometry, schedule)
        self.scale = float(scale)
        self.sample_schedule = sample_schedule or "linear"
        if self.sample_schedule not in SCHEDULES:
            raise ValueError(f"unknown sample schedule '{self.sample_schedule}'")

    @classmethod
    def from_args(cls, args, z1=None):
        """Resolve from parsed CLI args, measuring `scale` from the data.

        A legacy --interpolant wins over --path-geometry/--time-schedule, so
        existing commands and saved run configs keep their exact meaning.
        """
        interpolant = getattr(args, "interpolant", None)
        geometry, schedule = resolve_path(
            interpolant if interpolant in LEGACY_INTERPOLANTS else None,
            getattr(args, "path_geometry", "segment"),
            getattr(args, "time_schedule", "linear"))
        scale = getattr(args, "path_scale", 0.0) or 0.0
        if geometry == "data-arc" and scale <= 0.0:
            if z1 is None:
                raise ValueError("'data-arc' needs --path-scale or the training data")
            scale = data_scale(z1)
            print(f"  [path]      data-arc scale measured from the data: {scale:.4f}")
        return cls(geometry, schedule, scale or 1.0,
                   getattr(args, "sample_schedule", None))

    @property
    def kwargs(self):
        """What interpolate() and endpoint_from_velocity() take."""
        return {"geometry": self.geometry, "schedule": self.schedule,
                "scale": self.scale}

    def describe(self):
        legacy = [k for k, v in LEGACY_INTERPOLANTS.items()
                  if v == (self.geometry, self.schedule)]
        name = f"{self.geometry}/{self.schedule}"
        if legacy:
            name += f" (= --interpolant {legacy[0]})"
        if self.geometry == "data-arc":
            name += f"  scale {self.scale:.3f}"
        if self.sample_schedule != "linear":
            name += f"   sampling grid: {self.sample_schedule}"
        return name


def add_arguments(parser):
    """The probability-path flags, shared by every runner.

    Defined beside the paths themselves so the flag set and the mechanism cannot
    drift, and so each runner's own parser stays short.
    """
    parser.add_argument("--interpolant", default=None, choices=INTERPOLANTS,
                        help="LEGACY shorthand, kept so existing commands and saved "
                             "run configs keep their exact meaning: linear = "
                             "segment/linear, quadratic = segment/quadratic, trig = "
                             "arc/linear. It wins over the two flags below. Prefer "
                             "those: these three names hid the fact that geometry "
                             "and time are independent choices, and that "
                             "'quadratic' is not a third path at all -- its "
                             "coefficients sum to one exactly as 'linear's do, so it "
                             "traces the SAME straight segment, and "
                             "quadratic(t=0.5) is bit-identical to linear(s=0.25).")
    parser.add_argument("--path-geometry", default="segment", choices=GEOMETRIES,
                        help="WHERE the path goes (default: segment, the straight "
                             "line). 'arc' has a^2+b^2=1, so for independent "
                             "centered unit-variance endpoints the marginal "
                             "variance is preserved throughout, where the segment "
                             "dips to 0.5 at the midpoint. 'data-arc' generalizes "
                             "that to data whose variance is not 1: it holds the "
                             "second moment on a straight line between the prior's "
                             "and the data's. Peptide ESM latents measure global "
                             "variance 1.000, so arc's premise holds there and "
                             "data-arc reduces to it; MNIST as trained has mean "
                             "-0.741 and variance 0.377, where arc preserves a "
                             "variance the data does not have.")
    parser.add_argument("--time-schedule", default="linear", choices=SCHEDULES,
                        help="HOW FAST the path parameter s(t) moves during "
                             "TRAINING (default: linear, s=t). 'quadratic' (s=t^2) "
                             "leaves the prior slowly and accelerates; 'cosine' "
                             "(s=sin(pi t/2)) is the mirror image. On a straight "
                             "geometry this cannot change the final marginal in the "
                             "exact-ODE limit -- it reweights which noise levels the "
                             "loss sees and rescales the velocity target, nothing "
                             "more. Pair it with --sample-schedule to separate that "
                             "from step placement.")
    parser.add_argument("--sample-schedule", default=None, choices=SCHEDULES,
                        help="Where the Euler integrator puts its steps at SAMPLING "
                             "time (default: linear = the uniform grid used before "
                             "this flag existed, reproduced with the same "
                             "arithmetic). Kept separate from --time-schedule "
                             "because the two are measured differently: a training "
                             "schedule effect shows at any step count, a step-"
                             "placement effect only at low ones. Fused into one "
                             "flag, neither can be attributed.")
    parser.add_argument("--path-scale", type=float, default=0.0, metavar="S",
                        help="Data standard deviation for --path-geometry data-arc "
                             "(default: 0 = measure it from the training set).")
