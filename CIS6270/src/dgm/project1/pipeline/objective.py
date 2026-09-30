"""What guidance climbs: scalarized objectives, senses, setpoints, constraints.

Kept apart from the gradient machinery in guidance.py so the definition of the
objective can be read, and changed, without touching how it is differentiated.
"""
import torch

from .config import DEVICE

class Objective:
    """Turns raw reward-head outputs into the scalar guidance follows.

    The reward head predicts standardized property values. What guidance should
    climb is not always the raw prediction, and Lecture 3.4 separates the two
    cases this implements.

    Objectives are scalarized on a simplex: R_lambda = sum_m lambda_m * term_m,
    each term written so larger is better.

      setpoint    term_0 = -(r1 - y*)^2 rather than +r1. Steering brightness
                  "up" is a weak claim on GFP because nearly any substitution
                  makes it dimmer, so a dim output demonstrates nothing. Hitting
                  a requested value tests calibration: sweep y* and report the
                  slope of achieved against requested.
      sense       -1 for a property to be minimized. The slides: "For a property
                  that should be minimized, such as toxicity, we reverse its
                  sign so that larger normalized values consistently represent
                  more desirable outcomes."

    The constraint is deliberately NOT on that simplex. Lecture 3.4 keeps it
    separate -- "Scalarization tells the model which valid outcomes are
    preferred. A constraint specifies which outcomes are allowed" -- as a
    one-sided squared penalty subtracted from the scalarized reward:

        R = sum_m lambda_m * term_m  -  rho * max(0, r_c - threshold)^2

    so rho is a severity, not a tradeoff weight, and raising it cannot silently
    eat the objective budget.

    All thresholds and setpoints arrive in STANDARDIZED units. Callers convert
    from raw with from_raw(), because the reward head only ever sees the
    standardized targets load_data() built.
    """

    def __init__(self, n_props, setpoint=None, senses=None,
                 constraint_index=None, constraint_threshold=None, rho=0.0,
                 setpoint_saturation=4.0):
        self.n_props = n_props
        self.setpoint = setpoint
        self.setpoint_saturation = float(setpoint_saturation)
        # Properties past the constrained one are not objectives.
        self.n_obj = n_props if constraint_index is None else constraint_index
        self.senses = tuple(senses) if senses is not None else (1.0,) * self.n_obj
        if len(self.senses) != self.n_obj:
            raise ValueError(f"senses must have {self.n_obj} entries, got {len(self.senses)}")
        self.constraint_index = constraint_index
        self.constraint_threshold = constraint_threshold
        self.rho = float(rho)

    @property
    def constrained(self) -> bool:
        return (self.constraint_index is not None
                and self.constraint_threshold is not None and self.rho > 0.0)

    @staticmethod
    def from_raw(value, index, stats):
        """Convert a raw property value into the standardized space."""
        mean = float(stats["r_mean"][index])
        std = float(stats["r_std"][index])
        return (float(value) - mean) / std

    def term_scale(self, rewards, j):
        """Scalar multiplier for objective j, [B], bounded to [-1, 1].

        Exists because `normalize` and `setpoint` otherwise destroy each other.
        Normalization rescales a term's gradient to unit norm so lambda is not
        confounded with whatever scale a reward head learned. But the setpoint
        gradient is -2(r1 - y*) * d r1/dz, so dividing out its norm divides out
        the error factor too and leaves only its sign: every target above the
        current prediction collapses to plain maximize, every target below it to
        plain minimize, and a setpoint sweep returns one answer repeated.

        So normalization is applied to the RAW property gradient, and the
        objective's own factor is reapplied here. Clamping keeps the guidance
        term bounded by sum(lambda) = 1 (plus rho for the constraint), so the
        step stays interpretable and cannot blow up; within the clamp the
        controller still decelerates as it approaches the target and stops on
        it, which is the behaviour a setpoint is for.
        """
        if j == 0 and self.setpoint is not None:
            bound = self.setpoint_saturation
            return (2.0 * (self.setpoint - rewards[:, 0])).clamp(-bound, bound)
        return torch.full_like(rewards[:, j], self.senses[j])

    def terms(self, rewards):
        """Per-sample value of each scalarized objective, [B, n_obj].

        Larger is better in every column, so the caller can weight them directly.
        """
        columns = []
        for j in range(self.n_obj):
            if j == 0 and self.setpoint is not None:
                columns.append(-(rewards[:, 0] - self.setpoint) ** 2)
            else:
                columns.append(self.senses[j] * rewards[:, j])
        return torch.stack(columns, dim=1)

    def penalty(self, rewards):
        """One-sided squared violation of the constraint, [B]. Zero when satisfied."""
        excess = rewards[:, self.constraint_index] - self.constraint_threshold
        return excess.clamp_min(0.0) ** 2

    def describe(self, stats=None, names=None):
        names = names or (stats or {}).get("r_names") or [f"r{i+1}" for i in range(self.n_props)]
        parts = []
        for j in range(self.n_obj):
            if j == 0 and self.setpoint is not None:
                parts.append(f"-({names[0]} - {self.setpoint:+.3f})^2")
            else:
                parts.append(f"{'+' if self.senses[j] > 0 else '-'}{names[j]}")
        text = "  objectives : " + ", ".join(parts)
        if self.constrained:
            text += (f"\n  constraint : penalty {self.rho:g} * "
                     f"max(0, {names[self.constraint_index]} - "
                     f"{self.constraint_threshold:+.3f})^2")
        return text


def weight_vector(lambdas, objective=None):
    """Tradeoff weights on the simplex, one per scalarized objective.

    Lecture 3.4 requires nonnegative weights summing to one, so the mix is a
    tradeoff rather than a second guidance-strength dial confounded with eta.
    A shorter list is zero-padded, which is what lets the default lambdas=(1, 0)
    keep working when a third property is present but constrained rather than
    scalarized.
    """
    n = len(lambdas) if objective is None else objective.n_obj
    values = list(lambdas[:n]) + [0.0] * max(0, n - len(lambdas))
    if len(lambdas) > n:
        raise ValueError(f"Got {len(lambdas)} lambdas for {n} objectives: {lambdas}")
    if any(v < 0 for v in values):
        raise ValueError(f"lambdas must be nonnegative, got {lambdas}")
    lam = torch.tensor(values, dtype=torch.float32, device=DEVICE)
    total = lam.sum()
    if float(total) <= 0:
        raise ValueError(f"lambdas must not be all zero, got {lambdas}")
    return lam / total
