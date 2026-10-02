"""How a noise draw is paired with a data point.

Flow matching learns transport between the MARGINALS of the pairs it is trained
on, so the one hard rule here is that every coupling must be a bijection within
the batch. Each noise draw is used exactly once and each data point is used
exactly once, which leaves both marginals untouched. Nearest-neighbour pairing
-- every data point takes its closest noise draw -- is therefore NOT available:
it reuses some draws and discards others, changes the source marginal, and the
trained field no longer transports N(0, I) to the data.

Four mechanisms, all expressed as one assignment problem:

  independent   the identity permutation. Lecture 2.2's default, and what
                training.py did before this module existed.
  ot            assignment on ||z0 - z1||^2 (minibatch optimal transport).
  aux           assignment on ||z0 - z1||^2 + beta * ||a0 - a1||^2, where a1 is
                the data point's property vector and a0 is a target property
                assigned to the noise draw. beta=0 recovers `ot` exactly, which
                is what makes the ablation a single knob rather than a separate
                code path.
  informed      not a coupling at all: it REPLACES the Gaussian source with a
                map of the property vector (IDEA.md option C1). Kept here
                because it is the alternative to a coupling, and because the
                diagnostic has to measure both against the same baseline.

Minibatch OT is biased with respect to true OT and the bias shrinks with the
batch size, so batch size stops being an efficiency setting and becomes a
method hyperparameter. check_coupling.py sweeps it for exactly that reason.
"""
import numpy as np
import torch

KINDS = ("independent", "ot", "aux", "informed")


def add_arguments(parser):
    """The coupling flags, shared by every runner.

    Defined here rather than in each runner's parser so the flag set, its
    defaults and its documentation cannot drift between the peptide and image
    studies -- a difference there would read as a modality effect.
    """
    parser.add_argument("--coupling", default="independent", choices=KINDS,
                        help="How a noise draw is paired with a data point "
                             "(default: independent, what the lectures use). 'ot' "
                             "solves minimum-cost assignment on ||z0-z1||^2 within "
                             "each minibatch; 'aux' adds --coupling-beta times a "
                             "property-distance cost and reduces to 'ot' at beta=0; "
                             "'informed' is not a coupling but a replacement source "
                             "built from the property vector. FLOW ONLY -- the DDPM "
                             "forward process defines its own pairing, so changing "
                             "it there means changing the forward process.")
    parser.add_argument("--coupling-beta", type=float, default=0.0, metavar="B",
                        help="Weight of the property-distance term in the 'aux' "
                             "cost (default: 0 = plain OT). Both terms are divided "
                             "by their own mean first, for the reason guidance.py "
                             "gives for --normalize-guidance.")
    parser.add_argument("--coupling-columns", type=int, nargs="+", default=None,
                        metavar="IDX",
                        help="1-based property columns entering the 'aux' cost "
                             "(default: all). Pass 1 to couple on the first "
                             "property alone, which makes a requested value an "
                             "INPUT to generation rather than something guidance "
                             "has to climb to.")


def columns_from_args(args):
    """The 1-based --coupling-columns as 0-based indices, or None."""
    return ([c - 1 for c in args.coupling_columns]
            if args.coupling_columns else None)


def source_for(kind, z1, aux):
    """The fitted InformedSource this coupling needs, or None.

    Fitted once on the whole training set rather than per batch: the map is part
    of the model, and refitting per batch would make the source move during
    training.
    """
    if kind != "informed":
        return None
    source = InformedSource.fit(z1, aux)
    print(f"  [coupling]  informed source fitted, residual sigma "
          f"{source.sigma:.4f}")
    return source


def pair(z0, z1, kind="independent", beta=0.0, aux=None, columns=None):
    """`z0` reordered to line up with `z1` -- couple() plus the indexing."""
    return z0[couple(z0, z1, kind, beta, aux, columns)]


def cost_matrix(x, y):
    """Pairwise squared Euclidean cost, [B, B], between flattened batches.

    Written as ||x||^2 + ||y||^2 - 2 x.y rather than torch.cdist because the
    latents here are large (a 237x320 protein latent is 75,840 numbers) and this
    form is one matmul. The diagonal is not special: x[i] and y[i] are a
    candidate pair like any other.
    """
    a = x.flatten(1)
    b = y.flatten(1)
    return (a.pow(2).sum(1)[:, None] + b.pow(2).sum(1)[None, :]
            - 2.0 * (a @ b.T))


def assign(cost):
    """The minimum-cost bijection, as a permutation of the ROW index.

    Returns `perm` with the meaning: row perm[j] is matched to column j. The
    caller permutes its noise batch by this, so noise[perm] lines up with data.
    """
    from scipy.optimize import linear_sum_assignment

    rows, cols = linear_sum_assignment(cost.detach().float().cpu().numpy())
    perm = np.empty(len(cols), dtype=np.int64)
    perm[cols] = rows
    return torch.from_numpy(perm).to(cost.device)


def _normalized(cost):
    """Cost divided by its own mean, so two costs can be added meaningfully.

    The same argument guidance.py makes for `normalize`: without this, beta is
    confounded with whatever scale each cost happens to have. A latent cost on
    75,840 standardized dimensions is of order 2D; a property cost on two
    standardized columns is of order 4. Added raw, beta would have to span four
    orders of magnitude to mean anything.
    """
    return cost / cost.mean().clamp_min(1e-12)


def target_aux(aux):
    """A target property vector for each noise draw, [B, P].

    Drawn from the batch's own empirical property distribution by permuting it,
    which keeps the marginal exact -- the same reason the coupling itself is a
    permutation. This is what lets the source carry a requested property
    (IDEA.md option D) while remaining samplable at generation time: at sampling
    you supply the target yourself instead of drawing one.
    """
    return aux[torch.randperm(len(aux), device=aux.device)]


def couple(z0, z1, kind="independent", beta=0.0, aux=None, columns=None):
    """Permutation aligning the noise batch `z0` with the data batch `z1`.

    `aux` is the batch's standardized property matrix [B, P] -- the r_tilde the
    loader already yields. `columns` selects which of its columns enter the
    cost, so option D (one scalar property) and option C2 (the whole structural
    vector) are the same call with a different slice.
    """
    if kind not in KINDS:
        raise ValueError(f"unknown coupling '{kind}', expected one of {KINDS}")
    if kind in ("independent", "informed"):
        # `informed` changes the source, not the pairing: see informed_source().
        return torch.arange(len(z0), device=z0.device)
    cost = _normalized(cost_matrix(z0, z1))
    if kind == "aux":
        if aux is None:
            raise ValueError("coupling 'aux' needs the property matrix")
        a1 = aux if columns is None else aux[:, list(columns)]
        if beta:
            cost = cost + float(beta) * _normalized(cost_matrix(target_aux(a1), a1))
    return assign(cost)


# ══════════════════════════════════════════════════════════════════════════════
# Informed source (IDEA.md option C1)
# ══════════════════════════════════════════════════════════════════════════════

class InformedSource:
    """A source distribution built from the property vector, not from noise.

    z0 = A a + sigma * eps, with `a` the standardized property vector. The
    residual term is not optional: a peptide run has two property columns, so
    `A a` alone spans a rank-2 subspace of a 7,680-dimensional latent space and
    a field trained from it never sees most directions. sigma is set from the
    residual the fit leaves behind, so the source has the same total scale as
    the Gaussian baseline it replaces.

    At sampling time there is no protein to read properties from, which is the
    objection IDEA.md raises against this option. `draw` therefore samples `a`
    from a Gaussian fitted to the training property distribution, or from a
    caller-supplied target vector -- in which case the source is interpretable
    and the requested property is an input rather than a guidance term.
    """

    def __init__(self, A, shape, sigma, aux_mean, aux_std):
        self.A, self.shape, self.sigma = A, shape, float(sigma)
        self.aux_mean, self.aux_std = aux_mean, aux_std

    @classmethod
    def fit(cls, z1, aux, ridge=1.0):
        """Least-squares map from property vector to flattened latent."""
        shape = z1.shape[1:]
        X = torch.cat([aux, torch.ones(len(aux), 1, device=aux.device)], dim=1)
        Z = z1.flatten(1)
        gram = X.T @ X + ridge * torch.eye(X.shape[1], device=X.device)
        A = torch.linalg.solve(gram, X.T @ Z)
        residual = (Z - X @ A).pow(2).mean().sqrt()
        return cls(A, shape, residual, aux.mean(0), aux.std(0).clamp_min(1e-6))

    def paired(self, aux):
        """The source point belonging to these data points (training time)."""
        X = torch.cat([aux, torch.ones(len(aux), 1, device=aux.device)], dim=1)
        mean = (X @ self.A).reshape(-1, *self.shape)
        return mean + self.sigma * torch.randn_like(mean)

    def draw(self, n, device, aux=None):
        """A source batch with no data in hand (sampling time)."""
        if aux is None:
            aux = (self.aux_mean + self.aux_std
                   * torch.randn(n, len(self.aux_mean), device=device))
        return self.paired(aux.to(device))


# ══════════════════════════════════════════════════════════════════════════════
# What the diagnostic measures
# ══════════════════════════════════════════════════════════════════════════════

def target_moment(z0, z1):
    """Mean square of the flow-matching target z1 - z0.

    This is the quantity a coupling is supposed to reduce, and it is directly
    comparable to check_loss_floor.py's "predict 0" row: a model that outputs
    zero scores exactly this. A coupling that lowers it has lowered the ceiling
    on the regression problem, which is the mechanism behind every claim about
    straighter paths and fewer integration steps.
    """
    return float((z1 - z0).pow(2).mean())


def pair_distance(z0, z1):
    """Mean per-pair squared distance, normalized by dimension."""
    d = z0.flatten(1).shape[1]
    return float((z1 - z0).flatten(1).pow(2).sum(1).mean() / d)
