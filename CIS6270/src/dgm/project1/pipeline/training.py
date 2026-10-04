"""Training loops for the flow-matching and diffusion heads.

Each loop trains its generative network and its reward head together, so the
head sees exactly the noise levels guidance will later query it at. The
throughput and schedule machinery lives in optim.py.
"""
import torch
import torch.nn.functional as F

from .config import BATCH_SIZE, CONDITION_DROP, DEVICE, HIDDEN, LEARNING_RATE
from .coupling import pair, source_for
from .losses import generative_loss, reward_loss
from .nets import EMA, DiffusionModel, FlowModel, RewardModel
from .optim import (Batches, autocast, build_optimizer, enable_fast_matmul,
                    warmup_cosine)
from .paths import (PathSpec, interpolate, make_ddpm_schedule,
                    sample_timesteps)

# ══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ══════════════════════════════════════════════════════════════════════════════

class Report:
    """Per-epoch loss bookkeeping, split by head.

    The loops used to accumulate `generative + reward` into one number and
    print that. It is the only training signal the OSG logs carry, and it
    cannot be attributed: the 2026-10-03 H100 run reported diffusion at 0.1819
    / 0.1728 / 0.2086 / 0.1799 across 500 epochs with no way to tell whether
    the field had stalled or the reward head had. Keep them apart.
    """

    def __init__(self):
        self.generative, self.reward, self.validation = [], [], []

    def add(self, generative, reward, steps):
        self.generative.append(generative / max(1, steps))
        self.reward.append(reward / max(1, steps))

    def show(self, tag, epoch, epochs, every=None):
        """Print on the quarter marks, matching the previous cadence."""
        every = every or max(1, epochs // 4)
        if (epoch + 1) % every and epoch + 1 != epochs:
            return
        line = (f"  [{tag:<9}] epoch {epoch+1:>4}/{epochs}: "
                f"gen {self.generative[-1]:.4f}  reward {self.reward[-1]:.4f}")
        if self.validation and self.validation[-1] is not None:
            line += f"  val {self.validation[-1]:.4f}"
        print(line)

    @property
    def total(self):
        """Summed curve, so anything reading the old `losses` list still works."""
        return [g + r for g, r in zip(self.generative, self.reward)]


def condition_for(c, r_tilde, conditioning, n_cond=1):
    """The value the trunk is conditioned on, for either channel width.

    "binary" passes the stored bucket c = int(score > -1.0) through unchanged,
    which is the Stage 0.2 control. "continuous" passes the standardized
    property columns themselves, so the field is told HOW bright rather than
    merely bright-or-not -- without that there is no input at which to request
    an intermediate brightness, and "controllable" cannot be demonstrated
    through conditioning at all.
    """
    if conditioning == "none":
        return None
    if conditioning == "continuous":
        # Pad as well as slice. A validation CSV may carry fewer property
        # columns than the training CSV the model was sized from, and slicing
        # alone returned a narrower tensor that then indexed off the end of the
        # trunk's embedding list. Zero is the right filler because the
        # properties are standardized, so it requests the dataset mean for a
        # property this split does not measure.
        if r_tilde.shape[1] >= n_cond:
            return r_tilde[:, :n_cond]
        padded = r_tilde.new_zeros((len(r_tilde), n_cond))
        padded[:, :r_tilde.shape[1]] = r_tilde
        return padded
    return c


def drop_mask(n, device, drop=CONDITION_DROP):
    """Which rows train the unconditional branch this step.

    Ho & Salimans (arXiv:2207.12598): training the conditional and
    unconditional branches in one network requires dropping the condition at
    random, so the null token learns the marginal. Returned as a mask rather
    than applied here, because the two conditioning channels null themselves
    differently -- a class index versus a learned null embedding.
    """
    return torch.rand(n, device=device) < drop


# Noise levels the validation pass is averaged over. Fixed rather than freshly
# sampled: a validation number has to be comparable between epochs, and
# resampling t adds variance that swamps the difference being measured. There
# was no validation signal at all before Stage 0.5.3, which is why no epoch
# budget in the sweep could be justified from anything but the training curve.
VAL_LEVELS = (0.1, 0.3, 0.5, 0.7, 0.9)


@torch.no_grad()
def _validate_flow(model, dataset, path, batch_size, conditioning, n_cond, seed=0):
    """Held-out flow-matching loss, averaged over a fixed time grid."""
    if dataset is None:
        return None
    model.eval()
    total, steps = 0.0, 0
    for z1, c, r_tilde in Batches(dataset, batch_size, seed=seed):
        cond = condition_for(c, r_tilde, conditioning, n_cond)
        for level in VAL_LEVELS:
            t = torch.full((len(z1),), level, device=DEVICE)
            z0 = torch.randn_like(z1)
            zt, target = interpolate(z0, z1, t, **path.kwargs)
            with autocast():
                total += F.mse_loss(model(zt, t, cond).float(), target).item()
            steps += 1
    model.train()
    return total / max(1, steps)


@torch.no_grad()
def _validate_diffusion(model, dataset, alpha_bars, batch_size, predict,
                        conditioning, n_cond, seed=0):
    """Held-out denoising loss on a fixed ladder of noise levels."""
    if dataset is None:
        return None
    model.eval()
    K = model.K
    total, steps = 0.0, 0
    for z0, c, r_tilde in Batches(dataset, batch_size, seed=seed):
        cond = condition_for(c, r_tilde, conditioning, n_cond)
        for fraction in VAL_LEVELS:
            k = torch.full((len(z0),), int(fraction * K), device=DEVICE,
                           dtype=torch.long).clamp(1, K)
            t = k.float() / K
            a = alpha_bars[k, None, None]
            eps = torch.randn_like(z0)
            zk = a.sqrt() * z0 + (1 - a).sqrt() * eps
            with autocast():
                prediction = model(zk, t, cond)
            total += generative_loss(prediction.float(), z0, eps, zk, a,
                                     predict).item()
            steps += 1
    model.train()
    return total / max(1, steps)


# ══════════════════════════════════════════════════════════════════════════════
# Flow matching
# ══════════════════════════════════════════════════════════════════════════════

def train_flow(dataset, epochs, batch_size=BATCH_SIZE, hidden=HIDDEN,
               path=None, arch="mlp", coupling="independent",
               coupling_beta=0.0, coupling_columns=None, ema_decay=0.0,
               lr=LEARNING_RATE, reward_lr=None, warmup=0, amp=True,
               val_dataset=None, censor_floor=None, conditioning="binary",
               modulation="adaln", rope=True, n_cond=1, seed=0):
    """Train the velocity field and the reward head together.

    `path` is a PathSpec (geometry, schedule, scale); the default straight
    segment on linear time is what the lectures use. `coupling` selects how
    each noise draw is paired with a data point; see coupling.py. It returns
    the fitted InformedSource alongside the model when coupling='informed',
    because sampling then has to draw from that same source rather than from
    N(0, I), and None otherwise.

    `ema_decay` was previously accepted by train_diffusion only and silently
    ignored here, even though flow is the loop whose loss rose on the final
    epoch in both long runs. It now applies to both.
    """
    enable_fast_matmul()
    path = path or PathSpec()
    _, length, dim = dataset.tensors[0].shape
    n_props = dataset.tensors[2].shape[1]
    n_cond = min(n_cond, n_props)
    batches = Batches(dataset, batch_size, seed=seed)
    model = FlowModel(length, dim, hidden, arch, conditioning, modulation,
                      rope, n_cond).to(DEVICE)
    reward = RewardModel(length, dim, hidden, arch, n_props).to(DEVICE)
    optimizer = build_optimizer(model, reward, lr, reward_lr)
    schedule = warmup_cosine(optimizer, epochs * len(batches), warmup)
    ema = EMA(model, ema_decay) if ema_decay > 0 else None
    source = source_for(coupling, dataset.tensors[0].float().to(DEVICE),
                        dataset.tensors[2].float().to(DEVICE))
    report = Report()

    for epoch in range(epochs):
        gen_total, rew_total = 0.0, 0.0
        for z1, c, r_tilde in batches:
            # Where z0 comes from is the coupling's whole job: an identity
            # pairing is the lecture default, OT re-pairs within the batch, and
            # an informed source draws from a property-derived distribution the
            # field was trained against.
            z0 = (source.paired(r_tilde) if source is not None else
                  pair(torch.randn_like(z1), z1, coupling, coupling_beta,
                       r_tilde, coupling_columns))
            t = torch.rand(len(z1), device=DEVICE)
            zt, target = interpolate(z0, z1, t, **path.kwargs)
            cond = condition_for(c, r_tilde, conditioning, n_cond)
            with autocast(amp):
                velocity = model(zt, t, cond, drop_mask(len(z1), DEVICE))
                predicted = reward(zt, t)
            gen = F.mse_loss(velocity.float(), target)
            rew = reward_loss(predicted.float(), r_tilde, censor_floor)
            optimizer.zero_grad(set_to_none=True)
            (gen + rew).backward()
            optimizer.step()
            schedule.step()
            if ema is not None:
                ema.update(model)
            gen_total += gen.item()
            rew_total += rew.item()
        report.add(gen_total, rew_total, len(batches))
        report.validation.append(
            _validate_flow(model, val_dataset, path, batch_size, conditioning,
                           n_cond, seed))
        report.show("flow", epoch, epochs)

    if ema is not None:
        ema.copy_to(model)              # sample from the averaged weights
    model.eval().requires_grad_(False)
    reward.eval().requires_grad_(False)
    return model, reward, report.total, source


# ══════════════════════════════════════════════════════════════════════════════
# Diffusion
# ══════════════════════════════════════════════════════════════════════════════

def train_diffusion(dataset, epochs, batch_size=BATCH_SIZE, hidden=HIDDEN,
                    arch="mlp", predict="x0", steps=1000, stratified=False,
                    ema_decay=0.0, lr=LEARNING_RATE, reward_lr=None, warmup=0,
                    amp=True, val_dataset=None, censor_floor=None,
                    weighting="none", beta_schedule="linear",
                    conditioning="binary", modulation="adaln", rope=True,
                    n_cond=1, seed=0):
    """Train the denoiser and the reward head together.

    `predict` defaults to "x0" rather than "eps". Lecture 3.2's AMP-Diffusion
    section -- latent diffusion on ESM-2 embeddings, which is exactly this
    setting -- specifies "directly predict denoised latent and train using MSE
    loss". Under "eps" the output is
    sqrt(1-abar) z + sqrt(abar) net(...), so at high noise the network is
    handed a nearly correct answer and contributes nothing; see nets.py. The
    2026-10-03 H100 run is the measurement: 500 epochs of eps moved the loss
    from 0.1819 (epoch 125) to 0.1799 (epoch 500), non-monotonically.

    `weighting` selects the per-noise-level loss weight; see losses.py.
    """
    enable_fast_matmul()
    _, length, dim = dataset.tensors[0].shape
    betas, alphas, alpha_bars, post_vars = make_ddpm_schedule(steps, beta_schedule)
    K = len(betas) - 1
    n_props = dataset.tensors[2].shape[1]
    n_cond = min(n_cond, n_props)
    batches = Batches(dataset, batch_size, seed=seed)
    model = DiffusionModel(length, dim, alpha_bars, hidden, arch, predict,
                           conditioning, modulation, rope, n_cond).to(DEVICE)
    reward = RewardModel(length, dim, hidden, arch, n_props).to(DEVICE)
    optimizer = build_optimizer(model, reward, lr, reward_lr)
    schedule = warmup_cosine(optimizer, epochs * len(batches), warmup)
    ema = EMA(model, ema_decay) if ema_decay > 0 else None
    report = Report()

    for epoch in range(epochs):
        gen_total, rew_total = 0.0, 0.0
        for z0, c, r_tilde in batches:
            k = sample_timesteps(len(z0), K, DEVICE, stratified)
            t = k.float() / K
            a = alpha_bars[k, None, None]
            eps = torch.randn_like(z0)
            # Lecture 3.2: jump straight to any noise level rather than
            # simulating the earlier steps.
            zk = a.sqrt() * z0 + (1 - a).sqrt() * eps
            cond = condition_for(c, r_tilde, conditioning, n_cond)
            with autocast(amp):
                prediction = model(zk, t, cond, drop_mask(len(z0), DEVICE))
                predicted_reward = reward(zk, t)
            gen = generative_loss(prediction.float(), z0, eps, zk, a,
                                  predict, weighting)
            rew = reward_loss(predicted_reward.float(), r_tilde, censor_floor)
            optimizer.zero_grad(set_to_none=True)
            (gen + rew).backward()
            optimizer.step()
            schedule.step()
            if ema is not None:
                ema.update(model)
            gen_total += gen.item()
            rew_total += rew.item()
        report.add(gen_total, rew_total, len(batches))
        report.validation.append(
            _validate_diffusion(model, val_dataset, alpha_bars, batch_size,
                                predict, conditioning, n_cond, seed))
        report.show("diffusion", epoch, epochs)

    if ema is not None:
        ema.copy_to(model)              # sample from the averaged weights
    model.eval().requires_grad_(False)
    reward.eval().requires_grad_(False)
    return model, reward, report.total, alpha_bars, betas, alphas, post_vars
