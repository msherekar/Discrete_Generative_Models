"""Throughput and optimization machinery shared by both training loops.

Separated from training.py so the loops stay readable as loops: everything
here is about how fast a step runs and how its learning rate moves, not about
what is being regressed onto.
"""
import math
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# ══════════════════════════════════════════════════════════════════════════════
# Device setup
# ══════════════════════════════════════════════════════════════════════════════

def enable_fast_matmul():
    """Turn on the tensor-core matmul paths that are off by default.

    Measured on OSPool job 15833935 (H100, avGFP, transformer d_model=256,
    batch 128): 324 steps/epoch for 500 epochs took 290 minutes, i.e. 108 ms
    per step. Forward and backward for that model over 128 x 237 tokens is
    roughly 1 TFLOP, which an H100 at plain float32 (~60 TFLOPS) should finish
    in under 20 ms. The gap was launch overhead and host-side batching, not
    arithmetic, but float32 matmul was leaving the tensor cores idle on top of
    that.

    TF32 keeps float32 storage and signature while running the multiply on
    tensor cores at ~10 bits of mantissa. For a denoising regression whose
    targets are unit-variance Gaussian noise that is far more precision than
    the objective carries.
    """
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


def autocast(enabled=True):
    """bfloat16 autocast on CUDA, a no-op context everywhere else.

    bfloat16 rather than float16 because it keeps float32's exponent range, so
    there is no loss scaling to tune and no silent overflow in the reward head
    when a setpoint term is squared. Hopper and Blackwell both have native
    bf16 tensor cores; on anything older this simply runs slower, not wrong.
    """
    if enabled and torch.cuda.is_available():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast("cpu", enabled=False)


# ══════════════════════════════════════════════════════════════════════════════
# Batching
# ══════════════════════════════════════════════════════════════════════════════

def _host_loader(dataset, batch_size):
    """DataLoader for latents that did not fit on the GPU.

    Worker count comes from the CPUs actually available -- sched_getaffinity
    respects the cgroup a Condor slot imposes, where os.cpu_count() would
    report the whole machine and oversubscribe a two-core slot.
    """
    try:
        available = len(os.sched_getaffinity(0))
    except AttributeError:                              # not Linux
        available = os.cpu_count() or 1
    workers = max(0, min(4, available - 1))
    extra = {"num_workers": workers, "prefetch_factor": 2} if workers else {}
    return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                      pin_memory=torch.cuda.is_available(), **extra)


class Batches:
    """Shuffled minibatches, drawn on the GPU when the data already lives there.

    The host path gathers `batch_size` random rows out of a multi-gigabyte CPU
    tensor once per step, synchronously with that step. Measured on an OSPool
    A40 (job 15833630) that held GPU utilization at 0.57 with one core pinned
    for the whole run, and the 2026-10-03 H100 run still showed `gpu 0%` at
    several five-minute heartbeats for the same reason.

    When data.load_data() has already placed the latents in VRAM there is
    nothing to gather: a permutation and an index select stay on the device and
    the loop never touches the host. That is the whole optimization.

    This deliberately breaks bit-for-bit comparability with runs made through
    the DataLoader, because the random stream differs. Runs from before the
    architecture change are not comparable anyway; pass float32/cpu to
    load_data and the old path returns.
    """

    def __init__(self, dataset, batch_size, seed=0):
        self.tensors = dataset.tensors
        self.batch_size = batch_size
        self.device = self.tensors[0].device
        self.n = len(self.tensors[0])
        self.on_device = self.device.type == "cuda"
        self.loader = None if self.on_device else _host_loader(dataset, batch_size)
        self.generator = None
        if self.on_device:
            self.generator = torch.Generator(device=self.device)
            self.generator.manual_seed(seed)

    def __len__(self):
        """Number of steps per epoch, matching what the loops divide by."""
        if self.loader is not None:
            return len(self.loader)
        return math.ceil(self.n / self.batch_size)

    def __iter__(self):
        if self.loader is not None:
            for batch in self.loader:
                yield tuple(t.to(self.device, non_blocking=True) for t in batch)
            return
        order = torch.randperm(self.n, device=self.device, generator=self.generator)
        for start in range(0, self.n, self.batch_size):
            index = order[start:start + self.batch_size]
            # Latents are stored float16 to halve the buffer; promote per batch
            # so every downstream interpolation and loss runs in float32.
            yield (self.tensors[0][index].float(),
                   self.tensors[1][index],
                   self.tensors[2][index].float())


# ══════════════════════════════════════════════════════════════════════════════
# Optimizer and schedule
# ══════════════════════════════════════════════════════════════════════════════

def build_optimizer(field, reward, lr, reward_lr=None, weight_decay=0.0):
    """One Adam over two parameter groups that need different learning rates.

    The generative field and the reward head used to share a single group at
    lr=1e-3. They are not the same problem: the field regresses onto a target
    whose scale is set by the probability path, while the head regresses onto
    standardized properties and converges much sooner. Sharing one rate means
    whichever converges first spends the rest of training being pushed around
    by the other's gradient noise.

    `reward_lr=None` keeps the historical behaviour of one rate for both.
    """
    groups = [{"params": list(field.parameters()), "lr": lr, "name": "field"}]
    groups.append({"params": list(reward.parameters()),
                   "lr": lr if reward_lr is None else reward_lr, "name": "reward"})
    return torch.optim.AdamW(groups, lr=lr, weight_decay=weight_decay,
                             betas=(0.9, 0.999))


def warmup_cosine(optimizer, total_steps, warmup_steps=0, floor=0.05):
    """Linear warmup, then cosine decay to `floor` times the base rate.

    Constant 1e-3 for 500 epochs is what produced the loss ticking UP on the
    final epoch in both earlier long runs (flow: 0.4090 -> 0.4117 at epoch 125,
    0.3094 -> 0.3156 at 250). Decay removes most of that; EMA in nets.py
    handles the rest. Warmup matters once the batch grows past a few hundred,
    where the first steps are otherwise large enough to move the zero-init
    output projection off its carefully chosen starting point.
    """
    warmup_steps = max(0, int(warmup_steps))
    total_steps = max(1, int(total_steps))

    def factor(step):
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def add_arguments(parser):
    """Stage 0.5 flags: throughput, learning-rate schedule, validation."""
    group = parser.add_argument_group("optimization and throughput (Stage 0.5)")
    group.add_argument("--lr", type=float, default=None, metavar="R",
                       help="Learning rate for the generative field (default: "
                            "1e-3, the value hardcoded before this flag). With "
                            "--scale-lr it is multiplied by batch_size/128.")
    group.add_argument("--reward-lr", type=float, default=None, metavar="R",
                       help="Separate learning rate for the reward head "
                            "(default: follow --lr). They are not the same "
                            "problem -- the head regresses onto standardized "
                            "properties and converges much sooner -- so one "
                            "shared rate means whichever converges first is "
                            "then pushed around by the other's gradient noise.")
    group.add_argument("--warmup", type=int, default=0, metavar="N",
                       help="Linear warmup steps before cosine decay "
                            "(default: 0). Matters once the batch passes a few "
                            "hundred, where the first steps would otherwise "
                            "move the zero-initialized output projection off "
                            "its chosen starting point.")
    group.add_argument("--lr-floor", type=float, default=0.05, metavar="F",
                       help="Final learning rate as a fraction of the base "
                            "(default: 0.05). A constant 1e-3 for 500 epochs "
                            "is what produced the training loss ticking UP on "
                            "the final epoch of both long runs.")
    group.add_argument("--scale-lr", action="store_true",
                       help="Apply the linear scaling rule when --batch-size "
                            "differs from 128, so raising the batch for "
                            "throughput does not silently also shrink the "
                            "effective step (Goyal et al., arXiv:1706.02677).")
    group.add_argument("--no-amp", dest="amp", action="store_false",
                       help="Disable bfloat16 autocast. On by default: the "
                            "H100 run was host-bound at 108 ms/step for a 5M "
                            "parameter model, roughly 6x slower than the "
                            "arithmetic requires.")
    group.add_argument("--val-dataset", type=Path, default=None, metavar="CSV",
                       help="Held-out CSV to report validation loss on, e.g. "
                            "data/avgfp_val.csv. Without it the epoch budget "
                            "can only be justified from the training curve, "
                            "which is how 500 epochs came to be spent when "
                            "diffusion stopped improving after 125.")
    group.add_argument("--latent-dtype", default="float16",
                       choices=("float16", "float32"),
                       help="Storage dtype for the encoded latents (default: "
                            "float16). avGFP at float32 is 12.55 GB per copy, "
                            "and four live copies in load_data is what held "
                            "OSG job 15833935 at 63.6 GB against a 60 GB "
                            "request. float32 reproduces the older numerics.")
    group.add_argument("--latent-device", default=None,
                       choices=("cuda", "cpu"),
                       help="Where the latent buffer lives (default: auto, "
                            "GPU when it fits in half the free VRAM). Keeping "
                            "it resident is what removes the host-side "
                            "batching bottleneck.")
    parser.set_defaults(amp=True)
    return parser


def scale_lr_for_batch(lr, batch_size, reference=128):
    """Linear scaling rule: proportionally larger batches want larger steps.

    Goyal et al., "Accurate, Large Minibatch SGD" (arXiv:1706.02677). Applied
    so that raising --batch-size for throughput does not silently also mean
    training for fewer, unchanged-size steps.
    """
    return lr * (batch_size / reference)
