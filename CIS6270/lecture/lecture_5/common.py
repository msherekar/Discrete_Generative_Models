"""CPU teaching utilities shared by the two lecture folders."""
import csv
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from lecture_core import DNA, K, MASK, encode


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def decode(tokens):
    alphabet = 'ACGTm'
    return [''.join(alphabet[int(i)] for i in row) for row in tokens]


def make_data(count=512, length=8, seed=7):
    """Four synthetic motif families, with independent 8% base mutations."""
    rng = np.random.default_rng(seed)
    motifs = ['ACGT', 'CGTA', 'TATA', 'GCGC']
    strings, labels = [], []
    for _ in range(count):
        motif = motifs[int(rng.integers(4))]
        seq = list((motif * ((length + 3) // 4))[:length])
        for j in range(length):
            if rng.random() < .08:
                seq[j] = 'ACGT'[int(rng.integers(4))]
        strings.append(''.join(seq))
        labels.append(int(sum(x in 'GC' for x in seq) / length >= .6))
    return encode(strings), torch.tensor(labels)


def load_data(path, length):
    if path is None:
        return make_data(length=length)
    with open(path) as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    strings = [r['sequence'].strip().upper() for r in rows]
    if not strings or any(len(s) != length or set(s) - set('ACGT') for s in strings):
        raise ValueError('All DNA sequences must contain only A/C/G/T and have --length bases.')
    labels = [int(r.get('label', sum(c in 'GC' for c in s) / length >= .6))
              for r, s in zip(rows, strings)]
    if set(labels) - {0, 1}:
        raise ValueError('Labels must be 0 or 1.')
    return encode(strings), torch.tensor(labels)


class ConditionalDNA(DNA):
    """The slide network plus a label embedding; label 2 means unconditional."""
    def __init__(self, width=32, max_len=64):
        super().__init__(width, max_len)
        self.condition = nn.Embedding(3, width)

    def forward(self, z, t=None, label=None):
        h = self.token(z) if z.ndim == 2 else self.soft(z)
        pos = torch.arange(z.shape[1], device=z.device)
        h = h + self.position(pos)[None]
        if t is not None:
            h = h + self.time(t[:, None])[:, None]
        if label is None:
            label = torch.full((len(z),), 2, dtype=torch.long, device=z.device)
        h = h + self.condition(label)[:, None]
        return self.output(self.context(h))


def objectives(tokens):
    """Both toy objectives are maximized: GC fraction and ATAT agreement."""
    onehot = torch.nn.functional.one_hot(tokens.long(), 4).float()
    return soft_objectives(onehot)


def soft_objectives(z):
    gc = (z[..., 1] + z[..., 2]).mean(-1)
    motif = torch.tensor([0, 3], device=z.device).repeat((z.shape[1] + 1) // 2)[:z.shape[1]]
    match = z.gather(-1, motif[None, :, None].expand(z.shape[0], -1, 1)).squeeze(-1).mean(-1)
    return torch.stack((gc, match), -1)


def metrics(tokens):
    strings = decode(tokens)
    score = objectives(tokens).mean(0)
    return dict(valid_dna=all(set(s) <= set('ACGT') for s in strings),
                unique_fraction=len(set(strings)) / len(strings),
                mean_gc=float(score[0]), mean_atat_match=float(score[1]))


def save_run(out, config, losses, samples, extra=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / ('sample_config.json' if config.get('mode') == 'sample' else 'config.json')).write_text(json.dumps(config, indent=2))
    if losses or not (out / 'losses.json').exists():
        (out / 'losses.json').write_text(json.dumps(losses, indent=2))
    (out / 'samples.txt').write_text('\n'.join(decode(samples)) + '\n')
    report = {'metrics': metrics(samples), **(extra or {})}
    (out / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'output': str(out), **report}, indent=2))
    return report


def optimize(model, loss_fn, data, steps, batch_size=32, lr=.002):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    model.train()
    for step in range(steps):
        idx = torch.randint(len(data), (batch_size,))
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model, data[idx], idx)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite loss at step {step}')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        losses.append(float(loss.detach()))
    model.eval()
    return losses


def project_simplex(z, eps=1e-6):
    """Euclidean simplex projection, followed by a small interior floor."""
    sorted_z = z.sort(-1, descending=True).values
    cssv = sorted_z.cumsum(-1) - 1.
    k = torch.arange(1, z.shape[-1] + 1, dtype=z.dtype, device=z.device)
    active = sorted_z - cssv / k > 0
    rho = active.sum(-1, keepdim=True).clamp_min(1)
    theta = cssv.gather(-1, rho - 1) / rho
    result = (z - theta).clamp_min(eps)
    return result / result.sum(-1, keepdim=True)
