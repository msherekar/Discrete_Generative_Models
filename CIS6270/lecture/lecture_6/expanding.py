"""Expanding Flow Maps on variable-length text with learned gap insertions.

Based on paper Algorithms 2-4. The state carries birth times alongside token
coordinates. Finite transport uses local clocks on a shared destination canvas.
The insertion head learns both remaining-count and interval-count expectations.
"""
import torch
from torch import nn
from torch.nn import functional as F

from common import ema_copy, mlp, optimize


class ExpandingNet(nn.Module):
    def __init__(self, length, vocab, width):
        super().__init__()
        self.length, self.vocab = length, vocab
        # Every token contributes state, two local clocks, birth, and active mask.
        self.denoiser = mlp(length * (vocab+4), length*vocab, width)
        self.insertion = mlp(length*(vocab+2)+2, length+1, width)

    def predict(self, x, local_s, local_t, births, mask):
        features = torch.cat([x, local_s[..., None], local_t[..., None],
                              births[..., None], mask[..., None].to(x)], -1)
        return self.denoiser(features.flatten(1)).reshape(-1, self.length, self.vocab)

    def counts(self, x, births, mask, s, t, diagonal=False):
        features = torch.cat([x, births[..., None], mask[..., None].to(x)], -1).flatten(1)
        features = torch.cat([features, s, t], -1)
        remaining = F.softplus(self.insertion(features)) + 1e-5
        # Linear birth CDF: rho=(t-s)/(1-s). The diagonal predicts the
        # remaining count; off-diagonal predictions include the interval factor.
        return remaining if diagonal else (t-s)/(1-s)*remaining


def local_clock(t, births):
    return ((t-births)/(1-births)).clamp(0, 1)


def local_map(model, x, a, b, births, mask):
    p = model.predict(x, a, b, births, mask).softmax(-1)
    eta = ((b-a)/(1-a).clamp_min(1e-6))[..., None]
    y = (x + eta*(p-x)) * mask[..., None]
    return y, p


def gap_counts(active_indices, born_indices, length):
    """Assign each missing index to a gap while retaining original token order."""
    result = torch.zeros(len(active_indices)+1, device=active_indices.device)
    if len(born_indices):
        gap = torch.searchsorted(active_indices, born_indices)
        result.scatter_add_(0, gap, torch.ones_like(gap, dtype=result.dtype))
    return result


def compact(full, births, selected, max_length):
    x = full.new_zeros(max_length, full.shape[-1])
    b = births.new_zeros(max_length)
    mask = torch.zeros(max_length, dtype=torch.bool, device=full.device)
    n = len(selected)
    x[:n], b[:n], mask[:n] = full[selected], births[selected], True
    return x, b, mask


def count_divergence(target, mean):
    # Poisson NLL up to target-only constants. Finite at target=0.
    # Avoid xlogy(0,0)'s undefined intermediate derivative in autodiff.
    return mean - target + target * (target.clamp_min(1e-12).log()-mean.log())


def expansion_batch(ids, lengths, vocab, batch_size):
    index = torch.randint(len(ids), (batch_size,), device=ids.device)
    fields = {k: [] for k in ['xs','bs','ms','xe','be','me','ys','ye','missing','interval','gapmask']}
    pairs = (.96*torch.rand(batch_size, 2, device=ids.device)).sort(-1).values
    s, t = pairs[:, :1], pairs[:, 1:]
    t = torch.maximum(t, s+1e-4)
    L = ids.shape[1]
    for j, row in enumerate(index):
        n = int(lengths[row])
        clean = F.one_hot(ids[row, :n], vocab).float()
        births = .999*torch.rand(n, device=ids.device)
        noise = torch.randn_like(clean)
        clock = local_clock(s[j], births)
        full = (1-clock[:, None])*noise+clock[:, None]*clean
        active = torch.where(births <= s[j])[0]
        later = torch.where(births <= t[j])[0]
        xs, bs, ms = compact(full, births, active, L)
        xe, be, me = compact(full, births, later, L)
        ys, _, _ = compact(clean, births, active, L)
        ye, _, _ = compact(clean, births, later, L)
        missing = full.new_zeros(L+1)
        interval = full.new_zeros(L+1)
        gm = torch.arange(L+1, device=ids.device) <= len(active)
        missing[gm] = gap_counts(active, torch.where(births>s[j])[0], n)
        interval[gm] = gap_counts(active, torch.where((births>s[j]) & (births<=t[j]))[0], n)
        for k, value in zip(fields, [xs,bs,ms,xe,be,me,ys,ye,missing,interval,gm]):
            fields[k].append(value)
    return {k: torch.stack(v) for k,v in fields.items()}, s, t


def train_expanding(ids, lengths, vocab, args):
    model = ExpandingNet(ids.shape[1], len(vocab), args.width).to(ids.device)
    teacher = ema_copy(model)
    def objective(step):
        b, s, t = expansion_batch(ids, lengths, len(vocab), args.batch_size)
        a = local_clock(s, b['bs'])
        logits = model.predict(b['xs'], a, a, b['bs'], b['ms'])
        ce = -(b['ys']*logits.log_softmax(-1)).sum(-1)
        diagonal = (ce*b['ms']).sum()/b['ms'].sum().clamp_min(1)
        # Lift both paths to the same destination canvas, with the same noise.
        la, lb = local_clock(s,b['be']), local_clock(t,b['be'])
        lu = local_clock((s+t)/2,b['be'])
        with torch.no_grad():
            middle, p1 = local_map(teacher,b['xe'],la,lu,b['be'],b['me'])
            _, p2 = local_map(teacher,middle,lu,lb,b['be'],b['me'])
            gamma = ((1-lb)*(lu-la)/((1-lu)*(lb-la)).clamp_min(1e-6)).clamp(0,1)
            target = gamma[...,None]*p1+(1-gamma[...,None])*p2
        logits = model.predict(b['xe'],la,lb,b['be'],b['me'])
        kl = F.kl_div(logits.log_softmax(-1),target,reduction='none').sum(-1)
        finite = (kl*b['me']).sum()/b['me'].sum().clamp_min(1)
        m = model.counts(b['xs'],b['bs'],b['ms'],s,s,diagonal=True)
        interval = model.counts(b['xs'],b['bs'],b['ms'],s,t)
        count = count_divergence(b['missing'],m)+count_divergence(b['interval'],interval)
        insertion = (count*b['gapmask']).sum()/b['gapmask'].sum()
        ramp = min(1.,(step+1)/max(1,args.train_steps//4))
        return diagonal+ramp*finite+insertion, {'diagonal_ce':diagonal,'finite':finite,'insertion':insertion}
    logs = optimize(model,objective,args.train_steps,args.lr,teacher)
    return model, {'model':model.state_dict(),'length':ids.shape[1],'vocab':vocab}, logs


def bounded_counts(means, budget):
    """Independent binomial proposals, then left-to-right joint budget capping.

    Before capping, each proposal has its predicted mean after [0,budget]
    clipping. Conditional means alone do not determine the joint count law.
    """
    if budget == 0:
        return torch.zeros_like(means, dtype=torch.long), 0
    raw = torch.distributions.Binomial(budget, probs=means.clamp(0,budget)/budget).sample().long()
    kept = raw.clone()
    remaining = budget
    for i in range(len(kept)):
        kept[i] = min(int(kept[i]),remaining)
        remaining -= int(kept[i])
    return kept, int((raw-kept).sum())


def insert_tokens(state, births, counts, noise, birth_time):
    pieces, times, offset = [], [], 0
    for gap, count in enumerate(counts.tolist()):
        if count:
            pieces.append(noise[offset:offset+count])
            times.append(births.new_full((count,),birth_time))
            offset += count
        if gap < len(state):
            pieces.append(state[gap:gap+1]); times.append(births[gap:gap+1])
    if offset != len(noise):
        raise ValueError('Noise rows must match insertion counts.')
    if not pieces:
        return state, births
    return torch.cat(pieces),torch.cat(times)


@torch.no_grad()
def sample_expanding(model,count,steps,device):
    result = torch.zeros(count,model.length,dtype=torch.long,device=device)
    lengths = torch.zeros(count,dtype=torch.long,device=device)
    capped = 0
    traces = []
    for j in range(count):
        x = torch.zeros(0,model.vocab,device=device)
        births = torch.zeros(0,device=device)
        trace = [0]
        for k in range(steps):
            s,t = k/steps,(k+1)/steps
            padded,bt,mask = compact(x,births,torch.arange(len(x),device=device),model.length)
            start,end = torch.tensor([[s]],device=device),torch.tensor([[t]],device=device)
            means = model.counts(padded[None],bt[None],mask[None],start,end)[0,:len(x)+1]
            counts,discarded = bounded_counts(means,model.length-len(x));capped+=discarded
            noise = torch.randn(int(counts.sum()),model.vocab,device=device)
            # Finite expand-then-transport convention: new coordinates enter at s.
            x,births = insert_tokens(x,births,counts,noise,s)
            if len(x):
                padded,bt,mask=compact(x,births,torch.arange(len(x),device=device),model.length)
                y,_=local_map(model,padded[None],local_clock(start,bt[None]),
                              local_clock(end,bt[None]),bt[None],mask[None])
                x=y[0,:len(x)]
            trace.append(len(x))
        lengths[j]=len(x)
        if len(x):result[j,:len(x)]=x.argmax(-1)
        traces.append(trace)
    return result,lengths,{'capped_proposed_insertions':capped,'length_trajectories':traces}
