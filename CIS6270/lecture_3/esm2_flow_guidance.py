"""Standalone ESM-2 flow matching guidance example for CIS 6270.
Input CSV: sequence,c,r1,r2. Sequences have one fixed length; c is 0 or 1.
Both objectives are oriented so larger is better. The bundled CSV is synthetic.
Run: python esm2_flow_guidance.py --data esm2_example.csv --epochs 200
Install: pip install torch transformers==4.57.6
Outputs: guided residue latents, model weights, and decoded amino-acid sequences.
"""
import argparse
import csv
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, EsmForMaskedLM

ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ESM_NAME = "facebook/esm2_t6_8M_UR50D"
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
POLAR_RESIDUES = "DEHKNQRST"  # Operational polar/charged set; not a solubility assay.
BATCH_SIZE, HIDDEN, LEARNING_RATE = 16, 128, 1e-3
CONDITION_DROP = 0.2  # Drop c during training so the same model learns the null condition.


def composition_proxies(sequences):
    # Transparent teaching rewards; these are not measured activity or solubility.
    return torch.tensor([
        [(sum(a in "KR" for a in s) - sum(a in "DE" for a in s)) / len(s),
         sum(a in POLAR_RESIDUES for a in s) / len(s)]
        for s in sequences
    ], dtype=torch.float32)


# 1. Load annotated sequences and encode frozen ESM-2 residue vectors.
@torch.no_grad()
def load_data(path):
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    sequences = [row["sequence"].strip().upper() for row in rows]
    if len(rows) < 4 or any(not s or set(s) - set(AMINO_ACIDS) for s in sequences):
        raise ValueError("Supply at least four sequences using the 20 standard amino acids")
    lengths = {len(s) for s in sequences}
    if len(lengths) != 1 or max(lengths) > 128:
        raise ValueError("This compact example requires one fixed sequence length, at most 128")
    c = torch.tensor([int(row["c"]) for row in rows], dtype=torch.long)
    # With sequence,c only, compute example rewards directly from residue counts.
    # Optional r1/r2 columns override them with supplied labels, such as measurements.
    if {"r1", "r2"}.issubset(rows[0]):
        r = torch.tensor([[float(row["r1"]), float(row["r2"])] for row in rows])
    else:
        r = composition_proxies(sequences)
    if set(c.tolist()) != {0, 1} or not torch.isfinite(r).all():
        raise ValueError("Include both c=0 and c=1, with finite r1 and r2")

    tokenizer = AutoTokenizer.from_pretrained(ESM_NAME, cache_dir=ROOT / ".esm2_cache")
    esm = EsmForMaskedLM.from_pretrained(
        ESM_NAME, cache_dir=ROOT / ".esm2_cache", use_safetensors=True
    ).to(DEVICE).eval().requires_grad_(False)
    encoded = []
    for start in range(0, len(sequences), BATCH_SIZE):
        tokens = tokenizer(sequences[start:start + BATCH_SIZE], return_tensors="pt")
        tokens = {key: value.to(DEVICE) for key, value in tokens.items()}
        hidden = esm.esm(**tokens).last_hidden_state
        encoded.append(hidden[:, 1:-1].cpu())  # Remove BOS/EOS; retain all L residue positions.
    z = torch.cat(encoded)                    # [N, L, D], with D=320 for this checkpoint.
    z_mean = z.mean((0, 1), keepdim=True)
    z_std = z.std((0, 1), correction=0, keepdim=True).clamp_min(1e-4)
    r_mean, r_std = r.mean(0), r.std(0, correction=0).clamp_min(1e-6)
    dataset = TensorDataset((z - z_mean) / z_std, c, (r - r_mean) / r_std)
    stats = {"z_mean": z_mean, "z_std": z_std, "r_mean": r_mean, "r_std": r_std}
    return dataset, esm, tokenizer, stats


# 2. Small model classes: flattening lets each output depend on the entire sequence.
class FlowModel(nn.Module):
    def __init__(self, length, dim):
        super().__init__()
        self.length, self.dim = length, dim
        self.time = nn.Sequential(nn.Linear(1, 32), nn.SiLU(), nn.Linear(32, 32))
        self.skip = nn.Linear(32, 1)           # Preserve full-dimensional state/noise through a time gate.
        self.condition = nn.Embedding(3, 16)    # Indices 0,1 are classes; 2 is the null class.
        self.net = nn.Sequential(
            nn.Linear(length * dim + 48, HIDDEN), nn.SiLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.SiLU(),
            nn.Linear(HIDDEN, length * dim),
        )

    def forward(self, z, t, c):
        time = self.time(t[:, None])
        inputs = torch.cat([z.flatten(1), time, self.condition(c)], dim=1)
        return self.skip(time)[:, :, None] * z + self.net(inputs).reshape_as(z)
        # A time-dependent linear part plus a learned nonlinear velocity correction.


class RewardModel(nn.Module):
    def __init__(self, length, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(length * dim + 1, HIDDEN), nn.SiLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.SiLU(), nn.Linear(HIDDEN, 2),
        )

    def forward(self, z, t):
        return self.net(torch.cat([z.flatten(1), t[:, None]], dim=1))
        # Two predicted standardized endpoint objectives, given the current latent and time.


# 3. Flow matching: Z_t=(1-t)Z_0+tZ_1; target velocity is Z_1-Z_0.
def train(dataset, epochs):
    _, length, dim = dataset.tensors[0].shape
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = FlowModel(length, dim).to(DEVICE)
    reward_model = RewardModel(length, dim).to(DEVICE)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(reward_model.parameters()), lr=LEARNING_RATE)
    for epoch in range(epochs):
        total = 0.0
        for z1, c, r_tilde in loader:
            z1, c, r_tilde = z1.to(DEVICE), c.to(DEVICE), r_tilde.to(DEVICE)
            z0 = torch.randn_like(z1)          # Flow convention: Z_0=noise, Z_1=clean latent.
            t = torch.rand(len(z1), device=DEVICE)
            zt = (1 - t[:, None, None]) * z0 + t[:, None, None] * z1
            dropped = c.masked_fill(torch.rand(len(c), device=DEVICE) < CONDITION_DROP, 2)
            loss_flow = F.mse_loss(model(zt, t, dropped), z1 - z0)
            loss_reward = F.mse_loss(reward_model(zt, t), r_tilde)
            loss = loss_flow + loss_reward    # Independent parameter sets; one optimizer suffices.
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item()
        if (epoch + 1) % 50 == 0 or epoch + 1 == epochs:
            print(f"epoch {epoch+1}: combined training loss {total / len(loader):.4f}")
    for network in (model, reward_model):
        network.eval().requires_grad_(False)   # Freeze weights; still allow gradients of input z.
        for parameter in network.parameters():
            parameter.grad = None
    return model, reward_model


# 4. R_lambda = sum_m lambda_m * r_tilde_m; differentiate the current latent only.
def reward_gradient(reward_model, z, t, lambdas):
    with torch.enable_grad():                  # Re-enable input gradients inside sampling.
        state = z.detach().requires_grad_(True)
        R_lambda = (reward_model(state, t) * lambdas).sum(dim=1)
        grad = torch.autograd.grad(R_lambda.sum(), state)[0]
    return grad.detach()                      # Do not retain graphs across sampling steps.


def normalize_weights(lambdas):
    values = torch.as_tensor(lambdas, dtype=torch.float32, device=DEVICE)
    if values.shape != (2,) or not torch.isfinite(values).all() or (values < 0).any() or values.sum() <= 0:
        raise ValueError("Use two finite, nonnegative weights with a positive sum")
    return values / values.sum()              # lambda controls tradeoffs, eta controls strength.


# 5. Sampling: CFG first; optional reward gradient then changes the velocity.
@torch.no_grad()
def sample(model, reward_model, n=8, c=1, w=0.0, eta=0.0, lambdas=(1.0, 0.0), steps=200):
    if c not in (0, 1) or steps < 1 or w < 0 or eta < 0:
        raise ValueError("Use c=0/1, positive steps, and nonnegative w and eta")
    lambdas = normalize_weights(lambdas)
    torch.manual_seed(123)                     # Compare methods from the same initial noise.
    z = torch.randn(n, model.length, model.dim, device=DEVICE)
    null = torch.full((n,), 2, dtype=torch.long, device=DEVICE)
    condition = torch.full((n,), c, dtype=torch.long, device=DEVICE)
    dt = 1.0 / steps
    for step in range(steps):
        t = torch.full((n,), step * dt, device=DEVICE)
        v_uncond = model(z, t, null)
        v = v_uncond
        if w != 0:
            v = v_uncond + w * (model(z, t, condition) - v_uncond)
        if eta != 0:
            grad_R = reward_gradient(reward_model, z, t, lambdas)
            kappa = eta * 4 * t[:, None, None] * (1 - t[:, None, None])
            v = v + kappa * grad_R             # Chosen steering rule; not exact conditional transport.
        z = z + dt * v                         # Integrate forward from t=0 to t=1.
    return z


# 6. One decoder, used only AFTER the flow or diffusion trajectory is complete.
@torch.no_grad()
def decode(z, esm, tokenizer, stats, min_polar=12):
    latent = z * stats["z_std"].to(z.device) + stats["z_mean"].to(z.device)
    logits = esm.lm_head(latent)               # Frozen head produces one vocabulary distribution per residue.
    aa_ids = torch.tensor(tokenizer.convert_tokens_to_ids(list(AMINO_ACIDS)), device=z.device)
    logits = logits.index_select(-1, aa_ids)   # Restrict the vocabulary to the 20 amino acids.
    if not 0 <= min_polar <= z.shape[1]:
        raise ValueError("min_polar must lie between zero and the sequence length")
    polar = torch.tensor([a in POLAR_RESIDUES for a in AMINO_ACIDS], device=z.device)
    polar_ids = polar.nonzero().flatten()
    best_scores, choices = logits.max(dim=-1)  # Start from unrestricted amino-acid argmax.
    polar_scores, local = logits[..., polar_ids].max(dim=-1)
    polar_choices = polar_ids[local]          # Best polar/charged amino acid at each position.
    for i in range(len(z)):
        already_polar = polar[choices[i]]
        missing = max(0, min_polar - int(already_polar.sum()))
        if missing:
            cost = (best_scores[i] - polar_scores[i]).masked_fill(already_polar, float("inf"))
            positions = cost.topk(missing, largest=False).indices
            choices[i, positions] = polar_choices[i, positions]
    return ["".join(AMINO_ACIDS[i] for i in row) for row in choices.cpu().tolist()]
    # Exact maximum-logit decode subject to at least min_polar selected residues.
    # This enforces composition, not measured solubility; no gradient through argmax.


# 7. Run CFG, single-objective steering, and scalarized multi-objective steering in order.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "esm2_example.csv")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--min-polar", type=int, default=12)
    parser.add_argument("--output", type=Path, default=ROOT / "esm2_flow_outputs")
    args = parser.parse_args()
    if args.epochs < 1 or args.samples < 1:
        parser.error("epochs and samples must be positive")
    torch.manual_seed(7)
    if DEVICE.type == "cpu":
        torch.set_num_threads(2)
    dataset, esm, tokenizer, stats = load_data(args.data)
    if not 0 <= args.min_polar <= dataset.tensors[0].shape[1]:
        parser.error("min-polar must lie between zero and the sequence length")
    model, reward_model = train(dataset, args.epochs)
    outputs = {
        "cfg": sample(model, reward_model, args.samples, c=1, w=2.0),
        "single": sample(model, reward_model, args.samples, eta=1.0, lambdas=(1.0, 0.0)),
        "multi": sample(model, reward_model, args.samples, eta=1.0, lambdas=(0.7, 0.3)),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for name, latent in outputs.items():
        if not torch.isfinite(latent).all():
            raise RuntimeError(f"Nonfinite {name} output; reduce guidance or check training")
        sequences = decode(latent, esm, tokenizer, stats, args.min_polar)
        assert all(sum(a in POLAR_RESIDUES for a in s) >= args.min_polar for s in sequences)
        fasta = "".join(f">{name}_{i+1}\n{seq}\n" for i, seq in enumerate(sequences))
        (args.output / f"{name}.fasta").write_text(fasta)
        count = sum(a in POLAR_RESIDUES for a in sequences[0])
        print(f"{name}: {sequences[0]}  polar/charged residues={count}")
        print("  decoded mean composition proxies:", composition_proxies(sequences).mean(0).tolist())
    torch.save({"standardized_latents": {k: v.cpu() for k, v in outputs.items()},
                "model": model.state_dict(), "reward_model": reward_model.state_dict(),
                "stats": stats, "esm_name": ESM_NAME,
                "length": model.length, "dim": model.dim, "min_polar": args.min_polar,
                "polar_residues": POLAR_RESIDUES}, args.output / "results.pt")
    print(f"Saved latent tensors and FASTA sequences to {args.output}")


if __name__ == "__main__":
    main()
