#!/usr/bin/env python3
"""Does the reward model still work where the generator actually samples?

Guidance climbs the reward model's prediction, so a guidance result means
nothing unless that prediction tracks the property ON GENERATED LATENTS. The
reward head is trained on real encoded sequences, and a generator that lands
off that manifold gets a confident prediction that carries no information.

Measured on outputs/esm2_8m_calib2 (60 epochs, transformer, hidden 256):

    reward head vs true r1 on REAL      latents:  +0.834 flow, +0.867 diffusion
    reward head vs METL oracle on GEN   latents:  -0.186 flow, +0.003 diffusion

so the head had learned the task almost perfectly and was still useless for
steering. The visible symptom was reward hacking exactly as Lecture 3.4
describes it: over eta 30 -> 100 -> 300 the flow's predicted brightness climbed
-0.230 -> +0.157 -> +0.680 while oracle brightness stayed flat near -1.50.

Run this before trusting any guidance comparison. A generated-latent
correlation near zero means the guidance arms are measuring nothing, however
clean their error bars look.

  python diagnostics/check_reward_transfer.py outputs/esm2_8m_calib2
  python diagnostics/check_reward_transfer.py outputs/esm2_8m_calib2 --n 3000
"""
import argparse
import csv
import random
from pathlib import Path


import numpy as np
import torch
from scipy.stats import spearmanr

from dgm.project1 import run_experiment as R

from dgm.common.paths import project_dir

ROOT = project_dir()

# Below this, guidance arms are not measuring the property they claim to.
USABLE = 0.5


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path,
                        help="A run directory holding flow/ and/or diffusion/.")
    parser.add_argument("--dataset", type=Path,
                        default=ROOT / "data" / "avgfp_train_props.csv",
                        help="CSV the reward head was trained on.")
    parser.add_argument("--n", type=int, default=1500,
                        help="Real sequences to sample for the reference check.")
    parser.add_argument("--esm", default="facebook/esm2_t6_8M_UR50D")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--arch", default="transformer")
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def encode(sequences, esm_id, cache, device, batch=64):
    from transformers import AutoTokenizer, EsmForMaskedLM
    tokenizer = AutoTokenizer.from_pretrained(esm_id, cache_dir=str(cache))
    esm = EsmForMaskedLM.from_pretrained(
        esm_id, cache_dir=str(cache), use_safetensors=True).to(device).eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(sequences), batch):
            toks = tokenizer(sequences[start:start + batch], return_tensors="pt")
            toks = {k: v.to(device) for k, v in toks.items()}
            out.append(esm.esm(**toks).last_hidden_state[:, 1:-1].cpu())
    return torch.cat(out)


def predict(reward, z, device, batch=64):
    with torch.no_grad():
        return torch.cat([
            reward(z[i:i + batch].to(device),
                   torch.ones(len(z[i:i + batch]), device=device))[:, 0].cpu()
            for i in range(0, len(z), batch)
        ]).numpy()


def main():
    args = parse_args()
    device = R.DEVICE

    rows = list(csv.DictReader(args.dataset.open(newline="")))
    random.seed(args.seed)
    rows = random.sample(rows, min(args.n, len(rows)))
    true = np.array([float(r["r1"]) for r in rows])
    print(f"  encoding {len(rows)} real sequences from {args.dataset.name}...")
    z_real = encode([r["sequence"] for r in rows], args.esm,
                    ROOT / "cache", device)

    worst = 1.0
    for method in ("flow", "diffusion"):
        path = args.run_dir / method / "results.pt"
        if not path.is_file():
            continue
        saved = torch.load(path, weights_only=False)
        stats = saved["stats"]
        reward = R.RewardModel(saved["length"], saved["dim"], args.hidden,
                               args.arch, saved["reward_model"]["head.bias"].numel()
                               if "head.bias" in saved["reward_model"] else 3).to(device)
        reward.load_state_dict(saved["reward_model"])
        reward.eval()
        mean, std = float(stats["r_mean"][0]), float(stats["r_std"][0])

        zs = (z_real - stats["z_mean"]) / stats["z_std"]
        on_real = spearmanr(predict(reward, zs, device) * std + mean, true).statistic

        predicted, observed, gen_z, gen_seqs = [], [], [], []
        for arm, z in saved["standardized_latents"].items():
            predicted += (predict(reward, z, device) * std + mean).tolist()
            observed += list(np.asarray(saved["oracle_brightness"][arm]))
            gen_z.append(z)
            gen_seqs += list(saved["sequences"][arm])
        on_generated = spearmanr(predicted, observed).statistic
        worst = min(worst, on_generated)

        # Decoding is a projection onto "exactly k substitutions from the
        # permitted support", not a small perturbation. Re-encode what actually
        # came out and measure how far it landed from the latent guidance was
        # steering. On calib3 this was 77.2, about 27.5% of |z| and roughly ten
        # times the distance at which the decoder's output changes at all: the
        # reward head scored +0.0001 against the oracle on the generated latent
        # and +0.7056 on the re-encoding of that latent's own decoded sequence.
        # The head is fine. It is being asked about a point the output no longer
        # corresponds to.
        gen_z = torch.cat(gen_z)
        z_back = encode(gen_seqs, args.esm, ROOT / "cache", device)
        z_back = (z_back - stats["z_mean"]) / stats["z_std"]
        trip = float((gen_z - z_back).flatten(1).norm(dim=1).mean())
        scale = float(gen_z.flatten(1).norm(dim=1).mean())
        on_decoded = spearmanr(predict(reward, z_back, device) * std + mean,
                               observed).statistic

        print(f"\n  {method}")
        print(f"    reward head vs true r1,   REAL latents      : {on_real:+.4f}")
        print(f"    reward head vs oracle,    GENERATED latents : {on_generated:+.4f}"
              f"   ({len(predicted)} samples)")
        print(f"    reward head vs oracle,    DECODED+re-encoded: {on_decoded:+.4f}")
        print(f"    decode round-trip distance                  : {trip:.1f}"
              f"  ({trip / max(scale, 1e-9):.1%} of |z| = {scale:.1f})")
        decode_explains = on_decoded >= USABLE > on_generated
        if decode_explains:
            print(f"    -> the reward head works on the sequence that actually comes "
                  f"out, and not on the latent guidance steers. The decode is a "
                  f"projection, not a perturbation, so the guidance signal does not "
                  f"survive it. Loosening the decoder (--mut-budget, "
                  f"--restrict-support) shrinks the projection; tightening it "
                  f"cannot be fixed by raising eta.")
        if on_generated < USABLE <= on_real and not decode_explains:
            print(f"    -> the head learned the task and does not transfer to its "
                  f"own generator's output.")
            print(f"       Guidance here optimizes a signal uncorrelated with "
                  f"brightness. Bring the generator back toward the data manifold "
                  f"(--anchor-strength), score the predicted endpoint rather than "
                  f"the noisy state (--endpoint-guidance), or train longer, and "
                  f"re-check before reading any guidance arm.")
        elif on_generated >= USABLE:
            print(f"    -> usable: guidance arms measure the property they claim to.")

    print()
    if worst < USABLE:
        print(f"VERDICT: generated-latent correlation {worst:+.4f} is below "
              f"{USABLE}. Guidance comparisons from this run are not meaningful.")
        return 1
    print(f"VERDICT: reward signal transfers to generated samples.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
