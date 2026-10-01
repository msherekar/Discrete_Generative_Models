#!/usr/bin/env python3
"""Fréchet Inception Distance between generated images and real MNIST.

FID is the standard fidelity measure for image generation. Both sets of images
are passed through a fixed pretrained network, the resulting feature vectors are
summarized by their mean and covariance, and FID is the Fréchet distance between
those two Gaussians:

    FID = ||mu_g - mu_r||^2 + Tr(S_g + S_r - 2 (S_g S_r)^{1/2})

Lower is better; 0 means the two feature distributions match. Unlike the
saturation proxy used earlier in this study, it is sensitive to structure rather
than to brightness alone, so a uniformly bright blob cannot score well by
matching a summary statistic.

Two feature extractors, because the usual one is a poor fit for MNIST:

  inception  InceptionV3 pool3, the literature-standard 2048-d features. MNIST
             is 28x28 grayscale and has to be upscaled to 299x299 RGB first, so
             the absolute numbers are not comparable to FID on natural images --
             but they rank configurations consistently, which is what is needed.
  mnist      A small classifier trained here on MNIST, using its penultimate
             layer. Cheaper, and its features are actually about digit identity.

Report which one was used; do not mix the two scales in one table.

Usage:
  dgm-compute-fid --prefix mn --seeds 11 12 13 14 15 --etas 1 2 5 10
  dgm-compute-fid --prefix mn --features mnist --etas 1 5 20 50
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from dgm.common.paths import SHARED_DATA, project_dir

ROOT = project_dir()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODES = ("cfg", "single", "multi")


def load_real(data_dir, n):
    from torchvision import datasets
    from torchvision.transforms import v2
    tf = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
                     v2.Normalize((0.5,), (0.5,))])
    mnist = datasets.MNIST(root=str(data_dir), train=True, download=True, transform=tf)
    idx = torch.randperm(len(mnist))[:n]
    x = torch.stack([mnist[int(i)][0] for i in idx])
    y = torch.tensor([mnist[int(i)][1] for i in idx])
    return x, y


class SmallNet(nn.Module):
    """Classifier whose penultimate layer supplies MNIST-native features."""

    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU())
        self.head = nn.Linear(128, 10)

    def forward(self, x, features=False):
        h = self.body(x)
        return h if features else self.head(h)


def train_small_net(data_dir, epochs=2, cache=ROOT / "data" / "mnist_featnet.pt"):
    if cache.is_file():
        net = SmallNet().to(DEVICE)
        net.load_state_dict(torch.load(cache, map_location=DEVICE))
        return net.eval().requires_grad_(False)
    x, y = load_real(data_dir, 20000)
    net = SmallNet().to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(epochs):
        order = torch.randperm(len(x))
        for i in range(0, len(x), 256):
            b = order[i:i + 256]
            loss = F.cross_entropy(net(x[b].to(DEVICE)), y[b].to(DEVICE))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    with torch.no_grad():
        acc = (net(x[:2000].to(DEVICE)).argmax(1).cpu() == y[:2000]).float().mean()
    print(f"  feature network trained, train accuracy {acc:.3f}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), cache)
    return net.eval().requires_grad_(False)


def frechet(a, b):
    """Fréchet distance between the Gaussians fitted to two feature sets."""
    from scipy import linalg
    mu_a, mu_b = a.mean(0), b.mean(0)
    cov_a = np.cov(a, rowvar=False)
    cov_b = np.cov(b, rowvar=False)
    covmean, _ = linalg.sqrtm(cov_a.dot(cov_b), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu_a - mu_b
    return float(diff.dot(diff) + np.trace(cov_a) + np.trace(cov_b) - 2 * np.trace(covmean))


def features_mnist(net, images, batch=256):
    out = []
    with torch.no_grad():
        for i in range(0, len(images), batch):
            out.append(net(images[i:i + batch].to(DEVICE), features=True).cpu().numpy())
    return np.concatenate(out)


def fid_inception(generated, real):
    """torchmetrics FID: expects uint8 RGB, so rescale [-1,1] and repeat channels."""
    from torchmetrics.image.fid import FrechetInceptionDistance
    metric = FrechetInceptionDistance(feature=2048, normalize=False).to(DEVICE)

    def to_uint8(x):
        x = ((x.clamp(-1, 1) + 1) * 127.5).to(torch.uint8)
        return x.repeat(1, 3, 1, 1)

    for i in range(0, len(real), 128):
        metric.update(to_uint8(real[i:i + 128]).to(DEVICE), real=True)
    for i in range(0, len(generated), 128):
        metric.update(to_uint8(generated[i:i + 128]).to(DEVICE), real=False)
    return float(metric.compute())


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prefix", default="mn")
    p.add_argument("--outputs", type=Path, default=ROOT / "outputs")
    p.add_argument("--data-dir", type=Path, default=SHARED_DATA)
    p.add_argument("--variants", nargs="+", default=["st", "ep"])
    p.add_argument("--etas", nargs="+", default=["1", "2", "5", "10"])
    p.add_argument("--seeds", type=int, nargs="+", default=[11, 12, 13, 14, 15])
    p.add_argument("--methods", nargs="+", default=["flow", "diffusion"])
    p.add_argument("--features", default="mnist", choices=("mnist", "inception"))
    p.add_argument("--real-n", type=int, default=5000)
    p.add_argument("--outdir", type=Path, default=None)
    args = p.parse_args()
    outdir = args.outdir or ROOT / "plots" / f"{args.prefix}_sweep"
    outdir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    real, _ = load_real(args.data_dir, args.real_n)
    print(f"real reference: {len(real)} images, features = {args.features}")
    net = train_small_net(args.data_dir) if args.features == "mnist" else None
    real_feat = features_mnist(net, real) if net is not None else None

    rows = []
    print(f"\n{'method':<11}{'var':<5}{'eta':>5}  {'mode':<9}{'FID':>10}{'seeds':>7}")
    print("-" * 48)
    for method in args.methods:
        for v in args.variants:
            for e in args.etas:
                for mode in ("cfg", "single"):
                    per_seed = []
                    for s in args.seeds:
                        f = args.outputs / f"{args.prefix}_{v}_{e}_s{s}" / "results.pt"
                        if not f.is_file():
                            continue
                        saved = torch.load(f, weights_only=False, map_location="cpu")
                        img = saved["results"][method][mode]["images"]
                        if net is not None:
                            per_seed.append(frechet(features_mnist(net, img), real_feat))
                        else:
                            per_seed.append(fid_inception(img, real))
                    if not per_seed:
                        continue
                    m = float(np.mean(per_seed))
                    rows.append({"method": method, "variant": v, "eta": e,
                                 "mode": mode, "fid": m, "n_seeds": len(per_seed),
                                 "features": args.features})
                    print(f"{method:<11}{v:<5}{e:>5}  {mode:<9}{m:>10.2f}"
                          f"{len(per_seed):>7}")
    if not rows:
        raise SystemExit("no runs found")
    path = outdir / f"{args.prefix}_fid_{args.features}.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"\nWrote {path}")
    print("cfg rows are the unguided reference: FID rising above them is the cost "
          "of guidance.")


if __name__ == "__main__":
    main()
