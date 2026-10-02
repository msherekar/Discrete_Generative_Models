"""MNIST tensors and image properties: the image side's data.py.

Split out of run_mnist.py for the same reason data.py is separate on the peptide
side -- the diagnostics load a modality's data without wanting its training
loop, and check_coupling.py imports from here rather than pulling in the U-Net.
"""
import torch
from torch.utils.data import TensorDataset


def image_properties(x):
    """[B,1,28,28] in [-1,1] -> [B,2] of (mean intensity, mirror symmetry).

    Both are exact functions of the image, like the residue counts on the
    peptide side, so evaluating a generated sample carries no oracle error.
    """
    ink = x.flatten(1).mean(1)
    mirror = torch.flip(x, dims=[-1])
    # Correlation with the left-right flip, normalized per image.
    a = (x - x.flatten(1).mean(1)[:, None, None, None]).flatten(1)
    b = (mirror - mirror.flatten(1).mean(1)[:, None, None, None]).flatten(1)
    symmetry = (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1)).clamp_min(1e-8)
    return torch.stack([ink, symmetry], dim=1)


def load_mnist(limit, data_dir):
    """Standardized properties alongside the images, as the peptide loader does."""
    from torchvision import datasets
    from torchvision.transforms import v2
    transform = v2.Compose([
        v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.5,), std=(0.5,)),
    ])
    mnist = datasets.MNIST(root=data_dir, train=True, download=True,
                           transform=transform)
    n = min(limit, len(mnist))
    x = torch.stack([mnist[i][0] for i in range(n)])
    props = image_properties(x)
    # Binary class from the first property, mirroring the peptide setup.
    c = (props[:, 0] > props[:, 0].median()).long()
    mean, std = props.mean(0), props.std(0, correction=0).clamp_min(1e-6)
    return TensorDataset(x, c, (props - mean) / std), {"mean": mean, "std": std}
