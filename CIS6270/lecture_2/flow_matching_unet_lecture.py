"""CIS 6270, Lecture 2: unconditional MNIST generation with flow matching.

Install: python -m pip install torch torchvision
Run: python flow_matching_unet_lecture.py
Notation: x0 = prior noise, x1 = data, xt = state, ut = target, v_pred = prediction.
"""

# %% 01. Imports
from pathlib import Path  # Dataset and output paths.

import torch  # Tensors, gradients, and optimization.
import torch.nn as nn  # Layers and model classes.
import torch.nn.functional as F  # Resize feature maps in the decoder.
from torch.utils.data import DataLoader  # Shuffle and batch real images.
from torchvision import datasets
from torchvision.transforms import v2  # Convert and normalize image pixels.
from torchvision.utils import save_image  # Export tensors as PNG grids.


# %% 02. Configuration
BATCH_SIZE = 128  # Images per training batch.
EPOCHS = 20  # Passes through the training set.
LEARNING_RATE = 2e-4  # Optimizer step size.
BASE_CHANNELS = 32  # U-Net widths: 32, 64, 128.
SAMPLE_STEPS = 100  # Euler steps from t=0 to t=1.
NUM_SAMPLES = 16  # Generate a 4-by-4 image grid.
SEED = 7  # Seed for initialization and training.
DATA_DIR = Path("data")  # MNIST download location.
OUTPUT_DIR = Path("flow_matching_outputs")  # Checkpoint and generated image folder.

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()  # NVIDIA GPU.
    else "mps" if torch.backends.mps.is_available()  # Apple GPU.
    else "cpu"  # CPU fallback.
)


# %% 03. Load real images
def make_loader():
    transform = v2.Compose([
        v2.ToImage(),  # Channel-first image: [1, 28, 28].
        v2.ToDtype(torch.float32, scale=True),  # Integer pixels 0-255 -> floats 0-1.
        v2.Normalize(mean=(0.5,), std=(0.5,)),  # Real data pixels: [0, 1] -> [-1, 1].
    ])
    dataset = datasets.MNIST(
        root=DATA_DIR, train=True, download=True, transform=transform,  # Use the training split.
    )
    return DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,  # Batch shape: [B, 1, 28, 28].
        num_workers=0, pin_memory=(DEVICE.type == "cuda"),  # Load in the main process.
    )


# %% 04. Sample the conditional path
def sample_conditional_path(x1):  # x1 contains normalized real images.
    x0 = torch.randn_like(x1)  # Independent Gaussian prior; same shape as x1.
    t = torch.rand(x1.shape[0], device=x1.device)  # One random time per image; [B].
    t_image = t[:, None, None, None]  # [B, 1, 1, 1]; share time across pixels.
    xt = (1.0 - t_image) * x0 + t_image * x1  # Noise at t=0; real image at t=1.
    ut = x1 - x0  # Derivative of the straight interpolation.
    return xt, t, ut  # Network input, time, and target velocity.


# %% 05. Define a simple convolution block
def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.SiLU(),  # Keep H and W.
        nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.SiLU(),  # Refine features.
    )


# %% 06. Define the small U-Net
class FlowUNet(nn.Module):
    def __init__(self, base_channels=BASE_CHANNELS):
        super().__init__()
        c = base_channels  # Base width; c=32 by default.
        self.encoder1 = conv_block(2, c)  # Image + time -> c channels.
        self.encoder2 = conv_block(c, 2 * c)  # Encoder: c -> 2c.
        self.middle = conv_block(2 * c, 4 * c)  # Bottleneck: 2c -> 4c.
        self.decoder2 = conv_block(6 * c, 2 * c)  # Upsampled 4c + skip 2c -> 2c.
        self.decoder1 = conv_block(3 * c, c)  # Upsampled 2c + skip c -> c.
        self.output = nn.Conv2d(c, 1, 1)  # One signed velocity per pixel.
        self.pool = nn.MaxPool2d(2)  # Halve the image height and width.

    # %% 07. Compute the velocity field
    def forward(self, xt, t):
        t_image = t[:, None, None, None].expand_as(xt)  # Constant time channel.
        x = torch.cat([xt, t_image], dim=1)  # [B, 2, 28, 28].
        skip1 = self.encoder1(x)  # Keep full-resolution features.
        skip2 = self.encoder2(self.pool(skip1))  # 28x28 -> 14x14.
        x = self.middle(self.pool(skip2))  # 14x14 -> 7x7.
        x = F.interpolate(x, size=skip2.shape[-2:], mode="nearest")  # 7x7 -> 14x14.
        x = self.decoder2(torch.cat([x, skip2], dim=1))  # Restore encoder detail.
        x = F.interpolate(x, size=skip1.shape[-2:], mode="nearest")  # 14x14 -> 28x28.
        x = self.decoder1(torch.cat([x, skip1], dim=1))  # Restore fine detail.
        v_pred = self.output(x)  # Predict v_theta(xt, t).
        return v_pred  # [B, 1, 28, 28], matching xt.


# %% 08. Train for one epoch
def train_one_epoch(model, loader, optimizer):
    model.train()  # Select training mode.
    total_loss, num_images = 0.0, 0  # Accumulate an image-weighted epoch loss.
    for x1, _ in loader:  # Ignore digit labels: unconditional generation.
        x1 = x1.to(DEVICE)  # Move real images to the model device.
        xt, t, ut = sample_conditional_path(x1)  # Fresh noise and times for this batch.
        v_pred = model(xt, t)  # The model receives only xt and t.
        squared_error = (v_pred - ut).square()  # Velocity error at every pixel.
        loss = squared_error.flatten(1).sum(1).mean()  # Sum pixels, average images.
        optimizer.zero_grad(set_to_none=True)  # Clear old gradients.
        loss.backward()  # Compute gradients with respect to weights.
        optimizer.step()  # Update the U-Net weights.
        total_loss += loss.item() * x1.shape[0]  # Weight by the actual batch size.
        num_images += x1.shape[0]  # Include the smaller final batch.
    return total_loss / num_images  # Average training loss per image.

# Save snapshots every quarter of the default 100-step run; keep ODE states unclipped.
# %% 09. Generate images with Euler integration
@torch.no_grad()  # Sampling needs no gradient tracking.
def sample_images(model, num_samples=NUM_SAMPLES, steps=SAMPLE_STEPS):
    if num_samples < 1 or steps < 1:
        raise ValueError("num_samples and steps must be positive")
    model.eval()  # Select evaluation mode.
    device = next(model.parameters()).device  # Use the model device.
    xt = torch.randn(num_samples, 1, 28, 28, device=device)  # Initial x0.
    dt = 1.0 / steps  # Positive time increment.
    snapshots = [xt.cpu().clone()]  # Keep the initial noise for plotting.
    for step in range(steps):  # Integrate forward in time.
        t = torch.full((num_samples,), step * dt, device=device)  # Shared time grid.
        v_pred = model(xt, t)  # Reevaluate velocity at the current state.
        xt = xt + dt * v_pred  # Euler: state += time increment * velocity.
        if (step + 1) % max(steps // 4, 1) == 0 or step + 1 == steps:
            snapshots.append(xt.cpu().clone())  # Save intermediate and final states.
    return xt, torch.stack(snapshots, dim=1)  # Raw images; [B, snapshots, 1, 28, 28].


# %% 10. Set up the complete run
def main():
    torch.manual_seed(SEED)  # Seed training randomness.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # Create the output folder.
    loader = make_loader()  # Download, normalize, and batch MNIST.
    model = FlowUNet().to(DEVICE)  # Create the velocity network.
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=0.0,  # Optimize the flow loss.
    )
    num_parameters = sum(p.numel() for p in model.parameters())  # Count weights and biases.
    print(f"device={DEVICE} parameters={num_parameters:,}")

    # %% 11. Train and save the learned parameters
    losses = []  # Record the training loss after each epoch.
    for epoch in range(1, EPOCHS + 1):  # Complete EPOCHS passes through the data.
        loss = train_one_epoch(model, loader, optimizer)  # Run the training loop.
        losses.append(loss)  # Keep the loss history.
        print(f"epoch={epoch:02d} squared_norm_loss={loss:.3f}")
    torch.save({
        "model": model.state_dict(), "losses": losses,  # Weights and training curve.
        "base_channels": BASE_CHANNELS,  # Width needed to reconstruct the model.
        "seed": SEED,  # Record the training seed.
    }, OUTPUT_DIR / "flow_unet_mnist.pt")

    # %% 12. Sample and save images
    torch.manual_seed(SEED + 1)  # Fix the starting noise for generation.
    samples, trajectory = sample_images(model)  # Integrate the learned ODE.
    display_samples = ((samples.cpu() + 1.0) / 2.0).clamp(0.0, 1.0)  # Clip for display only.
    save_image(display_samples, OUTPUT_DIR / "samples.png", nrow=4)  # Final images.
    display_path = ((trajectory[:4] + 1.0) / 2.0).clamp(0.0, 1.0)  # First four sample paths.
    save_image(
        display_path.flatten(0, 1), OUTPUT_DIR / "trajectory.png",  # One row per sample.
        nrow=trajectory.shape[1],  # Time increases across columns.
    )


# %% 13. Run the script
if __name__ == "__main__":  # Execute when launched directly.
    main()  # Load data, train, sample, and save.
