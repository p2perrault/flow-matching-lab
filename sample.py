import argparse
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os

from models import *
from utils import *

def sample(model, n_samples, dim, n_steps, device, c, DDIM):
    model.eval()
    x = torch.randn(n_samples, *dim, device=device)
    time_steps = torch.linspace(0., 1., n_steps + 1, device=device)
    cond = None
    if c > 0:
        cond = torch.randint(0, max(c,2), (n_samples, 1)).to(device)

    xs = [x.detach().cpu()]
    with torch.no_grad():
        for i in range(n_steps):
            x = model.step(
                x_t=x,
                t_start=time_steps[i],
                t_end=time_steps[i + 1],
                cond=F.one_hot(cond.squeeze(), num_classes=c) if c > 1 else cond,
                DDIM=DDIM,
            )
            xs.append(x.detach().cpu())
    return xs, cond, time_steps.cpu()

def save_samples(xs, cond=None, ts=None, save_dir="samples", cmap="gray", vmin=0, vmax=1):
    os.makedirs(save_dir, exist_ok=True)

    # Color palette for 2D points
    palette = ["red", "green", "blue", "orange", "purple", "brown", "cyan", "magenta"]

    for i, (t,x) in enumerate(zip(ts, xs)):

        # ---------- 2D points ----------
        if x.ndim == 2 and x.shape[1] == 2:
            fig, ax = plt.subplots()
            colors = [palette[lbl % len(palette)] for lbl in cond.squeeze().tolist()] if cond is not None else "blue"
            ax.scatter(x[:, 0], x[:, 1], s=5, c=colors)
            ax.set_xlim(-3, 3)
            ax.set_ylim(-3, 3)
            ax.set_aspect("equal")
            
            ax.set_title(f"t={t:.2f}")
            fig.savefig(
                f"{save_dir}/{i:03d}.png",
                bbox_inches="tight",
                pad_inches=0
            )
            plt.close(fig)

        # ---------- Image grids ----------
        elif x.ndim == 4:  # (N, C, H, W)
            N, C, H, W = x.shape
            ncols = int(N**0.5)
            nrows = (N + ncols - 1) // ncols

            # Large figure size for grids
            if N > 1:
                fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*2, nrows*2))
                axes = axes.flatten()
            else:
                dpi = 100
                fig = plt.figure(figsize=(W/dpi, H/dpi), dpi=dpi)
                fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
                ax = fig.add_axes([0,0,1,1])
                ax.axis("off")
                axes = [ax]

            # Display images and add conditional labels
            for j, ax in enumerate(axes[:N]):
                img = x[j].permute(1,2,0).squeeze()  # H,W,C -> H,W
                ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
                ax.axis("off")
                ax.text(0.5, -0.1, f"{cond[j].item() if cond is not None else ' '}", transform=ax.transAxes,
                            ha="center", va="top", fontsize=20)

            # Turn off extra axes
            for ax_extra in axes[N:]:
                ax_extra.axis("off")

            fig.suptitle(f"t={t:.2f}", fontsize=20)
            plt.tight_layout()
            fig.savefig(f"{save_dir}/{i:03d}.png")
            plt.close(fig)

def run_sampling(
    checkpoint="output/TwoMoons.pth",
    n_samples=300,
    n_steps=100,
    device="cuda" if torch.cuda.is_available() else "cpu",
    model_config='{"model" : "MLP", "h" : 16}',
    gif=None,
    slider=False,
):
    print(f"using {device}")
    img_dir = checkpoint.split('.')[0]
    checkpoint = torch.load(checkpoint, map_location=device, weights_only=False)
    dim = checkpoint["dim"]
    c = checkpoint["c"]
    model = load(model_config, dim, c).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    DDIM = checkpoint["DDIM"] if "DDIM" in checkpoint else False

    xs, cond, time_steps = sample(model, n_samples, dim, n_steps, device, c, DDIM)
    save_samples(xs, cond, time_steps, img_dir)
    if gif:
        create_bouncing_gif(img_dir, img_dir+'.gif')
    if slider:
        slider(img_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample from trained flow")
    parser.add_argument("--checkpoint", type=str, default="output/TwoMoons.pth")
    parser.add_argument("--gif", action="store_true")
    parser.add_argument("--n_samples", type=int, default=300)
    parser.add_argument("--n_steps", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model_config", type=str, default='{"model" : "MLP", "h" : 16}')
    parser.add_argument("--slider", action="store_true")

    args = parser.parse_args()
    args.slider = display_image_cli if args.slider else False
    run_sampling(**vars(args))