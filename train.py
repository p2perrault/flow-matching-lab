import torch
import argparse
import torch.nn as nn
import numpy as np
import time
import os
from pathlib import Path

from models import *
from utils import *
from data import *

def save_checkpoint(model, optimizer, epoch, dim, c, to_plot, save_path, DDIM):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "DDIM": DDIM,
        "dim": dim,
        "c": c,
        "loss_history": to_plot,
    }
    torch.save(checkpoint, save_path)
    print(f"\nModel saved to {save_path}")

def train(
    epochs,
    loader,
    dim,
    model,
    save_path,
    lr=1e-3,
    device='cuda:0',
    live_plot=False,
    checkpoint=None,
    c=0,
    print_frequency=50,
    DDIM=False,
):
    conditional = c > 0
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    start_epoch = 0
    to_plot = {'train loss': []}

    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        to_plot = checkpoint["loss_history"]
        print(f"Resuming from epoch {checkpoint['epoch']}")

    start_time = time.time()
    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        losses = []

        for iter_step, (x_1, y) in enumerate(loader):
            x_1 = x_1.to(device)
            if conditional:
                y = y.to(device)
            x_0 = torch.randn_like(x_1)
            batch_size = x_0.shape[0]
            extra_dims = (1,) * (x_0.dim() - 1)
            t = torch.rand((batch_size, *extra_dims), device=x_0.device)

            optimizer.zero_grad()
            if DDIM:
                alpha = ALPHA(1-t)
                x_t = torch.sqrt(alpha) * x_1 + torch.sqrt(1 - alpha) * x_0
                eps = model(t=t, x_t=x_t, cond=y if conditional else None)
                loss = loss_fn(eps, x_0)
            else:
                x_t = (1 - t) * x_0 + t * x_1
                v = model(t=t, x_t=x_t, cond=y if conditional else None)
                loss = loss_fn(v, x_1 - x_0)

            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            if (iter_step+1) % print_frequency == 0:
                elapsed = time.time() - start_time
                rate = iter_step / (time.time() - epoch_start)
                remaining = ((epochs - epoch) * len(loader) - iter_step) / rate
                print(
                    f"{epoch}/{epochs-1} "
                    f"{iter_step}/{len(loader)} "
                    f"[{format_time(elapsed)}<{format_time(remaining)}, {rate:5.2f}it/s] "
                    f"| loss={losses[-1]:.6f}"
                )

        with torch.no_grad():
            epoch_loss = np.mean(losses)
            to_plot['train loss'].append(epoch_loss)

        if live_plot:
            live_plot(to_plot)
    
    name = loader.__class__.__name__
    if conditional:
        name += '_cond'
    if DDIM:
        name += '_ddim'
    save_checkpoint(model, optimizer, epoch, dim, c, to_plot, save_path(name), DDIM)

def run_training(
    epochs=10,
    lr=1e-2,
    batch_size=4096,
    dataset="TwoMoons",
    device=None,
    live_plot=False,
    conditional=False,
    DDIM=False,
    model_config='{"model" : "MLP", "h" : 16}',
    resume=None,
    outdir='output',
    print_frequency=50
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # create output folder
    os.makedirs(outdir, exist_ok=True)
    save_path = lambda epoch: Path(outdir) / f"{epoch}.pth"

    # load dataset
    loader = eval(dataset)(batch_size=batch_size)

    # determine conditional
    dim = next(iter(loader))[0].shape[1:]
    c = 0
    if conditional:
        c = next(iter(loader))[1].shape[1:]
        if len(c)==0:
            c = 1
        elif len(c)==1:
            c = c[0]

    # load model
    model = load(model_config, dim, c)
    model = model.to(device)

    # checkpoint
    checkpoint = None
    if resume is not None:
        checkpoint = torch.load(save_path(resume), map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

    # training
    train(
        epochs=epochs,
        loader=loader,
        dim=dim,
        model=model,
        save_path=save_path,
        lr=lr,
        device=device,
        live_plot=live_plot,
        checkpoint=checkpoint,
        c=c,
        print_frequency=print_frequency,
        DDIM=DDIM,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--dataset", type=str, default="TwoMoons")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--live_plot", action="store_true")
    parser.add_argument("--conditional", action="store_true")
    parser.add_argument("--DDIM", action="store_true")
    parser.add_argument("--model_config", type=str, default='{"model" : "MLP", "h" : 16}')
    parser.add_argument("--resume", type=str)
    parser.add_argument("--outdir", type=str, default='output')
    parser.add_argument("--print_frequency", type=int, default=50)

    args = parser.parse_args()
    args.live_plot = live_plot_cli if args.live_plot else False

    run_training(**vars(args))