import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn
import numpy as np
import json

def load(model_config, dim, c):
    model_config = json.loads(model_config)
    model_name = model_config.pop("model")
    return eval(model_name)(dim=dim, c=c, **model_config)

def add_emb(x, emb):
    return x + emb[:, :, None, None]

# For DDIM
T = lambda t: t*.1 + t**2/2 * (20 - .1)
ALPHA = lambda t: torch.exp(-T(t)/2)

class Flow(nn.Module):
    def __init__(self):
        super().__init__()

    def step(self, x_t: Tensor, t_start: Tensor, t_end: Tensor, method='euler', cond=None, DDIM=False) -> Tensor:
        t_start = t_start.view(1, 1).expand(x_t.shape[0], *((1,) * (x_t.dim() - 1)))
        if cond is not None:
            cond = cond.view(x_t.shape[0], cond.shape[-1])
        if DDIM:
            alpha_start = ALPHA(1-t_start)
            alpha_end = ALPHA(1-t_end)
            x_0 = self(x_t=x_t, t=t_start, cond=cond)
            x_1 = (x_t - torch.sqrt(1 - alpha_start) * x_0) / torch.sqrt(alpha_start)
            return torch.sqrt(alpha_end) * x_1 + torch.sqrt(1 - alpha_end) * x_0
        else:
            if method == 'euler':
                return x_t + (t_end - t_start) * self(x_t=x_t, t=t_start, cond=cond)
            if method == 'midpoint':
                return x_t + (t_end - t_start) * self(t=t_start + (t_end - t_start) / 2, x_t= x_t + (t_end - t_start) / 2 * self(x_t=x_t, t=t_start, cond=cond), cond=cond)
        
class MLP(Flow):
    def __init__(self, dim, h=16, c = 0, n_layer=2):
        self.dim = dim
        self.flat_dim = np.prod(dim)
        self.c = c
        super().__init__()

        in_dim = self.flat_dim + 1 + c

        layers = []

        layers.append(nn.Linear(in_dim, h))
        layers.append(nn.ReLU())

        for _ in range(n_layer):
            layers.append(nn.Linear(h, h))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(h, self.flat_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, t: Tensor, x_t: Tensor, cond=None) -> Tensor:
        if cond is None:
            out = self.net(torch.cat((t.view(-1, 1), x_t.view(-1, self.flat_dim)), -1))
        else:
            out = self.net(torch.cat((t.view(-1, 1), x_t.view(-1, self.flat_dim), cond.view(-1, self.c)), -1))
        return out.view(-1, *self.dim)
      
class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)

    def forward(self, x):
        x = F.relu(self.norm1(self.conv1(x)))
        x = F.relu(self.norm2(self.conv2(x)))
        return x

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.SiLU(),
            nn.Linear(dim*4, dim)
        )
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2

        freqs = torch.exp(
            -torch.arange(half, device=device) * torch.log(torch.tensor(10000.0)) / half
        )

        args = t[:, None] * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        return self.mlp(emb)

class UNet(Flow):
    def __init__(self, in_ch=1, base=64, h=128, dim=(28,28), c=0):
        super().__init__()
        assert all(d % 8 == 0 for d in dim[-2:]), f"Spatial dims must be divisible by 8, got {dim[-2:]}"
        self.cond_dim = c

        # Time
        self.time_emb = TimeEmbedding(h)

        self.emb_proj1 = nn.Linear(128, 64)
        self.emb_proj2 = nn.Linear(128, 128)
        self.emb_proj3 = nn.Linear(128, 256)
        self.emb_proj_mid = nn.Linear(128, 512)

        # Encoder
        self.enc1 = Block(in_ch, base)
        self.enc2 = Block(base, base*2)
        self.enc3 = Block(base*2, base*4)

        # Downsampling
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.mid = Block(base*4, base*8)

        # Decoder
        self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.dec3 = Block(base*8, base*4)

        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.dec2 = Block(base*4, base*2)

        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.dec1 = Block(base*2, base)

        self.out = nn.Conv2d(base, in_ch, 1)

        if c > 0:
            self.cond_emb = nn.Linear(c, h)

    def forward(self, t: Tensor, x_t: Tensor, cond=None):

        emb = self.time_emb(t.view(-1))

        if cond is not None:
            emb = emb + self.cond_emb(cond.float())

        e1 = self.enc1(x_t)

        e1 = add_emb(e1, self.emb_proj1(emb))

        e2 = self.enc2(self.pool(e1))
        e2 = add_emb(e2, self.emb_proj2(emb))

        e3 = self.enc3(self.pool(e2))
        e3 = add_emb(e3, self.emb_proj3(emb))

        m = self.mid(self.pool(e3))
        m = add_emb(m, self.emb_proj_mid(emb))

        d3 = self.up3(m)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out(d1)
    
class Affine(Flow):
    def __init__(self, dim, c):
        super().__init__()
        self.dim = dim
        self.flat_dim = np.prod(dim)
        self.net = nn.Linear(1 + c, 2*self.flat_dim)

    def forward(self, x_t, t, cond):
        if cond is None:
            a, b = self.net(t).chunk(2, dim=-1)
        else:
            a, b = self.net(torch.cat([t, cond], dim=-1)).chunk(2, dim=-1)
        a = a.view(-1, *self.dim)
        b = b.view(-1, *self.dim)
        return a * x_t + b