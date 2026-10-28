"""The network: a U-Net that steps the atmosphere forward by `step_hours`.

Inputs per step:  state(t-1), state(t), boundary ring of state(t+1), boundary mask,
                  static fields (orography, land-sea mask, lat, lon), time features.
Output:           state(t+1) = state(t) + scaled change. The boundary ring is then
                  overwritten with the boundary data, like a limited-area NWP model.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import boundary_mask, grid, processed_dir
from .dataset import Stats


def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU(),
        nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU(),
    )


class UNet(nn.Module):
    def __init__(self, cin, cout, base=32, mults=(1, 2, 4, 4)):
        super().__init__()
        chs = [base * m for m in mults]
        self.depth = len(chs) - 1
        self.inc = conv_block(cin, chs[0])
        self.downs = nn.ModuleList(conv_block(chs[i], chs[i + 1]) for i in range(self.depth))
        self.ups = nn.ModuleList(conv_block(chs[i + 1] + chs[i], chs[i]) for i in reversed(range(self.depth)))
        self.out = nn.Conv2d(chs[0], cout, 1)
        nn.init.zeros_(self.out.weight)  # start as "persistence" and learn the change
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        x = self.inc(x)
        skips = [x]
        for down in self.downs:
            x = down(F.avg_pool2d(x, 2))
            skips.append(x)
        skips.pop()
        for up in self.ups:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            x = up(torch.cat([x, skips.pop()], 1))
        return self.out(x)


class Stepper(nn.Module):
    def __init__(self, n_chan, static, mask, tend_scale, base=32):
        super().__init__()
        self.register_buffer("static", static[None])                    # (1,S,H,W)
        self.register_buffer("mask", mask[None, None])                  # (1,1,H,W)
        self.register_buffer("tend_scale", tend_scale[None, :, None, None])
        H, W = mask.shape
        m = 8
        self.pad = (0, math.ceil(W / m) * m - W, 0, math.ceil(H / m) * m - H)
        cin = 3 * n_chan + 1 + static.shape[0] + 4
        self.net = UNet(cin, n_chan, base=base)

    def forward(self, prev, now, next_boundary, tfeat):
        """All states normalised, shape (B,C,H,W). Only the ring of next_boundary is used."""
        B, C, H, W = now.shape
        bnd = next_boundary * self.mask
        x = torch.cat([
            prev, now, bnd,
            self.mask.expand(B, -1, H, W),
            self.static.expand(B, -1, H, W),
            tfeat[:, :, None, None].expand(B, -1, H, W),
        ], 1)
        x = F.pad(x, self.pad, mode="replicate")
        change = self.net(x)[..., :H, :W]
        pred = now + change * self.tend_scale
        return bnd + (1 - self.mask) * pred


def build_stepper(cfg):
    static = np.load(processed_dir(cfg) / "static.npy")                # orography (m), lsm
    lat, lon = grid(cfg)
    H, W = len(lat), len(lon)
    lat2 = np.broadcast_to(((lat - lat.mean()) / lat.std())[:, None], (H, W))
    lon2 = np.broadcast_to(((lon - lon.mean()) / lon.std())[None, :], (H, W))
    static = np.stack([static[0] / 1000.0, static[1], lat2, lon2]).astype(np.float32)
    stats = Stats(cfg)
    tend_scale = stats.diff_std / stats.std[:, 0, 0]
    return Stepper(len(cfg["channels"]), torch.from_numpy(static), torch.from_numpy(boundary_mask(cfg)),
                   torch.from_numpy(tend_scale.astype(np.float32)), base=cfg["model"]["base_channels"])


def load_checkpoint(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = build_stepper(ck["cfg"])
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck["cfg"]
