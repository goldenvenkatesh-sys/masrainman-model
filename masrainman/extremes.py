"""Finding cyclones (deep, compact lows) so training can pay them extra attention.

A cyclone shows up as a pressure minimum much lower than its surroundings. The "low depth"
of a cell is the mean MSLP of the 5 x 5 degree box around it minus its own MSLP: ~1-2 hPa
on ordinary days, 5-25 hPa inside a cyclone. Broad monsoon troughs barely register, because
their surroundings are low as well. Only ~3% of 2018-2021 samples hold a cyclone, so an
MSE-trained model learns to fill them in; these helpers let training sample those cases
more often and weight the cells around them more.
"""
import numpy as np
import torch
import torch.nn.functional as F

from .config import boundary_mask

BOX_DEG = 5.0
DEPTH_START = 2.0   # hPa: below this nothing is boosted
DEPTH_FULL = 8.0    # hPa: full boost from here (a well-formed cyclone)


def box_cells(cfg):
    return int(round(BOX_DEG / cfg["domain"]["grid"])) // 2 * 2 + 1


def low_depth(msl, cells):
    """msl (..., H, W) in hPa, torch or numpy -> low depth in hPa, same shape and type."""
    is_np = isinstance(msl, np.ndarray)
    x = torch.from_numpy(np.array(msl, np.float32)) if is_np else msl.float()
    shape = x.shape
    x = x.reshape(-1, 1, *shape[-2:])
    r = cells // 2
    mean = F.avg_pool2d(F.pad(x, (r, r, r, r), mode="replicate"), cells, stride=1)
    d = (mean - x).reshape(shape)
    return d.numpy() if is_np else d


def ramp(depth):
    """0 below DEPTH_START, 1 above DEPTH_FULL, linear between."""
    return ((depth - DEPTH_START) / (DEPTH_FULL - DEPTH_START)).clip(0, 1)


def sample_weights(ds, cfg, boost):
    """One weight per training window: 1 + boost x (how cyclonic its deepest state is).
    All states in the window count, so cases where a low deepens are boosted too."""
    arch, ch = ds.arch, cfg["channels"]
    im, cells = ch.index("msl"), box_cells(cfg)
    inner = boundary_mask(cfg) == 0
    depth = {}
    for t in arch.times:
        depth[t] = low_depth(arch.get(t)[im], cells)[inner].max()
    steps = range(-1, ds.rollout + 1)
    d = np.array([max(depth[t + k * arch.step] for k in steps) for t in ds.inits])
    return 1 + boost * ramp(d), d


def pixel_weight(target_msl_hpa, cells, gain):
    """(B, 1, H, W) extra loss weight around deep lows in the target: 1 + gain x ramp(depth),
    spread over the surrounding box so winds and rain around the centre count as well."""
    w = ramp(low_depth(target_msl_hpa, cells))[:, None]
    spread = cells // 2 | 1  # about 2.5 degrees around the centre
    r = spread // 2
    w = F.max_pool2d(F.pad(w, (r, r, r, r), mode="replicate"), spread, stride=1)
    return 1 + gain * w
