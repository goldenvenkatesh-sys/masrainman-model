"""The boundary zone: outer cells come from boundary data, a blend zone fades into the model."""
import numpy as np
import torch

from masrainman.config import boundary_mask
from masrainman.model import Stepper
from masrainman.train import loss_weight


def make(cfg):
    C = len(cfg["channels"])
    mask = torch.from_numpy(boundary_mask(cfg))
    return Stepper(C, torch.zeros(4, *mask.shape), mask, torch.ones(C), base=8), mask, C


def test_mask_profile(cfg):
    cfg["domain"]["boundary_width"] = 4
    m = boundary_mask(cfg)  # 9 x 9 grid
    np.testing.assert_allclose(m[4, :5], [1.0, 1.0, 1.0, 0.5, 0.0])  # 3 forced, 1 blended, then model
    assert m[4, 4] == 0 and (m == m.T).all() and m.min() == 0 and m.max() == 1


def test_starts_as_persistence_with_forced_ring(cfg):
    model, mask, C = make(cfg)
    now, prev, bnd = torch.randn(2, C, 9, 9), torch.randn(2, C, 9, 9), torch.randn(2, C, 9, 9)
    out = model(prev, now, bnd, torch.zeros(2, 4))
    assert out.shape == (2, C, 9, 9)
    forced, inside = mask == 1, mask == 0
    assert torch.allclose(out[..., forced], bnd[..., forced])
    assert torch.allclose(out[..., inside], now[..., inside])  # output layer starts at zero


def test_blend_zone_mixes_boundary_and_model(cfg):
    cfg["domain"]["boundary_width"] = 4
    model, mask, C = make(cfg)
    torch.nn.init.normal_(model.net.out.weight)
    now, bnd = torch.randn(1, C, 9, 9), torch.randn(1, C, 9, 9)
    out = model(now, now, bnd, torch.zeros(1, 4))
    w = mask[4, 3]  # a blend cell
    assert 0 < w < 1
    # neither pure boundary data nor the plain model prediction, but a mix of the two
    assert not torch.allclose(out[..., 4, 3], bnd[..., 4, 3])
    assert not torch.allclose(out[..., 4, 3], now[..., 4, 3])
    forced = mask == 1
    assert torch.allclose(out[..., forced], bnd[..., forced])


def test_loss_weight_ignores_forced_cells(cfg):
    w = loss_weight(cfg)[0, 0]
    m = torch.from_numpy(boundary_mask(cfg))
    assert (w[m == 1] == 0).all() and (w[m == 0] > 0).all()
    assert torch.isclose(w.mean(), torch.tensor(1.0))
