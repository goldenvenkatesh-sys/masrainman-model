"""The stepper must keep the boundary ring equal to the boundary data."""
import torch

from masrainman.config import boundary_mask
from masrainman.model import Stepper


def make(cfg):
    C = len(cfg["channels"])
    mask = torch.from_numpy(boundary_mask(cfg))
    return Stepper(C, torch.zeros(4, *mask.shape), mask, torch.ones(C), base=8), mask, C


def test_starts_as_persistence_with_forced_ring(cfg):
    model, mask, C = make(cfg)
    now, prev, bnd = torch.randn(2, C, 9, 9), torch.randn(2, C, 9, 9), torch.randn(2, C, 9, 9)
    out = model(prev, now, bnd, torch.zeros(2, 4))
    assert out.shape == (2, C, 9, 9)
    ring, inside = mask.bool(), ~mask.bool()
    assert torch.allclose(out[..., ring], bnd[..., ring])
    assert torch.allclose(out[..., inside], now[..., inside])  # output layer starts at zero


def test_ring_forced_after_training_changes(cfg):
    model, mask, C = make(cfg)
    torch.nn.init.normal_(model.net.out.weight)
    now, bnd = torch.randn(1, C, 9, 9), torch.randn(1, C, 9, 9)
    out = model(now, now, bnd, torch.zeros(1, 4))
    ring = mask.bool()
    assert torch.allclose(out[..., ring], bnd[..., ring])
    assert not torch.allclose(out[..., ~ring], now[..., ~ring])
