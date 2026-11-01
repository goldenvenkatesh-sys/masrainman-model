"""Cyclone detection and the extra training attention it drives."""
import numpy as np
import torch

from masrainman.cyclones import track
from masrainman.extremes import low_depth, pixel_weight, ramp, sample_weights


def dip(n=41, depth=10.0, at=(20, 20), width=2.0):
    y, x = np.mgrid[:n, :n]
    return (1010 - depth * np.exp(-((y - at[0]) ** 2 + (x - at[1]) ** 2) / (2 * width ** 2))).astype(np.float32)


def test_low_depth_finds_compact_lows_only():
    assert np.allclose(low_depth(np.full((41, 41), 1005, np.float32), 21), 0, atol=1e-4)
    trough = np.tile(np.linspace(1000, 1010, 41, dtype=np.float32), (41, 1))  # broad gradient
    assert np.abs(low_depth(trough, 21)[5:-5, 15:-15]).max() < 0.1
    d = low_depth(dip(), 21)
    assert np.unravel_index(d.argmax(), d.shape) == (20, 20) and 7 < d.max() < 10
    # torch in, torch out, same numbers
    assert torch.allclose(low_depth(torch.from_numpy(dip()), 21), torch.from_numpy(d))


def test_ramp_and_pixel_weight():
    np.testing.assert_allclose(ramp(np.array([0.0, 2.0, 5.0, 8.0, 20.0])), [0, 0, 0.5, 1, 1])
    w = pixel_weight(torch.from_numpy(np.stack([dip(), np.full((41, 41), 1008, np.float32)])), 21, 3.0)
    assert w.shape == (2, 1, 41, 41)
    assert w[0, 0, 20, 20] == 4 and w[0, 0, 20, 24] == 4  # centre and its surroundings
    assert w[0, 0, 0, 0] == 1 and (w[1] == 1).all()       # far away / no low


def test_sample_weights_boost_windows_holding_a_cyclone(cfg):
    class Arch:
        step = np.timedelta64(6, "h")
        times = np.arange(np.datetime64("2019-01-01T00"), np.datetime64("2019-01-03T00"), step)

        def get(self, t):
            x = np.zeros((len(cfg["channels"]), 9, 9), np.float32)
            x[cfg["channels"].index("msl")] = dip(9, 12.0, (4, 4), 1.0) if t == self.times[4] else 1008
            return x

    class DS:
        arch, rollout = Arch(), 1
        inits = list(Arch.times[1:-1])

    w, depth = sample_weights(DS(), cfg, 5.0)
    hit = [i for i, t in enumerate(DS.inits) if t in Arch.times[3:6]]  # windows that contain step 4
    assert (w[hit] > 5).all() and np.delete(w, hit).max() == 1


def test_track_follows_the_moving_low():
    lat = lon = np.arange(41) * 0.25
    msl = np.stack([dip(at=(10 + k, 10 + 2 * k)) for k in range(6)])
    inner = np.ones((41, 41), bool)
    tr = track(msl, 21, inner, lat, lon, (2.5, 2.5))
    np.testing.assert_allclose(tr[:, 0], (10 + np.arange(6)) * 0.25)
    np.testing.assert_allclose(tr[:, 1], (10 + 2 * np.arange(6)) * 0.25)
    assert (tr[:, 2] < 1001).all()
