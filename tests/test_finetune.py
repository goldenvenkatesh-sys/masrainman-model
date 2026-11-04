"""Fine-tuning data handling and the promotion rule (no network)."""
import numpy as np
import pandas as pd

from masrainman import finetune as ft


class FakeStats:
    mean = np.full((3, 1, 1), 1000.0, np.float32)
    std = np.full((3, 1, 1), 5.0, np.float32)

    def norm(self, x):
        return (x - self.mean) / self.std

    def denorm(self, x):
        return x * self.std + self.mean


def test_saved_state_comes_back_within_float16_precision(tmp_path):
    st = FakeStats()
    x = (1000 + np.random.default_rng(0).normal(0, 5, (3, 4, 5))).astype(np.float32)
    ft.save_analysis(tmp_path, "2026-11-04T06", x, st)
    a = ft.Analyses(tmp_path, st)
    t = np.datetime64("2026-11-04T06", "h")
    assert a.has(t) and np.allclose(a.get(t), x, atol=0.02)


def test_windows_need_every_6h_state_and_respect_the_split(tmp_path):
    st = FakeStats()
    times = pd.date_range("2026-11-01T00", "2026-11-03T00", freq="6h").drop(pd.Timestamp("2026-11-02T18"))
    for t in times:
        ft.save_analysis(tmp_path, t, np.full((3, 2, 2), 1000.0, np.float32), st)
    a = ft.Analyses(tmp_path, st)
    inits = ft.windows_inside(a)
    assert np.datetime64("2026-11-01T06", "h") in inits
    assert all(not (t - ft.STEP <= np.datetime64("2026-11-02T18") <= t + 4 * ft.STEP) for t in inits)
    assert all(t + 4 * ft.STEP < np.datetime64("2026-11-02T00") for t in ft.windows_inside(a, end=np.datetime64("2026-11-02T00", "h")))


def rep(rain, field, n=5):
    return {"n": n, "rain_rmse": rain, "rain_ets": [None] * 3, "field": {"t2m": field}}


def test_promotion_rule():
    old = rep([10.0, 11.0, 12.0], [1.0, 1.0])
    assert ft.better(old, rep([9.0, 10.0, 11.0], [1.0, 1.005]))[0]
    assert not ft.better(old, rep([9.0, 10.0, 11.0], [1.1, 1.1]))[0]        # other fields worse
    assert not ft.better(old, rep([10.0, 11.0, 11.9], [0.9, 0.9]))[0]       # rain gain too small
    assert not ft.better(rep([10.0] * 3, [1.0], n=2), rep([5.0] * 3, [0.5]))[0]   # too few tests
