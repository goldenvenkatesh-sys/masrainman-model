"""Reading processed data: archive lookup by time, normalisation, training windows."""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import processed_dir


def to_hour(t):
    return np.datetime64(t, "h")


def time_features(t):
    """Time of day and season as sin/cos pairs."""
    ts = pd.Timestamp(t)
    h = 2 * np.pi * ts.hour / 24
    d = 2 * np.pi * (ts.dayofyear - 1) / 365.25
    return np.array([np.sin(h), np.cos(h), np.sin(d), np.cos(d)], np.float32)


class Archive:
    """All processed years, addressable by valid time. Arrays are memory-mapped,
    so the whole record never has to fit in RAM."""

    def __init__(self, cfg, years):
        proc = processed_dir(cfg)
        self.step = np.timedelta64(cfg["step_hours"], "h")
        self.paths, self.lookup = [], {}
        for year in years:
            path = proc / f"{year}.npy"
            if not path.exists():
                continue
            k = len(self.paths)
            self.paths.append(path)
            for i, t in enumerate(np.load(proc / f"{year}_times.npy")):
                self.lookup[to_hour(t)] = (k, i)
        if not self.lookup:
            raise RuntimeError(f"No processed data for years {list(years)} in {proc}")
        self.times = np.array(sorted(self.lookup))
        self._arrays = None

    def __getstate__(self):  # DataLoader workers re-open the memmaps themselves
        state = self.__dict__.copy()
        state["_arrays"] = None
        return state

    def has(self, t):
        return to_hour(t) in self.lookup

    def get(self, t):
        if self._arrays is None:
            self._arrays = [np.load(p, mmap_mode="r") for p in self.paths]
        k, i = self.lookup[to_hour(t)]
        return np.asarray(self._arrays[k][i], dtype=np.float32)


class Stats:
    def __init__(self, cfg):
        s = np.load(processed_dir(cfg) / "stats.npz")
        if list(s["channels"]) != list(cfg["channels"]):
            raise ValueError("stats.npz was made for different channels - rerun preprocessing")
        self.mean = s["mean"][:, None, None]
        self.std = s["std"][:, None, None]
        self.diff_std = s["diff_std"]

    def norm(self, x):
        return (x - self.mean) / self.std

    def denorm(self, x):
        return x * self.std + self.mean


class Windows(Dataset):
    """Sequences of states t-1, t, t+1 ... t+rollout (normalised) for training."""

    def __init__(self, cfg, years, rollout=1):
        self.arch = Archive(cfg, years)
        self.stats = Stats(cfg)
        self.rollout = rollout
        step = self.arch.step
        self.inits = [t for t in self.arch.times
                      if all(self.arch.has(t + k * step) for k in range(-1, rollout + 1))]

    def __len__(self):
        return len(self.inits)

    def __getitem__(self, i):
        t, step = self.inits[i], self.arch.step
        seq = np.stack([self.stats.norm(self.arch.get(t + k * step)) for k in range(-1, self.rollout + 1)])
        tf = np.stack([time_features(t + k * step) for k in range(1, self.rollout + 1)])
        return torch.from_numpy(seq), torch.from_numpy(tf)
