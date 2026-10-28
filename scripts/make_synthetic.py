"""Create a small fake dataset so the whole pipeline can be tested before ERA5 arrives.

  python scripts/make_synthetic.py --out D:/masrainman_synthetic
  python -m masrainman.train --config config.synthetic.yaml --epochs 1 --steps-per-epoch 20 --val-steps 5

The fields are moving waves with a daily cycle - not weather, just the right shapes.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from masrainman.config import grid, load_config, processed_dir  # noqa: E402
from masrainman.preprocess import compute_stats  # noqa: E402

# rough typical value and spread per channel
TYPICAL = {"t2m": (27, 4), "d2m": (20, 4), "u10": (0, 4), "v10": (0, 4), "msl": (1008, 4),
           "tcwv": (45, 10), "tp": (0.5, 0.8), "z500": (5850, 30), "t850": (18, 3), "q850": (12, 3),
           "u850": (0, 8), "v850": (0, 6), "u200": (-5, 15), "v200": (0, 8)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--days", type=int, default=30, help="days per fake year")
    args = ap.parse_args()

    cfg = load_config()
    cfg["data_dir"] = args.out
    cfg["run_dir"] = str(Path(args.out) / "runs")
    cfg["years"] = {"train": [2001, 2001], "val": [2002, 2002], "test": [2003, 2003]}
    with open(ROOT / "config.synthetic.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    lat, lon = grid(cfg)
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    rng = np.random.default_rng(0)
    proc = processed_dir(cfg)
    land = ((LA > 8) & (LA < 22) & (LO > 74) & (LO < 86)).astype(np.float32)
    np.save(proc / "static.npy", np.stack([land * 600, land]).astype(np.float32))

    step = cfg["step_hours"]
    for year in (2001, 2002, 2003):
        times = np.arange(np.datetime64(f"{year}-01-01T00"), np.datetime64(f"{year}-01-01T00") +
                          np.timedelta64(args.days * 24, "h"), np.timedelta64(step, "h"))
        data = np.empty((len(times), len(cfg["channels"]), len(lat), len(lon)), np.float32)
        for c, name in enumerate(cfg["channels"]):
            mean, spread = TYPICAL[name]
            kx, ky, phase = rng.uniform(0.1, 0.4), rng.uniform(0.1, 0.4), rng.uniform(0, 6.28)
            for i, t in enumerate(times):
                hours = (t - times[0]).astype(int)
                wave = np.sin(kx * LO + ky * LA - 0.05 * hours + phase)
                daily = 0.3 * np.sin(2 * np.pi * (hours % 24) / 24 + LO / 15) * land
                data[i, c] = mean + spread * (wave + daily)
        tp = cfg["channels"].index("tp")
        data[:, tp] = np.log1p(np.clip(data[:, tp], 0, None))
        np.save(proc / f"{year}.npy", data)
        np.save(proc / f"{year}_times.npy", times.astype("datetime64[h]"))
        print(f"wrote fake {year}: {data.shape}")
    compute_stats(cfg)
    print(f"config written to {ROOT / 'config.synthetic.yaml'}")


if __name__ == "__main__":
    main()
