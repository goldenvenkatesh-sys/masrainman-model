"""Score how well the model keeps and moves real cyclones.

  python -m masrainman.cyclones --ckpt D:/masrainman_runs/full/best.pt

Starts a 5-day forecast from the young low of each named cyclone below and follows the
deepest low near it (see extremes.py) in the forecast and in ERA5, step by step. Reports per storm the
deepest central pressure reached (model vs ERA5), how deep the low got relative to its
surroundings, and the mean track error in km over days 1-3. Writes cyclones.csv next
to the checkpoint. 2022 storms are validation cases, 2023 storms are the unseen test year.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from .config import boundary_mask, grid
from .dataset import Archive, Stats
from .extremes import box_cells, low_depth
from .forecast import needed_times, rollout, to_physical
from .model import load_checkpoint

# Start time and the ERA5 position of the young low (a weak depression) at that time.
# Bay of Bengal storms only: Arabian Sea ones such as Biparjoy (2023) stayed on the
# western boundary strip, where ERA5 drives the forecast, so they test nothing.
STORMS = {
    "Asani 2022": ("2022-05-07T12", 10.5, 90.5),
    "Sitrang 2022": ("2022-10-22T12", 14.5, 89.8),
    "Mandous 2022": ("2022-12-06T12", 7.8, 88.5),
    "Mocha 2023": ("2023-05-10T00", 8.5, 89.8),
    "Hamoon 2023": ("2023-10-21T12", 14.0, 87.2),
    "Michaung 2023": ("2023-12-01T00", 9.5, 86.0),
}
SEARCH_DEG = 2.0  # the low may move at most this far between 6-h steps of the track


def track(msl, cells, inner, lat, lon, start):
    """Follow the deepest low from `start` (lat, lon). msl (T, H, W) in hPa.
    Returns per step: (lat, lon, central msl, local depth)."""
    depth = low_depth(msl, cells)
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    out, (y, x) = [], start
    for k in range(len(msl)):
        near = inner & (np.abs(la - y) <= SEARCH_DEG) & (np.abs(lo - x) <= SEARCH_DEG)
        i = np.unravel_index(np.where(near, depth[k], -np.inf).argmax(), depth[k].shape)
        y, x = lat[i[0]], lon[i[1]]
        out.append((y, x, msl[k][i], depth[k][i]))
    return np.array(out)


def km(a, b):
    """Great-circle distance between (lat, lon) arrays."""
    p1, p2 = np.radians(a[:, 0]), np.radians(b[:, 0])
    dl = np.radians(b[:, 1] - a[:, 1])
    c = np.sin(p1) * np.sin(p2) + np.cos(p1) * np.cos(p2) * np.cos(dl)
    return 6371 * np.arccos(np.clip(c, -1, 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--days", type=float, default=5)
    args = ap.parse_args()

    model, cfg = load_checkpoint(args.ckpt)
    torch.set_num_threads(cfg["train"]["threads"])
    stats = Stats(cfg)
    lat, lon = grid(cfg)
    inner = boundary_mask(cfg) == 0
    cells, im = box_cells(cfg), cfg["channels"].index("msl")
    n_steps = int(args.days * 24 // cfg["step_hours"])
    days3 = int(3 * 24 // cfg["step_hours"])
    arch = Archive(cfg, [2021, 2022, 2023, 2024])

    rows = []
    for name, (init, y0, x0) in STORMS.items():
        t0 = np.datetime64(init, "h")
        if not all(arch.has(t) for t in needed_times(arch, t0, n_steps)):
            print(f"{name}: no data, skipped")
            continue
        fc = to_physical(cfg, stats, rollout(model, arch, stats, [t0], n_steps))[0, :, im]
        truth = np.stack([arch.get(t0 + (k + 1) * arch.step)[im] for k in range(n_steps)])
        start = (y0, x0)
        tm, tt = track(fc, cells, inner, lat, lon, start), track(truth, cells, inner, lat, lon, start)
        err = km(tm[:days3, :2], tt[:days3, :2]).mean()
        row = {"storm": name, "init": init, **{k: float(v) for k, v in {
               "era5_min_hpa": tt[:, 2].min(), "model_min_hpa": tm[:, 2].min(),
               "era5_max_depth_hpa": tt[:, 3].max(), "model_max_depth_hpa": tm[:, 3].max(),
               "track_err_km_day1_3": err}.items()}}
        rows.append(row)
        print(f"{name:16s} lowest MSLP model {row['model_min_hpa']:6.1f} vs ERA5 {row['era5_min_hpa']:6.1f} hPa | "
              f"depth {row['model_max_depth_hpa']:4.1f} vs {row['era5_max_depth_hpa']:4.1f} hPa | "
              f"track error days 1-3 {err:4.0f} km", flush=True)

    if not rows:
        raise SystemExit("No storm cases with data.")
    ratio = np.mean([r["model_max_depth_hpa"] / r["era5_max_depth_hpa"] for r in rows])
    track_err = np.mean([r["track_err_km_day1_3"] for r in rows])
    print(f"mean: model keeps {100 * ratio:.0f}% of the real depth, track error {track_err:.0f} km")
    out = Path(args.ckpt).parent / "cyclones.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.2f}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
