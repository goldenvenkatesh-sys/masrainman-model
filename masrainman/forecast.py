"""Make a 0-10 day forecast from an ERA5 start time (hindcast mode).

  python -m masrainman.forecast --ckpt D:/masrainman_runs/baseline/best.pt --init 2023-11-20T00 --days 10

Writes two netCDF files to <run_dir>/forecasts/:
  fc_<init>.nc           6-hourly, 0-240 h
  fc_<init>_hourly24.nc  hourly for the first 24 h (time-interpolated from the 6-h steps)

Hindcast mode takes the boundary ring from ERA5, i.e. "perfect boundaries".
Live forecasts take it from ECMWF open data instead (see live_forecast.py).
"""
import argparse

import numpy as np
import pandas as pd
import torch
import xarray as xr

from .config import CHANNEL_INFO, grid, processed_dir, run_dir
from .dataset import Archive, Stats, time_features, to_hour
from .model import load_checkpoint


@torch.no_grad()
def rollout(model, arch, stats, inits, n_steps):
    """Batched forecast for several start times. Returns normalised (B, n_steps, C, H, W)."""
    step = arch.step

    def batch(times):
        return torch.from_numpy(np.stack([stats.norm(arch.get(t)) for t in times]).astype(np.float32))

    prev, now = batch([t - step for t in inits]), batch(inits)
    outs = []
    for k in range(1, n_steps + 1):
        valid = [t + k * step for t in inits]
        tf = torch.from_numpy(np.stack([time_features(t) for t in valid]))
        nxt = model(prev, now, batch(valid), tf)
        outs.append(nxt.numpy())
        prev, now = now, nxt
    return np.stack(outs, 1)


def to_physical(cfg, stats, x):
    """Normalised -> physical units; rain becomes mm per step."""
    x = stats.denorm(x)
    if "tp" in cfg["channels"]:
        i = cfg["channels"].index("tp")
        x[..., i, :, :] = np.expm1(np.clip(x[..., i, :, :], 0, None))
    return x


def needed_times(arch, init, n_steps):
    return [init + k * arch.step for k in range(-1, n_steps + 1)]


def make_dataset(cfg, fields, init, lead_hours, note):
    lat, lon = grid(cfg)
    lsm = np.load(processed_dir(cfg) / "static.npy")[1]
    data = {"lsm": (("lat", "lon"), lsm, {"long_name": "land-sea mask"})}
    for i, name in enumerate(cfg["channels"]):
        info = CHANNEL_INFO[name]
        units = f"mm per {cfg['step_hours']} h" if name == "tp" else info["units"]
        data[name] = (("lead", "lat", "lon"), fields[:, i], {"units": units, "long_name": info["long"]})
    valid = pd.Timestamp(init) + pd.to_timedelta(lead_hours, unit="h")
    return xr.Dataset(data, coords={"lead": ("lead", lead_hours, {"units": "hours"}), "valid_time": ("lead", valid),
                                    "lat": lat, "lon": lon},
                      attrs={"init_time": str(init), "model": "MasRainMan AI", "note": note})


def hourly_first_day(cfg, fc6):
    """Linear interpolation between 6-h steps for 0-24 h. Rain is spread evenly
    over the hours of each step (so it becomes mm per hour)."""
    step = cfg["step_hours"]
    hours = np.arange(0, 25)
    src = np.arange(0, 25, step)
    f = fc6[: len(src)]
    out = np.empty((len(hours),) + f.shape[1:], np.float32)
    for j, h in enumerate(hours):
        k = min(h // step, len(src) - 2)
        w = (h - src[k]) / step
        out[j] = (1 - w) * f[k] + w * f[k + 1]
    if "tp" in cfg["channels"]:
        i = cfg["channels"].index("tp")
        k_end = np.clip(np.ceil(hours / step).astype(int), 0, len(src) - 1)
        out[:, i] = f[k_end, i] / step
    return hours, out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--init", required=True, help="start time, e.g. 2023-11-20T00")
    ap.add_argument("--days", type=float, default=10)
    args = ap.parse_args()

    model, cfg = load_checkpoint(args.ckpt)
    init = to_hour(args.init)
    year = pd.Timestamp(init).year
    arch = Archive(cfg, [year - 1, year, year + 1])
    stats = Stats(cfg)
    n_steps = int(args.days * 24 // cfg["step_hours"])
    missing = [t for t in needed_times(arch, init, n_steps) if not arch.has(t)]
    if missing:
        raise SystemExit(f"ERA5 missing for {len(missing)} times (first {missing[0]}); "
                         "hindcasts need data from init-6h to the last valid time.")

    run_and_save(cfg, model, arch, stats, init, n_steps, run_dir(cfg, "forecasts"),
                 "hindcast with ERA5 boundaries")


def run_and_save(cfg, model, arch, stats, init, n_steps, out, note):
    """Forecast from `init` using `arch` for the start and boundary data; write both netCDF files."""
    fc = to_physical(cfg, stats, rollout(model, arch, stats, [init], n_steps)[0])
    analysis = to_physical(cfg, stats, stats.norm(arch.get(init))[None])
    fields = np.concatenate([analysis, fc])
    lead = np.arange(n_steps + 1) * cfg["step_hours"]

    tag = pd.Timestamp(init).strftime("%Y%m%d%H")
    path = out / f"fc_{tag}.nc"
    make_dataset(cfg, fields, init, lead, note).to_netcdf(path)
    hours, hourly = hourly_first_day(cfg, fields)
    ds_h = make_dataset(cfg, hourly, init, hours, note + "; hourly values interpolated from 6-hourly model steps")
    if "tp" in cfg["channels"]:
        ds_h["tp"].attrs["units"] = "mm per h"
    ds_h.to_netcdf(out / f"fc_{tag}_hourly24.nc")
    print(f"wrote {path}")
    print(f"wrote {out / f'fc_{tag}_hourly24.nc'}")
    return path


if __name__ == "__main__":
    main()
