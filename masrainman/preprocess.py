"""Turn downloaded ERA5 netCDF files into model-ready arrays.

Writes to <data_dir>/processed/:
  {year}.npy         float32 (time, channel, lat, lon) in friendly units
  {year}_times.npy   datetime64[h] valid times
  static.npy         (orography in m, land-sea mask)
  stats.npz          per-channel mean/std and std of one-step changes (train years)

Usage:
  python -m masrainman.preprocess
  python -m masrainman.preprocess --years 2022 2023
"""
import argparse

import numpy as np
import xarray as xr

from .config import CHANNEL_INFO, G, all_years, grid, load_config, processed_dir, raw_dir, year_range


def open_nc(path):
    ds = xr.open_dataset(path)
    # The new CDS writes `valid_time` / `pressure_level`; older files use `time` / `level`.
    rename = {k: v for k, v in {"valid_time": "time", "pressure_level": "level"}.items() if k in ds.variables}
    ds = ds.rename(rename)
    ds = ds.drop_vars([v for v in ("expver", "number") if v in ds.variables])
    return ds.sortby("latitude", ascending=False).sortby("longitude")


def check_grid(cfg, ds, path):
    lat, lon = grid(cfg)
    if ds.sizes["latitude"] != len(lat) or ds.sizes["longitude"] != len(lon) \
            or not np.allclose(ds.latitude, lat) or not np.allclose(ds.longitude, lon):
        raise ValueError(f"{path} grid does not match the domain in config.yaml")


def process_static(cfg):
    path = raw_dir(cfg) / "static.nc"
    ds = open_nc(path)
    check_grid(cfg, ds, path)
    orog = ds["z"].squeeze().values / G
    lsm = ds["lsm"].squeeze().values
    np.save(processed_dir(cfg) / "static.npy", np.stack([orog, lsm]).astype(np.float32))


def process_year(cfg, year, tp_tail=None):
    """Returns the last hours of hourly rain so the next year can continue the sums."""
    step = cfg["step_hours"]
    chans = cfg["channels"]
    src = raw_dir(cfg) / str(year)
    paths = {k: src / f"{k}_{year}.nc" for k in ("sfc", "pl", "tp")}
    if not all(p.exists() for p in paths.values()):
        print(f"  {year}: files missing, skipped")
        return None
    # Files hold a whole year; xarray loads lazily, so we read one month at a time.
    sfc_y, pl_y, tp_y = open_nc(paths["sfc"]), open_nc(paths["pl"]), open_nc(paths["tp"])["tp"]
    check_grid(cfg, sfc_y, paths["sfc"])
    frames, times = [], []
    for month in range(1, 13):
        tag = f"{year}{month:02d}"
        sfc = sfc_y.sel(time=sfc_y.time.dt.month == month)
        pl = pl_y.sel(time=pl_y.time.dt.month == month)
        t_inst = sfc.time.values
        if len(t_inst) == 0:
            continue

        # Rain over the `step` hours ending at each time. ERA5 hourly tp at hour h
        # is the total for (h-1, h], in metres.
        tp_h = tp_y.sel(time=tp_y.time.dt.month == month)
        if tp_tail is not None:
            tp_h = xr.concat([tp_tail, tp_h], "time")
        acc = (tp_h.rolling(time=step).sum() * 1000.0).reindex(time=t_inst)
        tp_tail = tp_h.isel(time=slice(len(tp_h.time) - (step - 1), None)) if step > 1 else None

        frame = np.empty((len(t_inst), len(chans), sfc.sizes["latitude"], sfc.sizes["longitude"]), np.float32)
        for ci, name in enumerate(chans):
            info = CHANNEL_INFO[name]
            if name == "tp":
                a = np.log1p(np.clip(acc.values, 0, None))
            elif info["level"] is None:
                a = sfc[info["var"]].values
            else:
                a = pl[info["var"]].sel(level=info["level"]).values
            frame[:, ci] = info["convert"](a) if info["convert"] else a
            if name != "tp" and np.isnan(frame[:, ci]).all():
                raise ValueError(f"{tag}: {name} is missing for the whole month - the raw file is "
                                 "incomplete; delete it and download it again")

        ok = ~np.isnan(frame).any(axis=(1, 2, 3))  # first steps of the record have no rain history
        frames.append(frame[ok])
        times.append(t_inst[ok])
        print(f"  {tag}: {ok.sum()} times")

    if frames:
        out = processed_dir(cfg)
        np.save(out / f"{year}.npy", np.concatenate(frames))
        np.save(out / f"{year}_times.npy", np.concatenate(times).astype("datetime64[h]"))
    return tp_tail


def compute_stats(cfg, stride=4):
    """Channel mean/std over the training years, plus the std of one-step changes
    (used to scale what the network predicts)."""
    proc = processed_dir(cfg)
    step = np.timedelta64(cfg["step_hours"], "h")
    C = len(cfg["channels"])
    s, ss, n = np.zeros(C), np.zeros(C), 0
    ds, dss, dn = np.zeros(C), np.zeros(C), 0
    for year in year_range(cfg["years"]["train"]):
        path = proc / f"{year}.npy"
        if not path.exists():
            continue
        data = np.load(path, mmap_mode="r")
        times = np.load(proc / f"{year}_times.npy")
        idx = np.arange(0, len(data) - 1, stride)
        x = np.asarray(data[idx], dtype=np.float64)
        s += x.sum((0, 2, 3))
        ss += (x ** 2).sum((0, 2, 3))
        n += x.shape[0] * x.shape[2] * x.shape[3]
        idx = idx[times[idx + 1] - times[idx] == step]
        d = np.asarray(data[idx + 1], dtype=np.float64) - np.asarray(data[idx], dtype=np.float64)
        ds += d.sum((0, 2, 3))
        dss += (d ** 2).sum((0, 2, 3))
        dn += d.shape[0] * d.shape[2] * d.shape[3]
    if n == 0:
        raise RuntimeError("No processed training years found - run preprocessing first.")
    mean = s / n
    std = np.sqrt(np.maximum(ss / n - mean ** 2, 1e-12))
    dmean = ds / dn
    diff_std = np.sqrt(np.maximum(dss / dn - dmean ** 2, 1e-12))
    np.savez(proc / "stats.npz", mean=mean.astype(np.float32), std=std.astype(np.float32),
             diff_std=diff_std.astype(np.float32), channels=np.array(cfg["channels"]))
    for c, m, sd, dsd in zip(cfg["channels"], mean, std, diff_std):
        print(f"  {c:5s} mean {m:9.2f}  std {sd:8.3f}  step-change std {dsd:7.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--years", type=int, nargs=2, metavar=("FIRST", "LAST"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    years = range(args.years[0], args.years[1] + 1) if args.years else all_years(cfg)
    process_static(cfg)
    tail = None
    for year in years:
        print(f"{year}:")
        tail = process_year(cfg, year, tail)
    print("statistics:")
    compute_stats(cfg)


if __name__ == "__main__":
    main()
