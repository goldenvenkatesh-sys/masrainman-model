"""ERA5 files in the new CDS layout -> processed arrays, stats and training windows."""
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from masrainman.config import grid, processed_dir
from masrainman.dataset import Windows
from masrainman.preprocess import compute_stats, process_static, process_year


def write_fake_year(cfg, year=2019, rain_m_per_hour=0.001):
    lat, lon = grid(cfg)
    raw = Path(cfg["data_dir"]) / "raw"
    (raw / str(year)).mkdir(parents=True)
    H, W = len(lat), len(lon)
    dims = ("valid_time", "latitude", "longitude")
    coords = lambda t: {"valid_time": t, "latitude": lat, "longitude": lon}
    t6 = pd.date_range(f"{year}-01-01", f"{year}-02-28 18:00", freq="6h")
    th = pd.date_range(f"{year}-01-01", f"{year}-02-28 23:00", freq="1h")
    full = lambda t, v: np.full((len(t), H, W), v, np.float32)

    xr.Dataset({"z": (dims, full(t6[:1], 9.80665 * 100)), "lsm": (dims, full(t6[:1], 1.0))},
               coords=coords(t6[:1])).to_netcdf(raw / "static.nc")
    xr.Dataset({"t2m": (dims, full(t6, 300.0)), "d2m": (dims, full(t6, 290.0)),
                "u10": (dims, full(t6, 1.0)), "v10": (dims, full(t6, -1.0)),
                "msl": (dims, full(t6, 101000.0)), "tcwv": (dims, full(t6, 40.0))},
               coords=coords(t6)).to_netcdf(raw / str(year) / f"sfc_{year}.nc")
    pl_dims = ("valid_time", "pressure_level", "latitude", "longitude")
    pl = {v: (pl_dims, np.full((len(t6), 3, H, W), val, np.float32))
          for v, val in {"z": 57000.0, "t": 290.0, "q": 0.01, "u": 5.0, "v": 2.0}.items()}
    xr.Dataset(pl, coords={**coords(t6), "pressure_level": [200, 500, 850]}).to_netcdf(
        raw / str(year) / f"pl_{year}.nc")
    xr.Dataset({"tp": (dims, full(th, rain_m_per_hour))},
               coords=coords(th)).to_netcdf(raw / str(year) / f"tp_{year}.nc")


def test_units_rain_and_month_boundary(cfg):
    write_fake_year(cfg)
    process_static(cfg)
    process_year(cfg, 2019)
    proc = processed_dir(cfg)
    data = np.load(proc / "2019.npy")
    times = np.load(proc / "2019_times.npy")
    ch = cfg["channels"]

    # 00 UTC on 1 Jan has no rain history, so the record starts at 06 UTC
    assert times[0] == np.datetime64("2019-01-01T06", "h")
    assert times[-1] == np.datetime64("2019-02-28T18", "h")
    assert len(times) == (31 + 28) * 4 - 1
    np.testing.assert_allclose(data[:, ch.index("t2m")], 26.85, atol=1e-4)
    np.testing.assert_allclose(data[:, ch.index("msl")], 1010.0, atol=1e-3)
    np.testing.assert_allclose(data[:, ch.index("z500")], 57000 / 9.80665, rtol=1e-6)
    np.testing.assert_allclose(data[:, ch.index("q850")], 10.0, rtol=1e-6)
    rain_mm = np.expm1(data[:, ch.index("tp")])
    np.testing.assert_allclose(rain_mm, 6.0, rtol=1e-5)  # 1 mm/h for 6 h, including across 1 Feb

    static = np.load(proc / "static.npy")
    np.testing.assert_allclose(static[0], 100.0, rtol=1e-6)  # orography in metres


def test_stats_and_windows(cfg):
    write_fake_year(cfg)
    process_static(cfg)
    process_year(cfg, 2019)
    compute_stats(cfg)
    ds = Windows(cfg, [2019], rollout=2)
    seq, tf = ds[0]
    C = len(cfg["channels"])
    assert seq.shape == (4, C, 9, 9)       # t-1, t, t+1, t+2
    assert tf.shape == (2, 4)
    assert np.isfinite(seq.numpy()).all()
    assert len(ds) == len(np.load(processed_dir(cfg) / "2019_times.npy")) - 3
