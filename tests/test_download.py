"""Split CDS downloads (month chunks, one file per level) must recombine into the
pl_YYYY.nc layout that preprocessing reads."""
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from masrainman.config import grid, processed_dir, raw_dir
from masrainman.download_era5 import combine, pressure_by_level
from masrainman.preprocess import process_static, process_year
from test_preprocess import write_fake_year

VALUES = {"z": 57000.0, "t": 290.0, "q": 0.01, "u": 5.0, "v": 2.0}
SHORT = {"geopotential": "z", "temperature": "t", "specific_humidity": "q",
         "u_component_of_wind": "u", "v_component_of_wind": "v"}


def write_level_chunk(path, cfg, level, variables, times):
    lat, lon = grid(cfg)
    dims = ("valid_time", "pressure_level", "latitude", "longitude")
    data = {SHORT[v]: (dims, np.full((len(times), 1, len(lat), len(lon)), VALUES[SHORT[v]], np.float32))
            for v in variables}
    xr.Dataset(data, coords={"valid_time": times, "pressure_level": [int(level)],
                             "latitude": lat, "longitude": lon}).to_netcdf(path)


def test_chunks_and_levels_recombine(cfg):
    write_fake_year(cfg)  # sfc, tp and a reference pl file
    out = raw_dir(cfg) / "2019"
    (out / "pl_2019.nc").unlink()
    jan = pd.date_range("2019-01-01", "2019-01-31 18:00", freq="6h")
    feb = pd.date_range("2019-02-01", "2019-02-28 18:00", freq="6h")

    level_files = []
    for level, variables in pressure_by_level(cfg).items():
        parts = [out / f"pl_2019_{level}hPa_m01-01.nc", out / f"pl_2019_{level}hPa_m02-02.nc"]
        write_level_chunk(parts[0], cfg, level, variables, jan)
        write_level_chunk(parts[1], cfg, level, variables, feb)
        target = out / f"pl_2019_{level}hPa.nc"
        combine(parts, target, "time")
        assert not any(p.exists() for p in parts)
        level_files.append(target)
    combine(level_files, out / "pl_2019.nc", "merge")
    assert sorted(p.name for p in out.iterdir()) == ["pl_2019.nc", "sfc_2019.nc", "tp_2019.nc"]

    with xr.open_dataset(out / "pl_2019.nc") as pl:
        assert list(pl.pressure_level.values) == [200, 500, 850]
        assert pl.sizes["valid_time"] == len(jan) + len(feb)

    process_static(cfg)
    process_year(cfg, 2019)
    data = np.load(processed_dir(cfg) / "2019.npy")
    ch = cfg["channels"]
    np.testing.assert_allclose(data[:, ch.index("z500")], 57000 / 9.80665, rtol=1e-6)
    np.testing.assert_allclose(data[:, ch.index("t850")], 290 - 273.15, rtol=1e-5)
    np.testing.assert_allclose(data[:, ch.index("u200")], 5.0)
    assert np.isfinite(data).all()
