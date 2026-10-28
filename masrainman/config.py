"""Config loading, grid definition and the channel catalogue."""
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent

# ERA5 short name -> CDS request name
CDS_NAMES = {
    "t2m": "2m_temperature",
    "d2m": "2m_dewpoint_temperature",
    "u10": "10m_u_component_of_wind",
    "v10": "10m_v_component_of_wind",
    "msl": "mean_sea_level_pressure",
    "tcwv": "total_column_water_vapour",
    "tp": "total_precipitation",
    "z": "geopotential",
    "t": "temperature",
    "q": "specific_humidity",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
}

G = 9.80665


def _ch(var, level, convert, units, long):
    return dict(var=var, level=level, convert=convert, units=units, long=long)


# Model channel -> where it comes from in ERA5 and how it is stored.
# Values are stored in friendly units (degC, hPa, g/kg ...). Rain is stored as
# log1p(mm per step) because rain is very skewed; forecasts convert it back.
CHANNEL_INFO = {
    "t2m": _ch("t2m", None, lambda a: a - 273.15, "degC", "2 m temperature"),
    "d2m": _ch("d2m", None, lambda a: a - 273.15, "degC", "2 m dewpoint"),
    "u10": _ch("u10", None, None, "m/s", "10 m eastward wind"),
    "v10": _ch("v10", None, None, "m/s", "10 m northward wind"),
    "msl": _ch("msl", None, lambda a: a / 100.0, "hPa", "mean sea level pressure"),
    "tcwv": _ch("tcwv", None, None, "kg/m2", "total column water vapour"),
    "tp": _ch("tp", None, None, "mm", "precipitation in the last step"),
    "z500": _ch("z", 500, lambda a: a / G, "m", "500 hPa geopotential height"),
    "t850": _ch("t", 850, lambda a: a - 273.15, "degC", "850 hPa temperature"),
    "q850": _ch("q", 850, lambda a: a * 1000.0, "g/kg", "850 hPa specific humidity"),
    "u850": _ch("u", 850, None, "m/s", "850 hPa eastward wind"),
    "v850": _ch("v", 850, None, "m/s", "850 hPa northward wind"),
    "u200": _ch("u", 200, None, "m/s", "200 hPa eastward wind"),
    "v200": _ch("v", 200, None, "m/s", "200 hPa northward wind"),
}


def load_config(path=None):
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    unknown = [c for c in cfg["channels"] if c not in CHANNEL_INFO]
    if unknown:
        raise ValueError(f"Unknown channels in config: {unknown}")
    return cfg


def raw_dir(cfg):
    p = Path(cfg["data_dir"]) / "raw"
    p.mkdir(parents=True, exist_ok=True)
    return p


def processed_dir(cfg):
    p = Path(cfg["data_dir"]) / "processed"
    p.mkdir(parents=True, exist_ok=True)
    return p


def run_dir(cfg, name):
    p = Path(cfg["run_dir"]) / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def year_range(pair):
    return list(range(pair[0], pair[1] + 1))


def all_years(cfg):
    ys = cfg["years"].values()
    return list(range(min(y[0] for y in ys), max(y[1] for y in ys) + 1))


def grid(cfg):
    """Latitudes (north -> south, like ERA5) and longitudes (west -> east)."""
    d = cfg["domain"]
    g = d["grid"]
    lat = np.round(np.arange(d["north"], d["south"] - g / 2, -g), 4)
    lon = np.round(np.arange(d["west"], d["east"] + g / 2, g), 4)
    return lat, lon


def boundary_mask(cfg):
    lat, lon = grid(cfg)
    w = cfg["domain"]["boundary_width"]
    m = np.zeros((len(lat), len(lon)), np.float32)
    m[:w] = m[-w:] = 1
    m[:, :w] = m[:, -w:] = 1
    return m


def focus_mask(cfg):
    lat, lon = grid(cfg)
    f = cfg["focus"]
    inside_lat = (lat <= f["north"]) & (lat >= f["south"])
    inside_lon = (lon >= f["west"]) & (lon <= f["east"])
    return (inside_lat[:, None] & inside_lon[None, :]).astype(np.float32)
