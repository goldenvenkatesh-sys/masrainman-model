"""Forecast maps: temperature + pressure, wind, rain, moisture.

  python -m masrainman.plot D:/masrainman_runs/forecasts/fc_2023112000.nc --leads 24 72 120 240
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
from scipy.ndimage import gaussian_filter
import xarray as xr

# MasRainMan standard rainfall colour bar, the same as the Rain Alarm web app (app.js):
# daily totals in mm; nothing is drawn below 1 mm, and 300 mm and more is grey.
RAIN_LEVELS = [1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 100, 150, 200, 300]
RAIN_COLORS = ["#a8c8f0", "#76a9dc", "#5b96d1", "#2679b9", "#1768a5", "#167a9f", "#188b45", "#35a843",
               "#73c53f", "#a8d62f", "#d8e82b", "#ffe51f", "#ffc21c", "#ff9e16", "#f47b16", "#e94b18",
               "#d72a1d", "#c51f50", "#e2477a", "#f080a0"]
RAIN_OVER = "#6b6b6b"


def rain_cmap():
    cmap = ListedColormap(RAIN_COLORS)
    cmap.set_over(RAIN_OVER)
    return cmap, BoundaryNorm(RAIN_LEVELS, cmap.N)  # above 300 mm -> the "over" grey


def rain_24h(ds, lead):
    """Rain total over the 24 h ending at `lead` (or since the start, for earlier leads)."""
    leads = ds.lead.values
    take = (leads > max(lead - 24, 0)) & (leads <= lead)
    return ds.tp.isel(lead=np.where(take)[0]).sum("lead"), lead - max(lead - 24, 0)


BOUNDARIES = Path(__file__).with_name("data") / "south_india_boundaries.json"
_lines = {}


def boundaries():
    """State lines (DataMeet), land borders as per the Government of India, coastline."""
    if not _lines:
        d = json.loads(BOUNDARIES.read_text())
        _lines.update({k: [np.array(c) for c in d[k]] for k in ("states", "borders", "coast")})
    return _lines


def draw_boundaries(ax, ds):
    b = boundaries()
    ax.add_collection(LineCollection(b["states"], colors="#555555", linewidths=0.45))
    ax.add_collection(LineCollection(b["coast"], colors="k", linewidths=0.7))
    ax.add_collection(LineCollection(b["borders"], colors="k", linewidths=0.9))
    ax.set_xlim(float(ds.lon.min()), float(ds.lon.max()))
    ax.set_ylim(float(ds.lat.min()), float(ds.lat.max()))
    ax.set_aspect("equal")


def smooth(field, sigma):
    """Light smoothing for display only: the 0.25 deg grid otherwise shows as blocks and
    grid-scale speckle. The netCDF files keep the raw model values."""
    return gaussian_filter(np.asarray(field, dtype=np.float32), sigma, mode="nearest")


def plot_lead(ds, lead, out):
    f = ds.sel(lead=lead)
    valid = str(f.valid_time.values)[:13].replace("T", " ")
    lon, lat = ds.lon, ds.lat
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    m = ax.pcolormesh(lon, lat, smooth(f.t2m, 0.7), cmap="RdYlBu_r", shading="gouraud")
    fig.colorbar(m, ax=ax, label="2 m temperature (degC)")
    cs = ax.contour(lon, lat, smooth(f.msl, 1.5), levels=np.arange(980, 1030, 2), colors="k", linewidths=0.7)
    ax.clabel(cs, fontsize=7, fmt="%d")
    ax.set_title("Temperature + MSLP (hPa)")

    ax = axes[0, 1]
    u, v = smooth(f.u10, 0.7), smooth(f.v10, 0.7)
    m = ax.pcolormesh(lon, lat, np.hypot(u, v), cmap="viridis", shading="gouraud")
    fig.colorbar(m, ax=ax, label="10 m wind speed (m/s)")
    s = slice(None, None, 6)
    ax.quiver(lon[s], lat[s], u[s, s], v[s, s], color="w", scale=300)
    ax.set_title("10 m wind")

    ax = axes[1, 0]
    total, hours = rain_24h(ds, lead)
    cmap, norm = rain_cmap()
    m = ax.contourf(lon, lat, smooth(total, 0.8), levels=RAIN_LEVELS, cmap=cmap, norm=norm, extend="max")
    cb = fig.colorbar(m, ax=ax, label="mm", ticks=RAIN_LEVELS, spacing="uniform")
    cb.ax.tick_params(labelsize=7)
    ax.set_title(f"Rainfall, {int(hours)} h total to +{int(lead)} h")

    ax = axes[1, 1]
    m = ax.pcolormesh(lon, lat, smooth(f.tcwv, 0.7), cmap="BrBG", shading="gouraud")
    fig.colorbar(m, ax=ax, label="kg/m2")
    ax.set_title("Total column water vapour")

    for ax in axes.flat:
        draw_boundaries(ax, ds)
    fig.suptitle(f"MasRainMan AI  init {ds.attrs['init_time']}  +{int(lead)} h  (valid {valid} UTC)")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    ap.add_argument("--leads", type=float, nargs="+", default=[24, 72, 120, 240])
    args = ap.parse_args()
    path = Path(args.file)
    ds = xr.open_dataset(path)
    for lead in args.leads:
        if lead in ds.lead.values:
            plot_lead(ds, lead, path.with_name(f"{path.stem}_{int(lead):03d}h.png"))
        else:
            print(f"lead {lead} h not in file")


if __name__ == "__main__":
    main()
