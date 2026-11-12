"""Forecast maps: temperature + pressure, wind, rain, moisture.

  python -m masrainman.plot D:/masrainman_runs/forecasts/fc_2023112000.nc --leads 24 72 120 240
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from . import masrainman_logo  # noqa: F401  (MasRainman logo + name on every saved map)
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


def panel_t2m_msl(fig, ax, ds, f, lead):
    m = ax.pcolormesh(ds.lon, ds.lat, smooth(f.t2m, 0.7), cmap="RdYlBu_r", shading="gouraud")
    fig.colorbar(m, ax=ax, label="2 m temperature (degC)")
    cs = ax.contour(ds.lon, ds.lat, smooth(f.msl, 1.5), levels=np.arange(980, 1030, 2), colors="k", linewidths=0.7)
    ax.clabel(cs, fontsize=7, fmt="%d")
    return "Temperature + MSLP (hPa)"


def panel_msl(fig, ax, ds, f, lead):
    msl = smooth(f.msl, 1.5)
    m = ax.contourf(ds.lon, ds.lat, msl, levels=np.arange(990, 1025, 1), cmap="RdYlBu_r", extend="both")
    fig.colorbar(m, ax=ax, label="hPa", ticks=np.arange(990, 1025, 2))
    cs = ax.contour(ds.lon, ds.lat, msl, levels=np.arange(980, 1030, 2), colors="k", linewidths=0.6)
    ax.clabel(cs, fontsize=7, fmt="%d")
    return "Mean sea level pressure"


def panel_t2m(fig, ax, ds, f, lead):
    m = ax.pcolormesh(ds.lon, ds.lat, smooth(f.t2m, 0.7), cmap="RdYlBu_r", shading="gouraud")
    fig.colorbar(m, ax=ax, label="degC")
    return "2 m temperature"


def panel_wind(fig, ax, ds, f, lead):
    u, v = smooth(f.u10, 0.7), smooth(f.v10, 0.7)
    m = ax.pcolormesh(ds.lon, ds.lat, np.hypot(u, v), cmap="viridis", shading="gouraud")
    fig.colorbar(m, ax=ax, label="m/s")
    s = slice(None, None, 6)
    ax.quiver(ds.lon[s], ds.lat[s], u[s, s], v[s, s], color="w", scale=300)
    return "10 m wind"


# 850 hPa wind speed bands (knots), same colours as the forecast-models site's 850 hPa maps
W850_LEVELS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80]
W850_COLORS = ["#f4f6fa", "#d6e6f5", "#a9cdee", "#72b4e3", "#3f97cf", "#43b07a", "#8fcf4c",
               "#e3e04b", "#f6b13b", "#ef7a2d", "#dd3c2a", "#b0195a", "#7a1f87"]


def panel_wind850(fig, ax, ds, f, lead):
    """850 hPa wind (about 1.5 km up): speed in knots shaded, streamlines for the monsoon flow."""
    u, v = smooth(f.u850, 0.8), smooth(f.v850, 0.8)
    kt = np.hypot(u, v) * 1.943844
    cmap = ListedColormap(W850_COLORS)
    cmap.set_over("#4a0f5a")
    m = ax.contourf(ds.lon, ds.lat, kt, levels=W850_LEVELS, cmap=cmap,
                    norm=BoundaryNorm(W850_LEVELS, cmap.N), extend="max")
    fig.colorbar(m, ax=ax, label="knots", ticks=W850_LEVELS)
    lat = np.asarray(ds.lat, dtype=float)
    k = slice(None, None, -1) if lat[0] > lat[-1] else slice(None)   # streamplot needs rising latitudes
    ax.streamplot(np.asarray(ds.lon, dtype=float), lat[k], u[k], v[k], density=1.6, color="#1a1a1a",
                  linewidth=0.7, arrowsize=0.9, arrowstyle="->")
    return "850 hPa wind + streamlines"


def panel_rain(fig, ax, ds, f, lead):
    total, hours = rain_24h(ds, lead)
    cmap, norm = rain_cmap()
    m = ax.contourf(ds.lon, ds.lat, smooth(total, 0.8), levels=RAIN_LEVELS, cmap=cmap, norm=norm, extend="max")
    cb = fig.colorbar(m, ax=ax, label="mm", ticks=RAIN_LEVELS, spacing="uniform")
    cb.ax.tick_params(labelsize=7)
    return f"Rainfall, {int(hours)} h total to +{int(lead)} h"


def panel_tcwv(fig, ax, ds, f, lead):
    m = ax.pcolormesh(ds.lon, ds.lat, smooth(f.tcwv, 0.7), cmap="BrBG", shading="gouraud")
    fig.colorbar(m, ax=ax, label="kg/m2")
    return "Total column water vapour"


# One map per parameter (the names are used in file names and by web pages)
PARAMS = {"rain": panel_rain, "mslp": panel_msl, "t2m": panel_t2m, "wind": panel_wind,
          "wind850": panel_wind850, "tcwv": panel_tcwv}


def _title(ds, f, lead):
    valid = str(f.valid_time.values)[:13].replace("T", " ")
    return f"MasRainMan AI  init {ds.attrs['init_time']}  +{int(lead)} h  (valid {valid} UTC)"


def plot_lead(ds, lead, out):
    """All four panels on one page."""
    f = ds.sel(lead=lead)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, panel in zip(axes.flat, (panel_t2m_msl, panel_wind, panel_rain, panel_tcwv)):
        ax.set_title(panel(fig, ax, ds, f, lead))
        draw_boundaries(ax, ds)
    fig.suptitle(_title(ds, f, lead))
    fig.masrainman_top = axes.flat[0]          # logo header above the page title
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"wrote {out}")


def plot_param(ds, lead, param, out):
    """One parameter, one large map."""
    f = ds.sel(lead=lead)
    fig, ax = plt.subplots(figsize=(10, 8.6))
    name = PARAMS[param](fig, ax, ds, f, lead)
    draw_boundaries(ax, ds)
    ax.set_title(f"{name}\n{_title(ds, f, lead)}", fontsize=12)
    fig.masrainman_top = ax                   # logo header centred above the map
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def rain_total(ds, hours):
    """Rain total from the start of the run to +hours (capped at the last lead)."""
    hours = min(hours, int(ds.lead.values.max()))
    take = (ds.lead.values > 0) & (ds.lead.values <= hours)
    return ds.tp.isel(lead=np.where(take)[0]).sum("lead"), hours


def total_label(hours):
    return f"{hours} h" if hours < 48 else f"{hours // 24} days"


def plot_rain_total(ds, hours, out):
    """Accumulated rain from the start of the run, one large map (24 h, 3 days, 7 days, 10 days)."""
    total, hours = rain_total(ds, hours)
    f = ds.sel(lead=hours)
    fig, ax = plt.subplots(figsize=(10, 8.6))
    cmap, norm = rain_cmap()
    m = ax.contourf(ds.lon, ds.lat, smooth(total, 0.8), levels=RAIN_LEVELS, cmap=cmap, norm=norm, extend="max")
    cb = fig.colorbar(m, ax=ax, label="mm", ticks=RAIN_LEVELS, spacing="uniform")
    cb.ax.tick_params(labelsize=7)
    draw_boundaries(ax, ds)
    ax.set_title(f"Total rainfall, {total_label(hours)} (+0 to +{hours} h)\n{_title(ds, f, hours)}", fontsize=12)
    fig.masrainman_top = ax                   # logo header centred above the map
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    return hours


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
