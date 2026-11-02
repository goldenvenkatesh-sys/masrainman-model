"""Forecast maps: temperature + pressure, wind, rain, moisture.

  python -m masrainman.plot D:/masrainman_runs/forecasts/fc_2023112000.nc --leads 24 72 120 240
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
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


def coast(ax, ds):
    ax.contour(ds.lon, ds.lat, ds.lsm, levels=[0.5], colors="k", linewidths=0.6)
    ax.set_aspect("equal")


def plot_lead(ds, lead, out):
    f = ds.sel(lead=lead)
    valid = str(f.valid_time.values)[:13].replace("T", " ")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    m = ax.pcolormesh(ds.lon, ds.lat, f.t2m, cmap="RdYlBu_r", shading="auto")
    fig.colorbar(m, ax=ax, label="2 m temperature (degC)")
    cs = ax.contour(ds.lon, ds.lat, f.msl, levels=np.arange(980, 1030, 2), colors="k", linewidths=0.7)
    ax.clabel(cs, fontsize=7, fmt="%d")
    ax.set_title("Temperature + MSLP (hPa)")

    ax = axes[0, 1]
    speed = np.hypot(f.u10, f.v10)
    m = ax.pcolormesh(ds.lon, ds.lat, speed, cmap="viridis", shading="auto")
    fig.colorbar(m, ax=ax, label="10 m wind speed (m/s)")
    s = slice(None, None, 6)
    ax.quiver(ds.lon[s], ds.lat[s], f.u10[s, s], f.v10[s, s], color="w", scale=300)
    ax.set_title("10 m wind")

    ax = axes[1, 0]
    total, hours = rain_24h(ds, lead)
    rain = np.where(total >= RAIN_LEVELS[0], total, np.nan)
    cmap, norm = rain_cmap()
    m = ax.pcolormesh(ds.lon, ds.lat, rain, cmap=cmap, norm=norm, shading="auto")
    cb = fig.colorbar(m, ax=ax, label="mm", ticks=RAIN_LEVELS, spacing="uniform", extend="max")
    cb.ax.tick_params(labelsize=7)
    ax.set_title(f"Rainfall, {int(hours)} h total to +{int(lead)} h")

    ax = axes[1, 1]
    m = ax.pcolormesh(ds.lon, ds.lat, f.tcwv, cmap="BrBG", shading="auto")
    fig.colorbar(m, ax=ax, label="kg/m2")
    ax.set_title("Total column water vapour")

    for ax in axes.flat:
        coast(ax, ds)
    fig.suptitle(f"MasRainMan AI  init {ds.attrs['init_time']}  +{int(lead)} h  (valid {valid} UTC)")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
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
