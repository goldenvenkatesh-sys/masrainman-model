"""Forecast maps: temperature + pressure, wind, rain, moisture.

  python -m masrainman.plot D:/masrainman_runs/forecasts/fc_2023112000.nc --leads 24 72 120 240
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import numpy as np
import xarray as xr

RAIN_LEVELS = [0.1, 1, 2.5, 5, 10, 20, 40, 80]


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
    rain = np.where(f.tp >= RAIN_LEVELS[0], f.tp, np.nan)
    m = ax.pcolormesh(ds.lon, ds.lat, rain, cmap="YlGnBu",
                      norm=BoundaryNorm(RAIN_LEVELS, 256, extend="max"), shading="auto")
    fig.colorbar(m, ax=ax, label=ds.tp.attrs.get("units", "mm"))
    ax.set_title("Rain")

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
