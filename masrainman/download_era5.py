"""Download ERA5 for the model domain from the Copernicus Climate Data Store.

One-time setup:
  1. Make a free account at https://cds.climate.copernicus.eu
  2. Accept the ERA5 licence on the dataset page.
  3. Put your API key in %USERPROFILE%\\.cdsapirc :
         url: https://cds.climate.copernicus.eu/api
         key: <your-key>

Usage:
  python -m masrainman.download_era5                     # all years in config.yaml
  python -m masrainman.download_era5 --years 2022 2023   # a range
Files already downloaded are skipped, so you can stop and restart any time.
"""
import argparse

import cdsapi

from .config import CDS_NAMES, CHANNEL_INFO, all_years, load_config, raw_dir

SINGLE = "reanalysis-era5-single-levels"
PRESSURE = "reanalysis-era5-pressure-levels"


def base_request(cfg, year, month, times):
    d = cfg["domain"]
    return {
        "product_type": ["reanalysis"],
        "year": [str(year)],
        "month": [f"{month:02d}"],
        "day": [f"{day:02d}" for day in range(1, 32)],
        "time": times,
        "area": [d["north"], d["west"], d["south"], d["east"]],
        "grid": [d["grid"], d["grid"]],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def retrieve(client, dataset, request, target):
    if target.exists():
        print(f"  have {target.name}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(".part")
    print(f"  requesting {target.name} ...")
    client.retrieve(dataset, request, str(part))
    part.replace(target)


def needed_variables(cfg):
    surface, pressure, levels = set(), set(), set()
    for name in cfg["channels"]:
        info = CHANNEL_INFO[name]
        if name == "tp":
            continue  # rain is downloaded hourly in its own file
        if info["level"] is None:
            surface.add(CDS_NAMES[info["var"]])
        else:
            pressure.add(CDS_NAMES[info["var"]])
            levels.add(str(info["level"]))
    return sorted(surface), sorted(pressure), sorted(levels, key=int)


def download_static(client, cfg):
    req = base_request(cfg, 2020, 1, ["00:00"])
    req["day"] = ["01"]
    req["variable"] = ["geopotential", "land_sea_mask"]
    retrieve(client, SINGLE, req, raw_dir(cfg) / "static.nc")


def download_month(client, cfg, year, month):
    surface, pressure, levels = needed_variables(cfg)
    step = cfg["step_hours"]
    inst_times = [f"{h:02d}:00" for h in range(0, 24, step)]
    out = raw_dir(cfg) / str(year)
    tag = f"{year}{month:02d}"

    req = base_request(cfg, year, month, inst_times)
    req["variable"] = surface
    retrieve(client, SINGLE, req, out / f"sfc_{tag}.nc")

    req = base_request(cfg, year, month, inst_times)
    req["variable"] = pressure
    req["pressure_level"] = levels
    retrieve(client, PRESSURE, req, out / f"pl_{tag}.nc")

    if "tp" in cfg["channels"]:
        # ERA5 rain is an hourly accumulation; we sum it to per-step totals later.
        req = base_request(cfg, year, month, [f"{h:02d}:00" for h in range(24)])
        req["variable"] = [CDS_NAMES["tp"]]
        retrieve(client, SINGLE, req, out / f"tp_{tag}.nc")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--years", type=int, nargs=2, metavar=("FIRST", "LAST"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    years = range(args.years[0], args.years[1] + 1) if args.years else all_years(cfg)
    client = cdsapi.Client()
    download_static(client, cfg)
    for year in years:
        print(f"{year}:")
        for month in range(1, 13):
            download_month(client, cfg, year, month)


if __name__ == "__main__":
    main()
