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
Files already downloaded are skipped, so you can stop and restart any time. Each
submitted request's ID is saved next to its file (<name>.job), so a restart picks
up the same queued request instead of sending a duplicate.
"""
import argparse
import os
from pathlib import Path

import requests
import xarray as xr
from ecmwf.datastores import Client

from .config import CDS_NAMES, CHANNEL_INFO, all_years, load_config, raw_dir

SINGLE = "reanalysis-era5-single-levels"
PRESSURE = "reanalysis-era5-pressure-levels"
DEAD = ("failed", "rejected", "dismissed", "deleted")


def cds_client():
    """Use the key in .cdsapirc explicitly. Otherwise the library may pick up an
    older ~/.ecmwfdatastoresrc first and fail with 401."""
    rc = Path(os.environ.get("CDSAPI_RC", Path.home() / ".cdsapirc"))
    conf = dict(line.split(":", 1) for line in rc.read_text().splitlines() if ":" in line)
    return Client(url=conf["url"].strip(), key=conf["key"].strip())


def base_request(cfg, year, months, times):
    d = cfg["domain"]
    return {
        "product_type": ["reanalysis"],
        "year": [str(year)],
        "month": [f"{m:02d}" for m in months],
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
    job = target.with_suffix(".job")
    remote = None
    if job.exists():
        try:
            remote = client.get_remote(job.read_text().strip())
            if remote.status in DEAD:
                print(f"  earlier request for {target.name} {remote.status}; sending a new one")
                remote = None
            else:
                print(f"  {target.name}: reattached to queued request {remote.request_id[:8]} ({remote.status})")
        except requests.HTTPError as e:
            # Only a missing request (expired or cancelled on the website) justifies a new one.
            # Anything else (403 terms not accepted, outages) must not create duplicates.
            if e.response is None or e.response.status_code != 404:
                raise
            print(f"  earlier request for {target.name} not found; sending a new one")
            remote = None
    if remote is None:
        print(f"  requesting {target.name} ...")
        remote = client.submit(dataset, request)
        job.write_text(remote.request_id)
    remote.download(str(part))  # waits in the CDS queue, then downloads
    part.replace(target)
    job.unlink()


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
    req = base_request(cfg, 2020, [1], ["00:00"])
    req["day"] = ["01"]
    req["variable"] = ["geopotential", "land_sea_mask"]
    retrieve(client, SINGLE, req, raw_dir(cfg) / "static.nc")


def pressure_by_level(cfg):
    """{level: CDS variable names needed at that level}, e.g. z only at 500 hPa."""
    out = {}
    for name in cfg["channels"]:
        info = CHANNEL_INFO[name]
        if info["level"] is not None:
            out.setdefault(str(info["level"]), set()).add(CDS_NAMES[info["var"]])
    return {lev: sorted(v) for lev, v in sorted(out.items(), key=lambda kv: int(kv[0]))}


def combine(parts, target, how):
    """Join downloaded pieces into one file: 'time' stacks month chunks, 'merge' joins levels."""
    dss = [xr.open_dataset(p) for p in parts]
    try:
        if how == "time":
            dim = "valid_time" if "valid_time" in dss[0].dims else "time"
            ds = xr.concat(dss, dim=dim)
        else:
            ds = xr.merge(dss, join="outer", compat="override")
        tmp = target.with_suffix(".tmp")
        ds.to_netcdf(tmp)
    finally:
        for d in dss:
            d.close()
    tmp.replace(target)
    for p in parts:
        p.unlink()
    print(f"  combined {len(parts)} pieces into {target.name}")


def fetch_year(client, dataset, make_request, target):
    """Download a whole year into `target`. If CDS says one request is too large,
    split it into equal month chunks that fit, then stack them in time."""
    if target.exists():
        print(f"  have {target.name}")
        return
    months = list(range(1, 13))
    for size in (12, 6, 4, 3, 2, 1):
        cost = client.estimate_costs(dataset, make_request(months[:size]))
        if cost["cost"] <= cost["limit"]:
            break
    if size == 12:
        retrieve(client, dataset, make_request(months), target)
        return
    parts = []
    for i in range(0, 12, size):
        chunk = months[i:i + size]
        part = target.with_name(f"{target.stem}_m{chunk[0]:02d}-{chunk[-1]:02d}.nc")
        retrieve(client, dataset, make_request(chunk), part)
        parts.append(part)
    combine(parts, target, "time")


def download_year(client, cfg, year):
    """One request per data type per year where CDS allows it (each request can
    wait in the queue for a long time); split only when a request is too large."""
    surface, _, _ = needed_variables(cfg)
    step = cfg["step_hours"]
    inst_times = [f"{h:02d}:00" for h in range(0, 24, step)]
    out = raw_dir(cfg) / str(year)

    def single(variables, times):
        return lambda months: {**base_request(cfg, year, months, times), "variable": variables}

    fetch_year(client, SINGLE, single(surface, inst_times), out / f"sfc_{year}.nc")

    # Pressure levels: one request per level with only the fields used there
    # (asking for every field at every level costs twice as much).
    pl_target = out / f"pl_{year}.nc"
    if pl_target.exists():
        print(f"  have {pl_target.name}")
    else:
        level_files = []
        for level, variables in pressure_by_level(cfg).items():
            def make(months, level=level, variables=variables):
                return {**base_request(cfg, year, months, inst_times),
                        "variable": variables, "pressure_level": [level]}
            target = out / f"pl_{year}_{level}hPa.nc"
            fetch_year(client, PRESSURE, make, target)
            level_files.append(target)
        combine(level_files, pl_target, "merge")

    if "tp" in cfg["channels"]:
        # ERA5 rain is an hourly accumulation; we sum it to per-step totals later.
        fetch_year(client, SINGLE, single([CDS_NAMES["tp"]], [f"{h:02d}:00" for h in range(24)]),
                   out / f"tp_{year}.nc")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--years", type=int, nargs=2, metavar=("FIRST", "LAST"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    years = range(args.years[0], args.years[1] + 1) if args.years else all_years(cfg)
    client = cds_client()
    download_static(client, cfg)
    for year in years:
        print(f"{year}:")
        download_year(client, cfg, year)


if __name__ == "__main__":
    main()
