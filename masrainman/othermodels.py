"""Other global models for the verification page: 24-h rain (and airport MSLP / wind where the
files are small enough), cut to the MasRainMan domain.

  gfs    NCEP GFS 0.25 deg       AWS open data (noaa-gfs-bdp-pds), only the needed GRIB records
  aigfs  NOAA AIGFS (AI GFS)     NOMADS, surface file (6 small fields) per 6-h step
  icon   DWD ICON 13 km          DWD open data, icosahedral cells regridded here; 00 UTC reaches 180 h
  gdps   CMC GDPS 15 km          MSC Datamart (dated folders)

Every fetcher returns (rain[days, H, W] mm, points[steps, airports, 3] or None), where rain day k
is lead 24k..24(k+1) h like the MasRainMan archive, and points hold MSLP hPa, u10, v10 every 6 h.
DWD and MSC keep only about a day of runs, so those two only work for the current day; GFS and
AIGFS can catch up a few missed days.
"""
import bz2
import re
import tempfile
import time

import eccodes
import numpy as np
import pandas as pd
import requests
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree

HTTP = requests.Session()
HTTP.headers["User-Agent"] = "masrainman-verify"
DAYS = 10
POINT_STEPS = list(range(6, 241, 6))


class NotAvailable(Exception):
    pass


def get(url, headers=None, timeout=120, tries=4):
    for attempt in range(tries):
        try:
            r = HTTP.get(url, headers=headers, timeout=timeout)
            if r.status_code in (403, 404):
                raise NotAvailable(f"{r.status_code} {url}")
            r.raise_for_status()
            return r.content
        except NotAvailable:
            raise
        except Exception as e:  # busy servers: wait and try again
            if attempt == tries - 1:
                raise
            print(f"  retry {attempt + 1}: {url.rsplit('/', 1)[-1][:70]} ({e})")
            time.sleep(10 * (attempt + 1))


# ---------------------------------------------------------------- GRIB reading

def read_grib(blob, lat, lon):
    """Every message of a GRIB blob -> list of dicts (name, level, start, end h, field on lat/lon)."""
    out = []
    with tempfile.NamedTemporaryFile(suffix=".grib2") as f:
        f.write(blob)
        f.flush()
        with open(f.name, "rb") as fh:
            while True:
                g = eccodes.codes_grib_new_from_file(fh)
                if g is None:
                    break
                try:
                    eccodes.codes_set(g, "stepUnits", "h")   # GFS writes "0-1 day acc" in days
                    m = {"name": eccodes.codes_get(g, "shortName"),
                         "level": eccodes.codes_get(g, "level"),
                         "start": int(eccodes.codes_get(g, "startStep")),
                         "end": int(eccodes.codes_get(g, "endStep"))}
                    if eccodes.codes_get(g, "gridType") == "regular_ll":
                        m["field"] = _regular(g, lat, lon)
                    else:
                        m["values"] = eccodes.codes_get_double_array(g, "values")
                    out.append(m)
                finally:
                    eccodes.codes_release(g)
    return out


def _regular(g, lat, lon):
    ni, nj = eccodes.codes_get(g, "Ni"), eccodes.codes_get(g, "Nj")
    v = eccodes.codes_get_double_array(g, "values").reshape(nj, ni)
    v[v == eccodes.codes_get(g, "missingValue")] = np.nan
    la0 = eccodes.codes_get(g, "latitudeOfFirstGridPointInDegrees")
    la1 = eccodes.codes_get(g, "latitudeOfLastGridPointInDegrees")
    lo0 = eccodes.codes_get(g, "longitudeOfFirstGridPointInDegrees")
    lo1 = eccodes.codes_get(g, "longitudeOfLastGridPointInDegrees")
    if lo1 < lo0:
        lo1 += 360.0
    glat, glon = np.linspace(la0, la1, nj), np.linspace(lo0, lo1, ni)
    glon = np.where(glon >= 180, glon - 360, glon)
    order = np.argsort(glon)
    glon, v = glon[order], v[:, order]
    if glat[0] > glat[-1]:
        glat, v = glat[::-1], v[::-1]
    j = np.where((glat >= lat.min() - 1) & (glat <= lat.max() + 1))[0]
    i = np.where((glon >= lon.min() - 1) & (glon <= lon.max() + 1))[0]
    f = RegularGridInterpolator((glat[j], glon[i]), v[np.ix_(j, i)], bounds_error=False, fill_value=np.nan)
    LON, LAT = np.meshgrid(lon, lat)
    return f((LAT, LON)).astype(np.float32)


def daily_from_accumulations(msgs, days=DAYS):
    """Rain messages with (start, end) hours -> 24-h totals for leads 0-24, 24-48 ...

    Works with running totals (0-N), buckets (N-6..N) or a mix: each message extends a running
    total from its start. Stops at the first day that cannot be built."""
    cum = {0: None}
    for m in sorted(msgs, key=lambda m: (m["end"], m["start"])):
        if m["start"] in cum and m["end"] not in cum:
            base = cum[m["start"]]
            cum[m["end"]] = m["field"] if base is None else base + m["field"]
    out = []
    for k in range(days):
        a, b = 24 * k, 24 * (k + 1)
        if b not in cum or a not in cum:
            break
        out.append(cum[b] if cum[a] is None else cum[b] - cum[a])
    if not out:
        raise NotAvailable("no complete 24-h rain")
    return np.clip(np.array(out, np.float32), 0, None)


def points_from(fields, lat, lon, airports, at_points):
    """{step: {'msl','u','v'}} -> (steps, airports, 3), or None if a step is missing."""
    if not all(s in fields and len(fields[s]) == 3 for s in POINT_STEPS):
        return None
    pts = [a[1:3] for a in airports]
    arr = np.stack([np.stack([fields[s][k] for s in POINT_STEPS]) for k in ("msl", "u", "v")])
    return np.stack([at_points(a, lat, lon, pts) for a in arr], -1)


def _point_fields(msgs, step, fields):
    for m in msgs:
        if m["end"] != step:
            continue
        if m["name"] in ("prmsl", "msl"):
            fields.setdefault(step, {})["msl"] = m["field"] / 100.0
        elif m["name"] in ("10u", "u10") or (m["name"] == "u" and m["level"] == 10):
            fields.setdefault(step, {})["u"] = m["field"]
        elif m["name"] in ("10v", "v10") or (m["name"] == "v" and m["level"] == 10):
            fields.setdefault(step, {})["v"] = m["field"]


# ---------------------------------------------------------------- NCEP GFS (AWS)

def idx_ranges(idx_text, wanted):
    """NCEP .idx -> byte ranges of the records whose 'VAR:level' is in `wanted`."""
    lines = [l.split(":") for l in idx_text.strip().splitlines()]
    out = []
    for n, l in enumerate(lines):
        if f"{l[3]}:{l[4]}" in wanted:
            end = int(lines[n + 1][1]) - 1 if n + 1 < len(lines) else ""
            out.append((int(l[1]), end))
    return out


def fetch_ncep_records(url, wanted):
    idx = get(url + ".idx").decode()
    blob = b""
    for a, b in idx_ranges(idx, wanted):
        blob += get(url, headers={"Range": f"bytes={a}-{b}"})
    if not blob:
        raise NotAvailable(f"no {wanted} in {url}")
    return blob


GFS_URL = "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{ymd}/{hh}/atmos/gfs.t{hh}z.pgrb2.0p25.f{s:03d}"
POINT_VARS = {"PRMSL:mean sea level", "UGRD:10 m above ground", "VGRD:10 m above ground"}


def fetch_gfs(cycle, lat, lon, airports, at_points):
    ymd, hh = cycle.strftime("%Y%m%d"), cycle.strftime("%H")
    rain, fields = [], {}
    for s in POINT_STEPS:
        want = set(POINT_VARS)
        if s % 24 == 0:
            want.add("APCP:surface")
        msgs = read_grib(fetch_ncep_records(GFS_URL.format(ymd=ymd, hh=hh, s=s), want), lat, lon)
        rain += [m for m in msgs if m["name"] == "tp" and m["start"] == 0]   # running total since the start
        _point_fields(msgs, s, fields)
    return daily_from_accumulations(rain), points_from(fields, lat, lon, airports, at_points)


# ---------------------------------------------------------------- NOAA AIGFS (NOMADS)

AIGFS_URL = ("https://nomads.ncep.noaa.gov/pub/data/nccf/com/aigfs/prod/aigfs.{ymd}/{hh}/model/atmos/grib2/"
             "aigfs.t{hh}z.sfc.f{s:03d}.grib2")


def fetch_aigfs(cycle, lat, lon, airports, at_points):
    ymd, hh = cycle.strftime("%Y%m%d"), cycle.strftime("%H")
    rain, fields = [], {}
    for s in POINT_STEPS:   # the surface file holds only 6 fields, so take it whole
        msgs = read_grib(get(AIGFS_URL.format(ymd=ymd, hh=hh, s=s)), lat, lon)
        rain += [m for m in msgs if m["name"] == "tp"]
        _point_fields(msgs, s, fields)
        time.sleep(0.5)     # NOMADS blocks clients that ask too fast
    return daily_from_accumulations(rain), points_from(fields, lat, lon, airports, at_points)


# ---------------------------------------------------------------- DWD ICON

ICON_URL = "https://opendata.dwd.de/weather/nwp/icon/grib/{hh}/{var}/"


def _icon(cycle, var, step=None):
    hh, ymdh = cycle.strftime("%H"), cycle.strftime("%Y%m%d%H")
    if step is None:
        name = f"icon_global_icosahedral_time-invariant_{ymdh}_{var.upper()}.grib2.bz2"
    else:
        name = f"icon_global_icosahedral_single-level_{ymdh}_{step:03d}_{var.upper()}.grib2.bz2"
    return bz2.decompress(get(ICON_URL.format(hh=hh, var=var) + name, timeout=300))


class IconRegrid:
    """ICON cells -> regular grid by inverse distance from the 4 nearest cells."""

    def __init__(self, clat, clon, lat, lon):
        sel = (clat > lat.min() - 1) & (clat < lat.max() + 1) & (clon > lon.min() - 1) & (clon < lon.max() + 1)
        self.sel = np.where(sel)[0]
        cos = np.cos(np.radians(np.mean(lat)))
        tree = cKDTree(np.c_[clon[sel] * cos, clat[sel]])
        LON, LAT = np.meshgrid(lon, lat)
        d, self.i = tree.query(np.c_[LON.ravel() * cos, LAT.ravel()], k=4)
        w = 1.0 / np.maximum(d, 1e-6) ** 2
        self.w, self.shape = w / w.sum(1, keepdims=True), LAT.shape

    def __call__(self, values):
        return (values[self.sel][self.i] * self.w).sum(1).reshape(self.shape).astype(np.float32)


def fetch_icon(cycle, lat, lon, airports, at_points):
    clat = read_grib(_icon(cycle, "clat"), lat, lon)[0]["values"]
    clon = read_grib(_icon(cycle, "clon"), lat, lon)[0]["values"]
    rg = IconRegrid(clat, clon, lat, lon)
    last = 180 if cycle.hour in (0, 12) else 120
    rain = []
    for s in range(24, min(last, 24 * DAYS) + 1, 24):
        try:
            m = read_grib(_icon(cycle, "tot_prec", s), lat, lon)[0]
        except NotAvailable:
            break
        rain.append({"start": 0, "end": s, "field": rg(m["values"])})
    return daily_from_accumulations(rain), None


# ---------------------------------------------------------------- CMC GDPS (MSC Datamart)

GDPS_DIRS = ("https://dd.weather.gc.ca/{ymd}/WXO-DD/model_gdps/15km/{hh}/{s:03d}/",
             "https://dd.weather.gc.ca/today/model_gdps/15km/{hh}/{s:03d}/")


def gdps_rain_file(listing, tag):
    """Pick the precipitation-accumulation file in a Datamart folder listing (plain total first)."""
    names = sorted(set(re.findall(rf'href="({tag}_MSC_GDPS_[^"]*Precip[^"]*Accum[^"]*_Sfc_[^"]*\.grib2)"', listing)))
    if not names:
        return None
    plain = [n for n in names if re.search(r"Accum_Sfc", n)]
    return (plain or names)[0]


def fetch_gdps(cycle, lat, lon, airports, at_points):
    ymd, hh = cycle.strftime("%Y%m%d"), cycle.strftime("%H")
    tag = cycle.strftime("%Y%m%dT%HZ")
    rain = []
    for s in range(24, 24 * DAYS + 1, 24):
        blob = None
        for d in GDPS_DIRS:
            folder = d.format(ymd=ymd, hh=hh, s=s)
            try:
                name = gdps_rain_file(get(folder).decode(errors="ignore"), tag)
            except NotAvailable:
                continue
            if name:
                blob = get(folder + name)
                break
        if blob is None:
            if s == 24:
                raise NotAvailable(f"GDPS {tag}: no rain file")
            break
        rain += [m for m in read_grib(blob, lat, lon) if m["end"] == s]
    return daily_from_accumulations(rain), None


FETCHERS = {"gfs": fetch_gfs, "aigfs": fetch_aigfs, "icon": fetch_icon, "gdps": fetch_gdps}


def fetch(model, cycle, lat, lon, airports, at_points):
    return FETCHERS[model](pd.Timestamp(cycle), np.asarray(lat), np.asarray(lon), airports, at_points)
