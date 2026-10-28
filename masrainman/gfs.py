"""NOAA GFS 0.25 deg data for live forecasts: start state and boundary ring.

Files come from the NOMADS grib filter, cut to the model domain (about 0.5 MB
each), and are cached in <data_dir>/gfs/<cycle>/. NOMADS keeps about 10 days.
"""
import time
import urllib.request
from pathlib import Path

import eccodes
import numpy as np
import pandas as pd

from .config import grid

NOMADS = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
VARS = ["TMP", "DPT", "UGRD", "VGRD", "PRMSL", "PWAT", "APCP", "HGT", "SPFH"]
LEVELS = ["2_m_above_ground", "10_m_above_ground", "mean_sea_level",
          "entire_atmosphere_(considered_as_a_single_layer)", "surface",
          "200_mb", "500_mb", "850_mb"]

# channel -> (GRIB shortName, typeOfLevel, level, conversion to stored units)
GFS_FIELDS = {
    "t2m": ("2t", "heightAboveGround", 2, lambda a: a - 273.15),
    "d2m": ("2d", "heightAboveGround", 2, lambda a: a - 273.15),
    "u10": ("10u", "heightAboveGround", 10, None),
    "v10": ("10v", "heightAboveGround", 10, None),
    "msl": ("prmsl", "meanSea", 0, lambda a: a / 100.0),
    "tcwv": ("pwat", "atmosphereSingleLayer", 0, None),
    "z500": ("gh", "isobaricInhPa", 500, None),          # already geopotential metres
    "t850": ("t", "isobaricInhPa", 850, lambda a: a - 273.15),
    "q850": ("q", "isobaricInhPa", 850, lambda a: a * 1000.0),
    "u850": ("u", "isobaricInhPa", 850, None),
    "v850": ("v", "isobaricInhPa", 850, None),
    "u200": ("u", "isobaricInhPa", 200, None),
    "v200": ("v", "isobaricInhPa", 200, None),
}


def cycle_dir(cfg, cycle):
    p = Path(cfg["data_dir"]) / "gfs" / pd.Timestamp(cycle).strftime("%Y%m%d%H")
    p.mkdir(parents=True, exist_ok=True)
    return p


def url(cfg, cycle, fhour):
    ts = pd.Timestamp(cycle)
    d = cfg["domain"]
    params = [f"dir=%2Fgfs.{ts:%Y%m%d}%2F{ts:%H}%2Fatmos", f"file=gfs.t{ts:%H}z.pgrb2.0p25.f{fhour:03d}"]
    params += [f"var_{v}=on" for v in VARS] + [f"lev_{lev}=on" for lev in LEVELS]
    params += ["subregion=", f"toplat={d['north']}", f"leftlon={d['west']}",
               f"rightlon={d['east']}", f"bottomlat={d['south']}"]
    return NOMADS + "?" + "&".join(params)


def download(cfg, cycle, fhour, retries=4):
    path = cycle_dir(cfg, cycle) / f"f{fhour:03d}.grib2"
    if path.exists() and path.stat().st_size > 0:
        return path
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url(cfg, cycle, fhour), timeout=120) as r:
                data = r.read()
            if not data.startswith(b"GRIB"):
                raise IOError("NOMADS did not return GRIB data (file not published yet?)")
            path.with_suffix(".part").write_bytes(data)
            path.with_suffix(".part").replace(path)
            time.sleep(1)  # be polite to NOMADS
            return path
        except Exception as e:  # network hiccups are common; retry with backoff
            if attempt == retries - 1:
                raise RuntimeError(f"GFS {cycle} f{fhour:03d}: {e}") from e
            time.sleep(10 * (attempt + 1))


def read_grib(path, cfg):
    """{(shortName, typeOfLevel, level): field} plus rain as {(start, end): field}.
    Fields are returned north -> south to match ERA5."""
    lat, lon = grid(cfg)
    fields, rain = {}, {}
    with open(path, "rb") as f:
        while True:
            g = eccodes.codes_grib_new_from_file(f)
            if g is None:
                break
            try:
                ni, nj = eccodes.codes_get(g, "Ni"), eccodes.codes_get(g, "Nj")
                if (nj, ni) != (len(lat), len(lon)):
                    raise ValueError(f"{path}: GFS grid {nj}x{ni} does not match the domain")
                a = eccodes.codes_get_values(g).reshape(nj, ni)
                if eccodes.codes_get(g, "jScansPositively") == 1:
                    a = a[::-1]
                name = eccodes.codes_get(g, "shortName")
                if name == "tp":
                    start, _, end = eccodes.codes_get(g, "stepRange").partition("-")
                    rain.setdefault((int(start), int(end or start)), a)
                else:
                    key = (name, eccodes.codes_get(g, "typeOfLevel"), eccodes.codes_get(g, "level"))
                    fields.setdefault(key, a)
            finally:
                eccodes.codes_release(g)
    return fields, rain


class GFSArchive:
    """Looks like dataset.Archive (get(t), has(t), step) but serves GFS:
    analyses for the start, forecasts of one cycle for the boundary ring."""

    def __init__(self, cfg, cycle, n_steps):
        self.cfg = cfg
        self.step = np.timedelta64(cfg["step_hours"], "h")
        self.cycle = np.datetime64(cycle, "h")
        self.n_steps = n_steps
        self._cache = {}

    def _read(self, cycle, fhour):
        key = (str(cycle), fhour)
        if key not in self._cache:
            self._cache[key] = read_grib(download(self.cfg, cycle, fhour), self.cfg)
        return self._cache[key]

    def _rain_6h(self, cycle, fhour):
        """mm in the 6 h ending at cycle+fhour, from 6-h buckets or differences of totals."""
        _, rain = self._read(cycle, fhour)
        if (fhour - 6, fhour) in rain:
            return rain[(fhour - 6, fhour)]
        if (0, fhour) in rain:
            _, prev = self._read(cycle, fhour - 6)
            if (0, fhour - 6) in prev:
                return rain[(0, fhour)] - prev[(0, fhour - 6)]
        for (start, end), a in rain.items():  # 3-h buckets after f120: add the two halves
            if end == fhour and start == fhour - 3:
                _, half = self._read(cycle, fhour - 3)
                if (fhour - 6, fhour - 3) in half:
                    return half[(fhour - 6, fhour - 3)] + a
        raise KeyError(f"no 6-h rain total for f{fhour:03d}")

    def source(self, t):
        """(cycle, forecast hour) used for valid time t."""
        t = np.datetime64(t, "h")
        if t <= self.cycle:
            return t, 0  # analyses: the cycle at t itself
        return self.cycle, int((t - self.cycle) / np.timedelta64(1, "h"))

    def has(self, t):
        return True

    def get(self, t):
        cycle, fh = self.source(t)
        fields, _ = self._read(cycle, fh)
        if fh == 0:  # analyses carry no rain; use the previous cycle's 0-6 h forecast
            rain = self._rain_6h(cycle - self.step, 6)
        else:
            rain = self._rain_6h(cycle, fh)
        out = []
        for name in self.cfg["channels"]:
            if name == "tp":
                out.append(np.log1p(np.clip(rain, 0, None)))
                continue
            short, level_type, level, convert = GFS_FIELDS[name]
            a = fields[(short, level_type, level)]
            out.append(convert(a) if convert else a)
        return np.stack(out).astype(np.float32)

    def prefetch(self):
        """Download everything a forecast will need, with progress output."""
        hours = list(range(6, self.n_steps * self.cfg["step_hours"] + 1, self.cfg["step_hours"]))
        needed = [(self.cycle - 2 * self.step, 6), (self.cycle - self.step, 0), (self.cycle - self.step, 6),
                  (self.cycle, 0)] + [(self.cycle, h) for h in hours]
        if any(h > 120 for h in hours):  # 3-hourly output after f120 may need the in-between files
            needed += [(self.cycle, h - 3) for h in hours if h > 120]
        for i, (c, h) in enumerate(needed, 1):
            download(self.cfg, c, h)
            print(f"\r  GFS files {i}/{len(needed)}", end="", flush=True)
        print()


def latest_cycle(cfg, max_back_hours=48):
    """Most recent GFS cycle whose last needed file (f240) is already published."""
    now = pd.Timestamp.utcnow().tz_localize(None).floor("6h")
    for back in range(0, max_back_hours + 1, 6):
        cycle = now - pd.Timedelta(hours=back)
        try:
            with urllib.request.urlopen(url(cfg, cycle, 240), timeout=60) as r:
                if r.read(4) == b"GRIB":
                    return np.datetime64(cycle, "h")
        except Exception:
            pass
    raise RuntimeError("No complete GFS cycle found on NOMADS in the last 48 h")
