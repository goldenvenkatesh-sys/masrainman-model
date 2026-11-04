"""ECMWF IFS open data (0.25 deg) for live forecasts: start state and boundary ring.

Open data is free (CC-BY-4.0, credit ECMWF). Files are global, about 10 MB per time
step, so each run is cut to the model domain right after download and saved as
<data_dir>/ecmwf/<cycle>/fields.npz (a few MB); the global GRIB files are deleted.
All four daily runs are used. 00 and 12 UTC (stream oper) reach 240 h. 06 and 18 UTC (stream
scda) reach only 90 h, so beyond that the boundary ring comes from the 00/12 UTC run 6 h older, at
the same valid time; its last 6 h (+234 to +240) hold the boundary of +234.
"""
from pathlib import Path

import eccodes
import numpy as np
import pandas as pd
from ecmwf.opendata import Client

from .config import grid

# channel -> (GRIB shortName, pressure level or 0 for surface, conversion to stored units)
ECMWF_FIELDS = {
    "t2m": ("2t", 0, lambda a: a - 273.15),
    "d2m": ("2d", 0, lambda a: a - 273.15),
    "u10": ("10u", 0, None),
    "v10": ("10v", 0, None),
    "msl": ("msl", 0, lambda a: a / 100.0),
    "tcwv": ("tcwv", 0, None),
    "z500": ("gh", 500, None),          # already geopotential metres
    "t850": ("t", 850, lambda a: a - 273.15),
    "q850": ("q", 850, lambda a: a * 1000.0),
    "u850": ("u", 850, None),
    "v850": ("v", 850, None),
    "u200": ("u", 200, None),
    "v200": ("v", 200, None),
}
RUN = np.timedelta64(12, "h")  # the run used for the state 6 h before the start is this much older
H6 = np.timedelta64(6, "h")
LAST = {"oper": 240, "scda": 90}  # last forecast step in open data


def stream(cycle):
    return "oper" if pd.Timestamp(cycle).hour in (0, 12) else "scda"


def cycle_dir(cfg, cycle):
    p = Path(cfg["data_dir"]) / "ecmwf" / pd.Timestamp(cycle).strftime("%Y%m%d%H")
    p.mkdir(parents=True, exist_ok=True)
    return p


def _requests(cfg):
    sfc = sorted({ECMWF_FIELDS[c][0] for c in cfg["channels"] if c in ECMWF_FIELDS and ECMWF_FIELDS[c][1] == 0})
    pl = sorted({ECMWF_FIELDS[c][0] for c in cfg["channels"] if c in ECMWF_FIELDS and ECMWF_FIELDS[c][1] > 0})
    levels = sorted({ECMWF_FIELDS[c][1] for c in cfg["channels"] if c in ECMWF_FIELDS and ECMWF_FIELDS[c][1] > 0})
    if "tp" in cfg["channels"]:
        sfc.append("tp")
    return {"sfc": dict(param=sfc), "pl": dict(param=pl, levelist=levels, levtype="pl")}


def _crop(path, cfg, out):
    """Cut every message in a global GRIB file to the domain; key 'short:level:step'."""
    lat, lon = grid(cfg)
    with open(path, "rb") as f:
        while True:
            g = eccodes.codes_grib_new_from_file(f)
            if g is None:
                break
            try:
                ni, nj = eccodes.codes_get(g, "Ni"), eccodes.codes_get(g, "Nj")
                a = eccodes.codes_get_values(g).reshape(nj, ni)
                if eccodes.codes_get(g, "jScansPositively") == 1:
                    a = a[::-1]
                lat0 = eccodes.codes_get(g, "latitudeOfFirstGridPointInDegrees")
                lon0 = eccodes.codes_get(g, "longitudeOfFirstGridPointInDegrees")
                lat0 = max(lat0, eccodes.codes_get(g, "latitudeOfLastGridPointInDegrees"))
                lon0 = lon0 - 360 if lon0 >= 180 else lon0
                dx = 360.0 / ni
                rows = np.rint((lat0 - lat) / dx).astype(int)
                cols = np.rint((lon - lon0) / dx).astype(int) % ni
                level = eccodes.codes_get(g, "level") if eccodes.codes_get(g, "typeOfLevel") == "isobaricInhPa" else 0
                key = f"{eccodes.codes_get(g, 'shortName')}:{level}:{eccodes.codes_get(g, 'step')}"
                out[key] = a[np.ix_(rows, cols)].astype(np.float32)
            finally:
                eccodes.codes_release(g)


def download_cycle(cfg, cycle, steps):
    """Fetch the given forecast steps of one run, cropped. Skips what is already cached."""
    d = cycle_dir(cfg, cycle)
    cache = d / "fields.npz"
    fields = dict(np.load(cache)) if cache.exists() else {}
    have = {int(k.rsplit(":", 1)[1]) for k in fields}
    missing = sorted(set(steps) - have)
    if not missing:
        return fields
    ts = pd.Timestamp(cycle)
    client = Client(source="ecmwf", model="ifs", resol="0p25")
    for name, req in _requests(cfg).items():
        target = d / f"{name}.grib2"
        print(f"  ECMWF {ts:%Y-%m-%d %H}Z {name}: {len(missing)} steps ...", flush=True)
        client.retrieve(date=ts.strftime("%Y%m%d"), time=ts.hour, stream=stream(cycle), type="fc",
                        step=missing, target=str(target), **req)
        _crop(target, cfg, fields)
        target.unlink()
    np.savez_compressed(cache, **fields)
    return fields


class ECMWFArchive:
    """Looks like dataset.Archive (get(t), has(t), step) but serves ECMWF open data:
    the run at `cycle` for the start and boundary ring; the run 12 h earlier for the
    state 6 h before the start and for rain at the start (analyses carry no rain).
    For 06/18 UTC runs, boundaries after +90 h come from the 00/12 UTC run 6 h earlier."""

    def __init__(self, cfg, cycle, n_steps):
        self.cfg = cfg
        self.step = np.timedelta64(cfg["step_hours"], "h")
        self.cycle = np.datetime64(cycle, "h")
        if pd.Timestamp(self.cycle).hour % 6:
            raise ValueError("ECMWF runs start at 00, 06, 12 or 18 UTC")
        self.n_steps = n_steps
        self.prev = self.cycle - RUN
        self.own_last = LAST[stream(self.cycle)]
        self.base = self.cycle if self.own_last >= 240 else self.cycle - H6   # an oper run
        self._fields = {}

    def _needs(self):
        """{run: steps} for the whole forecast."""
        sh = self.cfg["step_hours"]
        need = {self.prev: {0, sh, 2 * sh}, self.cycle: set()}
        for k in range(self.n_steps + 1):
            run, step = self.source(self.cycle + k * self.step)
            need.setdefault(run, set()).add(step)
            if step >= sh:
                need[run].add(step - sh)   # rain is a difference of running totals
        return need

    def prefetch(self):
        for run, steps in self._needs().items():
            self._fields[run] = download_cycle(self.cfg, run, sorted(steps))

    def _f(self, cycle, short, level, step):
        if cycle not in self._fields:
            self.prefetch()
        return self._fields[cycle][f"{short}:{level}:{step}"]

    def source(self, t):
        """(run, step) used for valid time t."""
        t = np.datetime64(t, "h")
        if t < self.cycle:
            return self.prev, int((t - self.prev) / np.timedelta64(1, "h"))
        lead = int((t - self.cycle) / np.timedelta64(1, "h"))
        if lead <= self.own_last:
            return self.cycle, lead
        return self.base, min(int((t - self.base) / np.timedelta64(1, "h")), LAST["oper"])

    def has(self, t):
        return True

    def get(self, t):
        run, step = self.source(t)
        sh = self.cfg["step_hours"]
        if run == self.cycle and step == 0:  # rain in the 6 h before the start
            rain_run, rain_step = self.prev, int(RUN / np.timedelta64(1, "h"))
        else:
            rain_run, rain_step = run, step
        out = []
        for name in self.cfg["channels"]:
            if name == "tp":  # ECMWF rain is a running total in metres
                mm = (self._f(rain_run, "tp", 0, rain_step) - self._f(rain_run, "tp", 0, rain_step - sh)) * 1000.0
                out.append(np.log1p(np.clip(mm, 0, None)))
                continue
            short, level, convert = ECMWF_FIELDS[name]
            a = self._f(run, short, level, step)
            out.append(convert(a) if convert else a)
        return np.stack(out).astype(np.float32)


def latest_cycle(cfg):
    """Most recent run (00, 06, 12 or 18 UTC) with all its steps published."""
    from .latest import latest
    return np.datetime64(pd.Timestamp(latest()).tz_localize(None), "h")
