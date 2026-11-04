"""Check live forecasts against what was observed.

  python -m masrainman.verify archive --nc out/live/fc_2026110400.nc --data vd
  python -m masrainman.verify update --data vd --site vs

archive  keeps the small part of a 00 UTC forecast that is needed later: 24-h rain and the
         MSLP / wind at the airports, for MasRainMan and for ECMWF IFS and AIFS (rain).
update   collects the gauges and airport reports, scores every forecast with observations
         and writes the Verification page's data and maps to the site folder.

Rain: 24-h totals of the state gauges (TN-SMART, Kerala, Karnataka, Andhra Pradesh, Telangana;
the same files as the MasRainman rain reports) averaged onto the model's 0.25 deg cells. Rain-day D ends
08:30 IST on D; forecast Day k of a 00 UTC run ends 05:30 IST, so 21 of the 24 hours overlap.
Scores are stored as sums so the page can add up any period and region.
MSLP and wind: airport reports (METAR), whose QNH pressure equals MSLP near sea level.
The gridded gauge rain under <data>/obs_rain is also the target for later training.
"""
import argparse
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates, uniform_filter
import xarray as xr

from .config import grid, load_config

THRESHOLDS = (2.5, 15.6, 64.5)   # IMD: rainy day, moderate, heavy
SCALES = (1, 3, 5, 9)            # near-miss squares in cells: ~25, 75, 125, 225 km
KEEP_DAYS = 180
MAP_DAYS = 30
MAP_LEADS = (1, 2, 3, 5)
REGIONS = {"tn": 1, "kl": 2, "ap": 3, "tg": 4, "ka": 5}
REGION_NAMES = {"all": "South India", "tn": "Tamil Nadu", "kl": "Kerala", "ka": "Karnataka", "ap": "Andhra Pradesh",
                "tg": "Telangana"}
MODEL_NAMES = {"masrainman": "MasRainMan AI", "ifs": "ECMWF IFS", "aifs": "ECMWF AIFS"}

# ICAO: name, lat, lon, elevation (m). Above about 300 m QNH is not quite MSLP: read the bias with care.
AIRPORTS = {
    "VOMM": ("Chennai", 12.99, 80.17, 16), "VOTR": ("Tiruchirappalli", 10.77, 78.71, 88),
    "VOMD": ("Madurai", 9.83, 78.09, 141), "VOCB": ("Coimbatore", 11.03, 77.04, 404),
    "VOTV": ("Thiruvananthapuram", 8.48, 76.92, 5), "VOCI": ("Kochi", 10.15, 76.40, 9),
    "VOCL": ("Kozhikode", 11.14, 75.96, 104), "VOML": ("Mangaluru", 12.96, 74.89, 103),
    "VOBL": ("Bengaluru", 13.20, 77.71, 915), "VOHS": ("Hyderabad", 17.24, 78.43, 617),
    "VOVZ": ("Visakhapatnam", 17.72, 83.22, 3), "VOTP": ("Tirupati", 13.63, 79.54, 107),
}
METAR_URL = "https://aviationweather.gov/api/data/metar"
TN_URL = "https://beta-tnsmart.rimes.int/Rainfall_Python_IDW/Hourly_json/hourly_rainfall_{d}.json"
REPORT_DATA = "https://raw.githubusercontent.com/goldenvenkatesh-sys/masrainman-rain-report/main/data"
KT = 0.514444


def at_points(field, lat, lon, points):
    """Bilinear values of (..., H, W) arrays at (lat, lon) points; lat runs north to south."""
    field = np.asarray(field, np.float32)
    dy, dx = lat[1] - lat[0], lon[1] - lon[0]
    rows = (np.array([p[0] for p in points]) - lat[0]) / dy
    cols = (np.array([p[1] for p in points]) - lon[0]) / dx
    flat = field.reshape(-1, *field.shape[-2:])
    out = [map_coordinates(f, [rows, cols], order=1, mode="nearest") for f in flat]
    return np.array(out).reshape(*field.shape[:-2], len(points))


# ---------------------------------------------------------------- forecast archive

def daily_rain(tp, lead):
    """(T, H, W) 6-h rain at lead hours -> ten 24-h totals, each ending at a multiple of 24 h."""
    out = [tp[(lead > k * 24) & (lead <= (k + 1) * 24)].sum(0) for k in range(int(lead.max() // 24))]
    return np.array(out, np.float32)


def fc_path(data, model, day):
    return Path(data) / "fc" / f"{model}_{day:%Y%m%d}.npz"


def save_fc(data, model, day, rain, points=None):
    p = fc_path(data, model, day)
    p.parent.mkdir(parents=True, exist_ok=True)
    extra = {} if points is None else {"points": points.astype(np.float32)}
    np.savez_compressed(p, rain=np.round(np.clip(rain, 0, 6000) * 10).astype(np.uint16), **extra)


def load_fc(path):
    z = np.load(path)
    return z["rain"] / 10.0, (z["points"] if "points" in z else None)


def ecmwf_rain(cfg, model, cycle):
    """24-h rain of an ECMWF run from open data, cropped to the domain (about 1 MB per step)."""
    import tempfile

    from ecmwf.opendata import Client

    from .ecmwf import _crop
    ts = pd.Timestamp(cycle)
    steps = list(range(24, 241, 24))
    client = Client(source="ecmwf", model={"ifs": "ifs", "aifs": "aifs-single"}[model], resol="0p25")
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "tp.grib2"
        client.retrieve(date=ts.strftime("%Y%m%d"), time=ts.hour, stream="oper", type="fc",
                        step=steps, param="tp", target=str(target))
        fields = {}
        _crop(target, cfg, fields)
    total = np.stack([fields[f"tp:0:{s}"] for s in steps])
    if model == "ifs":
        total = total * 1000.0   # IFS gives metres, AIFS kg/m2 (= mm)
    return np.clip(np.diff(total, axis=0, prepend=0.0), 0, None)


def ecmwf_points(cfg, cycle, lat, lon, cache):
    """MSLP and 10 m wind of the IFS run at the airports, if the live forecast cached that run."""
    path = Path(cache) / pd.Timestamp(cycle).strftime("%Y%m%d%H") / "fields.npz"
    if not path.exists():
        return None
    f = np.load(path)
    pts = [a[1:3] for a in AIRPORTS.values()]
    steps = range(6, 241, 6)
    try:
        msl = np.stack([f[f"msl:0:{s}"] for s in steps]) / 100.0
        u = np.stack([f[f"10u:0:{s}"] for s in steps])
        v = np.stack([f[f"10v:0:{s}"] for s in steps])
    except KeyError:
        return None
    return np.stack([at_points(a, lat, lon, pts) for a in (msl, u, v)], -1)


def archive(args):
    cfg = load_config()
    ds = xr.open_dataset(args.nc)
    cycle = pd.Timestamp(ds.attrs["init_time"])
    if cycle.hour != 0:
        print(f"{cycle}: only 00 UTC runs are scored")
        return
    day = cycle.date()
    lat, lon = ds.lat.values, ds.lon.values
    lead = ds.lead.values
    tp = ds.tp.values
    keep = lead > 0
    pts = [a[1:3] for a in AIRPORTS.values()]
    points = np.stack([at_points(ds[v].values[keep], lat, lon, pts) for v in ("msl", "u10", "v10")], -1)
    save_fc(args.data, "masrainman", day, daily_rain(tp, lead), points)
    print(f"archived MasRainMan {day}")
    for model in ("ifs", "aifs"):
        if fc_path(args.data, model, day).exists():
            continue
        try:
            rain = ecmwf_rain(cfg, model, cycle)
            ifs_pts = ecmwf_points(cfg, cycle, lat, lon, args.ecmwf_cache) if model == "ifs" else None
            save_fc(args.data, model, day, rain, ifs_pts)
            print(f"archived {model} {day}")
        except Exception as e:  # the comparison model must never stop the check of our own
            print(f"{model} {day}: not available ({e})")


# ---------------------------------------------------------------- observations

def _json(url, timeout=120):
    import requests
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "masrainman-verify"})
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()


def clean(rows):
    return [(la, lo, r) for la, lo, r in rows if None not in (la, lo, r) and 0 <= r < 700]


def gauges(day):
    """{region: [(lat, lon, 24-h rain mm)]} for the rain-day ending 08:30 IST on `day`."""
    d = day.isoformat()
    out = {}
    js = _json(TN_URL.format(d=d))
    if js:
        out["tn"] = clean([(s.get("latitude"), s.get("longitude"), s.get("total")) for s in js])
    js = _json(f"{REPORT_DATA}/kerala/daily/{d}.json")
    if js:
        out["kl"] = clean([(s.get("lat"), s.get("lon"), s.get("rain")) for s in js["stations"]])
    js = _json(f"{REPORT_DATA}/ap/daily/{d}.json")
    if js:
        known = _json(f"{REPORT_DATA}/ap/stations.json") or {}
        out["ap"] = clean([(known[k]["lat"], known[k]["lon"], r) for k, r in js["rain"].items() if k in known])
    js = _json(f"{REPORT_DATA}/tg/daily/{d}.json")
    if js and "points" in js:
        out["tg"] = clean([(s.get("lat"), s.get("lon"), s.get("rain")) for s in js["points"]])
    js = _json(f"{REPORT_DATA}/ka/daily/{d}.json")  # running totals up to the day's last live check
    if js:
        out["ka"] = clean([tuple(p) for p in js["points"]])
    return {k: v for k, v in out.items() if v}


def grid_gauges(by_region, lat, lon):
    """Mean gauge rain per 0.25 deg cell. Returns rain (NaN = no gauge), region code, gauges per cell."""
    H, W = len(lat), len(lon)
    total, n = np.zeros((H, W)), np.zeros((H, W), np.uint16)
    region = np.zeros((H, W), np.uint8)
    dy, dx = lat[1] - lat[0], lon[1] - lon[0]
    for name, rows in by_region.items():
        for la, lo, r in rows:
            i, j = int(round((la - lat[0]) / dy)), int(round((lo - lon[0]) / dx))
            if 0 <= i < H and 0 <= j < W:
                total[i, j] += r
                n[i, j] += 1
                region[i, j] = REGIONS[name]
    rain = np.where(n > 0, total / np.maximum(n, 1), np.nan).astype(np.float32)
    return rain, region, n


def obs_path(data, day):
    return Path(data) / "obs_rain" / f"{day:%Y%m%d}.npz"


def update_rain_obs(data, lat, lon, now, days=7):
    """Fetch the last rain-days; recent ones again, as late states may have arrived.
    `now` is IST; today's rain-day is only complete after 08:30 (files appear by about 09:30)."""
    first = 0 if (now.hour, now.minute) >= (9, 30) else 1
    for k in range(first, days):
        day = now.date() - timedelta(days=k)
        p = obs_path(data, day)
        if p.exists() and k > 2:
            continue
        try:
            by_region = gauges(day)
        except Exception as e:
            print(f"gauges {day}: {e}")
            continue
        if not by_region:
            continue
        rain, region, n = grid_gauges(by_region, lat, lon)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, rain=rain, region=region, n=n)
        print(f"gauges {day}: " + ", ".join(f"{k} {len(v)}" for k, v in by_region.items()))


def parse_metars(rows):
    """{icao: {'YYYY-MM-DDTHH': (pressure hPa, wind m/s)}} at 00/06/12/18 UTC only."""
    out = {}
    for r in rows:
        t = pd.Timestamp(r["reportTime"]).tz_localize(None) if pd.Timestamp(r["reportTime"]).tz is None \
            else pd.Timestamp(r["reportTime"]).tz_convert(None)
        if t.hour % 6 or t.minute:
            continue
        p, w = r.get("altim"), r.get("wspd")
        out.setdefault(r["icaoId"], {})[t.strftime("%Y-%m-%dT%H")] = (p, None if w is None else round(w * KT, 2))
    return out


def update_metar(data, hours=168):
    """Add airport reports to <data>/obs_metar.json (the feed keeps only the last few days)."""
    p = Path(data) / "obs_metar.json"
    have = json.loads(p.read_text()) if p.exists() else {}
    for code in AIRPORTS:  # one at a time: the feed returns at most about 400 rows per request
        try:
            rows = _json(f"{METAR_URL}?ids={code}&format=json&hours={hours}")
        except Exception as e:
            print(f"METAR {code}: {e}")
            continue
        for icao, obs in parse_metars(rows or []).items():
            have.setdefault(icao, {}).update(obs)
    p.write_text(json.dumps(have, separators=(",", ":")))
    return have


# ---------------------------------------------------------------- scores

def contingency(f, o, t):
    a, b = int(((f >= t) & (o >= t)).sum()), int(((f >= t) & (o < t)).sum())
    c, d = int(((f < t) & (o >= t)).sum()), int(((f < t) & (o < t)).sum())
    return [a, b, c, d]   # hits, false alarms, misses, correct negatives


def fractions(f, o, mask, t, size):
    """Numerator and denominator of the Fractions Skill Score over the gauge cells."""
    w = uniform_filter(mask.astype(float), size, mode="constant")
    ok = mask & (w > 0)
    ff = uniform_filter(((f >= t) & mask).astype(float), size, mode="constant")[ok] / w[ok]
    fo = uniform_filter(((o >= t) & mask).astype(float), size, mode="constant")[ok] / w[ok]
    return [round(float(((ff - fo) ** 2).sum()), 3), round(float((ff ** 2 + fo ** 2).sum()), 3)]


def score_rain(fc, obs, mask):
    f, o = fc[mask], obs[mask]
    rec = {"n": int(mask.sum()), "sf": round(float(f.sum()), 1), "so": round(float(o.sum()), 1),
           "sse": round(float(((f - o) ** 2).sum()), 1), "sff": round(float((f * f).sum()), 1),
           "soo": round(float((o * o).sum()), 1), "sfo": round(float((f * o).sum()), 1),
           "ct": {str(t): contingency(f, o, t) for t in THRESHOLDS},
           "fss": {str(t): {str(s): fractions(fc, obs, mask, t, s) for s in SCALES} for t in THRESHOLDS}}
    return rec


def rain_records(data, cutoff):
    records = []
    for path in sorted((Path(data) / "fc").glob("*.npz")):
        model, tag = path.stem.rsplit("_", 1)
        start = datetime.strptime(tag, "%Y%m%d").date()
        rain = load_fc(path)[0]
        for k in range(len(rain)):
            day = start + timedelta(days=k + 1)
            p = obs_path(data, day)
            if day < cutoff or not p.exists():
                continue
            z = np.load(p)
            obs, region = z["rain"], z["region"]
            have = np.isfinite(obs)
            masks = {"all": have, **{r: have & (region == c) for r, c in REGIONS.items()}}
            for name, mask in masks.items():
                if mask.sum() >= 5:
                    records.append({"day": day.isoformat(), "lead": k + 1, "model": model, "region": name,
                                    **score_rain(rain[k], obs, mask)})
    return records


def point_records(data, metars, cutoff):
    """Error sums per airport, model, valid day and forecast day (lead 6-24 h = Day 1 ...)."""
    names = list(AIRPORTS)
    sums = {}
    for path in sorted((Path(data) / "fc").glob("*.npz")):
        model, tag = path.stem.rsplit("_", 1)
        pts = load_fc(path)[1]
        if pts is None:
            continue
        start = pd.Timestamp(datetime.strptime(tag, "%Y%m%d"))
        for k in range(pts.shape[0]):
            lead = (k + 1) * 6
            t = start + pd.Timedelta(hours=lead)
            key_t = t.strftime("%Y-%m-%dT%H")
            if t.date() < cutoff:
                continue
            for j, icao in enumerate(names):
                ob = metars.get(icao, {}).get(key_t)
                if ob is None:
                    continue
                p_o, w_o = ob
                msl, u, v = (float(x) for x in pts[k, j])
                r = sums.setdefault((t.date().isoformat(), (lead + 23) // 24, model, icao),
                                    dict(np_=0, sp=0.0, ssp=0.0, nw=0, sw=0.0, ssw=0.0))
                if p_o is not None:
                    e = msl - p_o
                    r["np_"] += 1; r["sp"] += e; r["ssp"] += e * e
                if w_o is not None:
                    e = float(np.hypot(u, v)) - w_o
                    r["nw"] += 1; r["sw"] += e; r["ssw"] += e * e
    return [{"day": d, "lead": l, "model": m, "icao": i, **{k: round(v, 2) for k, v in r.items()}}
            for (d, l, m, i), r in sorted(sums.items())]


# ---------------------------------------------------------------- site files

def check_map(day, lead, obs, fields, lat, lon, out):
    """Observed gauge cells beside the forecasts of one rain-day, on the rain colour bar."""
    from .plot import RAIN_LEVELS, draw_boundaries, rain_cmap, smooth
    import matplotlib.pyplot as plt
    cmap, norm = rain_cmap()
    n = 1 + len(fields)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n + 1, 5.2), squeeze=False)
    box = xr.Dataset(coords={"lat": [7.0, 21.0], "lon": [72.5, 86.5]})
    ax = axes[0, 0]
    ax.set_facecolor("#ffffff")
    ax.pcolormesh(lon, lat, np.ma.masked_invalid(np.where(obs < 1, 1.0, np.nan)), cmap="Greys", vmin=0, vmax=8,
                  shading="nearest")
    m = ax.pcolormesh(lon, lat, np.ma.masked_invalid(np.where(obs >= 1, obs, np.nan)), cmap=cmap, norm=norm,
                      shading="nearest")
    ax.set_title(f"Observed gauges\nrain-day to 08:30 IST {day:%d %b %Y}", fontsize=10)
    draw_boundaries(ax, box)
    for ax, (name, f) in zip(axes[0, 1:], fields.items()):
        ax.contourf(lon, lat, smooth(f, 0.8), levels=RAIN_LEVELS, cmap=cmap, norm=norm, extend="max")
        ax.set_title(f"{MODEL_NAMES[name]}\nDay {lead} forecast", fontsize=10)
        draw_boundaries(ax, box)
    fig.tight_layout(rect=(0, 0, 0.93, 1))
    cax = fig.add_axes([0.94, 0.12, 0.012, 0.76])
    cb = fig.colorbar(m, cax=cax, label="mm in 24 h", ticks=RAIN_LEVELS, spacing="uniform", extend="max")
    cb.ax.tick_params(labelsize=6)
    fig.savefig(out, dpi=95)
    plt.close(fig)


def write_maps(data, site, lat, lon, today):
    (Path(site) / "maps").mkdir(parents=True, exist_ok=True)
    index = []
    for k in range(MAP_DAYS + 1):
        day = today - timedelta(days=k)
        p = obs_path(data, day)
        if not p.exists():
            continue
        obs = np.load(p)["rain"]
        leads = []
        for lead in MAP_LEADS:
            start = day - timedelta(days=lead)
            fields = {}
            for model in MODEL_NAMES:
                fp = fc_path(data, model, start)
                if fp.exists():
                    fields[model] = load_fc(fp)[0][lead - 1]
            if "masrainman" not in fields:
                continue
            name = f"{day:%Y%m%d}_d{lead}.png"
            check_map(day, lead, obs, fields, lat, lon, Path(site) / "maps" / name)
            leads.append({"lead": lead, "file": f"maps/{name}"})
        if leads:
            index.append({"day": day.isoformat(), "maps": leads})
    return index


def update(args):
    cfg = load_config()
    lat, lon = grid(cfg)
    now = datetime.now(timezone(timedelta(hours=5, minutes=30)))
    today = now.date()
    Path(args.data).mkdir(parents=True, exist_ok=True)
    update_rain_obs(args.data, lat, lon, now)
    metars = update_metar(args.data)
    cutoff = today - timedelta(days=KEEP_DAYS)
    rain = rain_records(args.data, cutoff)
    points = point_records(args.data, metars, cutoff)
    site = Path(args.site)
    site.mkdir(parents=True, exist_ok=True)
    days = write_maps(args.data, site, lat, lon, today)
    (site / "scores.json").write_text(json.dumps(
        {"updated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M"),
         "thresholds": THRESHOLDS, "scales": SCALES, "regions": REGION_NAMES, "models": MODEL_NAMES,
         "airports": {k: {"name": v[0], "elev": v[3]} for k, v in AIRPORTS.items()},
         "rain": rain, "points": points, "days": days}, separators=(",", ":")))
    print(f"{len(rain)} rain scores, {len(points)} airport scores, {len(days)} days with maps")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("archive")
    a.add_argument("--nc", required=True)
    a.add_argument("--data", required=True)
    a.add_argument("--ecmwf-cache", default="operational/ecmwf")
    a.set_defaults(fn=archive)
    u = sub.add_parser("update")
    u.add_argument("--data", required=True)
    u.add_argument("--site", required=True)
    u.set_defaults(fn=update)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
