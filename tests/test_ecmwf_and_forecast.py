"""ECMWF live data mapping (no network) and the hourly first-day output."""
import numpy as np

from masrainman.ecmwf import ECMWF_FIELDS, ECMWFArchive
from masrainman.forecast import hourly_first_day

CYCLE = np.datetime64("2026-10-28T00", "h")


def fake_fields(rain_m):
    """One run's cropped fields; rain_m maps step -> running total in metres."""
    out = {}
    for step, total in rain_m.items():
        for short, level, _ in ECMWF_FIELDS.values():
            value = {"2t": 300.0, "2d": 290.0, "msl": 101000.0, "q": 0.01}.get(short, 7.0)
            out[f"{short}:{level}:{step}"] = np.full((9, 9), value, np.float32)
        out[f"tp:0:{step}"] = np.full((9, 9), total, np.float32)
    return out


def archive(cfg):
    a = ECMWFArchive(cfg, CYCLE, n_steps=2)
    a._fields = {a.prev: fake_fields({0: 0.0, 6: 0.002, 12: 0.005}),
                 a.cycle: fake_fields({0: 0.0, 6: 0.003, 12: 0.004})}
    return a


def test_rain_windows(cfg):
    a, h6 = archive(cfg), np.timedelta64(6, "h")
    tp = cfg["channels"].index("tp")
    rain = lambda t: float(np.expm1(a.get(t)[tp]).mean())
    assert np.isclose(rain(CYCLE - h6), 2.0)      # previous run 0-6 h
    assert np.isclose(rain(CYCLE), 3.0)           # previous run 6-12 h (analysis has no rain)
    assert np.isclose(rain(CYCLE + h6), 3.0)      # this run 0-6 h
    assert np.isclose(rain(CYCLE + 2 * h6), 1.0)  # this run 6-12 h


def test_units(cfg):
    x = archive(cfg).get(CYCLE)
    ch = cfg["channels"]
    assert np.isclose(x[ch.index("t2m")].mean(), 26.85)
    assert np.isclose(x[ch.index("msl")].mean(), 1010.0)
    assert np.isclose(x[ch.index("q850")].mean(), 10.0)
    assert np.isclose(x[ch.index("z500")].mean(), 7.0)  # gh is already in metres


def test_hourly_first_day(cfg):
    C = len(cfg["channels"])
    leads = np.arange(0, 25, 6)
    fields = np.broadcast_to(leads[:, None, None, None], (5, C, 2, 2)).astype(np.float32).copy()
    tp = cfg["channels"].index("tp")
    fields[:, tp] = 6.0  # 6 mm in every 6-h step
    hours, hourly = hourly_first_day(cfg, fields)
    assert list(hours) == list(range(25))
    t2m = cfg["channels"].index("t2m")
    np.testing.assert_allclose(hourly[:, t2m, 0, 0], hours, atol=1e-5)  # linear in time
    np.testing.assert_allclose(hourly[:, tp], 1.0)                      # 1 mm per hour


def test_06_utc_run_takes_late_boundaries_from_the_older_long_run(cfg):
    c = np.datetime64("2026-11-04T06", "h")
    a = ECMWFArchive(cfg, c, n_steps=40)
    h = lambda n: np.timedelta64(n, "h")
    base = np.datetime64("2026-11-04T00", "h")
    assert a.source(c - h(6)) == (np.datetime64("2026-11-03T18", "h"), 6)
    assert a.source(c + h(90)) == (c, 90)
    assert a.source(c + h(96)) == (base, 102)
    assert a.source(c + h(234)) == (base, 240)
    assert a.source(c + h(240)) == (base, 240)       # held at the last step
    need = a._needs()
    assert max(need[c]) == 90 and min(need[base]) == 96 and max(need[base]) == 240


def test_00_utc_run_uses_its_own_run_to_the_end(cfg):
    a = ECMWFArchive(cfg, CYCLE, n_steps=40)
    assert a.source(CYCLE + np.timedelta64(240, "h")) == (CYCLE, 240)
    assert set(a._needs()) == {a.prev, CYCLE}


def test_latest_run_picks_the_newest_complete_one():
    from datetime import datetime, timezone
    from masrainman.latest import index_url, latest
    now = datetime(2026, 11, 4, 14, 20, tzinfo=timezone.utc)
    out = {index_url(datetime(2026, 11, 4, 0, tzinfo=timezone.utc))}
    assert latest(now, exists=out.__contains__).hour == 0          # 12 not run yet, 06 not complete
    out.add(index_url(datetime(2026, 11, 4, 6, tzinfo=timezone.utc)))
    assert latest(now, exists=out.__contains__).hour == 6
    assert "/06z/ifs/0p25/scda/2026110406" in index_url(datetime(2026, 11, 4, 6)) and "-90h-scda" in index_url(datetime(2026, 11, 4, 6))
