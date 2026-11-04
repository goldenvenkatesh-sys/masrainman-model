"""Scoring forecasts against gauges and airports."""
from datetime import date

import numpy as np
import pytest

from masrainman import verify as v


def test_daily_rain_sums_each_24h_window():
    lead = np.arange(0, 49, 6)
    tp = np.ones((len(lead), 2, 2))
    tp[0] = 100  # the start state's rain is not part of the forecast
    out = v.daily_rain(tp, lead)
    assert out.shape == (2, 2, 2) and (out == 4).all()


def test_grid_gauges_averages_per_cell_and_marks_the_region():
    lat, lon = np.array([1.0, 0.75, 0.5]), np.array([0.0, 0.25, 0.5])
    rain, region, n = v.grid_gauges({"tn": [(1.0, 0.0, 10), (1.02, 0.03, 20)], "kl": [(0.5, 0.5, 4)]}, lat, lon)
    assert rain[0, 0] == 15 and n[0, 0] == 2 and region[0, 0] == v.REGIONS["tn"]
    assert rain[2, 2] == 4 and region[2, 2] == v.REGIONS["kl"]
    assert np.isnan(rain[1, 1]) and n[1, 1] == 0


def test_clean_drops_missing_and_impossible_values():
    assert v.clean([(1, 2, 3), (None, 2, 3), (1, 2, None), (1, 2, -1), (1, 2, 900)]) == [(1, 2, 3)]


def test_scores_of_a_perfect_and_a_shifted_forecast():
    o = np.zeros((20, 20))
    o[8:12, 8:12] = 30
    mask = np.ones_like(o, bool)
    good = v.score_rain(o, o, mask)
    assert good["sse"] == 0 and good["sf"] == good["so"]
    assert good["ct"]["15.6"] == [16, 0, 0, 384]
    assert good["fss"]["15.6"]["1"][0] == 0            # FSS numerator 0 = perfect
    shifted = np.roll(o, 1, axis=1)
    bad = v.score_rain(shifted, o, mask)
    a, b, c, d = bad["ct"]["15.6"]
    assert a == 12 and b == 4 and c == 4
    # a near miss is forgiven more on a bigger square
    fss = lambda s: 1 - bad["fss"]["15.6"][s][0] / bad["fss"]["15.6"][s][1]
    assert fss("5") > fss("1")


def test_gaps_in_the_gauges_are_not_scored():
    o = np.full((10, 10), np.nan)
    o[4:6, 4:6] = 20
    mask = np.isfinite(o)
    r = v.score_rain(np.full((10, 10), 20.0), o, mask)
    assert r["n"] == 4 and r["sse"] == 0 and r["fss"]["15.6"]["3"][0] == pytest.approx(0)


def test_parse_metars_keeps_synoptic_hours_in_utc():
    rows = [{"icaoId": "VOMM", "reportTime": "2026-11-04T00:00:00.000Z", "altim": 1012, "wspd": 10},
            {"icaoId": "VOMM", "reportTime": "2026-11-04T00:30:00.000Z", "altim": 1013, "wspd": 2},
            {"icaoId": "VOMM", "reportTime": "2026-11-04T03:00:00.000Z", "altim": 1013, "wspd": 2},
            {"icaoId": "VOBL", "reportTime": "2026-11-04T06:00:00.000Z", "altim": None, "wspd": None}]
    out = v.parse_metars(rows)
    assert out["VOMM"] == {"2026-11-04T00": (1012, 5.14)}
    assert out["VOBL"] == {"2026-11-04T06": (None, None)}


def test_at_points_is_bilinear_with_lat_running_south():
    lat, lon = np.array([2.0, 1.0, 0.0]), np.array([0.0, 1.0])
    f = np.array([[2.0, 2.0], [1.0, 1.0], [0.0, 0.0]])
    assert v.at_points(f, lat, lon, [(0.5, 0.5)])[0] == pytest.approx(0.5)


def test_records_line_up_forecast_days_with_rain_days(tmp_path):
    day = date(2026, 11, 2)
    rain = np.zeros((10, 6, 6))
    rain[0] = 20  # Day 1 of the 2 Nov run = the rain-day ending 3 Nov
    v.save_fc(tmp_path, "masrainman", day, rain)
    obs = np.full((6, 6), 20.0, np.float32)
    region = np.full((6, 6), v.REGIONS["tn"], np.uint8)
    p = v.obs_path(tmp_path, date(2026, 11, 3))
    p.parent.mkdir()
    np.savez_compressed(p, rain=obs, region=region, n=np.ones((6, 6)))
    recs = v.rain_records(tmp_path, date(2026, 1, 1))
    assert {(r["lead"], r["region"]) for r in recs} == {(1, "all"), (1, "tn")}
    assert all(r["sse"] == 0 and r["day"] == "2026-11-03" for r in recs)


def test_airport_errors_go_to_the_forecast_day_of_their_lead(tmp_path):
    pts = np.zeros((40, len(v.AIRPORTS), 3), np.float32)
    pts[..., 0] = 1010
    pts[..., 1] = 3
    pts[..., 2] = 4
    v.save_fc(tmp_path, "masrainman", date(2026, 11, 2), np.zeros((10, 2, 2)), pts)
    metars = {"VOMM": {"2026-11-03T00": (1008, 3.0), "2026-11-03T06": (None, 5.0)}}
    recs = v.point_records(tmp_path, metars, date(2026, 1, 1))
    day1 = [r for r in recs if r["day"] == "2026-11-03" and r["lead"] == 1][0]
    day2 = [r for r in recs if r["day"] == "2026-11-03" and r["lead"] == 2][0]
    assert day1["np_"] == 1 and day1["sp"] == 2 and day1["ssw"] == 4   # wind 5 m/s forecast, 3 observed
    assert day2["np_"] == 0 and day2["nw"] == 1 and day2["sw"] == 0


def test_gauges_reads_every_state_file(monkeypatch):
    files = {"kerala": {"stations": [{"lat": 10.0, "lon": 76.0, "rain": 5}]},
             "ap/daily": {"rain": {"7": 2.0, "unknown": 9.0}}, "ap/stations": {"7": {"lat": 16.0, "lon": 80.0}},
             "tg/daily": {"points": [{"lat": 17.0, "lon": 78.0, "rain": 1}]},
             "ka/daily": {"points": [[13.0, 76.0, 3.0], [13.1, 76.1, None]]},
             "hourly": [{"latitude": 11.0, "longitude": 79.0, "total": 4}]}
    monkeypatch.setattr(v, "_json", lambda url, timeout=120: next((b for k, b in files.items() if k in url), None))
    g = v.gauges(date(2026, 11, 3))
    assert g == {"tn": [(11.0, 79.0, 4)], "kl": [(10.0, 76.0, 5)], "ap": [(16.0, 80.0, 2.0)],
                 "tg": [(17.0, 78.0, 1)], "ka": [(13.0, 76.0, 3.0)]}
