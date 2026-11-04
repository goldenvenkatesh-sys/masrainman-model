"""Comparison-model readers (offline: synthetic GRIB and listings)."""
import eccodes
import numpy as np
import pytest

from masrainman import othermodels as om


def fld(v):
    return np.full((2, 2), float(v), np.float32)


def test_running_totals_become_daily_rain():
    msgs = [{"start": 0, "end": 24 * k, "field": fld(10 * k)} for k in (1, 2, 3)]
    out = om.daily_from_accumulations(msgs)
    assert out.shape == (3, 2, 2) and (out == 10).all()


def test_six_hour_buckets_and_duplicates_are_added_once():
    msgs = []
    for s in range(6, 49, 6):
        msgs.append({"start": s - 6, "end": s, "field": fld(1)})
        msgs.append({"start": s - 6, "end": s, "field": fld(1)})   # AIGFS lists APCP twice
    msgs.append({"start": 0, "end": 12, "field": fld(2)})          # a running total beside buckets
    out = om.daily_from_accumulations(msgs)
    assert out.shape == (2, 2, 2) and (out == 4).all()


def test_no_complete_day_is_not_available():
    with pytest.raises(om.NotAvailable):
        om.daily_from_accumulations([{"start": 0, "end": 18, "field": fld(1)}])


def test_idx_ranges_pick_the_wanted_records():
    idx = ("1:0:d=2026110400:PRMSL:mean sea level:24 hour fcst:\n"
           "2:100:d=2026110400:APCP:surface:18-24 hour acc fcst:\n"
           "3:250:d=2026110400:APCP:surface:0-1 day acc fcst:\n"
           "4:400:d=2026110400:UGRD:10 m above ground:24 hour fcst:\n")
    assert om.idx_ranges(idx, {"APCP:surface"}) == [(100, 249), (250, 399)]
    assert om.idx_ranges(idx, {"UGRD:10 m above ground"}) == [(400, "")]


def test_gdps_listing_prefers_the_total_since_the_start():
    tag = "20261104T00Z"
    listing = "".join(f'<a href="{tag}_MSC_GDPS_{v}_LatLon0.15_PT024H.grib2">x</a>' for v in
                      ("AirTemp_AGL-2m", "Precip-Accum3h_Sfc", "Precip-Accum_Sfc"))
    assert om.gdps_rain_file(listing, tag) == f"{tag}_MSC_GDPS_Precip-Accum_Sfc_LatLon0.15_PT024H.grib2"
    assert om.gdps_rain_file("nothing here", tag) is None


def test_read_grib_cuts_a_global_field_to_the_domain():
    g = eccodes.codes_grib_new_from_samples("GRIB2")
    try:
        ni, nj = 360, 181   # 1 degree, north to south, longitudes 0..359 like GFS
        for k, v in {"Ni": ni, "Nj": nj, "latitudeOfFirstGridPointInDegrees": 90,
                     "latitudeOfLastGridPointInDegrees": -90, "longitudeOfFirstGridPointInDegrees": 0,
                     "longitudeOfLastGridPointInDegrees": 359, "iDirectionIncrementInDegrees": 1,
                     "jDirectionIncrementInDegrees": 1}.items():
            eccodes.codes_set(g, k, v)
        lat2, lon2 = np.meshgrid(np.linspace(90, -90, nj), np.arange(ni), indexing="ij")
        eccodes.codes_set_values(g, (lat2 + lon2 / 1000).ravel())
        blob = eccodes.codes_get_message(g)
    finally:
        eccodes.codes_release(g)
    lat, lon = np.array([20.0, 10.5, 0.0]), np.array([65.0, 80.25, 95.0])
    m = om.read_grib(blob, lat, lon)[0]
    want = lat[:, None] + lon[None, :] / 1000
    assert m["field"].shape == (3, 3) and np.allclose(m["field"], want, atol=1e-3)


def test_icon_regrid_takes_the_nearest_cells():
    clat = np.array([0.0, 0.0, 1.0, 1.0, 50.0])
    clon = np.array([70.0, 71.0, 70.0, 71.0, 10.0])
    rg = om.IconRegrid(clat, clon, np.array([1.0, 0.0]), np.array([70.0, 71.0]))
    out = rg(np.array([1.0, 2.0, 3.0, 4.0, 99.0]))
    assert np.allclose(out, [[3, 4], [1, 2]], atol=1e-3)
