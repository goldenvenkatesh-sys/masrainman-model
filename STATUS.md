# MasRainMan — where we stopped (28 Oct 2026)

## Done
- Full pipeline built and tested on fake data: download → preprocess → train → evaluate → forecast → maps.
- Repo: https://github.com/goldenvenkatesh-sys/masrainman-model (private). The live app repo
  `goldenvenkatesh-sys/masrainman` is separate and must not be force-pushed.
- Years in `config.yaml`: train 2018–2021, val 2022, test 2023.
- ERA5 download for 2018–2023 started 28 Oct 2026 15:27. `static.nc` arrived and matches the grid.
- Live forecasts from ECMWF open data work (`masrainman/ecmwf.py`, `masrainman/live_forecast.py`),
  tested on the 2026-10-28 00 UTC run with a test checkpoint. About 430 MB of downloads per run, cut
  to the domain. GFS was tried first and dropped in favour of ECMWF (same model family as ERA5). They need a real trained checkpoint to be useful:
  `python -m masrainman.live_forecast --ckpt D:/masrainman_runs/baseline/best.pt --plot`

## Download progress (28 Oct 2026, 20:05 IST)
- The CDS Terms of Use rev. 13 block is resolved (the account shows rev. 13 since ~19:50).
- Done: `static.nc`, `sfc_2018.nc` (203 MB).
- Each year needs 6 requests: sfc, pressure levels at 200 / 500 hPa (one each), 850 hPa
  (two half-year chunks, merged automatically), and hourly rain. 36 requests for 2018-2023.
- 22:15: 2018 surface, 200/500 hPa and 850 hPa Jan-Jun done; 850 hPa Jul-Dec queued. Live log: `D:\masrainman_data\download.log`.

## Resume from the last point
**Automatic since 28 Oct 2026:** the download starts by itself at every Windows login
(shortcut "MasRainMan ERA5 download" in the Startup folder, made by
`scripts\install_autostart.ps1`; remove with `-Remove`). It runs from the permanent copy
`D:\masrainman-model` (a clone of this repo; `D:\masrainman` is a separate folder of the
user's web app). Just switch the laptop on and log in.

The downloader skips every finished file and rejoins queued requests via `.job` files, so
restarts never send duplicates; a second copy exits at once if one is already running.
To start it by hand: `powershell -ExecutionPolicy Bypass -File D:\masrainman-model\scripts\download_era5.ps1`
Log: `D:\masrainman_data\download.log`. Files land in `D:\masrainman_data
aw\<year>\` as
`sfc_YYYY.nc`, `pl_YYYY.nc` and `tp_YYYY.nc` (18 in total, plus `static.nc`; pressure levels
arrive as per-level pieces first and are merged automatically).

## Next steps after the download
```
python -m masrainman.preprocess --years 2018 2023
python -m masrainman.train --name baseline --epochs 20
python -m masrainman.train --name baseline --resume --rollout 4 --epochs 5 --lr 5e-5
python -m masrainman.evaluate --ckpt D:/masrainman_runs/baseline/best.pt
```
Training also resumes: stop with Ctrl+C, then rerun with `--resume`.

Then daily live forecasts: `powershell -ExecutionPolicy Bypass -File scripts\live_daily.ps1`
(add it to Windows Task Scheduler at 14:00 IST for the 00 UTC run, 02:00 IST for 12 UTC).

Checks: `python -m pytest -q tests` (8 tests, all passing on 28 Oct 2026).

## Gotchas found so far
- CDS key: the downloader reads `%USERPROFILE%\.cdsapirc` explicitly, because an old
  `%USERPROFILE%\.ecmwfdatastoresrc` holds an outdated key that the library would use first (401 error).
- Run only one download at a time (Copernicus may disable accounts that send parallel requests).
- Data must stay on D: (C: has about 6 GB free).
