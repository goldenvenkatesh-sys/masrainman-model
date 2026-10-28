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

## Resume from the last point (e.g. Saturday night)
The downloader skips every file that is already complete, so just run it again.
Half-finished files end in `.part` and are downloaded again automatically. Each queued
request's ID is kept in a `.job` file, so a restart rejoins the same request (no duplicates).

```
git clone https://github.com/goldenvenkatesh-sys/masrainman-model.git D:\masrainman
cd D:\masrainman
powershell -ExecutionPolicy Bypass -File scripts\download_era5.ps1
```
The script runs in its own window (it survives Claude app disconnects), retries after errors
every 5 minutes, and logs to `D:\masrainman_data\download.log`.
Check progress: files land in `D:\masrainman_data\raw\<year>\` as `sfc_YYYY.nc`,
`pl_YYYY.nc` and `tp_YYYY.nc` - one file per type per year (18 files in total, plus `static.nc`).

## Next steps after the download
```
python -m masrainman.preprocess --years 2018 2023
python -m masrainman.train --name baseline --epochs 20
python -m masrainman.train --name baseline --resume --rollout 4 --epochs 5 --lr 5e-5
python -m masrainman.evaluate --ckpt D:/masrainman_runs/baseline/best.pt
```
Training also resumes: stop with Ctrl+C, then rerun with `--resume`.

## Gotchas found so far
- CDS key: the downloader reads `%USERPROFILE%\.cdsapirc` explicitly, because an old
  `%USERPROFILE%\.ecmwfdatastoresrc` holds an outdated key that the library would use first (401 error).
- Run only one download at a time (Copernicus may disable accounts that send parallel requests).
- Data must stay on D: (C: has about 6 GB free).
