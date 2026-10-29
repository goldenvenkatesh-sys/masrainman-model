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

## Download progress (29 Oct 2026, 05:50 IST)
- 2018: `sfc_2018.nc` and `pl_2018.nc` done; `tp_2018` being processed at Copernicus.
- 29 Oct fix: the first `pl_2018.nc` had empty 850 hPa winds (merge bug, now fixed and
  tested); it was rebuilt from the finished Copernicus results and checked field by field.
- Each year needs 6 requests: sfc, pressure levels at 200 / 500 hPa (one each), 850 hPa
  (two half-year chunks, merged automatically), and hourly rain. 36 requests for 2018-2023.
- Auto-start fires about 5 minutes after Windows login (Windows delays Startup apps).

## Trial run (29 Oct 2026): train 2018 only, 6 epochs, ~42 min on CPU
Config: `D:\masrainman_runs\trial\config.trial.yaml`; model `D:\masrainman_runs\trial\best.pt`
(epoch 5, val loss 0.114); scores in `D:\masrainman_runs\trial\scores.csv/.png` (71 forecasts in 2019).
- The pipeline works end to end on real ERA5.
- Learned: daily cycles. 6-h errors are 3-4x below persistence for t2m, msl and z500; rain
  beats persistence for days 1-3 (15% / 9% lower RMSE; CSI >=2.5 mm/6h: 0.35 day 1, 0.16 day 10).
- Not learned yet: moving systems. One-step changes in winds/moisture are only 10-25% of
  the real ones (u850, u200, v200 are near persistence). Cyclone Fani (init 2019-04-29) stays
  in place and weakens (1004 hPa vs <960 hPa observed). Errors at whole-day leads exceed
  persistence for t2m and z500 beyond ~2 days (single-step training drifts).
- Visible seam at the boundary ring.
- Expected fixes: 4 training years, more epochs, multi-step (rollout) fine-tuning,
  per-channel loss weights, a smooth blend at the boundary instead of a hard ring.

## Trial 2 (29 Oct 2026): tendency-unit loss + smooth boundary, 6 epochs, no fine-tune yet
Model `D:\masrainman_runs\trial2\best.pt` (epoch 5); scores `D:\masrainman_runs\trial2\scores.csv`.
- Better: rain amount RMSE beats persistence at every lead (+19% day 1 ... +24% day 10; trial 1
  only days 1-3); z500 day 10 27.3 -> 22.8 m; no boundary seam.
- Worse: heavy-rain CSI (day 1 0.35 -> 0.27, rain smoother); t2m beyond day 1 (day 3 2.10 -> 2.75 C).
- Winds/moisture near persistence: investigated 29 Oct evening, NOT a bug.
  * Data aligned in time (u10-u850 correlation peaks at lag 0).
  * 6-h changes are mostly unpredictable in the tropics: a linear model (2018 -> 2019, per
    point, all 14 fields + tendency + time of day) explains msl 87%, z500 87%, t2m 58%, tp 28%,
    v850 17%, u850 16%, q850 13%, tcwv 8%, u200 4%, v200 4% of the 6-h change.
  * Simple advection by the 850 hPa wind is worse than persistence.
  * Trial 2 explains only ~4% for u850 vs 16% for the linear model -> under-trained for winds
    (1,100 steps), so more training/data should help.
- Not done yet: stage 2 (4-step / 24-h fine-tune). Rerun with
  `python -m masrainman.train --config D:\masrainman_runs\trial\config.trial.yaml --name trial2 --resume --rollout 4 --epochs 1 --steps-per-epoch 120 --val-steps 15 --lr 5e-5`

## Trial 3 (29 Oct 2026 night): trial 2 + 6 more epochs + 24-h (4-step) fine-tune
Model `D:/masrainman_runs/trial3/best.pt` (one-step best before fine-tune: `best_stage1.pt`).
Much better than trials 1-2 and now beats persistence at most leads (days 3-10):
msl +18-26%, z500 +19-31%, rain RMSE +22-26%, v850 +14-15%, u850 +5-7%, tcwv +5-11%;
heavy-rain CSI best so far (day 1 0.36, day 3 0.34, day 10 0.23). t2m at whole days still
21-30% worse than persistence (but day 10 error 3.95 -> 1.75 C); u200 ~ persistence.
=> more training was the cure for the winds, as the diagnosis said.
Still weak: Cyclone Fani (init 2019-04-29) is filled in and not moved (truth 975 hPa at
15.8N 84.5E on day 3; model ~1004 hPa, broad weak swirl further south). MSE training damps
rare extremes and one year holds few cyclones -> more data, then consider extreme-aware
losses/oversampling of cyclone cases. A faint frame is still visible where the blend starts.

## Resume from the last point
**Automatic since 28 Oct 2026:** the download starts by itself at every Windows login
(shortcut "MasRainMan ERA5 download" in the Startup folder, made by
`scripts\install_autostart.ps1`; remove with `-Remove`). It runs from the permanent copy
`D:\masrainman-model` (a clone of this repo; `D:\masrainman` is a separate folder of the
user's web app). Just switch the laptop on and log in.

The downloader skips every finished file and rejoins queued requests via `.job` files, so
restarts never send duplicates; a second copy exits at once if one is already running.
To start it by hand: `powershell -ExecutionPolicy Bypass -File D:\masrainman-model\scripts\download_era5.ps1`
Log: `D:\masrainman_data\download.log`. Files land in `D:\masrainman_data\raw\<year>\` as
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
