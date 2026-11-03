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

## Download: COMPLETE (1 Nov 2026, 03:34 IST)
- All 18 yearly files for 2018-2023 (sfc / pl / tp) are in `D:/masrainman_data/raw`, 3.6 GB.
  Checked 1 Nov: every file on the right grid, all time steps present, zero missing values.
- The auto-start shortcut can stay (it now finds every file and exits) or be removed with
  `powershell -ExecutionPolicy Bypass -File D:/masrainman-model/scripts/install_autostart.ps1 -Remove`.

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

## Full run (1 Nov 2026): train 2018-2021, check 2022, score 2023 -- DONE
`scriptsull_run.ps1`, run dir `D:/masrainman_runs/full`. Warm start from trial 3, 12 epochs
over all 5841 training samples (~20-28 min each), then the 24-h fine-tune; total ~6.5 h
including a reboot (it resumed by itself at epoch 2). Best one-step val 0.378 (epoch 11;
trial 3 ~0.55). Model: `D:/masrainman_runs/full/best.pt` (`best_stage1.pt` = before fine-tune).
Skill vs persistence on the unseen year 2023 (RMSE gain, focus box), days 1 / 3 / 5 / 10:

| field | day 1 | day 3 | day 5 | day 10 |
|---|---|---|---|---|
| msl | +25% | +37% | +38% | +37% |
| z500 | +17% | +24% | +23% | +14% |
| rain (tp) | +22% | +23% | +33% | +33% |
| u850 / v850 | +28 / +23% | +38 / +28% | +40 / +35% | +43 / +36% |
| u10 / v10 | +25 / +22% | +35 / +36% | +39 / +38% | +45 / +38% |
| tcwv / q850 | +32 / +32% | +29 / +33% | +29 / +33% | +29 / +33% |
| t850 / d2m | +15 / +20% | +19 / +29% | +21 / +29% | +20 / +27% |
| u200 / v200 | +5 / +3% | +12 / +6% | +14 / +6% | +15 / +4% |
| t2m | -14% | -13% | -11% | -13% |

Rain CSI (>=2.5 mm/6 h): day 1 0.44, day 3 0.30, day 5 0.28, day 10 0.27.
Every field now beats persistence except t2m at whole days (error ~1.0 C day 1, 1.4 C day 5,
against 0.84 / 1.24 C for persistence: the daily cycle is easy to persist, so the model
needs to learn it better). Trial 3 was scored on 2019, so the comparison is indicative.

Cyclone Michaung (init 2023-12-01 00 UTC; maps `D:/masrainman_runs/forecasts/fc_2023120100_*.png`):
the model moves the low north-west towards Tamil Nadu for ~2 days but never deepens it:
lowest 1006 hPa vs ERA5 992 hPa near 13N 81.5E on 4 Dec; after day 3 it stalls near Sri Lanka.
Same weakness as Fani: extremes are smoothed away. Faint frame from the blend zone still shows.
Next ideas: cyclone/heavy-rain oversampling or a weighted loss, learn the daily t2m cycle
(more time features or hourly data), longer rollout fine-tune, IMD obs for verification.

## Cyclone fine-tune (1 Nov 2026 evening) -- DONE, new best model
`scripts\cyclone_run.ps1`, run dir `D:/masrainman_runs/cyclone`, ~2.75 h. Starts from the full
run's one-step model; `--cyclone-boost 5` draws windows holding a cyclone 13% of the time
instead of 3%, and `--cyclone-weight 3` counts errors within ~2.5 deg of a deep low up to 4x.
"Low depth" = 5x5 deg box mean MSLP minus the cell's MSLP (masrainman/extremes.py); it picks
out the named cyclones and ignores monsoon troughs. Model: `D:/masrainman_runs/cyclone/best.pt`.

Storm scorecard (`python -m masrainman.cyclones --ckpt ...`, 5-day forecasts from the young
low; lowest MSLP model / ERA5, track error days 1-3):

| storm | full run | cyclone run | ERA5 |
|---|---|---|---|
| Asani 2022 | 991 hPa, 265 km | 981 hPa, 379 km | 974 |
| Sitrang 2022 | 1002, 300 km | 993, 209 km | 995 |
| Mandous 2022 | 997, 223 km | 995, 344 km | 988 |
| Mocha 2023 | 974, 148 km | 973, 101 km | 960 |
| Hamoon 2023 | 1007, 426 km | 1003, 311 km | 1001 |
| Michaung 2023 | 1006, 252 km | 1001, 152 km | 992 |
| depth kept (capped at 100%) | 42% | 72% | |
| mean track error | 269 km | 249 km | |

Whole of 2023 vs the full run (skill vs persistence, percentage points): t2m much better
(day 5 -11% -> +2%, day 10 -13% -> +7%: now beats persistence after day 4), msl +7-8,
u200/v200 +3-10, rain CSI up (0.46/0.32/0.32/0.30 at days 1/3/5/10); u850 4-8 points worse
(still +20-39%), q850 and z500 2-8 points worse. A good trade for cyclone season.
Michaung maps: `D:/masrainman_runs/cyclone/forecasts/` -- closed low and Andhra coast rain band
on day 3, but ~2 deg north of the real centre and ~12 hPa too shallow.
Next: more years of ERA5 (2010-2017) for more storms, a bigger network, longer rollouts.

## Resume from the last point
**Download (finished 1 Nov):** the download started by itself at every Windows login
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

**Daily live forecasts: on GitHub Actions since 2 Nov 2026 (runs with the laptop off).**
`.github/workflows/daily_forecast.yml` runs daily at 08:30 UTC (14:00 IST), on every push of
`operational/`, or by hand ("Run workflow" on the Actions tab). It uses the model packed in
`operational/` (now the cyclone model; repack a new one with
`python -m masrainman.export_model --ckpt <best.pt> --out operational`, commit, push) and the
latest ECMWF open-data run. Maps go to the `live` branch (README shows them; replaced daily),
full netCDF output is the run's artifact (kept 14 days). ~5-10 Actions minutes per run, well
inside the 2000 free minutes a month for private repos.
Backup run times 15:47 and 18:13 IST (GitHub started the first scheduled run 10 h late);
scheduled runs stop at once when `cycle.txt` on `live` already has the newest ECMWF run. Each run
publishes a map per day (+24 ... +240 h) and `manifest.json`.
**On the WeatherNext3 site** (Cloudflare project masrainman-wn3): page `<site>/ai/`, button
"MasRainMan AI (10-day)" on the main page. Made by `masrainman-weathernext3` repo's
`.github/workflows/masrainman-ai.yml` (+ `web/ai.html`), which copies the `live` branch maps into
site-data and redeploys (14:55, 16:35, 19:05 IST). Needs the secret MASRAINMAN_MODEL_TOKEN there
(fine-grained token, read-only Contents on masrainman-model).
Rain maps use the MasRainMan standard colour bar (same as the Rain Alarm app's app.js):
24-h totals, 1-300+ mm. The laptop version (`scripts\live_daily.ps1`; its Task Scheduler task was removed 2 Nov, reinstall via
`scripts\install_daily_forecast.ps1`) still works for local runs. Live maps are a little noisier
than hindcasts: ECMWF's live analysis is sharper than ERA5 (a fine-tune on ECMWF data could fix it).

Checks: `python -m pytest -q tests` (8 tests, all passing on 28 Oct 2026).

## Gotchas found so far
- CDS key: the downloader reads `%USERPROFILE%\.cdsapirc` explicitly, because an old
  `%USERPROFILE%\.ecmwfdatastoresrc` holds an outdated key that the library would use first (401 error).
- Run only one download at a time (Copernicus may disable accounts that send parallel requests).
- Data must stay on D: (C: has about 6 GB free).
