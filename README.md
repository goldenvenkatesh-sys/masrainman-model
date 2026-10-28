# MasRainMan — AI weather model for South India

A data-driven regional weather model for South India. It forecasts from 0 to 10 days in
6-hour steps, with hourly output for the first 24 h. Variables:

| Channel | Meaning | Units |
|---|---|---|
| t2m, d2m | 2 m temperature, dewpoint | °C |
| u10, v10 | 10 m wind | m/s |
| msl | mean sea level pressure | hPa |
| tp | rain | mm per 6 h |
| tcwv | total column water vapour | kg/m² |
| z500 | 500 hPa height | m |
| t850, q850, u850, v850 | 850 hPa temp, humidity, wind (monsoon flow) | °C, g/kg, m/s |
| u200, v200 | 200 hPa wind (jets) | m/s |

## How it works
- **Domain** 0–25°N, 65–95°E at 0.25° (101 × 121 grid). This covers the Arabian Sea and
  Bay of Bengal so incoming systems are on the grid. South India (8–20°N, 72–88°E) is
  weighted double in training and is the box used for scoring.
- **Model**: a U-Net (1.2 M parameters) that predicts the 6-hour change of every field
  from the last two states. It is run 40 times for a 10-day forecast.
- **Boundaries**: like a regional NWP model, the outer 2° ring is taken from a global
  source: ERA5 in training and hindcasts, and ECMWF open data (00/12 UTC runs) for live forecasts.
- **Data**: ERA5 reanalysis, 2010–2023 (train 2010–20, validate 2021, test 2022–23).

## Hardware notes (this laptop: i3-1115G4, 8 GB RAM, no GPU)
- Everything runs on the CPU. One training step (batch of 8) takes about 2.5 s, so one
  epoch of 2000 steps takes about 1.5 h. A useful first model takes about 1–2 days of training.
- Data goes on **D:** (about 1 GB per year processed, plus raw files). C: is too full.
- Arrays are memory-mapped, so 8 GB of RAM is enough.

## Setup
```
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
```
For ERA5 you need a free Copernicus account (https://cds.climate.copernicus.eu). Accept
the ERA5 licence, then create `%USERPROFILE%\.cdsapirc`:
```
url: https://cds.climate.copernicus.eu/api
key: <your-key>
```

## Run
```
# 0. Optional: test the pipeline on fake data (about 2 minutes)
python scripts/make_synthetic.py --out D:/masrainman_synthetic
python -m masrainman.train --config config.synthetic.yaml --name smoke --epochs 1 --steps-per-epoch 10 --val-steps 3

# 1. Download ERA5 (slow: the CDS queue; start with a few years)
python -m masrainman.download_era5 --years 2018 2023

# 2. Build training arrays + statistics
python -m masrainman.preprocess --years 2018 2023

# 3. Train: single step, then multi-step fine-tuning for stable 10-day runs
python -m masrainman.train --name baseline --epochs 20
python -m masrainman.train --name baseline --resume --rollout 4 --epochs 5 --lr 5e-5

# 4. Score against persistence on the test years
python -m masrainman.evaluate --ckpt D:/masrainman_runs/baseline/best.pt

# 5. Forecast + maps
python -m masrainman.forecast --ckpt D:/masrainman_runs/baseline/best.pt --init 2023-11-20T00 --days 10
python -m masrainman.plot D:/masrainman_runs/forecasts/fc_2023112000.nc --leads 24 72 120 240
```
If you start with fewer years, edit `years:` in `config.yaml` to match (for example train
2018–2021, val 2022, test 2023).

## Project layout
```
config.yaml              domain, years, channels, training settings
masrainman/download_era5.py ERA5 download (CDS API)
masrainman/preprocess.py    netCDF -> arrays, unit conversion, 6-h rain totals, stats
masrainman/dataset.py       memory-mapped archive + training windows
masrainman/model.py         U-Net stepper with boundary forcing
masrainman/train.py         CPU training with rollout fine-tuning
masrainman/forecast.py      10-day forecast + hourly first-24-h file (netCDF)
masrainman/evaluate.py      RMSE vs persistence, rain CSI, per lead time
masrainman/plot.py          maps (temperature/MSLP, wind, rain, moisture)
scripts/make_synthetic.py fake data for testing
```

## Roadmap
1. ✅ Pipeline: download → preprocess → train → evaluate → forecast → maps
2. Train on real ERA5 and beat persistence at every lead time
3. ✅ Live forecasts from ECMWF open data (`python -m masrainman.live_forecast --ckpt ... --plot`);
   still to do: run them on a schedule
4. Check against IMD station and rain-gauge observations
5. A true hourly model for 0–24 h (train with `step_hours: 1`)
6. Better rain: ensembles or a probabilistic loss
