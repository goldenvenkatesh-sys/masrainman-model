# MasRainMan — where we stopped (28 Oct 2026)

## Done
- Full pipeline built and tested on fake data: download → preprocess → train → evaluate → forecast → maps.
- Repo: https://github.com/goldenvenkatesh-sys/masrainman-model (private). The live app repo
  `goldenvenkatesh-sys/masrainman` is separate and must not be force-pushed.
- Years in `config.yaml`: train 2018–2021, val 2022, test 2023.
- ERA5 download for 2018–2023 started 28 Oct 2026 15:27 (first request succeeded).

## Resume from the last point (e.g. Saturday night)
The downloader skips every file that is already complete, so just run it again.
Half-finished files end in `.part` and are downloaded again automatically.

```
git clone https://github.com/goldenvenkatesh-sys/masrainman-model.git D:\masrainman
cd D:\masrainman
python -m masrainman.download_era5 --years 2018 2023
```
Check progress: files land in `D:\masrainman_data\raw\<year>\` as `sfc_YYYYMM.nc`,
`pl_YYYYMM.nc` and `tp_YYYYMM.nc` (216 files in total, plus `static.nc`).

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
