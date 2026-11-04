"""Live 10-day forecast from the latest ECMWF open-data run.

  python -m masrainman.live_forecast --ckpt D:/masrainman_runs/baseline/best.pt --plot
  python -m masrainman.live_forecast --ckpt ... --cycle 2026-10-28T00
  python -m masrainman.live_forecast --ckpt operational/model.pt --data-dir operational --run-dir out --plot
      (the GitHub Actions setup: everything it needs is in the repo's operational/ folder)

ECMWF gives the start state and the boundary ring at every step, exactly like ERA5
does in hindcasts (ERA5 is made with the same ECMWF model, so the two match well).
Each run downloads about 430 MB, which is cut to the domain and cached as a few MB.
Output goes to <run_dir>/live/. Data: ECMWF open data, CC-BY-4.0.
"""
import argparse

import torch

from .config import run_dir
from .dataset import Stats, to_hour
from .ecmwf import ECMWFArchive, latest_cycle
from .forecast import run_and_save
from .model import load_checkpoint


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cycle", help="ECMWF run (00, 06, 12 or 18 UTC), e.g. 2026-10-28T06 (default: latest complete run)")
    ap.add_argument("--days", type=float, default=10)
    ap.add_argument("--plot", action="store_true", help="also draw a map per parameter for every day (+24 h ... +240 h) and a maps.json list")
    ap.add_argument("--data-dir", help="folder with processed/stats.npz and static.npy; also caches "
                                       "ECMWF data (default: data_dir of the checkpoint's config)")
    ap.add_argument("--run-dir", help="output root; maps go to <run-dir>/live (default: from the config)")
    ap.add_argument("--save-analysis", help="folder to keep this run's start state in, for fine-tuning "
                                            "(masrainman/finetune.py); ECMWF deletes its runs after a few days")
    args = ap.parse_args()

    model, cfg = load_checkpoint(args.ckpt, args.data_dir)
    cfg["run_dir"] = args.run_dir or cfg["run_dir"]
    torch.set_num_threads(cfg["train"]["threads"])
    cycle = to_hour(args.cycle) if args.cycle else latest_cycle(cfg)
    n_steps = int(args.days * 24 // cfg["step_hours"])
    print(f"ECMWF run {cycle}, {n_steps} steps")

    arch = ECMWFArchive(cfg, cycle, n_steps)
    arch.prefetch()
    if args.save_analysis:
        from .finetune import save_analysis
        print(f"kept start state {save_analysis(args.save_analysis, cycle, arch.get(cycle), Stats(cfg))}")
    path = run_and_save(cfg, model, arch, Stats(cfg), cycle, n_steps, run_dir(cfg, "live"),
                        "live forecast with ECMWF open-data start and boundaries")
    if args.plot:
        import json

        import pandas as pd
        import xarray as xr

        from .plot import PARAMS, plot_param
        ds = xr.open_dataset(path)
        maps = []
        for lead in range(24, int(ds.lead.values.max()) + 1, 24):  # one map per day
            files = {}
            for param in PARAMS:
                png = path.with_name(f"{path.stem}_{param}_{lead:03d}h.png")
                plot_param(ds, lead, param, png)
                files[param] = png.name
            valid = pd.Timestamp(ds.valid_time.sel(lead=lead).values)
            maps.append({"lead": lead, "valid": valid.strftime("%Y-%m-%dT%H"), "files": files})
            print(f"maps for +{lead} h")
        # read by web pages that show the maps (e.g. the WeatherNext3 site's MasRainMan AI page)
        manifest = {"init": pd.Timestamp(cycle).strftime("%Y-%m-%dT%H"), "model": "MasRainMan AI",
                    "made_utc": pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M"), "maps": maps}
        path.with_name(f"{path.stem}_maps.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
