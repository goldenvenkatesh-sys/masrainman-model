"""Live 10-day forecast from the latest NOAA GFS run.

  python -m masrainman.live_forecast --ckpt D:/masrainman_runs/baseline/best.pt
  python -m masrainman.live_forecast --ckpt ... --cycle 2026-10-28T00 --plot

GFS gives the start state (analyses at t-6 h and t) and the boundary ring at every
step, exactly like ERA5 does in hindcasts. Output goes to <run_dir>/live/.
Note: the model is trained on ERA5, so starting from GFS adds a small mismatch.
"""
import argparse

import torch

from .config import run_dir
from .dataset import Stats, to_hour
from .forecast import run_and_save
from .gfs import GFSArchive, latest_cycle
from .model import load_checkpoint


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cycle", help="GFS run, e.g. 2026-10-28T00 (default: latest complete run)")
    ap.add_argument("--days", type=float, default=10)
    ap.add_argument("--plot", action="store_true", help="also draw maps at 24/72/120/240 h")
    args = ap.parse_args()

    model, cfg = load_checkpoint(args.ckpt)
    torch.set_num_threads(cfg["train"]["threads"])
    cycle = to_hour(args.cycle) if args.cycle else latest_cycle(cfg)
    n_steps = int(args.days * 24 // cfg["step_hours"])
    print(f"GFS cycle {cycle}, {n_steps} steps")

    arch = GFSArchive(cfg, cycle, n_steps)
    arch.prefetch()
    path = run_and_save(cfg, model, arch, Stats(cfg), cycle, n_steps, run_dir(cfg, "live"),
                        "live forecast with GFS start and boundaries")
    if args.plot:
        from .plot import plot_lead
        import xarray as xr
        ds = xr.open_dataset(path)
        for lead in (24, 72, 120, 240):
            if lead in ds.lead.values:
                plot_lead(ds, lead, path.with_name(f"{path.stem}_{lead:03d}h.png"))


if __name__ == "__main__":
    main()
