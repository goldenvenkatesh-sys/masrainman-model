"""Pack a trained model for live forecasts anywhere (e.g. GitHub Actions).

  python -m masrainman.export_model --ckpt D:/masrainman_runs/cyclone/best.pt --out operational

Writes <out>/model.pt (weights and config only, no optimiser state) and copies the
normalisation statistics and static fields to <out>/processed/. Commit the folder;
the daily workflow runs  live_forecast --ckpt operational/model.pt --data-dir operational.
"""
import argparse
import shutil
from pathlib import Path

import torch

from .config import processed_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="operational")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    out = Path(args.out)
    (out / "processed").mkdir(parents=True, exist_ok=True)
    slim = {k: ck[k] for k in ("model", "cfg", "epoch", "rollout", "best_val") if k in ck}
    slim["source"] = str(args.ckpt)
    torch.save(slim, out / "model.pt")
    for name in ("stats.npz", "static.npy"):
        shutil.copyfile(processed_dir(ck["cfg"]) / name, out / "processed" / name)
    print(f"wrote {out / 'model.pt'} ({(out / 'model.pt').stat().st_size / 1e6:.1f} MB) and {out / 'processed'}")


if __name__ == "__main__":
    main()
