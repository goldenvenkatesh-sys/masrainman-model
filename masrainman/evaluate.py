"""Score the model on the test years over the South India box.

  python -m masrainman.evaluate --ckpt D:/masrainman_runs/baseline/best.pt

For every lead time (6 h ... 10 days) it reports RMSE of the model and of
persistence ("weather stays as it is now"). The model must beat persistence.
For rain it also reports CSI (critical success index) for >= 2.5 mm per step.
Writes scores.csv and scores.png next to the checkpoint.
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .config import CHANNEL_INFO, focus_mask, year_range
from .dataset import Archive, Stats
from .forecast import needed_times, rollout, to_physical
from .model import load_checkpoint

RAIN_THRESHOLD = 2.5


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--days", type=float, default=10)
    ap.add_argument("--every-days", type=int, default=5, help="spacing between start dates")
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()

    model, cfg = load_checkpoint(args.ckpt)
    torch.set_num_threads(cfg["train"]["threads"])
    test_years = year_range(cfg["years"]["test"])
    arch = Archive(cfg, [test_years[0] - 1] + test_years + [test_years[-1] + 1])
    stats = Stats(cfg)
    n_steps = int(args.days * 24 // cfg["step_hours"])
    every = np.timedelta64(args.every_days * 24, "h")

    inits, last = [], None
    for t in arch.times:
        if int(str(t)[:4]) not in test_years or t.astype(object).hour != 0:
            continue
        if last is not None and t - last < every:
            continue
        if all(arch.has(x) for x in needed_times(arch, t, n_steps)):
            inits.append(t)
            last = t
    if not inits:
        raise SystemExit("No complete test cases found.")
    print(f"{len(inits)} forecasts from {inits[0]} to {inits[-1]}")

    chans = cfg["channels"]
    focus = focus_mask(cfg)
    nf = focus.sum()
    C = len(chans)
    se_model = np.zeros((n_steps, C))
    se_pers = np.zeros((n_steps, C))
    rain = np.zeros((n_steps, 3))  # hits, misses, false alarms
    ti = chans.index("tp") if "tp" in chans else None

    for b in range(0, len(inits), args.batch):
        batch = inits[b:b + args.batch]
        fc = to_physical(cfg, stats, rollout(model, arch, stats, batch, n_steps))
        for j, t0 in enumerate(batch):
            start = to_physical(cfg, stats, stats.norm(arch.get(t0)))
            for k in range(n_steps):
                truth = to_physical(cfg, stats, stats.norm(arch.get(t0 + (k + 1) * arch.step)))
                se_model[k] += ((fc[j, k] - truth) ** 2 * focus).sum((1, 2)) / nf
                se_pers[k] += ((start - truth) ** 2 * focus).sum((1, 2)) / nf
                if ti is not None:
                    f, o = fc[j, k, ti] >= RAIN_THRESHOLD, truth[ti] >= RAIN_THRESHOLD
                    rain[k] += [(f & o)[focus > 0].sum(), (~f & o)[focus > 0].sum(), (f & ~o)[focus > 0].sum()]
        print(f"  {min(b + args.batch, len(inits))}/{len(inits)}", flush=True)

    rmse_m = np.sqrt(se_model / len(inits))
    rmse_p = np.sqrt(se_pers / len(inits))
    lead = (np.arange(n_steps) + 1) * cfg["step_hours"]
    csi = rain[:, 0] / np.maximum(rain.sum(1), 1)

    out = Path(args.ckpt).parent
    with open(out / "scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lead_hours", "channel", "units", "rmse_model", "rmse_persistence"])
        for k, h in enumerate(lead):
            for c, name in enumerate(chans):
                w.writerow([h, name, CHANNEL_INFO[name]["units"], f"{rmse_m[k, c]:.4f}", f"{rmse_p[k, c]:.4f}"])
            if ti is not None:
                w.writerow([h, f"tp_csi_{RAIN_THRESHOLD}mm", "", f"{csi[k]:.4f}", ""])

    show = [c for c in ("t2m", "u10", "msl", "tp", "z500", "u850") if c in chans]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, name in zip(axes.flat, show):
        c = chans.index(name)
        ax.plot(lead / 24, rmse_m[:, c], label="MasRainMan")
        ax.plot(lead / 24, rmse_p[:, c], "--", label="persistence")
        units = f"mm/{cfg['step_hours']}h" if name == "tp" else CHANNEL_INFO[name]["units"]
        ax.set(title=f"{name} RMSE ({units})", xlabel="lead (days)")
        ax.grid(alpha=0.3)
    axes.flat[0].legend()
    fig.suptitle(f"South India box, {len(inits)} forecasts, test years {test_years[0]}-{test_years[-1]}")
    fig.tight_layout()
    fig.savefig(out / "scores.png", dpi=120)

    for name in show:
        c = chans.index(name)
        row = "  ".join(f"{d}d {rmse_m[min(int(d * 24 / cfg['step_hours']) - 1, n_steps - 1), c]:.2f}"
                        for d in (1, 3, 5, 10) if d <= args.days)
        print(f"{name:5s} {row}")
    print(f"wrote {out / 'scores.csv'} and {out / 'scores.png'}")


if __name__ == "__main__":
    main()
