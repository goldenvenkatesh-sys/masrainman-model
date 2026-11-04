"""Fine-tune the operational model on recent ECMWF analyses and the state rain gauges, and
replace it only if it verifies better on days it has not seen.

  python -m masrainman.finetune --analyses td/ana --obs vd/obs_rain --ckpt operational/model.pt \
      --data-dir operational --out ft

Data (kept by the daily workflow, see .github/workflows/daily_forecast.yml):
  <analyses>/YYYYMMDDHH.npz  start state of every ECMWF run (00/06/12/18 UTC), normalised, float16
  <obs>/YYYYMMDD.npz         24-h gauge rain on the model grid (masrainman/verify.py)

Training windows are 24 h (4 steps) of analyses, like the ERA5 rollout fine-tune. The loss is
the usual one (each field's error in units of its 6-h change, South India counted double) plus,
for windows starting at 00 UTC, the 24-h rain against the gauges, and a pull towards the old
weights so a few weeks of data cannot make the model forget what ERA5 taught it.

The last --test-days are kept out of training. Old and new model then forecast 5 days from each
00 UTC start in that period (analyses as boundaries) and are scored on gauge rain (Days 1-3) and on
all fields against the analyses. The new model is written to <out>/model.pt and promoted (copied
over --ckpt) only with --promote and if it is better on rain and not worse on the rest.
"""
import argparse
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .dataset import Stats, time_features, to_hour

STEP = np.timedelta64(6, "h")
ROLLOUT = 4
THRESH = 15.6   # IMD moderate rain, mm in 24 h


# ---------------------------------------------------------------- analysis archive

def save_analysis(folder, cycle, state, stats):
    """Keep one start state (stored units, as Archive.get returns them) as normalised float16."""
    p = Path(folder) / f"{pd.Timestamp(cycle):%Y%m%d%H}.npz"
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, x=np.clip(stats.norm(state), -60000, 60000).astype(np.float16))
    return p


class Analyses:
    """Archive-like (has/get/step/times) view of the saved start states."""

    def __init__(self, folder, stats):
        self.stats, self.step = stats, STEP
        self.files = {to_hour(datetime.strptime(f.stem, "%Y%m%d%H")): f
                      for f in sorted(Path(folder).glob("??????????.npz"))}
        self.times = np.array(sorted(self.files))
        self._cache = {}

    def has(self, t):
        return to_hour(t) in self.files

    def norm(self, t):
        t = to_hour(t)
        if t not in self._cache:
            self._cache[t] = np.load(self.files[t])["x"].astype(np.float32)
        return self._cache[t]

    def get(self, t):
        return self.stats.denorm(self.norm(t)).astype(np.float32)


def gauge_rain(obs_dir, day):
    """(rain mm, mask) of the rain-day ending 08:30 IST on `day`, or None."""
    p = Path(obs_dir) / f"{day:%Y%m%d}.npz"
    if not p.exists():
        return None
    r = np.load(p)["rain"]
    m = np.isfinite(r)
    return (np.nan_to_num(r).astype(np.float32), m) if m.sum() >= 5 else None


class Windows(torch.utils.data.Dataset):
    """24-h windows t-6h ... t+24h of analyses, with the gauges of the matching rain-day for 00 UTC
    starts (Day 1 of a 00 UTC run: 21 of its 24 hours are inside the 08:30-08:30 IST rain-day)."""

    def __init__(self, ana, inits, obs_dir):
        self.ana, self.inits = ana, list(inits)
        self.obs = {}
        for t in self.inits:
            ts = pd.Timestamp(t)
            if ts.hour == 0 and obs_dir:
                g = gauge_rain(obs_dir, (ts + timedelta(days=1)).date())
                if g is not None:
                    self.obs[t] = g

    def __len__(self):
        return len(self.inits)

    def __getitem__(self, i):
        t = self.inits[i]
        seq = np.stack([self.ana.norm(t + k * STEP) for k in range(-1, ROLLOUT + 1)])
        tf = np.stack([time_features(t + k * STEP) for k in range(1, ROLLOUT + 1)])
        rain, mask = self.obs.get(t, (np.zeros(seq.shape[-2:], np.float32), np.zeros(seq.shape[-2:], bool)))
        return torch.from_numpy(seq), torch.from_numpy(tf), torch.from_numpy(rain), torch.from_numpy(mask)


def windows_inside(ana, start=None, end=None):
    """Start times whose whole window (t-6h ... t+24h) is in the archive and in [start, end)."""
    out = []
    for t in ana.times:
        if start is not None and t - STEP < start:
            continue
        if end is not None and t + ROLLOUT * STEP >= end:
            continue
        if all(ana.has(t + k * STEP) for k in range(-1, ROLLOUT + 1)):
            out.append(t)
    return out


# ---------------------------------------------------------------- training

def rain_mm(model_tp, stats, i_tp):
    """Normalised tp channel -> mm per 6 h (differentiable)."""
    x = model_tp * float(stats.std[i_tp, 0, 0]) + float(stats.mean[i_tp, 0, 0])
    return torch.expm1(torch.clamp(x, min=0))


def train(model, cfg, stats, ds, steps, lr, batch, gauge_weight, anchor, log=print):
    from .train import loss_weight
    weight = loss_weight(cfg)
    i_tp = cfg["channels"].index("tp")
    p0 = [p.detach().clone() for p in model.net.parameters()]
    opt = torch.optim.AdamW(model.net.parameters(), lr=lr, weight_decay=0)
    sampler = torch.utils.data.RandomSampler(ds, replacement=True, num_samples=steps * batch)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, sampler=sampler, drop_last=True)
    model.train()
    t0, hist = time.time(), []
    for i, (seq, tf, obs, mask) in enumerate(dl):
        prev, now = seq[:, 0], seq[:, 1]
        loss, rain24 = 0.0, 0.0
        for k in range(ROLLOUT):
            target = seq[:, k + 2]
            pred = model(prev, now, target, tf[:, k])
            loss = loss + (((pred - target) / model.tend_scale) ** 2 * weight).mean()
            rain24 = rain24 + rain_mm(pred[:, i_tp], stats, i_tp)
            prev, now = now, pred
        loss = loss / ROLLOUT
        g = torch.tensor(0.0)
        if gauge_weight and mask.any():
            g = ((torch.log1p(rain24) - torch.log1p(obs)) ** 2)[mask].mean()
            loss = loss + gauge_weight * g
        if anchor:
            loss = loss + anchor * sum(((p - q) ** 2).sum() for p, q in zip(model.net.parameters(), p0))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        hist.append(loss.item())
        if i % 25 == 0:
            log(f"  step {i:4d}/{steps}  loss {np.mean(hist[-25:]):.4f}  gauge {float(g):.3f}  "
                f"({(time.time() - t0) / 60:.1f} min)")
    model.eval()
    return hist


# ---------------------------------------------------------------- scoring

@torch.no_grad()
def score(model, cfg, stats, ana, inits, obs_dir, days=5):
    """Forecasts from each 00 UTC start (analyses as boundaries). Returns
    {'rain_rmse': [d1, d2, d3], 'rain_ets': [...], 'field': {channel: [nrmse per day]}, 'n': ...}."""
    from .config import focus_mask
    from .forecast import rollout, to_physical
    from .verify import contingency
    n_steps = days * 24 // 6
    focus = focus_mask(cfg) > 0
    i_tp = cfg["channels"].index("tp")
    sums = {"se": np.zeros(3), "n": np.zeros(3), "ct": np.zeros((3, 4))}
    ferr = np.zeros((len(cfg["channels"]), days))
    used = 0
    for t in inits:
        if pd.Timestamp(t).hour != 0 or not all(ana.has(t + k * STEP) for k in range(-1, n_steps + 1)):
            continue
        fc = rollout(model, ana, stats, [t], n_steps)[0]                    # normalised (n, C, H, W)
        truth = np.stack([ana.norm(t + (k + 1) * STEP) for k in range(n_steps)])
        for d in range(days):
            k = (d + 1) * 4 - 1
            ferr[:, d] += np.sqrt((((fc[k] - truth[k]) / model.tend_scale[0].numpy()) ** 2)[:, focus].mean(1))
        rain = to_physical(cfg, stats, fc.copy())[:, i_tp]                  # mm per 6 h
        for d in range(3):
            g = gauge_rain(obs_dir, (pd.Timestamp(t) + timedelta(days=d + 1)).date()) if obs_dir else None
            if g is None:
                continue
            day = rain[d * 4:(d + 1) * 4].sum(0)
            o, m = g
            sums["se"][d] += ((day[m] - o[m]) ** 2).sum()
            sums["n"][d] += m.sum()
            sums["ct"][d] += contingency(day[m], o[m], THRESH)
        used += 1
    ets = []
    for a, b, c, dd in sums["ct"]:
        tot = a + b + c + dd
        ar = (a + b) * (a + c) / tot if tot else 0
        ets.append(float((a - ar) / (a + b + c - ar)) if (a + b + c - ar) > 0 else None)
    rmse = [float(np.sqrt(se / n)) if n else None for se, n in zip(sums["se"], sums["n"])]
    return {"n": used, "rain_rmse": rmse, "rain_ets": ets,
            "field": {c: (ferr[i] / max(used, 1)).round(4).tolist() for i, c in enumerate(cfg["channels"])}}


def better(old, new, rain_gain=0.01, field_slack=0.01):
    """New model wins if Day 1-3 gauge-rain RMSE is at least 1% lower and the mean field error
    over Days 1-5 is no more than 1% higher. Returns (yes/no, reason)."""
    if old["n"] < 3:
        return False, f"only {old['n']} test forecasts; need at least 3"
    ro = [x for x in old["rain_rmse"] if x is not None]
    rn = [x for x in new["rain_rmse"] if x is not None]
    if not ro or len(ro) != len(rn):
        return False, "no gauge rain in the test days"
    r_old, r_new = float(np.mean(ro)), float(np.mean(rn))
    f_old = float(np.mean([np.mean(v) for v in old["field"].values()]))
    f_new = float(np.mean([np.mean(v) for v in new["field"].values()]))
    msg = f"rain RMSE {r_old:.2f} -> {r_new:.2f} mm, field error {f_old:.3f} -> {f_new:.3f}"
    if r_new > r_old * (1 - rain_gain):
        return False, "rain not better: " + msg
    if f_new > f_old * (1 + field_slack):
        return False, "other fields worse: " + msg
    return True, msg


# ---------------------------------------------------------------- main

def main():
    from .model import load_checkpoint
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analyses", required=True)
    ap.add_argument("--obs", help="gauge rain folder (verif-data branch obs_rain/)")
    ap.add_argument("--ckpt", default="operational/model.pt")
    ap.add_argument("--data-dir", default="operational")
    ap.add_argument("--out", default="ft")
    ap.add_argument("--min-days", type=int, default=25, help="days of analyses needed before training")
    ap.add_argument("--test-days", type=int, default=8)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--gauge-weight", type=float, default=0.5)
    ap.add_argument("--anchor", type=float, default=1e-4, help="pull towards the old weights")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--promote", action="store_true", help="overwrite --ckpt if the new model is better")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {"started_utc": pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M"), "promoted": False}

    def finish(reason):
        report["decision"] = reason
        (out / "report.json").write_text(json.dumps(report, indent=1))
        print(reason)

    model, cfg = load_checkpoint(args.ckpt, args.data_dir)
    stats = Stats(cfg)
    ana = Analyses(args.analyses, stats)
    if not len(ana.times):
        return finish("no analyses saved yet")
    span = (ana.times[-1] - ana.times[0]) / np.timedelta64(1, "D")
    report["analyses"] = {"count": len(ana.times), "first": str(ana.times[0]), "last": str(ana.times[-1])}
    if span < args.min_days:
        return finish(f"only {span:.1f} days of analyses; fine-tuning starts at {args.min_days}")

    split = ana.times[-1] - np.timedelta64(args.test_days, "D")
    train_inits = windows_inside(ana, end=split)
    test_inits = [t for t in ana.times if t >= split]
    ds = Windows(ana, train_inits, args.obs)
    report["train"] = {"windows": len(ds), "with_gauges": len(ds.obs), "steps": args.steps, "lr": args.lr,
                       "gauge_weight": args.gauge_weight, "anchor": args.anchor}
    print(f"{len(ana.times)} analyses over {span:.1f} days; {len(ds)} training windows "
          f"({len(ds.obs)} with gauges), test from {split}")
    if len(ds) < 20:
        return finish(f"only {len(ds)} complete training windows")

    old = score(model, cfg, stats, ana, test_inits, args.obs)
    print("old model:", json.dumps({k: old[k] for k in ("n", "rain_rmse", "rain_ets")}))
    hist = train(model, cfg, stats, ds, args.steps, args.lr, args.batch, args.gauge_weight, args.anchor)
    report["train"]["loss_first_last"] = [float(np.mean(hist[:25])), float(np.mean(hist[-25:]))]
    new = score(model, cfg, stats, ana, test_inits, args.obs)
    print("new model:", json.dumps({k: new[k] for k in ("n", "rain_rmse", "rain_ets")}))
    report["old"], report["new"] = old, new

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    slim = {k: ck[k] for k in ("cfg", "epoch", "rollout", "best_val") if k in ck}
    slim["model"] = model.state_dict()
    slim["source"] = f"finetune {report['started_utc']} on {report['analyses']['first']}..{report['analyses']['last']}" \
                     f" (from: {ck.get('source', args.ckpt)})"
    slim["history"] = ck.get("history", []) + [slim["source"]]
    torch.save(slim, out / "model.pt")

    ok, why = better(old, new)
    if ok and args.promote:
        torch.save(slim, args.ckpt)
        report["promoted"] = True
        why = "promoted: " + why
    elif ok:
        why = "better (not promoted, no --promote): " + why
    else:
        why = "kept the old model: " + why
    finish(why)


if __name__ == "__main__":
    main()
