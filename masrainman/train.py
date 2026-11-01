"""Train the model on CPU.

Stage 1 - learn one step:
  python -m masrainman.train --name baseline --epochs 20
Stage 2 - fine-tune on multi-step rollouts (keeps 10-day forecasts stable):
  python -m masrainman.train --name baseline --resume --rollout 4 --epochs 5 --lr 5e-5

Training stops and resumes safely (Ctrl+C, then --resume).
"""
import argparse
import csv
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import boundary_mask, focus_mask, load_config, run_dir, year_range
from .dataset import Windows
from .extremes import box_cells, pixel_weight, sample_weights
from .model import build_stepper


def loss_weight(cfg):
    """Ignore the forced ring (and the blend zone in proportion), count South India double."""
    w = (1 - boundary_mask(cfg)) * (1 + focus_mask(cfg))
    return torch.from_numpy(w / w.mean())[None, None]


def init_network(model, path):
    """Copy only the network weights from another checkpoint; static fields, the boundary
    mask and tend_scale stay as built from this run's data and statistics."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = {k[len("net."):]: v for k, v in ck["model"].items() if k.startswith("net.")}
    model.net.load_state_dict(net)


def run_batches(model, loader, weight, rollout, max_steps, opt=None, extra=None):
    total, n = 0.0, 0
    for i, (seq, tf) in enumerate(loader):
        if max_steps and i >= max_steps:
            break
        prev, now = seq[:, 0], seq[:, 1]
        loss = 0.0
        for k in range(rollout):
            target = seq[:, k + 2]
            # target is passed as the boundary source; the model only sees its outer ring
            pred = model(prev, now, target, tf[:, k])
            # Errors in units of each field's typical one-step change: in plain normalised
            # units a field that changes little per step (winds aloft) hardly counts, and
            # the model learns to leave it unchanged.
            err = (pred - target) / model.tend_scale
            w = weight if extra is None else weight * extra(target)
            loss = loss + (err ** 2 * w).mean()
            prev, now = now, pred
        loss = loss / rollout
        if opt is not None:
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        total += loss.item()
        n += 1
        if opt is not None and i % 50 == 0:
            print(f"    step {i:5d}  loss {loss.item():.4f}", flush=True)
    return total / max(n, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--name", default="baseline")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--rollout", type=int, default=1, help="steps unrolled in the loss")
    ap.add_argument("--lr", type=float)
    ap.add_argument("--batch", type=int)
    ap.add_argument("--steps-per-epoch", type=int, default=2000, help="cap so an epoch fits a CPU budget (0 = all)")
    ap.add_argument("--val-steps", type=int, default=100)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init-from", help="start from another run's network weights (its statistics, "
                                        "grid buffers and optimiser state are not copied)")
    ap.add_argument("--cyclone-boost", type=float, default=0,
                    help="draw training windows with a cyclone up to 1+N times as often")
    ap.add_argument("--cyclone-weight", type=float, default=0,
                    help="count errors around deep lows up to 1+N times as much")
    args = ap.parse_args()

    cfg = load_config(args.config)
    tc = cfg["train"]
    torch.set_num_threads(tc["threads"])
    torch.manual_seed(0)
    batch = args.batch or tc["batch_size"]
    lr = args.lr or tc["lr"]
    out = run_dir(cfg, args.name)

    train_ds = Windows(cfg, year_range(cfg["years"]["train"]), args.rollout)
    val_ds = Windows(cfg, year_range(cfg["years"]["val"]), args.rollout)
    print(f"train samples {len(train_ds)}, val samples {len(val_ds)}")
    sampler = None
    if args.cyclone_boost:
        w, depth = sample_weights(train_ds, cfg, args.cyclone_boost)
        sampler = torch.utils.data.WeightedRandomSampler(torch.from_numpy(w), len(w), replacement=True)
        print(f"cyclone boost {args.cyclone_boost}: {(depth >= 6).sum()} windows with a low >= 6 hPa deep, "
              f"now {100 * (w * (depth >= 6)).sum() / w.sum():.1f}% of draws (was {100 * (depth >= 6).mean():.1f}%)")
    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=sampler is None, sampler=sampler,
                          num_workers=tc["num_workers"], drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch, shuffle=True, num_workers=tc["num_workers"],
                        generator=torch.Generator().manual_seed(1))

    model = build_stepper(cfg)
    print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
    best, start, ck = np.inf, 1, None
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    if args.resume:
        ck = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        if args.rollout == ck.get("rollout", 1):
            # Same stage: carry on after the last finished epoch with the same schedule.
            best = ck.get("best_val", np.inf)
            start = ck.get("epoch", 0) + 1
            if "opt" in ck:
                opt.load_state_dict(ck["opt"])
                sched.load_state_dict(ck["sched"])
            else:  # checkpoints from before optimiser state was saved
                for _ in range(start - 1):
                    sched.step()
        print(f"resumed from {out / 'last.pt'} at epoch {start} of {args.epochs}")
        if start > args.epochs:
            print("this stage is already complete")
            return
    elif args.init_from:
        init_network(model, args.init_from)
        print(f"network weights initialised from {args.init_from}")
    weight = loss_weight(cfg)
    extra = None
    if args.cyclone_weight:
        im, cells, st = cfg["channels"].index("msl"), box_cells(cfg), train_ds.stats
        mean, std = float(st.mean[im, 0, 0]), float(st.std[im, 0, 0])
        extra = lambda target: pixel_weight(target[:, im] * std + mean, cells, args.cyclone_weight)  # noqa: E731

    log_path = out / "log.csv"
    new_log = not log_path.exists()
    with open(log_path, "a", newline="") as f:
        log = csv.writer(f)
        if new_log:
            log.writerow(["epoch", "rollout", "train_loss", "val_loss", "minutes"])
        for epoch in range(start, args.epochs + 1):
            t0 = time.time()
            model.train()
            tr = run_batches(model, train_dl, weight, args.rollout, args.steps_per_epoch, opt, extra)
            sched.step()
            model.eval()
            with torch.no_grad():
                va = run_batches(model, val_dl, weight, args.rollout, args.val_steps, extra=extra)
            minutes = (time.time() - t0) / 60
            print(f"epoch {epoch}: train {tr:.4f}  val {va:.4f}  ({minutes:.1f} min)", flush=True)
            log.writerow([epoch, args.rollout, f"{tr:.5f}", f"{va:.5f}", f"{minutes:.1f}"])
            f.flush()
            ck = {"model": model.state_dict(), "cfg": cfg, "epoch": epoch, "rollout": args.rollout,
                  "best_val": min(best, va), "opt": opt.state_dict(), "sched": sched.state_dict()}
            torch.save(ck, out / "last.pt")
            if va < best:
                best = va
                torch.save(ck, out / "best.pt")
                print("  saved best.pt")


if __name__ == "__main__":
    main()
