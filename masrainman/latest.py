"""Newest ECMWF IFS open-data run that is completely published (no heavy imports).

  python -m masrainman.latest      -> e.g. 2026-11-04T06

00/12 UTC runs count once step 240 is out, 06/18 UTC runs once step 90 is out. ecmwf-opendata's
own latest() cannot search 06/18 runs, so this checks the index files directly.
"""
from datetime import datetime, timedelta, timezone

import requests

ROOT = "https://data.ecmwf.int/forecasts"
LAST = {"oper": 240, "scda": 90}


def index_url(run):
    st = "oper" if run.hour in (0, 12) else "scda"
    return (f"{ROOT}/{run:%Y%m%d}/{run:%H}z/ifs/0p25/{st}/"
            f"{run:%Y%m%d%H}0000-{LAST[st]}h-{st}-fc.index")


def latest(now=None, hours_back=48, exists=None):
    now = now or datetime.now(timezone.utc)
    run = now.replace(hour=now.hour - now.hour % 6, minute=0, second=0, microsecond=0)
    if exists is None:
        exists = lambda url: requests.head(url, timeout=30, allow_redirects=True).status_code == 200
    for _ in range(hours_back // 6 + 1):
        if exists(index_url(run)):
            return run
        run -= timedelta(hours=6)
    raise RuntimeError(f"no complete ECMWF run in the last {hours_back} h")


if __name__ == "__main__":
    print(latest().strftime("%Y-%m-%dT%H"))
