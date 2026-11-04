#!/usr/bin/env bash
# Daily verification on GitHub Actions, after the forecast (see .github/workflows/daily_forecast.yml).
# The forecasts and observations kept for scoring live on the verif-data branch; the page's data
# and maps go to verif-site, which holds only the latest copy.
# BACKFILL=true also archives earlier 00 UTC forecasts still kept as run artifacts (14 days).
set -euo pipefail
remote="https://x-access-token:${TOKEN}@github.com/${GITHUB_REPOSITORY}.git"
if ! git clone -q --depth 1 -b verif-data "$remote" vd 2>/dev/null; then
  mkdir -p vd && git -C vd init -q -b verif-data && git -C vd remote add origin "$remote"
fi
git -C vd config user.name "masrainman-bot"
git -C vd config user.email "masrainman-bot@users.noreply.github.com"

python -u -m masrainman.verify archive --nc "$(ls out/live/fc_??????????.nc | head -1)" --data vd

if [ "${BACKFILL:-false}" = "true" ]; then
  for run in $(gh run list -w daily_forecast.yml -s success -L 40 --json databaseId -q '.[].databaseId'); do
    rm -rf art
    gh run download "$run" -n masrainman-forecast -D art 2>/dev/null || continue
    for nc in art/fc_????????00.nc; do
      [ -e "$nc" ] || continue
      tag=$(basename "$nc" 00.nc); tag=${tag#fc_}
      [ -e "vd/fc/masrainman_${tag}.npz" ] || python -u -m masrainman.verify archive --nc "$nc" --data vd
    done
  done
  rm -rf art
fi

python -u -m masrainman.verify update --data vd --site vs

git -C vd add -A
git -C vd diff --cached --quiet || git -C vd commit -q -m "Verification data $(date -u '+%Y-%m-%d %H:%M')"
git -C vd push -q origin verif-data

cd vs
git init -q -b verif-site
git config user.name "masrainman-bot"
git config user.email "masrainman-bot@users.noreply.github.com"
git add .
git commit -q -m "Verification page $(date -u '+%Y-%m-%d %H:%M')"
git push -q --force "$remote" verif-site
