# Cyclone fine-tune: continue from the full run's one-step model, drawing training windows
# that hold a cyclone ~4x as often and counting errors around deep lows up to 4x as much,
# then the 24-h fine-tune with the same settings. Scores storms, the whole of 2023, and
# redraws the Michaung forecast, so the result can be compared with D:\masrainman_runs\full.
# Safe to start again at any time, like full_run.ps1 (stage markers + training resume).
#   powershell -ExecutionPolicy Bypass -File scripts\cyclone_run.ps1
# scripts\install_autostart.ps1 -Job cyclone   starts it at every Windows login.
param([string]$Run = "D:\masrainman_runs\cyclone",
      [string]$WarmStart = "D:\masrainman_runs\full\best_stage1.pt")

$others = Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" | Where-Object {
    $_.ProcessId -ne $PID -and $_.CommandLine -match '-File\s+"?[^"]*cyclone_run\.ps1'
}
if ($others) { Write-Host "The cyclone run is already going (process $($others[0].ProcessId))."; exit 0 }

Set-Location (Split-Path $PSScriptRoot -Parent)
New-Item -ItemType Directory -Force $Run | Out-Null
$Host.UI.RawUI.WindowTitle = "MasRainMan CYCLONE training"
$log = Join-Path $Run "run.log"
$ck = Join-Path $Run "best.pt"
$name = Split-Path $Run -Leaf
$cyclone = @("--cyclone-boost", "5", "--cyclone-weight", "3")
if (Test-Path (Join-Path $Run "ALL_DONE")) { Write-Host "The cyclone run is already complete."; exit 0 }

function Step($name, $title, $argv) {
    $marker = Join-Path $Run "$name.done"
    if (Test-Path $marker) { return }
    Add-Content $log "=== $(Get-Date -Format s) $title"
    & python -u @argv 2>&1 | ForEach-Object { "$_"; Add-Content -Path $log -Value "$_" }
    $code = $LASTEXITCODE
    Add-Content $log "=== $(Get-Date -Format s) exit $code ($title)"
    if ($code -ne 0) { Add-Content $log "=== FAILED at: $title"; exit 1 }
    New-Item -ItemType File $marker | Out-Null
}

$start = if (Test-Path (Join-Path $Run "last.pt")) { @("--resume") } else { @("--init-from", $WarmStart) }
Step "1_train" "cyclone training 6 epochs" (@("-m", "masrainman.train", "--name", $name, "--epochs", "6",
    "--steps-per-epoch", "0", "--val-steps", "60", "--lr", "1e-4") + $cyclone + $start)
if (-not (Test-Path (Join-Path $Run "best_stage1.pt"))) { Copy-Item $ck (Join-Path $Run "best_stage1.pt") }

Step "2_finetune" "24-h fine-tune (4 steps)" (@("-m", "masrainman.train", "--name", $name, "--resume",
    "--rollout", "4", "--epochs", "1", "--steps-per-epoch", "240", "--val-steps", "20", "--lr", "5e-5") + $cyclone)
Step "3_storms" "storm scores" @("-m", "masrainman.cyclones", "--ckpt", $ck)
Step "4_score" "score on 2023" @("-m", "masrainman.evaluate", "--ckpt", $ck, "--days", "10", "--every-days", "5", "--batch", "4")
$fc = Join-Path $Run "forecasts"
Step "5_michaung" "Cyclone Michaung forecast" @("-m", "masrainman.forecast", "--ckpt", $ck, "--init", "2023-12-01T00",
    "--days", "10", "--out", $fc)
Step "6_maps" "Michaung maps" @("-m", "masrainman.plot", (Join-Path $fc "fc_2023120100.nc"), "--leads", "24", "72", "96", "120")

New-Item -ItemType File (Join-Path $Run "ALL_DONE") | Out-Null
Add-Content $log "=== $(Get-Date -Format s) ALL DONE"
Write-Host "Cyclone run complete. You can close this window." -ForegroundColor Green
