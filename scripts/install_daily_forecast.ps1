# Run the live forecast (scripts\live_daily.ps1) every day via Windows Task Scheduler.
# ECMWF's 00 UTC run is fully published by about 13:30 IST, so the default is 14:00 IST.
# If the laptop is off at that time, the task runs as soon as it is switched on (it then
# uses the latest complete ECMWF run, which may be the previous day's 12 UTC one).
# Needs no administrator rights; runs only while you are logged in.
#   powershell -ExecutionPolicy Bypass -File scripts\install_daily_forecast.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\install_daily_forecast.ps1 -At 15:30
#   powershell -ExecutionPolicy Bypass -File scripts\install_daily_forecast.ps1 -Remove
param([string]$At = "14:00", [switch]$Remove)

$name = "MasRainMan daily forecast"
if ($Remove) {
    Unregister-ScheduledTask -TaskName $name -Confirm:$false -ErrorAction SilentlyContinue
    Write-Host "Removed task '$name'"
    exit 0
}

$script = Join-Path $PSScriptRoot "live_daily.ps1"
$action = New-ScheduledTaskAction -Execute "$env:WINDIR\System32\WindowsPowerShell\v1.0\powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Minimized -File `"$script`"" `
    -WorkingDirectory (Split-Path $PSScriptRoot -Parent)
$trigger = New-ScheduledTaskTrigger -Daily -At $At
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -RunOnlyIfNetworkAvailable `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Hours 2)
Register-ScheduledTask -TaskName $name -Action $action -Trigger $trigger -Settings $settings `
    -Description "MasRainMan AI: 10-day forecast from the latest ECMWF open-data run" -Force | Out-Null
Write-Host "Installed task '$name': daily at $At (or as soon as the laptop is on after that)"
Write-Host "  -> $script"
Write-Host "Output: D:\masrainman_runs\live\  Log: D:\masrainman_runs\live\live.log"
