# Start a MasRainMan job automatically every time you log in to Windows.
# Adds a shortcut to your Startup folder that runs the job's script minimized.
#   powershell -ExecutionPolicy Bypass -File scripts\install_autostart.ps1                  # ERA5 download
#   powershell -ExecutionPolicy Bypass -File scripts\install_autostart.ps1 -Job training    # full training run
#   add -Remove to uninstall that job's shortcut
# Both jobs skip finished work and resume where they stopped, so starting them at every
# login is safe; if one is already running, the second start exits at once.
param([ValidateSet("download", "training")][string]$Job = "download", [switch]$Remove)

$jobs = @{
    download = @{ Script = "download_era5.ps1"; Name = "MasRainMan ERA5 download" }
    training = @{ Script = "full_run.ps1"; Name = "MasRainMan full training" }
}
$link = Join-Path ([Environment]::GetFolderPath("Startup")) "$($jobs[$Job].Name).lnk"
if ($Remove) {
    Remove-Item $link -ErrorAction SilentlyContinue
    Write-Host "Removed $link"
    exit 0
}

$script = Join-Path $PSScriptRoot $jobs[$Job].Script
$shell = New-Object -ComObject WScript.Shell
$s = $shell.CreateShortcut($link)
$s.TargetPath = "$env:WINDIR\System32\WindowsPowerShell\v1.0\powershell.exe"
$s.Arguments = "-NoProfile -ExecutionPolicy Bypass -WindowStyle Minimized -File `"$script`""
$s.WorkingDirectory = Split-Path $PSScriptRoot -Parent
$s.WindowStyle = 7  # minimized
$s.Description = "MasRainMan: resume the $Job job at login"
$s.Save()
Write-Host "Installed $link"
Write-Host "  -> $script"
