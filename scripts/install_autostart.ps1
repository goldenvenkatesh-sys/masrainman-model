# Start the ERA5 download automatically every time you log in to Windows.
# Adds a shortcut to your Startup folder that runs scripts\download_era5.ps1 minimized.
#   powershell -ExecutionPolicy Bypass -File scripts\install_autostart.ps1            # install
#   powershell -ExecutionPolicy Bypass -File scripts\install_autostart.ps1 -Remove    # uninstall
# The download skips finished files and rejoins queued requests, so starting it at every
# login is safe; if one is already running, the second start exits at once.
param([switch]$Remove)

$link = Join-Path ([Environment]::GetFolderPath("Startup")) "MasRainMan ERA5 download.lnk"
if ($Remove) {
    Remove-Item $link -ErrorAction SilentlyContinue
    Write-Host "Removed $link"
    exit 0
}

$script = Join-Path $PSScriptRoot "download_era5.ps1"
$shell = New-Object -ComObject WScript.Shell
$s = $shell.CreateShortcut($link)
$s.TargetPath = "$env:WINDIR\System32\WindowsPowerShell\v1.0\powershell.exe"
$s.Arguments = "-NoProfile -ExecutionPolicy Bypass -WindowStyle Minimized -File `"$script`""
$s.WorkingDirectory = Split-Path $PSScriptRoot -Parent
$s.WindowStyle = 7  # minimized
$s.Description = "MasRainMan: resume the ERA5 download at login"
$s.Save()
Write-Host "Installed $link"
Write-Host "  -> $script"
