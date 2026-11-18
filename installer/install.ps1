# FastTrack installer for Windows. In PowerShell:
#
#   powershell -ExecutionPolicy ByPass -c "irm https://raw.githubusercontent.com/HenriArthurBerger/fasttrack/main/installer/install.ps1 | iex"
#
# Installs FastTrack into its own folder (nothing touches any other Python), with the PyTorch build that
# matches this machine: CUDA with an NVIDIA GPU, CPU-only otherwise. Adds FastTrack to the Start menu and
# the desktop. Run it again to update. Options (environment variables, set before running):
#   $env:FASTTRACK_SOURCE = "C:\path\to\checkout"   install from a local copy instead of GitHub
#   $env:FASTTRACK_REF = "v0.2.0"                     a tag, branch or commit on GitHub (default: main)
#   $env:FASTTRACK_TORCH = "cpu" | "cu126" | "auto"   force a PyTorch build (default: auto)
#   $env:FASTTRACK_HOME = "D:\FastTrack"              where to install
#   $env:FASTTRACK_NO_SHORTCUTS = "1"                 skip the shortcuts
# Uninstall: $env:FASTTRACK_UNINSTALL = "1" and run it again.
$ErrorActionPreference = "Stop"

$Repo = "https://github.com/HenriArthurBerger/fasttrack"
$PythonVersion = "3.12"
$AppHome = if ($env:FASTTRACK_HOME) { $env:FASTTRACK_HOME } else { Join-Path $env:LOCALAPPDATA "FastTrack" }
$EnvDir = Join-Path $AppHome "env"
$Scripts = Join-Path $EnvDir "Scripts"
$StartMenu = Join-Path ([Environment]::GetFolderPath("Programs")) "FastTrack.lnk"
$Desktop = Join-Path ([Environment]::GetFolderPath("Desktop")) "FastTrack.lnk"

function Say($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }

if ($env:FASTTRACK_UNINSTALL) {
    Say "Removing FastTrack from $AppHome"
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $AppHome, $StartMenu, $Desktop
    Say "Done. Your experiment files and results were not touched."
    return
}

# ---------------------------------------------------------------- uv (a fast, self-contained Python installer)
$Uv = (Get-Command uv -ErrorAction SilentlyContinue).Source
$UvHome = Join-Path $env:USERPROFILE ".local\bin\uv.exe"
if (-not $Uv -and (Test-Path $UvHome)) { $Uv = $UvHome }
if (-not $Uv) {
    Say "Installing uv (https://docs.astral.sh/uv) to manage Python for FastTrack"
    $env:UV_NO_MODIFY_PATH = "1"
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    $Uv = $UvHome
    if (-not (Test-Path $Uv)) { throw "uv did not install; see https://docs.astral.sh/uv/getting-started/installation/" }
}

# ---------------------------------------------------------------- environment
New-Item -ItemType Directory -Force $AppHome | Out-Null
$Python = Join-Path $Scripts "python.exe"
if (-not (Test-Path $Python)) {
    Say "Creating a Python $PythonVersion environment in $EnvDir"
    & $Uv venv --quiet --python $PythonVersion $EnvDir
    if ($LASTEXITCODE) { throw "uv venv failed" }
}

if ($env:FASTTRACK_SOURCE) {
    $Source = (Resolve-Path $env:FASTTRACK_SOURCE).Path -replace '\\', '/'
    $Spec = "fasttrack[all] @ file:///$Source"
} else {
    $Ref = if ($env:FASTTRACK_REF) { $env:FASTTRACK_REF } else { "main" }
    $Spec = "fasttrack[all] @ git+$Repo@$Ref"
}
$Torch = if ($env:FASTTRACK_TORCH) { $env:FASTTRACK_TORCH } else { "auto" }

Say "Installing FastTrack and PyTorch (build: $Torch). The first install downloads 1-3 GB."
& $Uv pip install --python $Python --torch-backend $Torch --reinstall-package fasttrack $Spec
if ($LASTEXITCODE) { throw "Installing FastTrack failed" }

& (Join-Path $Scripts "fasttrack.exe") version

# ---------------------------------------------------------------- shortcuts
if (-not $env:FASTTRACK_NO_SHORTCUTS) {
    Say "Adding FastTrack to the Start menu and the desktop"
    $Icon = & $Python -I -c "import fasttrack.gui, pathlib; print(pathlib.Path(fasttrack.gui.__file__).parent / 'assets' / 'icon.ico')"
    $Shell = New-Object -ComObject WScript.Shell
    foreach ($Link in @($StartMenu, $Desktop)) {
        $Shortcut = $Shell.CreateShortcut($Link)
        $Shortcut.TargetPath = Join-Path $Scripts "fasttrack-gui.exe"
        $Shortcut.WorkingDirectory = [Environment]::GetFolderPath("MyDocuments")
        $Shortcut.IconLocation = $Icon
        $Shortcut.Description = "Track and measure cells in timelapse microscopy"
        $Shortcut.Save()
    }
    Say "To use the fasttrack command in a terminal: $Scripts\fasttrack.exe"
}

Say "FastTrack is installed. Open it from the Start menu or the desktop."
