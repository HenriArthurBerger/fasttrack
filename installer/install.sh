#!/usr/bin/env bash
# FastTrack installer for Linux and macOS.
#
#   curl -LsSf https://raw.githubusercontent.com/HenriArthurBerger/fasttrack/main/installer/install.sh | bash
#
# Installs FastTrack into its own folder (nothing touches your system Python), with the PyTorch build
# that matches this machine: CUDA on Linux with an NVIDIA GPU, CPU-only otherwise, Apple GPU support on
# macOS. Then adds the app to your applications menu and the `fasttrack` command to ~/.local/bin.
#
# Run it again to update. Options (environment variables):
#   FASTTRACK_SOURCE=/path/to/checkout   install from a local copy instead of GitHub
#   FASTTRACK_REF=v0.2.0                 a tag, branch or commit on GitHub (default: main)
#   FASTTRACK_TORCH=cpu|cu126|auto       force a PyTorch build (default: auto)
#   FASTTRACK_HOME=/some/folder          where to install
#   FASTTRACK_NO_SHORTCUTS=1             skip the menu entry and the command link
# Uninstall: bash install.sh --uninstall
set -euo pipefail

REPO="https://github.com/HenriArthurBerger/fasttrack"
PYTHON_VERSION="3.12"
OS="$(uname -s)"

if [ "$OS" = "Darwin" ]; then
    HOME_DEFAULT="$HOME/Library/Application Support/FastTrack"
else
    HOME_DEFAULT="${XDG_DATA_HOME:-$HOME/.local/share}/fasttrack"
fi
APP_HOME="${FASTTRACK_HOME:-$HOME_DEFAULT}"
ENV="$APP_HOME/env"
BIN="$HOME/.local/bin"
DESKTOP_FILE="${XDG_DATA_HOME:-$HOME/.local/share}/applications/fasttrack.desktop"
MAC_APP="$HOME/Applications/FastTrack.app"

say() { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31mError:\033[0m %s\n' "$*" >&2; exit 1; }

if [ "${1:-}" = "--uninstall" ]; then
    say "Removing FastTrack from $APP_HOME"
    rm -rf "$APP_HOME" "$DESKTOP_FILE" "$MAC_APP"
    for cmd in fasttrack fasttrack-gui; do
        [ -L "$BIN/$cmd" ] && rm -f "$BIN/$cmd"
    done
    say "Done. Your experiment files and results were not touched."
    exit 0
fi

# ---------------------------------------------------------------- uv (a fast, self-contained Python installer)
UV="$(command -v uv || true)"
if [ -z "$UV" ] && [ -x "$HOME/.local/bin/uv" ]; then UV="$HOME/.local/bin/uv"; fi
if [ -z "$UV" ]; then
    say "Installing uv (https://docs.astral.sh/uv) to manage Python for FastTrack"
    command -v curl >/dev/null || die "curl is needed to download uv"
    curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
    UV="$HOME/.local/bin/uv"
    [ -x "$UV" ] || die "uv did not install; see https://docs.astral.sh/uv/getting-started/installation/"
fi

# ---------------------------------------------------------------- environment
mkdir -p "$APP_HOME"
if [ ! -x "$ENV/bin/python" ]; then
    say "Creating a Python $PYTHON_VERSION environment in $ENV"
    "$UV" venv --quiet --python "$PYTHON_VERSION" "$ENV"
fi

if [ -n "${FASTTRACK_SOURCE:-}" ]; then
    SPEC="fasttrack[all] @ file://$(cd "$FASTTRACK_SOURCE" && pwd)"
else
    SPEC="fasttrack[all] @ git+$REPO@${FASTTRACK_REF:-main}"
fi

# uv picks the PyTorch build from the NVIDIA driver it finds (auto); without one it takes the CPU build.
# macOS has one PyTorch build (CPU + Apple GPU), so the backend setting is only used elsewhere.
TORCH="${FASTTRACK_TORCH:-auto}"
say "Installing FastTrack and PyTorch (build: $TORCH). The first install downloads 1-3 GB."
if [ "$OS" = "Darwin" ]; then
    "$UV" pip install --python "$ENV/bin/python" --reinstall-package fasttrack "$SPEC"
else
    "$UV" pip install --python "$ENV/bin/python" --torch-backend "$TORCH" --reinstall-package fasttrack "$SPEC"
fi

"$ENV/bin/fasttrack" version

# ---------------------------------------------------------------- shortcuts
if [ -z "${FASTTRACK_NO_SHORTCUTS:-}" ]; then
    mkdir -p "$BIN"
    ln -sf "$ENV/bin/fasttrack" "$BIN/fasttrack"
    ln -sf "$ENV/bin/fasttrack-gui" "$BIN/fasttrack-gui"
    ICON_DIR="$("$ENV/bin/python" -I -c 'import fasttrack.gui, pathlib; print(pathlib.Path(fasttrack.gui.__file__).parent / "assets")')"

    if [ "$OS" = "Darwin" ]; then
        say "Adding FastTrack to ~/Applications"
        rm -rf "$MAC_APP"
        mkdir -p "$MAC_APP/Contents/MacOS" "$MAC_APP/Contents/Resources"
        cp "$ICON_DIR/icon.icns" "$MAC_APP/Contents/Resources/FastTrack.icns"
        cat > "$MAC_APP/Contents/MacOS/FastTrack" <<EOF
#!/bin/bash
exec "$ENV/bin/fasttrack-gui" "\$@"
EOF
        chmod +x "$MAC_APP/Contents/MacOS/FastTrack"
        cat > "$MAC_APP/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>FastTrack</string>
    <key>CFBundleDisplayName</key><string>FastTrack</string>
    <key>CFBundleIdentifier</key><string>io.github.henriarthurberger.fasttrack</string>
    <key>CFBundleExecutable</key><string>FastTrack</string>
    <key>CFBundleIconFile</key><string>FastTrack</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleShortVersionString</key><string>$("$ENV/bin/python" -I -c 'import fasttrack; print(fasttrack.__version__)')</string>
    <key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
EOF
        touch "$MAC_APP"
    else
        say "Adding FastTrack to the applications menu"
        mkdir -p "$(dirname "$DESKTOP_FILE")"
        cat > "$DESKTOP_FILE" <<EOF
[Desktop Entry]
Type=Application
Name=FastTrack
Comment=Track and measure cells in timelapse microscopy
Exec="$ENV/bin/fasttrack-gui" %f
Icon=$ICON_DIR/icon.png
Terminal=false
Categories=Science;Biology;Education;
StartupWMClass=fasttrack
MimeType=application/x-yaml;
EOF
        command -v update-desktop-database >/dev/null && update-desktop-database "$(dirname "$DESKTOP_FILE")" 2>/dev/null || true
    fi
    case ":$PATH:" in
        *":$BIN:"*) ;;
        *) say "Add $BIN to your PATH to use the fasttrack command in a terminal." ;;
    esac
fi

say "FastTrack is installed. Open it from your applications, or run: fasttrack gui"
