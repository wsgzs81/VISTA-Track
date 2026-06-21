#!/bin/bash
# setup_ue5.sh �?Download and build UE5 from source on Ubuntu 22.04
# Prerequisites: Epic Games GitHub access (https://www.unrealengine.com/en-US/linux)
set -euo pipefail

UE5_VERSION="${1:-5.4}"
INSTALL_DIR="/opt/UnrealEngine"
WORK_DIR="."

echo "============================================"
echo "UE5 Setup Script"
echo "Version: ${UE5_VERSION}"
echo "Install: ${INSTALL_DIR}"
echo "============================================"

# ── Step 1: Install build dependencies ─────────────────────────────────
echo "[1/6] Installing build dependencies..."
apt-get update -qq
apt-get install -y -qq \
    build-essential clang lld llvm cmake ninja-build git git-lfs \
    python3 python3-venv python3-pip \
    libvulkan-dev vulkan-tools \
    xvfb x11-xserver-utils \
    libssl-dev libcurl4-openssl-dev \
    libfreetype-dev libfontconfig1-dev \
    rsync wget curl jq

# Verify tools
clang++ --version | head -1
cmake --version | head -1
ninja --version
echo "[1/6] Dependencies OK"

# ── Step 2: Clone UE5 ──────────────────────────────────────────────────
if [ -d "${INSTALL_DIR}" ]; then
    echo "[2/6] UE5 directory already exists at ${INSTALL_DIR}"
else
    echo "[2/6] Cloning UE5 (this requires Epic Games GitHub access)..."
    echo ""
    echo "If this fails, you need to:"
    echo "  1. Go to https://www.unrealengine.com/en-US/linux"
    echo "  2. Sign in with your Epic Games account"
    echo "  3. Connect your GitHub account"
    echo "  4. Generate a GitHub Personal Access Token"
    echo "  5. Run: git config --global credential.helper store"
    echo "  6. Run: echo 'https://USERNAME:TOKEN@github.com' > ~/.git-credentials"
    echo ""

    git clone --depth 1 --branch ${UE5_VERSION}-release \
        https://github.com/EpicGames/UnrealEngine.git \
        "${INSTALL_DIR}" || {
        echo "ERROR: Clone failed. Check your Epic Games GitHub access."
        echo "Visit: https://www.unrealengine.com/en-US/linux"
        exit 1
    }
fi
echo "[2/6] Clone OK"

# ── Step 3: Setup UE5 ──────────────────────────────────────────────────
echo "[3/6] Running UE5 setup..."
cd "${INSTALL_DIR}"
./Setup.sh --force || true
./GenerateProjectFiles.sh || true
echo "[3/6] Setup OK"

# ── Step 4: Build UE5 Editor ───────────────────────────────────────────
echo "[4/6] Building UE5 Editor (this takes 2-6 hours)..."
echo "  Using: make -j$(nproc) UnrealEditor"
cd "${INSTALL_DIR}"
make -j$(nproc) UnrealEditor 2>&1 | tee /tmp/ue5_build.log
echo "[4/6] Build OK"

# ── Step 5: Create symlink ─────────────────────────────────────────────
echo "[5/6] Creating symlinks..."
UE5_BIN="${INSTALL_DIR}/Engine/Binaries/Linux/UnrealEditor-Cmd"
if [ -f "${UE5_BIN}" ]; then
    ln -sf "${UE5_BIN}" /usr/local/bin/UnrealEditor-Cmd
    echo "  UnrealEditor-Cmd -> ${UE5_BIN}"
else
    echo "  WARNING: UnrealEditor-Cmd not found at expected path"
    find "${INSTALL_DIR}/Engine/Binaries" -name "UnrealEditor*" -type f 2>/dev/null | head -5
fi
echo "[5/6] Symlinks OK"

# ── Step 6: Install Python deps for orchestrator ───────────────────────
echo "[6/6] Installing Python dependencies..."
pip3 install pyyaml numpy pillow opencv-python-headless
echo "[6/6] Python deps OK"

# ── Verify ─────────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "UE5 Setup Complete!"
echo ""
echo "Verify: UnrealEditor-Cmd --version"
echo ""
echo "To generate the dataset:"
echo "  cd ${WORK_DIR}"
echo "  python3 tools/orchestrator/orchestrator.py --config configs/dataset.yaml --mode generate --dry-run"
echo "============================================"
