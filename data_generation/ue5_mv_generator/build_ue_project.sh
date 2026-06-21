#!/bin/bash
# Build MVTrackGen UE5 project after engine setup completes
set -euo pipefail

UE5_DIR="${UE5_DIR:-/opt/UnrealEngine}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="${VISTA_UE_PROJECT_DIR:-${SCRIPT_DIR}/ue_project}"
BUILD_SCRIPT="${UE5_DIR}/Engine/Build/BatchFiles/Linux/Build.sh"

echo "=== Building MVTrackGen ==="
echo "Engine: ${UE5_DIR}"
echo "Project: ${PROJ_DIR}"

# Generate project files
echo "[1/3] Generating project files..."
cd "${UE5_DIR}"
./GenerateProjectFiles.sh -project="${PROJ_DIR}/MVTrackGen.uproject" -game -engine 2>&1 | tail -5

# Build Development Editor
echo "[2/3] Building MVTrackGen Editor..."
"${BUILD_SCRIPT}" MVTrackGen Linux Development \
    -project="${PROJ_DIR}/MVTrackGen.uproject" \
    -waitmutex -NoHotReloadFromIDE 2>&1 | tail -20

# Create symlink for easy access
EDITOR_BIN=$(find "${UE5_DIR}/Engine/Binaries/Linux" -name "UnrealEditor-Cmd" -type f 2>/dev/null | head -1)
if [ -n "${EDITOR_BIN}" ]; then
    ln -sf "${EDITOR_BIN}" /usr/local/bin/UnrealEditor-Cmd
    echo "[3/3] Symlink created: UnrealEditor-Cmd -> ${EDITOR_BIN}"
fi

echo "=== Build Complete ==="
echo "Test run:"
echo "  UnrealEditor-Cmd ${PROJ_DIR}/MVTrackGen.uproject /Game/Maps/Default -game -vulkan -RenderOffScreen -unattended -stdout -MVTrackJob=/tmp/test_job.json"
