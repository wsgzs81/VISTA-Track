#!/bin/bash
# Build the editor target used by UnrealEditor-Cmd data generation.
set -euo pipefail

UE5_DIR="${UE5_DIR:-/opt/UnrealEngine}"
PROJ_DIR="${PROJ_DIR:-/workspace/home/zhangboqiang/mvtrack_gen/ue_project}"
BUILD_SCRIPT="${UE5_DIR}/Engine/Build/BatchFiles/Linux/Build.sh"

echo "=== Building MVTrackGenEditor ==="
echo "Engine: ${UE5_DIR}"
echo "Project: ${PROJ_DIR}"

cd "${UE5_DIR}"
./GenerateProjectFiles.sh -project="${PROJ_DIR}/MVTrackGen.uproject" -game -engine

"${BUILD_SCRIPT}" MVTrackGenEditor Linux Development \
    -project="${PROJ_DIR}/MVTrackGen.uproject" \
    -waitmutex -NoHotReloadFromIDE

EDITOR_BIN=$(find "${UE5_DIR}/Engine/Binaries/Linux" -name "UnrealEditor-Cmd" -type f 2>/dev/null | head -1)
if [ -n "${EDITOR_BIN}" ]; then
    ln -sf "${EDITOR_BIN}" /usr/local/bin/UnrealEditor-Cmd
    echo "Symlink created: UnrealEditor-Cmd -> ${EDITOR_BIN}"
fi

echo "=== MVTrackGenEditor Build Complete ==="
