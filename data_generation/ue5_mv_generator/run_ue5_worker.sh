#!/bin/bash
# Wrapper to run UE5 as non-root user
JOB_PATH="$1"
LOG_PATH="$2"
TIMEOUT="${3:-600}"

UE5_BIN="${UE5_BIN:-/opt/UnrealEngine/Engine/Binaries/Linux/UnrealEditor-Cmd}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UPROJECT="${VISTA_UE_PROJECT:-${SCRIPT_DIR}/ue_project/MVTrackGen.uproject}"
UE_PROJECT_DIR="$(dirname "$UPROJECT")"

# Fix permissions
chmod -R 777 "$UE_PROJECT_DIR" /tmp/mvtrack_jobs 2>/dev/null || true

# Parse job to get output dir and resolution
OUTPUT_DIR=$(python3 -c "import json; j=json.load(open(\"$JOB_PATH\")); print(j[\"output_dir\"])")
RESX=$(python3 -c "import json; j=json.load(open(\"$JOB_PATH\")); print(j[\"resolution\"][0])")
RESY=$(python3 -c "import json; j=json.load(open(\"$JOB_PATH\")); print(j[\"resolution\"][1])")

mkdir -p "$OUTPUT_DIR"
chmod 777 "$OUTPUT_DIR"

# Run as ue5user
timeout "$TIMEOUT" su - ue5user -c "$UE5_BIN $UPROJECT /Game/Maps/Default \
    -game -vulkan -RenderOffScreen -unattended -stdout -FullStdOutLogOutput \
    -MVTrackJob=$JOB_PATH \
    -NoTextureStreaming -NoLoadingScreen \
    -resx=$RESX -resy=$RESY \
    -windowed -NOSPLASH -NOSOUND" > "$LOG_PATH" 2>&1

echo $?
