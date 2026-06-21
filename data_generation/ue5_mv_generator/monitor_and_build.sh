#!/bin/bash
# Monitor UE5 setup and auto-start build when ready
set -euo pipefail

LOG="/tmp/ue5_setup.log"
BUILD_LOG="/tmp/ue5_build.log"

echo "Monitoring UE5 setup progress..."
while true; do
    if ! pgrep -f "Setup.sh" > /dev/null 2>&1; then
        # Check if setup completed successfully
        if grep -q "Dependencies are up to date" "${LOG}" 2>/dev/null || \
           grep -q "Setup successful" "${LOG}" 2>/dev/null || \
           [ -f /opt/UnrealEngine/Engine/Build/BatchFiles/Linux/Build.sh ]; then
            echo "Setup appears complete! Starting build..."
            break
        fi
        # Check if setup failed
        if grep -qi "error\|failed\|fatal" "${LOG}" 2>/dev/null; then
            echo "Setup may have failed. Check ${LOG}"
            tail -10 "${LOG}"
            exit 1
        fi
    fi

    PCT=$(grep -oP "Updating dependencies:\s+\d+%" "${LOG}" 2>/dev/null | tail -1 || echo "?")
    SPEED=$(grep -oP "\d+\.\d+ MiB/s" "${LOG}" 2>/dev/null | tail -1 || echo "?")
    echo "$(date +%H:%M:%S) Setup progress: ${PCT} (${SPEED})"
    sleep 60
done

echo "Starting UE5 build..."
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash build_ue_project.sh 2>&1 | tee "${BUILD_LOG}"
