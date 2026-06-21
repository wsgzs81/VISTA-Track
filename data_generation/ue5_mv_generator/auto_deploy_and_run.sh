#!/bin/bash
# auto_deploy_and_run.sh �?Complete UE5 MVTrackGen build + test + dataset generation
# Run this on the server after UE5 engine is built
set -euo pipefail

UE5_DIR="/opt/UnrealEngine"
PROJ_DIR="ue_project"
TOOLS_DIR="tools"
OUTPUT_DIR="output"
UE5_BUILD="${UE5_DIR}/Engine/Build/BatchFiles/Linux/Build.sh"
UE5_EDITOR="${UE5_DIR}/Engine/Binaries/Linux/UnrealEditor-Cmd"

echo "============================================"
echo "MVTrackGen Auto Deploy & Run"
echo "============================================"

# ── Step 0: Verify UE5 engine ──────────────────────────────────────────
echo "[0/7] Verifying UE5 engine..."
if [ ! -f "${UE5_EDITOR}" ]; then
    echo "ERROR: UnrealEditor-Cmd not found at ${UE5_EDITOR}"
    echo "Build the engine first: cd ${UE5_DIR} && make -j\$(nproc) UnrealEditor"
    exit 1
fi
echo "  UnrealEditor-Cmd found: $(ls -lh ${UE5_EDITOR} | awk '{print $5}')"

# ── Step 1: Compile MVTrackGen project ─────────────────────────────────
echo "[1/7] Compiling MVTrackGen project..."
cd "${UE5_DIR}"
${UE5_BUILD} MVTrackGen Linux Development \
    -project="${PROJ_DIR}/MVTrackGen.uproject" \
    -waitmutex 2>&1 | tail -20
echo "  Project compiled."

# ── Step 2: Create test job manifest ───────────────────────────────────
echo "[2/7] Creating test job manifest..."
mkdir -p /tmp/mvtrack_jobs /tmp/mvtrack_test_output
python3 -c "
import json, pathlib
job = {
    'job_id': 'test_001',
    'sequence_id': 'seq_test_001',
    'sequence_index': 0,
    'seed': 42,
    'target_category': 'cube',
    'target_mesh': 'builtin_cube',
    'target_scale_m': 0.3,
    'target_motion_type': 'physics_rolling',
    'num_cameras': 4,
    'num_frames': 10,
    'fps': 30,
    'resolution': [1280, 720],
    'num_occluders': 3,
    'occluder_categories': ['wall_segment', 'pillar', 'box_obstacle'],
    'output_dir': '/tmp/mvtrack_test_output',
    'status': 'pending'
}
pathlib.Path('/tmp/mvtrack_jobs/test_001.json').write_text(json.dumps(job, indent=2))
print('Test job manifest created')
"

# ── Step 3: Create default map via UE Python ────────────────────────────
echo "[3/7] Creating default map..."
mkdir -p "${PROJ_DIR}/Content/Maps"

# Create a Python script that generates the default map
cat > /tmp/create_default_map.py << 'PYEOF'
import unreal
import os

# Create a new empty level
level_lib = unreal.EditorLevelLibrary
asset_lib = unreal.EditorAssetLibrary

# Create the map directory
map_path = "/Game/Maps"
if not asset_lib.does_directory_exist(map_path):
    asset_lib.make_directory(map_path)

# Create a new level
level_lib.new_level("/Game/Maps/Default")
level_lib.save_current_level()

# Add a floor plane
floor_actor = level_lib.spawn_actor_from_class(
    unreal.StaticMeshActor,
    unreal.Vector(0, 0, 0),
    unreal.Rotator(0, 0, 0)
)
if floor_actor:
    mesh_comp = floor_actor.static_mesh_component
    mesh_comp.set_world_scale3d(unreal.Vector(20, 20, 0.1))
    mesh_comp.set_collision_profile_name("BlockAll")

# Add a directional light
light_actor = level_lib.spawn_actor_from_class(
    unreal.DirectionalLight,
    unreal.Vector(0, 0, 500),
    unreal.Rotator(-45, 45, 0)
)

# Add a skylight
sky_actor = level_lib.spawn_actor_from_class(
    unreal.SkyLight,
    unreal.Vector(0, 0, 300),
    unreal.Rotator(0, 0, 0)
)

level_lib.save_current_level()
unreal.log("Default map created at /Game/Maps/Default")
PYEOF

# Try to run the map creation script
${UE5_EDITOR} "${PROJ_DIR}/MVTrackGen.uproject" \
    -run=pythonscript \
    -script="/tmp/create_default_map.py" \
    -unattended -stdout 2>&1 | tail -5 || echo "  Map creation script had issues (non-fatal)"

# ── Step 4: Test run single sequence ────────────────────────────────────
echo "[4/7] Running test sequence (10 frames, 4 cameras)..."
mkdir -p /tmp/mvtrack_test_output

timeout 300 ${UE5_EDITOR} "${PROJ_DIR}/MVTrackGen.uproject" \
    /Game/Maps/Default \
    -game \
    -vulkan \
    -RenderOffScreen \
    -unattended \
    -stdout \
    -FullStdOutLogOutput \
    -MVTrackJob=/tmp/mvtrack_jobs/test_001.json \
    -NoTextureStreaming \
    -NoLoadingScreen \
    -resx=1280 \
    -resy=720 \
    -windowed \
    -NOSPLASH \
    -NOSOUND 2>&1 | tee /tmp/mvtrack_test_run.log | tail -30

TEST_EXIT=$?
echo "  Test run exit code: ${TEST_EXIT}"

# ── Step 5: Validate test output ────────────────────────────────────────
echo "[5/7] Validating test output..."
if [ -f /tmp/mvtrack_test_output/seq_meta.json ]; then
    echo "  seq_meta.json exists"
    python3 -c "import json; m=json.load(open('/tmp/mvtrack_test_output/seq_meta.json')); print(f'  success={m.get(\"success\")}, frames={m.get(\"num_frames\")}, annotations={m.get(\"total_annotations\")}')"
else
    echo "  WARNING: seq_meta.json not found"
fi

ls -la /tmp/mvtrack_test_output/ 2>/dev/null || echo "  Output directory empty"

# Check camera calibrations
if [ -d /tmp/mvtrack_test_output/cameras ]; then
    echo "  Camera calibrations: $(ls /tmp/mvtrack_test_output/cameras/*.json 2>/dev/null | wc -l) files"
fi

# Check frames
if [ -d /tmp/mvtrack_test_output/frames ]; then
    echo "  Frame directories: $(ls -d /tmp/mvtrack_test_output/frames/cam_* 2>/dev/null | wc -l) cameras"
fi

# ── Step 6: Run offline QC ──────────────────────────────────────────────
echo "[6/7] Running offline QC..."
python3 "${TOOLS_DIR}/qc/offline_qc.py" \
    --input /tmp/mvtrack_test_output \
    --cameras 4 --frames 10 2>&1 || echo "  QC had issues"

# ── Step 7: Generate dataset (dry run first) ────────────────────────────
echo "[7/7] Dataset generation dry run..."
mkdir -p "${OUTPUT_DIR}/metadata" "${OUTPUT_DIR}/shards" "${OUTPUT_DIR}/logs"
python3 "${TOOLS_DIR}/orchestrator/orchestrator.py" \
    --config configs/dataset.yaml \
    --mode generate --dry-run 2>&1 | tail -10

echo ""
echo "============================================"
echo "Auto Deploy & Run Complete!"
echo ""
echo "If test passed, generate the full dataset:"
echo "  python3 ${TOOLS_DIR}/orchestrator/orchestrator.py \\"
echo "    --config configs/dataset.yaml \\"
echo "    --mode generate --workers 1"
echo "============================================"
