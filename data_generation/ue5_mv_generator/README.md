# MVTrackGen â€?Multi-View Single Object Tracking Dataset Generator (UE5)

## Architecture

```
External Python Orchestrator (tools/orchestrator/orchestrator.py)
  â†?Generates job manifests from configs/dataset.yaml
  â†?Launches UE5 worker per sequence (1-10 sequences per worker)
  â†?Validates output (offline QC)
  â†?Retries failures, writes reports

UE5 C++ Runtime (Source/MVTrackRuntime/)
  â†?AMVTrackSequenceRunner: Main orchestrator actor
  â†?UMVTrackJobConfig: Parses -MVTrackJob=<path> from CLI
  â†?UMVTrackCameraManager: Multi-camera hemisphere + sync capture
  â†?UMVTrackTargetController: Spawn, physics settle, motion, occluders
  â†?UMVTrackAnnotationWriter: Per-frame annotations (bbox/pose/visibility)
```

## Visibility Guarantee

**Every sequence guarantees:**
1. Target visible in ALL 4 cameras (line-of-sight check via raycast)
2. At least 1 camera has partial/heavy occlusion (occluder placed between camera and target)
3. Target moves via physics impulse (rolling/tumbling) for motion diversity

## Quick Start

### 1. Install UE5

```bash
# Link Epic Games + GitHub first (see below)
bash setup_ue5.sh 5.4
```

### 2. Build the project

```bash
cd .
/opt/UnrealEngine/Engine/Build/BatchFiles/Linux/Build.sh \
    MVTrackGen Linux Development \
    -project=ue_project/MVTrackGen.uproject
```

### 3. Generate dataset (dry run)

```bash
python3 tools/orchestrator/orchestrator.py \
    --config configs/dataset.yaml \
    --mode generate --dry-run
```

### 4. Generate dataset (actual)

```bash
python3 tools/orchestrator/orchestrator.py \
    --config configs/dataset.yaml \
    --mode generate --workers 1
```

### 5. Validate

```bash
python3 tools/qc/offline_qc.py \
    --input output/shards \
    --cameras 4 --frames 300
```

## Epic Games + GitHub Account Linking

1. Go to https://www.unrealengine.com/ and sign in/up
2. Go to https://www.unrealengine.com/account/connected
3. Click "Connect" next to GitHub
4. Authorize on GitHub
5. Wait 5-30 minutes for repo access
6. Generate GitHub token: https://github.com/settings/tokens (scope: repo)
7. Configure on server:

```bash
git config --global credential.helper store
echo "https://YOUR_USERNAME:YOUR_TOKEN@github.com" > ~/.git-credentials
```

8. Run: `bash setup_ue5.sh 5.4`

## Output Structure

```
output/shards/shard_000000/seq_000000/
  seq_meta.json
  cameras/cam_000.json ... cam_003.json
  frames/cam_000/
    rgb/000000.png ... 000299.png
    depth/000000.exr ... 000299.exr
    mask/000000.png ... 000299.png
    ann/000000.json ... 000299.json
```

## Config

- `configs/dataset.yaml` â€?sequence count, FPS, resolution, target categories, occluders
- `configs/cameras.yaml` â€?hemisphere placement, FOV, occlusion strategy
- `configs/render.yaml` â€?Cycles/EEVEE, samples, denoising
