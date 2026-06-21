# SynMVTrack v7.3 Generated Dataset Recovery

Date: 2026-06-14

This directory is a clean recovery package for the best available generated-data line:

- Main candidate: `v7_3_true_mv_photometric_pilot20`
- Original remote config path: `/root/datasets/SynMVTrack/configs/dataset_true_mv_v7_3_photometric_pilot20.yaml`
- Original remote output path: `/root/datasets/SynMVTrack/output/SynMVTrack_v7_3_true_mv_photometric_pilot20_20260606`
- Local patch source: `F:\datasets\SynMVTrack_v7_2_patch`

## What Was Recovered

- Base SynMVTrack code skeleton from `F:\datasets\SynMVTrack`
- v7.3 `generator/generate_dataset.py`
- v7.3 `generator/render_sequence.py`
- render-quality gate: `generator/audit_render_quality.py`
- Blender 3.0 compatibility patch for the Principled BSDF emission socket
- Optional Blender 3.0 camera compatibility switch: `SYNMVTRACK_CAMERA_TRACK_QUAT=1`
- v7.2/v7.3 configs:
  - `configs/dataset_true_mv_v7_2_realism_pilot.yaml`
  - `configs/dataset_true_mv_v7_3_photometric_pilot.yaml`
  - `configs/dataset_true_mv_v7_3_photometric_pilot20.yaml`
- visual references in `previews/`

## Known Result History

The project notes say:

- v7.2 true-MV geometry passed the geometry direction, but failed the photometric gate.
- v7.3 fixed the photometric gate enough for controlled pilot use:
  - audit issues: `0`
  - target cross-view color mean: `43.05`
  - max color distance: `70.50`
  - bbox bad ratio: `0.000`
  - mask leak bad ratio: `0.000`
- v7.3 pilot20 later filtered to 14 accepted sequences with audit issues `0`.
- Stage2 training on that filtered14 recipe collapsed on weak GMTD:
  - AUC/AO `6.9727`
  - OP50 `4.3796`
  - conclusion: do not scale that training recipe blindly.

This means v7.3 is the best recovered generated-data script line, but it was not promoted as the training route for the final best tracker.

## Missing Piece

The original `assets/index/*.json` and the 300+ target / 100+ scene asset distribution are not present in the current local workspace or in `D:\beifen\mv3dpt-datasets`.

The old backup at `D:\beifen\mv3dpt-datasets` is useful, but it is not the original SynMVTrack asset pack. It contains real multi-view data, including DexYCB RGB/depth/mask/calibration/tracks and `first_frame_mesh.obj` files. Use it as a seed/fallback or as a source for the next improved data line, not as proof that the original v7.3 assets were recovered.

## First Recovery Pass

From this directory:

```bash
python scripts/build_dexycb_seed_assets.py \
  --mv3dpt-root /path/to/mv3dpt-datasets/datasets \
  --output-assets assets
```

On the current Windows machine:

```powershell
python scripts\build_dexycb_seed_assets.py `
  --mv3dpt-root "D:\beifen\mv3dpt-datasets\datasets" `
  --output-assets assets
```

Then run a metadata-only smoke test first:

```bash
python generator/generate_dataset.py \
  --config configs/dataset_true_mv_v7_3_photometric_pilot.yaml \
  --output output/v7_3_smoke_metadata \
  --step metadata \
  --verbose
```

If metadata passes and Blender is available:

```bash
bash scripts/run_v7_3_pilot20.sh
```

The pilot20 gate should keep these important thresholds:

- `gate_color_max_rgb_dist: 72.0`
- `gate_bbox_min_iou: 0.58`
- `gate_bbox_max_area_ratio: 1.9`
- `gate_min_rendered_partial_occ: 0.20`
- `gate_max_rendered_all_blocked: 0.18`
- `min_azimuth_separation_deg: 55`
- `inside_room_margin_m: 0.35`

## Recommended Optimization Path

Use three stages rather than immediately scaling synthetic data:

1. Reproduce v7.3 pilot20 with a small accepted set.
2. Run the render audit and weak-GMTD sanity test before any full training.
3. Build a new `v7_4` line using `D:\beifen\mv3dpt-datasets` as real multi-view priors:
   - sample camera/FOV/baseline statistics from DexYCB/Panoptic calibration files
   - use real multi-view backgrounds or color/depth statistics to reduce gray-room bias
   - keep geometry-aware occlusion and reject cross-view identity drift
   - add depth-based occlusion checks where depth is available

Promotion rule:

- weak GMTD must improve before full GMTD evaluation
- full GMTD must improve over the v12/v13 baselines before a generated-data claim is made
