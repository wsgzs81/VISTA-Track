# Training

This directory contains the current public training recipe for VISTA-Track.

## Current Best Recipe

The active best recipe is `stage1_real_qclean`:

- real-data first-stage tracking
- DINOv2-style backbone
- GOT-10k + TrackingNet + LaSOT mixture
- query-clean temporal state handling
- confidence-gated dynamic template inference

## Files

- `configs/stage1_real_qclean.yaml`: current best public training config.
- `scripts/train_stage1_real_qclean.sh`: launch script for two-GPU training.
- `scripts/start_stage1_real_qclean_after_warmup20.sh`: safe continuation script after a warmup checkpoint.
- `notes/query_clean_temporal_learning.md`: implementation design notes.
- `patches/`: integration notes for the public `vista_track` modules.

## Public Components

The reusable VISTA-Track training components live in `vista_track/`:

- `query_clean.py`: reset/update policy for temporal query tokens.
- `dynamic_template.py`: confidence-gated dynamic template memory.
- `data_mixture.py`: current GOT-10k/LaSOT/TrackingNet mixture recipe.
- `trackingnet_zip.py`: TrackingNet frame reader for extracted frames or zip archives.
- `metrics.py`: AO and success-AUC helpers for ablations.

The public repository does not vendor third-party tracker source trees. Integrate these modules into your own tracker implementation or experiment workspace.

## Data

Datasets are not included. Configure the following paths in your local training workspace:

- GOT-10k
- TrackingNet
- LaSOT
- pretrained visual backbone weights

## Checkpoints

Checkpoints are not committed to Git. Store them in `checkpoints/` or another ignored directory.
