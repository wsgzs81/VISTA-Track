# Training Patches

This directory documents how to integrate the public VISTA-Track components into a tracker training codebase.

## Query-Clean Temporal Training

Use `vista_track.query_clean.QueryCleanState` or `forward_query_clean` when a model carries temporal query tokens between frames.

Rule:

- Reset state at the start of every independently sampled training clip.
- Propagate detached state only within that clip.
- Keep state across frames during inference.

This prevents hidden target history from one random batch contaminating the next batch.

## Dynamic Template Inference

Use `vista_track.dynamic_template.DynamicTemplateMemory` to keep the initial template and add only high-confidence updates.

Recommended default:

- `score_threshold=0.55`
- `update_interval=5`
- `max_dynamic=2`
- always retain the initial template

## Real-Data Mixture

The current first-stage recipe uses GOT-10k, LaSOT, and TrackingNet with a `4:2:3` sampling ratio. The helper in `vista_track.data_mixture` exposes the public recipe and normalization utilities.

## TrackingNet Zip Loading

`vista_track.trackingnet_zip.TrackingNetZipReader` supports both extracted frames and per-video zip archives. This avoids requiring a full TrackingNet extraction before training.
