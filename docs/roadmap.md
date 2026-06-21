# Roadmap

## Phase 1: Strong Single-View Foundation

Goal: build a stable first-stage tracker.

Tasks:

- Finish the current real-data query-clean training run.
- Evaluate checkpoints at epochs 5, 10, and 15.
- Compare with clean reproduced baselines.
- Record AO, AUC, success, precision, and normalized precision where available.
- Analyze failures under occlusion, distractors, fast motion, scale change, and appearance shift.

## Phase 2: View-Invariant Association

Goal: match the same target across views without relying on strict calibration.

Candidate directions:

- Cross-view target-token attention.
- Appearance and temporal consistency matching.
- Confidence-weighted view reliability.
- Dynamic memory shared across views.
- Pseudo-correspondence mining from synchronized videos.

## Phase 3: Multi-View Fusion

Goal: fuse per-view evidence into a more robust target state.

Candidate directions:

- Reliability-aware cross-view fusion.
- Occlusion-aware view dropout.
- Geometry-light association first, optional weak geometry later.
- Training losses that preserve single-view stability while improving multi-view consistency.

## Phase 4: Data Engine

Goal: generate or curate data that improves real-world tracking.

Candidate directions:

- UE5 scenes with realistic materials, lighting, occlusion, and motion.
- Multi-camera synchronized rendering with consistent target identity.
- Real video pseudo-label mining.
- Quality filters for box tightness, visibility, identity consistency, and motion diversity.

## Phase 5: Paper-Ready Package

Goal: prepare credible conference-level evidence.

Tasks:

- Reproduce public baselines.
- Add modern single-view and multi-view comparisons.
- Run ablations for query-clean training, dynamic memory, and cross-view fusion.
- Show real-data gains, not only synthetic-data improvements.
