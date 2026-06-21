# Roadmap

## Phase 1: Strong Single-View Foundation

Goal: build a stable tracker that is competitive before multi-view fusion.

Tasks:

- Finish `stable_real_qclean` training.
- Evaluate epoch 5, 10, and 15 checkpoints.
- Compare against official MITracker weights and clean baselines.
- Track AO, AUC, success, precision, and normalized precision where supported.
- Diagnose failure cases: occlusion, scale change, distractors, color shift, fast motion.

Expected outcome:

- A reliable first-stage checkpoint that can serve as the single-view expert for VISTA-Track.

## Phase 2: View-Invariant Association

Goal: associate the same target across views without requiring strict calibration.

Candidate directions:

- Appearance-token matching with DINO-style features.
- Temporal consistency constraints across views.
- Confidence-weighted cross-view evidence aggregation.
- Dynamic memory shared across views.
- Pseudo-correspondence mining from synchronized videos.

Expected outcome:

- A new cross-view association module that can be trained and ablated independently.

## Phase 3: Multi-View Fusion Tracker

Goal: fuse per-view evidence into a robust target estimate.

Candidate directions:

- Cross-view attention over target tokens.
- Reliability-aware view weighting.
- Occlusion-aware view dropout training.
- Geometry-free fusion first, optional weak-calibration fusion later.

Expected outcome:

- VISTA-Track multi-view model with measurable gains over single-view and MITracker-style baselines.

## Phase 4: Data Engine

Goal: produce or curate data that improves real-world generalization.

Candidate directions:

- High-quality UE5 scenes with realistic materials, lighting, occlusion, and multi-camera synchronization.
- Real video mining with pseudo labels from strong single-view trackers.
- World-model generated data only if it passes realism and tracking-label quality checks.
- Domain randomization that preserves target identity across views.

Expected outcome:

- A training dataset that improves validation/test metrics instead of only looking visually impressive.

## Phase 5: Paper-Ready Evaluation

Goal: prepare a credible AAAI/CVPR-style experimental package.

Tasks:

- Reproduce MITracker baseline metrics.
- Add modern tracker baselines.
- Run ablations for each proposed component.
- Report gains on real test data, not only synthetic data.
- Prepare qualitative visualizations with failure/success cases.

Expected outcome:

- A defensible SOTA claim or a clear gap analysis for the next iteration.
