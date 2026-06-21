# VISTA-Track

**View-Invariant Spatio-Temporal Association for Multi-View Single Object Tracking**

VISTA-Track is a research project for robust multi-view single object tracking, targeting uncalibrated or weakly calibrated real-world scenes. The goal is not to repackage MITracker, but to build a new tracker and training paradigm around stable temporal memory, cross-view target association, and high-quality real/synthetic data.

## Research Goal

The long-term goal is to push multi-view single object tracking toward SOTA-level performance and a publishable AAAI/CVPR-style contribution.

Core target:

- Single target tracking across multiple synchronized or loosely synchronized views.
- Robust tracking under occlusion, appearance change, scale change, and cross-view ambiguity.
- Uncalibrated or weak-calibrated deployment, instead of relying entirely on fixed camera parameters.
- Strong first-stage single-view tracking, followed by cross-view spatial-temporal fusion.

## Working Hypothesis

The first-stage tracker must be reliable before multi-view fusion can help. Our current direction is:

1. Build a strong single-view foundation on real tracking data.
2. Add query-clean temporal training to avoid cross-sample memory contamination.
3. Use confidence-gated dynamic templates to reduce drift.
4. Add multi-view target association and fusion as a second-stage method.
5. Generate or curate data only when it improves real-world generalization.

## Current Method Direction

VISTA-Track will explore three method components:

- **Query-clean temporal learning**: temporal query state should persist inside a sequence, not leak across unrelated training samples.
- **Confidence-gated template memory**: update dynamic templates only on high-confidence frames to balance adaptation and drift resistance.
- **View-invariant association**: align target evidence across views with appearance, geometry-free correspondence, and temporal consistency cues.

## Baselines

MITracker is treated as an external baseline and implementation reference, not as the project identity.

Planned comparisons:

- Original MITracker official weights.
- Clean first-stage baseline trained on real data.
- VISTA-Track single-view stage.
- VISTA-Track multi-view fusion stage.
- Additional modern trackers when evaluation scripts are ready.

## Repository Policy

This repository stores only code, configs, docs, and lightweight scripts. Datasets, checkpoints, pretrained weights, training outputs, and logs must stay outside Git.

## Status

Active research prototype. Current server experiments are training under a separate workspace and will be migrated into this repository once each component is clean and reproducible.
