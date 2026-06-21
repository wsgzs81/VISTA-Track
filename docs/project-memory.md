# Project Memory

## Project Name

**VISTA-Track**

Full name: **View-Invariant Spatio-Temporal Association for Multi-View Single Object Tracking**

## Research Objective

Build a publishable multi-view single object tracker with:

- a stable single-view foundation
- confidence-aware temporal memory
- view-invariant target association
- data generation that improves real-world generalization

## Current Best Training Strategy

The first-stage tracker must be strong before multi-view fusion is added.

Current recipe:

- Use a DINOv2-style visual backbone for the active real-data training recipe.
- Train on GOT-10k, TrackingNet, and LaSOT.
- Use query-clean temporal training so history tokens persist only inside a sampled sequence, not across unrelated training batches.
- Use confidence-gated dynamic template updates during inference.
- Evaluate early checkpoints before committing to long training.

## Current Best Data Strategy

Use synthetic data only when it improves validation on real data.

Current generation direction:

- UE5-based multi-view rendering.
- Synchronized cameras.
- Target motion rather than camera-only motion.
- Realistic object colors/materials.
- Distractors and occluders.
- Offline QC for bounding boxes, visibility, and cross-view consistency.

## Public Release Rules

The public repository should contain code, configs, scripts, and documentation only.

Do not commit:

- server addresses or local machine paths
- datasets or generated frames
- downloaded assets
- checkpoints or pretrained weights
- experiment logs
- local credentials or virtual environments

## Evaluation Rule

Do not claim SOTA without real evaluation.

Required checks:

- AO / AUC where supported
- success / precision / normalized precision where supported
- comparison against official or reproduced baselines
- ablation for each proposed component
- qualitative failure analysis
