# VISTA-Track

**View-Invariant Spatio-Temporal Association for Multi-View Single Object Tracking**

VISTA-Track is an open research codebase for robust multi-view single object tracking. The project focuses on strong single-view tracking, confidence-aware temporal memory, geometry-light cross-view association, and synthetic data generation that is useful for real-world tracking.

## What This Repository Contains

- `vista_track/`: VISTA-Track training components for query-clean temporal learning, confidence-gated dynamic templates, data-mixture recipes, and evaluation helpers.
- `training/`: current training configs, launch scripts, patches, and implementation notes.
- `data_generation/`: public UE5 and lightweight synthetic dataset-generation pipelines.
- `docs/`: project memory, roadmap, and reproducibility notes.
- `scripts/`: repository maintenance helpers.

Large generated data, checkpoints, pretrained weights, logs, downloaded assets, local credentials, and local environment files are intentionally excluded.

## Current Best Direction

The current best first-stage recipe is a real-data tracker with query-clean temporal learning:

- Train first on real single-object tracking data.
- Continue with a GOT-10k, TrackingNet, and LaSOT mixture.
- Reset temporal query state across unrelated random training batches.
- Keep temporal state inside a sampled video clip and during inference.
- Use confidence-gated dynamic templates during inference to adapt appearance while limiting drift.

The current best data-generation direction is a UE5 multi-view generator with synchronized cameras, target motion, realistic materials, occluders, and offline quality checks.

## Public Release Scope

This repository contains VISTA-Track code, configs, and data-generation tools. It does not vendor third-party datasets, generated frames, checkpoints, pretrained weights, or third-party baseline source trees.

## Public Code Policy

Do not commit:

- datasets
- generated frames or annotations
- downloaded assets
- checkpoints or pretrained weights
- experiment logs
- server addresses or local machine paths
- local credentials
- local virtual environments
