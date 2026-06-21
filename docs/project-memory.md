# Project Memory

## Project Name

**VISTA-Track**

Full name: **View-Invariant Spatio-Temporal Association for Multi-View Single Object Tracking**

## Research Positioning

This is our own multi-view single object tracking project. MITracker is only a baseline/reference implementation. The project should not be presented as a MITracker fork.

## Current High-Level Strategy

The first-stage single-view tracker is the foundation. If it is unstable, later multi-view fusion will amplify errors instead of fixing them.

Current first-stage strategy:

- Use DINOv2 rather than DINOv3 for now, because earlier experiments showed DINOv3 did not improve this codebase/task.
- Train on real tracking data: GOT-10k, TrackingNet, and LaSOT.
- Avoid synthetic data that reduces real-world generalization.
- Use query-clean temporal training to prevent target-history leakage across unrelated random batches.
- Use confidence-gated dynamic templates at inference to adapt appearance while limiting drift.

## Current Server Context

Main server:

- Host: `root@10.22.17.66`
- Port: `31102`
- Work root: `/workspace/home/zhangboqiang`

Active legacy training workspace:

- `/workspace/home/zhangboqiang/MITracker`

This directory is still needed while current experiments are running. Do not delete it until checkpoints, scripts, and results are migrated.

## Current Active Experiment

Experiment name:

- `stable_real_qclean`

Purpose:

- Continue from `stable_real_warmup` epoch 20.
- Train on GOT-10k + TrackingNet + LaSOT.
- Include query-clean training changes.
- Prepare stronger first-stage tracker for later multi-view fusion.

Important implementation changes already tested:

- PyTorch 2.5 compatibility patches.
- DDP `LOCAL_RANK` compatibility.
- TrackingNet zip-frame loader.
- Query-clean Stage1 temporal state handling.
- Multi-view query assignment fix.
- Confidence-gated dynamic template inference.

## Data Policy

GitHub must not contain:

- datasets
- checkpoints
- pretrained weights
- tensorboard logs
- training outputs
- private credentials

Only code, configs, scripts, and documentation should be committed.

## Research Taste

Avoid shallow engineering-only tricks. The project should emphasize:

- new network structure
- new training paradigm
- strong and fair baselines
- reproducible experiments
- clear evidence that synthetic or generated data improves real-world tracking

## Near-Term Decision Rule

Evaluate checkpoints at epochs 5, 10, and 15 before deciding whether to continue full training. If the model plateaus or degrades, analyze data mix and template/memory behavior before adding more complexity.
