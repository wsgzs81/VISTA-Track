#!/usr/bin/env python3
"""
write_mvtrack_format.py - Write final MVTrack-style directory structure.

Rearranges the pipeline output into the standard MVTrack layout:

    seq_XXXXXX/
      seq_XXXXXX-N/          # per-view subdirectory
        img/00001.jpg
        masks/00000001.png
        groundtruth.txt
        invisible.txt
        visibility.txt
        attributes.json
        full_projected_bbox.txt
      BEV/
      calibs.json
      meta.json

Also generates train_split.txt / val_split.txt / test_split.txt.
"""

import json
import math
import os
import random
import shutil

import numpy as np


# ---------------------------------------------------------------------------
# Per-view file assembly
# ---------------------------------------------------------------------------

def _copy_images(src_img_dir, dst_img_dir, n_frames):
    """Copy/rename images from {fi:06d}.jpg to the tracker-format {fi:05d}.jpg."""
    os.makedirs(dst_img_dir, exist_ok=True)
    for fi in range(n_frames):
        src = os.path.join(src_img_dir, f"{fi:06d}.jpg")
        dst = os.path.join(dst_img_dir, f"{fi + 1:05d}.jpg")
        if os.path.isfile(src):
            shutil.copy2(src, dst)


def _copy_masks(src_mask_dir, dst_mask_dir, n_frames):
    """Copy/rename masks from {fi:06d}.png to {fi:08d}.png."""
    os.makedirs(dst_mask_dir, exist_ok=True)
    for fi in range(n_frames):
        src = os.path.join(src_mask_dir, f"{fi:06d}.png")
        dst = os.path.join(dst_mask_dir, f"{fi + 1:08d}.png")
        if os.path.isfile(src):
            shutil.copy2(src, dst)


def _load_lines(path):
    """Load non-empty lines from a text file."""
    if not os.path.isfile(path):
        return []
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def _write_groundtruth(src_lines, dst_path, n_frames):
    """Write groundtruth.txt with per-frame bbox."""
    with open(dst_path, "w") as f:
        for fi in range(n_frames):
            if fi < len(src_lines):
                f.write(src_lines[fi] + "\n")
            else:
                f.write("0,0,0,0\n")


def _write_invisible(src_lines, dst_path, n_frames):
    """Write invisible.txt as one CSV row for tracker-format compatibility."""
    vals = []
    for fi in range(n_frames):
        vals.append(src_lines[fi] if fi < len(src_lines) else "0")
    with open(dst_path, "w") as f:
        f.write(",".join(vals) + "\n")


def _write_visibility(visibility_data, dst_path, view_idx, n_frames):
    """Write per-view visibility.txt as one CSV row."""
    vals = []
    for fi in range(n_frames):
        if view_idx < len(visibility_data) and fi < len(visibility_data[view_idx]):
            vals.append(f"{visibility_data[view_idx][fi]:.4f}")
        else:
            vals.append("1.0000")
    with open(dst_path, "w") as f:
        f.write(",".join(vals) + "\n")


def _write_view_attributes(frame_attrs, dst_path, n_frames):
    """Write per-view attributes.json.

    Format:
        {"BC": [0,0,1,...], "MB": [0,0,0,...], ...}
    """
    attr_names = ["BC", "MB", "POC", "FOC", "OV", "DEF", "LR", "ARC", "SV"]
    result = {name: [] for name in attr_names}

    for fi in range(n_frames):
        if fi < len(frame_attrs):
            fa = frame_attrs[fi]
            for name in attr_names:
                result[name].append(fa.get(name, 0))
        else:
            for name in attr_names:
                result[name].append(0)

    with open(dst_path, "w") as f:
        json.dump(result, f, indent=2)


def _write_full_bbox(src_lines, dst_path, n_frames):
    """Write full_projected_bbox.txt for one view."""
    with open(dst_path, "w") as f:
        for fi in range(n_frames):
            if fi < len(src_lines):
                f.write(src_lines[fi] + "\n")
            else:
                f.write("-1,-1,-1,-1\n")


# ---------------------------------------------------------------------------
# BEV data — MVTrack 400×400 grid (8m × 8m, 2cm cells)
# ---------------------------------------------------------------------------

BEV_RANGE_M = 4.0        # ±4m per axis
BEV_GRID_SIZE = 400      # 400 × 400
BEV_CELL_M = 0.02        # 2cm per cell
BEV_SCALE = 1000.0       # m → mm for target_3d.txt


def world_to_bev_grid(wx, wz):
    """Convert world (x, z) to BEV grid indices [gx, gy].

    Grid covers x ∈ [-4, 4], z ∈ [-4, 4].
    gx = int((x + 4) / 0.02),  gy = int((z + 4) / 0.02).
    Values clamped to [0, 399].
    """
    gx = int((wx + BEV_RANGE_M) / BEV_CELL_M)
    gy = int((wz + BEV_RANGE_M) / BEV_CELL_M)
    return max(0, min(gx, BEV_GRID_SIZE - 1)), max(0, min(gy, BEV_GRID_SIZE - 1))


def _write_bev(seq_dir, trajectory, scene):
    """Write BEV/ with target_3d.txt (mm), target_bev.txt (grid), xyz_index.txt."""
    bev_dir = os.path.join(seq_dir, "BEV")
    os.makedirs(bev_dir, exist_ok=True)

    positions = trajectory.get("positions_m", [])

    # target_3d.txt: x,y,z in mm (world units)
    with open(os.path.join(bev_dir, "target_3d.txt"), "w") as f:
        for pos in positions:
            f.write(f"{pos[0] * BEV_SCALE:.1f},"
                    f"{pos[1] * BEV_SCALE:.1f},"
                    f"{pos[2] * BEV_SCALE:.1f}\n")

    # target_bev.txt: grid_x,grid_y  (400×400 integer indices)
    with open(os.path.join(bev_dir, "target_bev.txt"), "w") as f:
        for pos in positions:
            gx, gy = world_to_bev_grid(pos[0], pos[2])
            f.write(f"{gx},{gy}\n")

    # xyz_index.txt: one line per frame — world x,y,z (mm) + bev gx,gy
    with open(os.path.join(bev_dir, "xyz_index.txt"), "w") as f:
        f.write(f"# {len(positions)} frames\n")
        f.write("# grid: 400x400, cell: 0.02m, range: +/-4m\n")
        f.write("# format: x_mm,y_mm,z_mm,bev_gx,bev_gy\n")
        for pos in positions:
            gx, gy = world_to_bev_grid(pos[0], pos[2])
            f.write(f"{pos[0] * BEV_SCALE:.1f},"
                    f"{pos[1] * BEV_SCALE:.1f},"
                    f"{pos[2] * BEV_SCALE:.1f},"
                    f"{gx},{gy}\n")


# ---------------------------------------------------------------------------
# Calibs and meta
# ---------------------------------------------------------------------------

def _write_calibs(seq_dir, calibs):
    """Write calibs.json to sequence root."""
    with open(os.path.join(seq_dir, "calibs.json"), "w") as f:
        json.dump(calibs, f, indent=2)


def _write_meta(seq_dir, seq_id, n_views, cfg, target, scene, trajectory):
    """Write meta.json with sequence metadata."""
    main_challenge = trajectory.get("main_challenge", "POC")

    # Determine recoverable occlusion: at least one view has POC/FOC
    # but not all views fully occluded simultaneously
    recoverable = main_challenge in ("POC", "FOC", "OV")

    meta = {
        "sequence_id": seq_id,
        "num_views": n_views,
        "fps": cfg.get("dataset", {}).get("fps", 30),
        "target_asset_id": target.get("target_id", "unknown"),
        "target_category": target.get("category", "misc"),
        "scene_id": scene.get("scene_id", "unknown"),
        "main_challenge": main_challenge,
        "camera_layout": "wide_baseline_room",
        "recoverable_occlusion": recoverable,
    }

    with open(os.path.join(seq_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


# ---------------------------------------------------------------------------
# Split files
# ---------------------------------------------------------------------------

def generate_splits(output_dir, train_ratio=0.7, val_ratio=0.15, seed=42,
                    split_cfg=None):
    """Generate train/val/test split files.

    Args:
        output_dir: root output directory
        train_ratio: fraction for training (used if split_cfg is None)
        val_ratio: fraction for validation (used if split_cfg is None)
        seed: random seed for reproducibility
        split_cfg: optional dict from config with exact counts and bias:
            {"train": 350, "val": 50, "test": 100,
             "test_challenge_bias": ["FOC", "OV", "BC"]}
    """
    seq_ids = sorted([
        d for d in os.listdir(output_dir)
        if os.path.isdir(os.path.join(output_dir, d)) and d.startswith("seq_")
    ])

    rng = random.Random(seed)

    if split_cfg and isinstance(split_cfg, dict) and "train" in split_cfg:
        # V1: exact counts with optional test challenge bias
        n_train = split_cfg.get("train", 350)
        n_val = split_cfg.get("val", 50)
        n_test = split_cfg.get("test", 100)
        test_bias = split_cfg.get("test_challenge_bias", [])

        # Load challenge info for bias-aware splitting
        seq_challenges = {}
        for sid in seq_ids:
            meta_path = os.path.join(output_dir, sid, "meta.json")
            if os.path.isfile(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                seq_challenges[sid] = meta.get("main_challenge", "unknown")
            else:
                attrs_path = os.path.join(output_dir, sid, "attributes.json")
                if os.path.isfile(attrs_path):
                    with open(attrs_path) as f:
                        attrs = json.load(f)
                    seq_challenges[sid] = attrs.get("main_challenge", "unknown")
                else:
                    seq_challenges[sid] = "unknown"

        if test_bias:
            # Separate biased and neutral sequences
            biased = [s for s in seq_ids if seq_challenges.get(s, "") in test_bias]
            neutral = [s for s in seq_ids if s not in set(biased)]

            rng.shuffle(biased)
            rng.shuffle(neutral)

            # Fill test with biased first, then neutral
            test_ids = biased[:n_test]
            if len(test_ids) < n_test:
                test_ids += neutral[:n_test - len(test_ids)]
                neutral = neutral[n_test - len(test_ids):]
            else:
                # Remaining biased go to neutral pool
                neutral += biased[n_test:]
                rng.shuffle(neutral)

            # Train + val from remaining
            train_ids = neutral[:n_train]
            val_ids = neutral[n_train:n_train + n_val]
        else:
            rng.shuffle(seq_ids)
            train_ids = seq_ids[:n_train]
            val_ids = seq_ids[n_train:n_train + n_val]
            test_ids = seq_ids[n_train + n_val:n_train + n_val + n_test]
    else:
        # V0: ratio-based
        rng.shuffle(seq_ids)
        n = len(seq_ids)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_ids = seq_ids[:n_train]
        val_ids = seq_ids[n_train:n_train + n_val]
        test_ids = seq_ids[n_train + n_val:]

    def _view_entries(sid):
        seq_dir = os.path.join(output_dir, sid)
        if not os.path.isdir(seq_dir):
            return []
        entries = sorted(
            d for d in os.listdir(seq_dir)
            if os.path.isdir(os.path.join(seq_dir, d)) and d.startswith(sid + "-")
        )
        return entries or [sid]

    for split_name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        entries = []
        for sid in ids:
            entries.extend(_view_entries(sid))
        path = os.path.join(output_dir, f"{split_name}_split.txt")
        with open(path, "w") as f:
            for entry in entries:
                f.write(entry + "\n")
        print(f"  {split_name}_split.txt: {len(entries)} views from {len(ids)} sequences")

    # Print test challenge distribution
    if split_cfg and split_cfg.get("test_challenge_bias"):
        test_challenges = {}
        for sid in test_ids:
            ch = seq_challenges.get(sid, "unknown")
            test_challenges[ch] = test_challenges.get(ch, 0) + 1
        print(f"  Test challenge distribution: {test_challenges}")

    return train_ids, val_ids, test_ids


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def write_mvtrack_format(seq_dir, cfg=None):
    """Write MVTrack-style directory structure from pipeline output.

    Reads the pipeline output in seq_dir (masks/, visibility.txt, etc.)
    and creates the per-view subdirectory structure.

    Args:
        seq_dir: sequence directory (e.g., output/SynMVTrack/seq_0000)
        cfg: dataset config dict

    Returns:
        seq_dir path
    """
    # Load metadata
    with open(os.path.join(seq_dir, "attributes.json")) as f:
        attributes = json.load(f)
    with open(os.path.join(seq_dir, "calibs.json")) as f:
        calibs = json.load(f)

    n_frames = attributes.get("num_frames", 0)
    n_views = attributes.get("num_views", len(calibs))

    # Override n_frames from trajectory if available
    render_meta_path = os.path.join(seq_dir, "render_meta.json")
    if os.path.isfile(render_meta_path):
        with open(render_meta_path) as f:
            render_meta = json.load(f)
        trajectory = render_meta.get("trajectory", {})
        positions = trajectory.get("positions_m", [])
        if positions:
            n_frames = len(positions)
        scene = render_meta.get("scene", {})
        target = render_meta.get("target", {})
    else:
        trajectory = {}
        scene = {}
        target = {}

    # Load visibility data (cross-view: one file for all views)
    vis_path = os.path.join(seq_dir, "visibility.txt")
    if os.path.isfile(vis_path):
        vis_rows = []
        with open(vis_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    vis_rows.append([float(v) for v in line.split(",")])
        # Transpose: vis[view][frame]
        visibility = [
            [vis_rows[fi][vi] for fi in range(len(vis_rows))]
            for vi in range(n_views)
        ] if vis_rows else [[] for _ in range(n_views)]
    else:
        visibility = [[] for _ in range(n_views)]

    # Load full projected bboxes
    full_bbox_path = os.path.join(seq_dir, "full_projected_bbox.txt")
    if os.path.isfile(full_bbox_path):
        full_bbox_lines = _load_lines(full_bbox_path)
    else:
        full_bbox_lines = []

    # Parse full bbox per view
    full_bbox_per_view = [[] for _ in range(n_views)]
    for line in full_bbox_lines:
        parts = line.split("|")
        for vi in range(n_views):
            if vi < len(parts):
                full_bbox_per_view[vi].append(parts[vi])
            else:
                full_bbox_per_view[vi].append("-1,-1,-1,-1")

    # Load frame attributes from attributes.json
    frame_attrs = attributes.get("frame_attributes", {})

    # Temporary staging dir for rearranged files
    staging_dir = seq_dir + "_staging"

    # Create per-view subdirectories
    for vi in range(n_views):
        seq_base = os.path.basename(seq_dir)
        view_dir = os.path.join(seq_dir, f"{seq_base}-{vi + 1}")

        # Create view subdirectory
        os.makedirs(view_dir, exist_ok=True)

        # Copy images: img/{vi:04d}/{fi:06d}.jpg -> seq_XXXXX-N/img/{fi:08d}.jpg
        src_img = os.path.join(seq_dir, "img", f"{vi:04d}")
        dst_img = os.path.join(view_dir, "img")
        _copy_images(src_img, dst_img, n_frames)

        # Copy masks
        src_mask = os.path.join(seq_dir, "masks", f"{vi:04d}")
        dst_mask = os.path.join(view_dir, "masks")
        _copy_masks(src_mask, dst_mask, n_frames)

        # groundtruth.txt
        gt_src_path = os.path.join(seq_dir, f"groundtruth_v{vi:04d}.txt")
        gt_lines = _load_lines(gt_src_path)
        _write_groundtruth(gt_lines, os.path.join(view_dir, "groundtruth.txt"), n_frames)

        # invisible.txt
        inv_src_path = os.path.join(seq_dir, f"invisible_v{vi:04d}.txt")
        inv_lines = _load_lines(inv_src_path)
        _write_invisible(inv_lines, os.path.join(view_dir, "invisible.txt"), n_frames)

        # visibility.txt
        _write_visibility(visibility, os.path.join(view_dir, "visibility.txt"), vi, n_frames)

        # full_projected_bbox.txt
        _write_full_bbox(
            full_bbox_per_view[vi],
            os.path.join(view_dir, "full_projected_bbox.txt"),
            n_frames,
        )

        # attributes.json (per-view)
        view_key = f"view_{vi}"
        view_fattrs = frame_attrs.get(view_key, [])
        _write_view_attributes(view_fattrs, os.path.join(view_dir, "attributes.json"), n_frames)

    # BEV directory
    _write_bev(seq_dir, trajectory, scene)

    # calibs.json (already in root, keep it)
    _write_calibs(seq_dir, calibs)

    # meta.json
    _write_meta(seq_dir, os.path.basename(seq_dir), n_views, cfg or {},
                target, scene, trajectory)

    # Clean up old per-view files from root (groundtruth_v*.txt, invisible_v*.txt)
    for vi in range(n_views):
        for prefix in ["groundtruth_v", "invisible_v"]:
            old = os.path.join(seq_dir, f"{prefix}{vi:04d}.txt")
            if os.path.isfile(old):
                os.remove(old)

    print(f"  MVTrack format written: {seq_dir} ({n_views} views x {n_frames} frames)")
    return seq_dir


def write_batch(output_dir, cfg=None, seq_ids=None):
    """Write MVTrack format for multiple sequences and generate splits."""
    if seq_ids is None:
        seq_ids = sorted([
            d for d in os.listdir(output_dir)
            if os.path.isdir(os.path.join(output_dir, d)) and d.startswith("seq_")
        ])

    for seq_id in seq_ids:
        seq_dir = os.path.join(output_dir, seq_id)
        if not os.path.isdir(seq_dir):
            continue
        if not os.path.isfile(os.path.join(seq_dir, "calibs.json")):
            print(f"Skipping {seq_id}: no calibs.json")
            continue
        try:
            write_mvtrack_format(seq_dir, cfg=cfg)
        except Exception as e:
            print(f"Error writing {seq_id}: {e}")
            import traceback
            traceback.print_exc()

    # Generate splits
    print("\nGenerating splits...")
    generate_splits(output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Write MVTrack format")
    parser.add_argument("--seq-dir", default=None, help="Single sequence directory")
    parser.add_argument("--output-dir", default="output/SynMVTrack",
                        help="Batch output directory")
    args = parser.parse_args()

    if args.seq_dir:
        write_mvtrack_format(args.seq_dir)
    else:
        write_batch(args.output_dir)
