#!/usr/bin/env python3
"""
check_visibility.py — Multi-view visibility analysis and sequence discard rules.

Per-frame stats:
  - visible_view_count
  - invisible_view_count
  - recoverable_frame_count (at least one other view sees target)

Discard rules:
  1. Only 1 view ever visible throughout entire sequence → DISCARD
  2. No occlusion / OV / BC challenge at all → DISCARD
  3. Target always too small (all views, all frames bbox_area < min_area) → DISCARD
  4. Bbox overflow: majority of frames have bbox out of bounds → DISCARD
  5. Calibration projection error: majority of sampled frames fail → DISCARD
"""

import json
import os
import shutil
import sys

import numpy as np


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_visibility(seq_dir):
    """vis[view][frame] from visibility.txt."""
    path = os.path.join(seq_dir, "visibility.txt")
    if not os.path.isfile(path):
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append([float(v) for v in line.split(",")])
    if not rows:
        return []
    n_f, n_v = len(rows), len(rows[0])
    return [[rows[fi][vi] for fi in range(n_f)] for vi in range(n_v)]


def load_lines(path):
    if not os.path.isfile(path):
        return []
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


def parse_gt_line(line):
    return [int(v) for v in line.strip().split(",")]


# ---------------------------------------------------------------------------
# Per-sequence analysis
# ---------------------------------------------------------------------------

def analyze_sequence(seq_dir, min_bbox_area=100, discard_dir=None, verbose=False):
    """Analyze one sequence and optionally discard it.

    Args:
        seq_dir: sequence directory
        min_bbox_area: minimum bbox area to consider "not too small"
        discard_dir: if set, move bad sequences here
        verbose: print details

    Returns:
        dict with stats and discard decision.
    """
    seq_base = os.path.basename(seq_dir)

    with open(os.path.join(seq_dir, "attributes.json")) as f:
        attrs = json.load(f)
    n_frames = attrs["num_frames"]
    n_views = attrs.get("num_views", 0)

    # Infer n_views from subdirs if metadata doesn't have it
    if n_views == 0:
        subdirs = [
            d for d in os.listdir(seq_dir)
            if os.path.isdir(os.path.join(seq_dir, d)) and d.startswith(seq_base + "-")
        ]
        n_views = len(subdirs)

    if n_views == 0:
        return {"discard": True, "reason": "no_views_found", "stats": {}}

    visibility = load_visibility(seq_dir)
    main_challenge = attrs.get("main_challenge", "unknown")
    resolution = attrs.get("resolution", [640, 480])
    res_w, res_h = resolution

    # Per-frame stats
    vis_threshold = 0.05
    frame_stats = []
    for fi in range(n_frames):
        visible_count = 0
        invisible_count = 0
        for vi in range(n_views):
            vis = 1.0
            if vi < len(visibility) and fi < len(visibility[vi]):
                vis = visibility[vi][fi]
            if vis >= vis_threshold:
                visible_count += 1
            else:
                invisible_count += 1

        recoverable = visible_count >= 2  # at least one OTHER view sees it
        frame_stats.append({
            "frame": fi,
            "visible_views": visible_count,
            "invisible_views": invisible_count,
            "recoverable": recoverable,
        })

    # Aggregate stats
    total_visible_views = sum(fs["visible_views"] for fs in frame_stats)
    min_visible = min(fs["visible_views"] for fs in frame_stats)
    max_invisible = max(fs["invisible_views"] for fs in frame_stats)
    frames_with_0_visible = sum(1 for fs in frame_stats if fs["visible_views"] == 0)
    frames_recoverable = sum(1 for fs in frame_stats if fs["recoverable"])
    frames_all_but_1_invisible = sum(
        1 for fs in frame_stats if fs["visible_views"] <= 1
    )

    # --- Discard rule 1: only 1 view ever visible ---
    max_views_ever_visible = max(fs["visible_views"] for fs in frame_stats)
    discard = False
    reason = None

    if max_views_ever_visible <= 1:
        discard = True
        reason = "single_view_only"

    # --- Discard rule 2: no challenges at all ---
    if not discard:
        challenge_attrs = {"POC", "FOC", "OV", "BC", "MB", "DEF", "LR", "ARC", "SV"}
        has_challenge = main_challenge in challenge_attrs
        if not has_challenge:
            # Check frame_attributes for any challenge
            frame_attrs = attrs.get("frame_attributes", {})
            for vk, fattrs in frame_attrs.items():
                for fa in fattrs:
                    if any(fa.get(c, 0) for c in challenge_attrs):
                        has_challenge = True
                        break
                if has_challenge:
                    break
        if not has_challenge:
            discard = True
            reason = "no_challenges"

    # --- Discard rule 3: target always too small ---
    if not discard:
        small_count = 0
        total_checked = 0
        sample_frames = list(range(0, n_frames, max(1, n_frames // 10)))
        for vi in range(n_views):
            view_dir = os.path.join(seq_dir, f"{seq_base}-{vi + 1}")
            gt_lines = load_lines(os.path.join(view_dir, "groundtruth.txt"))
            for fi in sample_frames:
                total_checked += 1
                if fi < len(gt_lines):
                    bbox = parse_gt_line(gt_lines[fi])
                    area = bbox[2] * bbox[3]
                    if area < min_bbox_area:
                        small_count += 1
                else:
                    small_count += 1
        if total_checked > 0 and small_count / total_checked > 0.9:
            discard = True
            reason = "target_always_too_small"

    # --- Discard rule 4: bbox overflow ---
    if not discard:
        overflow_count = 0
        total_checked = 0
        sample_frames = list(range(0, n_frames, max(1, n_frames // 10)))
        for vi in range(n_views):
            view_dir = os.path.join(seq_dir, f"{seq_base}-{vi + 1}")
            gt_lines = load_lines(os.path.join(view_dir, "groundtruth.txt"))
            for fi in sample_frames:
                total_checked += 1
                if fi < len(gt_lines):
                    bbox = parse_gt_line(gt_lines[fi])
                    gx, gy, gw, gh = bbox
                    if gw > 0 and gh > 0:
                        if gx < -gw // 2 or gy < -gh // 2 or \
                           gx + gw > res_w + gw // 2 or gy + gh > res_h + gh // 2:
                            overflow_count += 1
        if total_checked > 0 and overflow_count / total_checked > 0.5:
            discard = True
            reason = "bbox_overflow_majority"

    # --- Discard rule 5: calib projection errors (quick check) ---
    if not discard:
        try:
            from check_calibration import check_sequence as calib_check
            calib_result = calib_check(seq_dir, max_reproj_px=80.0)
            proj_errs = calib_result["issues"].get("projection_error", 0)
            sample_count = max(1, len(sample_frames)) * n_views
            if proj_errs > sample_count * 0.5:
                discard = True
                reason = "calibration_projection_errors"
        except Exception:
            pass  # Skip if calibration checker unavailable

    stats = {
        "n_frames": n_frames,
        "n_views": n_views,
        "main_challenge": main_challenge,
        "max_views_ever_visible": max_views_ever_visible,
        "min_visible_views": min_visible,
        "max_invisible_views": max_invisible,
        "frames_with_zero_visible": frames_with_0_visible,
        "frames_recoverable": frames_recoverable,
        "recoverable_ratio": round(frames_recoverable / max(n_frames, 1), 4),
        "frames_only_1_visible": frames_all_but_1_invisible,
    }

    result = {
        "discard": discard,
        "reason": reason,
        "stats": stats,
    }

    # Execute discard
    if discard and discard_dir:
        os.makedirs(discard_dir, exist_ok=True)
        dst = os.path.join(discard_dir, seq_base)
        if not os.path.exists(dst):
            shutil.move(seq_dir, dst)
            if verbose:
                print(f"  DISCARDED {seq_base}: {reason} → {discard_dir}/")
        result["moved_to"] = dst

    return result


# ---------------------------------------------------------------------------
# Batch / CLI
# ---------------------------------------------------------------------------

def check_batch(output_dir, seq_ids=None, min_bbox_area=100,
                discard=False, verbose=False):
    if seq_ids is None:
        seq_ids = sorted([
            d for d in os.listdir(output_dir)
            if os.path.isdir(os.path.join(output_dir, d)) and d.startswith("seq_")
        ])

    discard_dir = os.path.join(output_dir, "_discarded") if discard else None
    results = {}
    discard_counts = {}

    for seq_id in seq_ids:
        seq_dir = os.path.join(output_dir, seq_id)
        if not os.path.isfile(os.path.join(seq_dir, "attributes.json")):
            continue
        try:
            r = analyze_sequence(
                seq_dir, min_bbox_area=min_bbox_area,
                discard_dir=discard_dir, verbose=verbose,
            )
        except Exception as e:
            print(f"  ERROR {seq_id}: {e}")
            continue

        results[seq_id] = r
        if r["discard"]:
            reason = r["reason"]
            discard_counts[reason] = discard_counts.get(reason, 0) + 1

    n_total = len(results)
    n_discard = sum(1 for r in results.values() if r["discard"])

    print(f"\n=== visibility check: {n_total} sequences ===")
    print(f"  PASS: {n_total - n_discard}")
    print(f"  DISCARD: {n_discard}")
    for reason, cnt in sorted(discard_counts.items(), key=lambda x: -x[1]):
        print(f"    {reason}: {cnt}")

    if discard and n_discard > 0:
        remaining = [
            d for d in os.listdir(output_dir)
            if os.path.isdir(os.path.join(output_dir, d)) and d.startswith("seq_")
        ]
        print(f"  Remaining sequences: {len(remaining)}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Check visibility and discard bad sequences")
    parser.add_argument("--seq-dir", default=None)
    parser.add_argument("--output-dir", default="output/SynMVTrack")
    parser.add_argument("--min-bbox-area", type=int, default=100)
    parser.add_argument("--discard", action="store_true",
                        help="Move bad sequences to _discarded/")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.seq_dir:
        r = analyze_sequence(args.seq_dir, min_bbox_area=args.min_bbox_area, verbose=True)
        print(f"Discard: {r['discard']}  Reason: {r['reason']}")
        for k, v in r["stats"].items():
            print(f"  {k}: {v}")
    else:
        check_batch(args.output_dir, min_bbox_area=args.min_bbox_area,
                    discard=args.discard, verbose=args.verbose)
