#!/usr/bin/env python3
"""
compute_dataset_stats.py — Dataset-wide statistics for SynMVTrack.

Reports:
  - Total sequences / views / frames
  - 9-class attribute distribution (per-sequence and per-frame)
  - Bbox size distribution (area, aspect ratio)
  - Visibility distribution (mean, median, % invisible)
  - Challenge type distribution
  - Camera layout stats (azimuth spread, elevation range)
  - Sequence quality summary (from visibility check)
"""

import json
import math
import os
import sys
from collections import Counter, defaultdict

import numpy as np


# ---------------------------------------------------------------------------
# Single sequence stats
# ---------------------------------------------------------------------------

def sequence_stats(seq_dir):
    """Compute stats for one sequence."""
    seq_base = os.path.basename(seq_dir)

    with open(os.path.join(seq_dir, "attributes.json")) as f:
        attrs = json.load(f)
    n_frames = attrs["num_frames"]
    n_views = attrs.get("num_views", 0)

    if n_views == 0:
        subdirs = [
            d for d in os.listdir(seq_dir)
            if os.path.isdir(os.path.join(seq_dir, d)) and d.startswith(seq_base + "-")
        ]
        n_views = len(subdirs)

    main_challenge = attrs.get("main_challenge", "unknown")

    # Load calibs for camera stats
    cam_azimuths = []
    try:
        with open(os.path.join(seq_dir, "calibs.json")) as f:
            calibs = json.load(f)
        for ck in sorted(calibs.keys()):
            cam = calibs[ck]
            T = np.array(cam["T"], dtype=np.float64)
            # Rough azimuth from T vector
            az = math.degrees(math.atan2(T[0], T[2])) if abs(T[2]) > 0.01 else 0
            cam_azimuths.append(az)
    except Exception:
        pass

    # Per-view bbox stats
    bbox_areas = []
    bbox_ars = []  # aspect ratios
    vis_ratios = []
    inv_count = 0
    total_checked = 0

    for vi in range(n_views):
        view_dir = os.path.join(seq_dir, f"{seq_base}-{vi + 1}")
        if not os.path.isdir(view_dir):
            continue

        gt_path = os.path.join(view_dir, "groundtruth.txt")
        inv_path = os.path.join(view_dir, "invisible.txt")
        vis_path = os.path.join(view_dir, "visibility.txt")

        gt_lines = []
        if os.path.isfile(gt_path):
            with open(gt_path) as f:
                gt_lines = [l.strip() for l in f if l.strip()]

        inv_lines = []
        if os.path.isfile(inv_path):
            with open(inv_path) as f:
                inv_lines = [l.strip() for l in f if l.strip()]

        view_vis = []
        if os.path.isfile(vis_path):
            with open(vis_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            view_vis.append(float(line))
                        except ValueError:
                            pass

        for fi in range(n_frames):
            total_checked += 1

            # Bbox
            if fi < len(gt_lines):
                parts = [int(v) for v in gt_lines[fi].split(",")]
                x, y, w, h = parts
                area = w * h
                if area > 0:
                    bbox_areas.append(area)
                    bbox_ars.append(w / max(h, 1))

            # Invisible
            if fi < len(inv_lines):
                if int(inv_lines[fi]) == 1:
                    inv_count += 1

            # Visibility
            if fi < len(view_vis):
                vis_ratios.append(view_vis[fi])

    # Per-frame attributes
    attr_counts = Counter()
    frame_attrs = attrs.get("frame_attributes", {})
    attr_names = ["POC", "FOC", "OV", "BC", "MB", "DEF", "LR", "ARC", "SV"]
    for vk, fattrs in frame_attrs.items():
        for fa in fattrs:
            for an in attr_names:
                if fa.get(an, 0) == 1:
                    attr_counts[an] += 1

    return {
        "n_frames": n_frames,
        "n_views": n_views,
        "main_challenge": main_challenge,
        "bbox_areas": bbox_areas,
        "bbox_ars": bbox_ars,
        "vis_ratios": vis_ratios,
        "invisible_count": inv_count,
        "total_checked": total_checked,
        "attr_counts": dict(attr_counts),
        "cam_azimuths": cam_azimuths,
    }


# ---------------------------------------------------------------------------
# Batch stats
# ---------------------------------------------------------------------------

def compute_stats(output_dir, seq_ids=None):
    if seq_ids is None:
        seq_ids = sorted([
            d for d in os.listdir(output_dir)
            if os.path.isdir(os.path.join(output_dir, d)) and d.startswith("seq_")
        ])

    all_stats = []
    challenge_dist = Counter()

    for seq_id in seq_ids:
        seq_dir = os.path.join(output_dir, seq_id)
        if not os.path.isfile(os.path.join(seq_dir, "attributes.json")):
            continue
        try:
            s = sequence_stats(seq_dir)
            all_stats.append((seq_id, s))
            challenge_dist[s["main_challenge"]] += 1
        except Exception as e:
            print(f"  ERROR {seq_id}: {e}")

    n_seqs = len(all_stats)
    if n_seqs == 0:
        print("No sequences found.")
        return

    # Aggregate
    total_views = sum(s["n_views"] for _, s in all_stats)
    total_frames = sum(s["n_frames"] for _, s in all_stats)
    all_areas = []
    all_ars = []
    all_vis = []
    total_invisible = 0
    total_checked = 0
    attr_totals = Counter()
    azimuth_spreads = []

    for _, s in all_stats:
        all_areas.extend(s["bbox_areas"])
        all_ars.extend(s["bbox_ars"])
        all_vis.extend(s["vis_ratios"])
        total_invisible += s["invisible_count"]
        total_checked += s["total_checked"]
        for k, v in s["attr_counts"].items():
            attr_totals[k] += v

        az = s["cam_azimuths"]
        if len(az) >= 2:
            az_sorted = sorted(az)
            gaps = [
                (az_sorted[i + 1] - az_sorted[i]) for i in range(len(az_sorted) - 1)
            ]
            gaps.append(360 - az_sorted[-1] + az_sorted[0])
            azimuth_spreads.append(360 - max(gaps))

    # --- Report ---
    print("=" * 60)
    print("  SynMVTrack Dataset Statistics")
    print("=" * 60)

    print(f"\nSequences: {n_seqs}")
    print(f"Total views: {total_views}")
    print(f"Total frames: {total_frames}")
    if n_seqs > 0:
        print(f"Avg views/seq: {total_views / n_seqs:.1f}")
        print(f"Avg frames/seq: {total_frames / n_seqs:.0f}")

    print(f"\n--- Challenge Distribution ---")
    for ch, cnt in sorted(challenge_dist.items(), key=lambda x: -x[1]):
        print(f"  {ch:6s}: {cnt:4d} ({cnt / n_seqs * 100:.0f}%)")

    print(f"\n--- Attribute Distribution (per-frame) ---")
    attr_names = ["POC", "FOC", "OV", "BC", "MB", "DEF", "LR", "ARC", "SV"]
    total_label_frames = max(total_checked, 1)
    for an in attr_names:
        cnt = attr_totals.get(an, 0)
        print(f"  {an:4s}: {cnt:6d} frames ({cnt / total_label_frames * 100:.1f}%)")

    print(f"\n--- Bbox Statistics ---")
    if all_areas:
        areas = np.array(all_areas)
        print(f"  Area (px²):")
        print(f"    mean={areas.mean():.0f}  median={np.median(areas):.0f}  "
              f"min={areas.min():.0f}  max={areas.max():.0f}  "
              f"std={areas.std():.0f}")
        print(f"    p5={np.percentile(areas, 5):.0f}  "
              f"p25={np.percentile(areas, 25):.0f}  "
              f"p75={np.percentile(areas, 75):.0f}  "
              f"p95={np.percentile(areas, 95):.0f}")
    if all_ars:
        ars = np.array(all_ars)
        print(f"  Aspect ratio (w/h):")
        print(f"    mean={ars.mean():.2f}  median={np.median(ars):.2f}  "
              f"min={ars.min():.2f}  max={ars.max():.2f}")

    print(f"\n--- Visibility Statistics ---")
    if all_vis:
        vis = np.array(all_vis)
        invisible_pct = (vis < 0.05).sum() / len(vis) * 100
        print(f"  Ratio: mean={vis.mean():.3f}  median={np.median(vis):.3f}  "
              f"min={vis.min():.3f}  max={vis.max():.3f}")
        print(f"  Invisible (<0.05): {invisible_pct:.1f}%")
    print(f"  Total invisible labels: {total_invisible}/{total_checked} "
          f"({total_invisible / max(total_checked, 1) * 100:.1f}%)")

    print(f"\n--- Camera Layout ---")
    if azimuth_spreads:
        spreads = np.array(azimuth_spreads)
        print(f"  Azimuth spread (deg): mean={spreads.mean():.0f}  "
              f"min={spreads.min():.0f}  max={spreads.max():.0f}")

    return {
        "n_sequences": n_seqs,
        "n_views": total_views,
        "n_frames": total_frames,
        "challenge_dist": dict(challenge_dist),
        "attribute_dist": dict(attr_totals),
        "bbox_area_mean": float(np.mean(all_areas)) if all_areas else 0,
        "bbox_area_median": float(np.median(all_areas)) if all_areas else 0,
        "invisible_ratio": total_invisible / max(total_checked, 1),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute dataset statistics")
    parser.add_argument("--seq-dir", default=None)
    parser.add_argument("--output-dir", default="output/SynMVTrack")
    parser.add_argument("--json-out", default=None, help="Save stats as JSON")
    args = parser.parse_args()

    if args.seq_dir:
        s = sequence_stats(args.seq_dir)
        for k, v in s.items():
            if k not in ("bbox_areas", "bbox_ars", "vis_ratios", "cam_azimuths"):
                print(f"  {k}: {v}")
    else:
        result = compute_stats(args.output_dir)
        if args.json_out and result:
            with open(args.json_out, "w") as f:
                json.dump(result, f, indent=2)
            print(f"\nStats saved to {args.json_out}")
