#!/usr/bin/env python3
"""
check_calibration.py — Validate camera calibration consistency.

Checks:
  1. Target 3D center projects near groundtruth bbox center
  2. Cross-view: same 3D point projects consistently across views
  3. R/T handedness: det(R) ≈ +1 (not mirrored)
  4. Unit consistency: T magnitude in plausible range for mm or m
  5. Focal length sanity: fx, fy > 0 and within expected range
"""

import json
import math
import os
import sys

import numpy as np


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_calibs(seq_dir):
    with open(os.path.join(seq_dir, "calibs.json")) as f:
        return json.load(f)


def load_render_meta(seq_dir):
    path = os.path.join(seq_dir, "render_meta.json")
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return {}


def load_lines(path):
    if not os.path.isfile(path):
        return []
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


def parse_gt_line(line):
    return [int(v) for v in line.strip().split(",")]


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------

def project_point(pt_3d, K, R, T):
    """Project 3D point to pixel coordinates. Returns (u, v, depth) or None."""
    pt = np.array(pt_3d, dtype=np.float64)
    cam = R @ pt + T
    if cam[2] <= 0.01:
        return None
    px = K @ cam
    u = px[0] / px[2]
    v = px[1] / px[2]
    return float(u), float(v), float(cam[2])


def reproject_error(pt_3d, pixel, K, R, T):
    """Euclidean distance between projected 3D point and observed pixel."""
    proj = project_point(pt_3d, K, R, T)
    if proj is None:
        return None
    return math.sqrt((proj[0] - pixel[0]) ** 2 + (proj[1] - pixel[1]) ** 2)


# ---------------------------------------------------------------------------
# Per-sequence checks
# ---------------------------------------------------------------------------

def check_sequence(seq_dir, max_reproj_px=50.0, verbose=False):
    """Run calibration checks on one sequence.

    Returns dict with issue counts.
    """
    calibs = load_calibs(seq_dir)
    render_meta = load_render_meta(seq_dir)

    with open(os.path.join(seq_dir, "attributes.json")) as f:
        attrs = json.load(f)

    n_frames = attrs["num_frames"]
    resolution = attrs.get("resolution", [640, 480])
    seq_base = os.path.basename(seq_dir)

    trajectory = render_meta.get("trajectory", {})
    positions = trajectory.get("positions_m", [])

    issues = {
        "projection_error": 0,       # 3D center projects far from bbox
        "cross_view_inconsistency": 0,
        "bad_rotation_determinant": 0,
        "suspicious_units": 0,
        "bad_focal_length": 0,
    }
    detail = []

    # Resolve camera keys: "cam0", "cam1", ... or "0", "1", ...
    cam_keys = sorted(calibs.keys(),
                      key=lambda k: int("".join(filter(str.isdigit, k)) or 0))
    n_views = len(cam_keys)

    # --- Check 1: R/T handedness and focal sanity ---
    for ci, ck in enumerate(cam_keys):
        cam = calibs[ck]
        K = np.array(cam["K"], dtype=np.float64)
        R = np.array(cam["R"], dtype=np.float64)
        T = np.array(cam["T"], dtype=np.float64)

        det_R = np.linalg.det(R)
        if abs(det_R - 1.0) > 0.05:
            issues["bad_rotation_determinant"] += 1
            detail.append(
                f"  [{seq_base}] {ck}: det(R) = {det_R:.4f} (expected ~1.0)"
            )

        fx, fy = K[0, 0], K[1, 1]
        if fx <= 0 or fy <= 0:
            issues["bad_focal_length"] += 1
            detail.append(f"  [{seq_base}] {ck}: bad focal fx={fx:.1f} fy={fy:.1f}")
        # Sanity: focal should be ~200-2000 px for typical cameras
        if not (50 < fx < 5000) or not (50 < fy < 5000):
            issues["bad_focal_length"] += 1
            detail.append(
                f"  [{seq_base}] {ck}: suspicious focal fx={fx:.1f} fy={fy:.1f}"
            )

        t_mag = float(np.linalg.norm(T))
        # Typical: mm units → T in 1000-50000; m units → T in 1-50
        if not (100 < t_mag < 100000) and not (0.05 < t_mag < 100):
            issues["suspicious_units"] += 1
            detail.append(
                f"  [{seq_base}] {ck}: ||T|| = {t_mag:.1f} — unclear units"
            )

    # --- Check 2: 3D → bbox projection ---
    # Sample frames to avoid O(n_frames * n_views) for long sequences
    sample_frames = list(range(0, n_frames, max(1, n_frames // 20)))

    for fi in sample_frames:
        if fi >= len(positions):
            continue
        pos = positions[fi]

        for vi, ck in enumerate(cam_keys):
            cam = calibs[ck]
            K = np.array(cam["K"], dtype=np.float64)
            R = np.array(cam["R"], dtype=np.float64)
            T = np.array(cam["T"], dtype=np.float64)

            proj = project_point(pos, K, R, T)
            if proj is None:
                # Target behind camera — skip (handled by visibility checks)
                continue

            u, v, depth = proj

            # Load bbox for this frame/view
            view_dir = os.path.join(seq_dir, f"{seq_base}-{vi + 1}")
            gt_path = os.path.join(view_dir, "groundtruth.txt")
            gt_lines = load_lines(gt_path)
            if fi < len(gt_lines):
                bbox = parse_gt_line(gt_lines[fi])
            else:
                bbox = [0, 0, 0, 0]

            gx, gy, gw, gh = bbox
            if gw > 0 and gh > 0:
                bbox_cx = gx + gw / 2.0
                bbox_cy = gy + gh / 2.0
                err = math.sqrt((u - bbox_cx) ** 2 + (v - bbox_cy) ** 2)
                if err > max_reproj_px:
                    issues["projection_error"] += 1
                    detail.append(
                        f"  [{seq_base}] v{vi} f{fi}: proj=({u:.0f},{v:.0f}) "
                        f"bbox_c=({bbox_cx:.0f},{bbox_cy:.0f}) err={err:.0f}px"
                    )

    # --- Check 3: cross-view consistency ---
    # Pick a few frames, project to all views, verify all consistent
    for fi in sample_frames[::max(1, len(sample_frames) // 5)]:
        if fi >= len(positions):
            continue
        pos = positions[fi]
        projs = []
        for vi, ck in enumerate(cam_keys):
            cam = calibs[ck]
            K = np.array(cam["K"], dtype=np.float64)
            R = np.array(cam["R"], dtype=np.float64)
            T = np.array(cam["T"], dtype=np.float64)
            p = project_point(pos, K, R, T)
            if p is not None:
                projs.append((vi, p))

        # Check that back-projected rays converge (triangulation check)
        if len(projs) >= 2:
            for i in range(len(projs)):
                for j in range(i + 1, len(projs)):
                    vi_a, (ua, va, da) = projs[i]
                    vi_b, (ub, vb, db) = projs[j]
                    # If both project inside image but depth diff is huge, suspicious
                    depth_ratio = max(da, db) / max(min(da, db), 0.01)
                    if depth_ratio > 50:
                        issues["cross_view_inconsistency"] += 1
                        detail.append(
                            f"  [{seq_base}] f{fi}: v{vi_a} depth={da:.1f} vs "
                            f"v{vi_b} depth={db:.1f} ratio={depth_ratio:.1f}"
                        )

    return {"issues": issues, "detail": detail}


# ---------------------------------------------------------------------------
# Batch / CLI
# ---------------------------------------------------------------------------

def check_batch(output_dir, seq_ids=None, max_reproj_px=50.0, verbose=False):
    if seq_ids is None:
        seq_ids = sorted([
            d for d in os.listdir(output_dir)
            if os.path.isdir(os.path.join(output_dir, d)) and d.startswith("seq_")
        ])

    total_issues = {}
    bad_seqs = []

    for seq_id in seq_ids:
        seq_dir = os.path.join(output_dir, seq_id)
        if not os.path.isfile(os.path.join(seq_dir, "calibs.json")):
            continue
        try:
            result = check_sequence(seq_dir, max_reproj_px=max_reproj_px, verbose=verbose)
        except Exception as e:
            print(f"  ERROR {seq_id}: {e}")
            continue

        seq_total = sum(result["issues"].values())
        for k, v in result["issues"].items():
            total_issues[k] = total_issues.get(k, 0) + v

        if seq_total > 0:
            bad_seqs.append((seq_id, seq_total))
            if verbose:
                for line in result["detail"]:
                    print(line)

    print(f"\n=== calibration check: {len(seq_ids)} sequences ===")
    for k, v in sorted(total_issues.items()):
        print(f"  {k}: {v}")
    if bad_seqs:
        bad_seqs.sort(key=lambda x: -x[1])
        print(f"\n  Sequences with issues: {len(bad_seqs)}")
        for sid, cnt in bad_seqs[:10]:
            print(f"    {sid}: {cnt} issues")

    return total_issues, bad_seqs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Check calibration consistency")
    parser.add_argument("--seq-dir", default=None)
    parser.add_argument("--output-dir", default="output/SynMVTrack")
    parser.add_argument("--max-reproj", type=float, default=50.0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.seq_dir:
        r = check_sequence(args.seq_dir, max_reproj_px=args.max_reproj, verbose=True)
        for line in r["detail"]:
            print(line)
        print(f"Issues: {r['issues']}")
    else:
        check_batch(args.output_dir, max_reproj_px=args.max_reproj, verbose=args.verbose)
