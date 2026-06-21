#!/usr/bin/env python3
"""
check_bbox_mask.py — Validate bbox / mask / visibility consistency.

Checks per (sequence, view, frame):
  1. bbox tight around mask: IoU(mask_bbox, gt_bbox) > 0
  2. bbox within image bounds
  3. mask area > 0 when bbox is non-zero
  4. invisible=0  ⇒  bbox must be valid (non-zero)
  5. invisible=1  ⇒  visibility < 0.05
  6. invisible=0  ⇒  visibility >= 0.05
"""

import json
import os
import sys

import numpy as np


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_mask(path):
    """Load grayscale mask → bool array (True = target)."""
    try:
        from PIL import Image
        return np.array(Image.open(path).convert("L")) > 128
    except Exception:
        return None


def mask_bbox(mask):
    """Tight bbox from bool mask → [x, y, w, h] or [0,0,0,0]."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()),
            int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]


def parse_gt_line(line):
    """Parse 'x,y,w,h' → list[int]."""
    return [int(v) for v in line.strip().split(",")]


def load_lines(path):
    if not os.path.isfile(path):
        return []
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


def load_scalar_series(path):
    """Load either one-value-per-line or one CSV row annotation files."""
    lines = load_lines(path)
    if len(lines) == 1 and "," in lines[0]:
        return [v.strip() for v in lines[0].split(",") if v.strip()]
    return lines


def load_visibility(seq_dir):
    """Load visibility.txt → vis[view][frame]."""
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
    n_frames, n_views = len(rows), len(rows[0])
    return [[rows[fi][vi] for fi in range(n_frames)] for vi in range(n_views)]


# ---------------------------------------------------------------------------
# Core check
# ---------------------------------------------------------------------------

def check_sequence(seq_dir, verbose=False):
    """Run all bbox/mask checks on one sequence.

    Returns dict with issue counts and per-frame detail.
    """
    # Load metadata
    with open(os.path.join(seq_dir, "attributes.json")) as f:
        attrs = json.load(f)
    n_frames = attrs["num_frames"]
    n_views = attrs.get("num_views", 0)

    # Count views from subdirectories if not in metadata
    if n_views == 0:
        n_views = sum(
            1 for d in os.listdir(seq_dir)
            if os.path.isdir(os.path.join(seq_dir, d)) and "-" in d
        )

    visibility = load_visibility(seq_dir)

    issues = {
        "bbox_mask_mismatch": 0,
        "bbox_out_of_bounds": 0,
        "empty_mask": 0,
        "invisible_but_valid_bbox": 0,
        "visible_but_invalid_bbox": 0,
        "invisible_vis_mismatch": 0,
        "visible_vis_mismatch": 0,
    }
    detail = []
    resolution = attrs.get("resolution", [640, 480])
    res_w, res_h = resolution

    seq_base = os.path.basename(seq_dir)

    for vi in range(n_views):
        view_dir = os.path.join(seq_dir, f"{seq_base}-{vi + 1}")
        if not os.path.isdir(view_dir):
            issues["bbox_mask_mismatch"] += n_frames
            detail.append(f"  View {vi}: missing directory {view_dir}")
            continue

        gt_lines = load_lines(os.path.join(view_dir, "groundtruth.txt"))
        inv_lines = load_scalar_series(os.path.join(view_dir, "invisible.txt"))

        for fi in range(n_frames):
            # --- groundtruth bbox ---
            gt_bbox = parse_gt_line(gt_lines[fi]) if fi < len(gt_lines) else [0, 0, 0, 0]

            # --- invisible flag ---
            invisible = 0
            if fi < len(inv_lines):
                invisible = int(inv_lines[fi])

            # --- visibility ratio ---
            vis_ratio = 1.0
            if vi < len(visibility) and fi < len(visibility[vi]):
                vis_ratio = visibility[vi][fi]

            # --- mask ---
            mask_path = os.path.join(view_dir, "masks", f"{fi + 1:08d}.png")
            mask = load_mask(mask_path)
            if mask is None:
                # Try old format
                mask_path2 = os.path.join(seq_dir, "masks", f"{vi:04d}", f"{fi:06d}.png")
                mask = load_mask(mask_path2)

            m_bbox = [0, 0, 0, 0]
            m_area = 0
            if mask is not None:
                m_bbox = mask_bbox(mask)
                m_area = int(mask.sum())

            gx, gy, gw, gh = gt_bbox
            mx, my, mw, mh = m_bbox

            # --- Check 1: bbox-mask mismatch ---
            if gw > 0 and gh > 0 and mw > 0 and mh > 0:
                # IoU between gt_bbox and mask_bbox
                ix1 = max(gx, mx)
                iy1 = max(gy, my)
                ix2 = min(gx + gw, mx + mw)
                iy2 = min(gy + gh, my + mh)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                union = gw * gh + mw * mh - inter
                iou = inter / max(union, 1)
                if iou < 0.3:
                    issues["bbox_mask_mismatch"] += 1
                    detail.append(
                        f"  [{seq_base}] v{vi} f{fi}: bbox-mask IoU={iou:.3f} "
                        f"gt=[{gx},{gy},{gw},{gh}] mask=[{mx},{my},{mw},{mh}]"
                    )

            # --- Check 2: bbox out of bounds ---
            if gw > 0 and gh > 0:
                if gx < 0 or gy < 0 or gx + gw > res_w or gy + gh > res_h:
                    issues["bbox_out_of_bounds"] += 1
                    detail.append(
                        f"  [{seq_base}] v{vi} f{fi}: bbox [{gx},{gy},{gw},{gh}] "
                        f"outside {res_w}x{res_h}"
                    )

            # --- Check 3: empty mask when bbox says otherwise ---
            if gw > 0 and gh > 0 and m_area == 0 and mask is not None:
                issues["empty_mask"] += 1
                detail.append(
                    f"  [{seq_base}] v{vi} f{fi}: bbox [{gx},{gy},{gw},{gh}] "
                    f"but mask is empty"
                )

            # --- Check 4: invisible=0 but bbox is zero ---
            if invisible == 0 and (gw == 0 or gh == 0):
                issues["visible_but_invalid_bbox"] += 1
                detail.append(
                    f"  [{seq_base}] v{vi} f{fi}: invisible=0 but bbox is zero"
                )

            # --- Check 5: invisible=1 but bbox is valid ---
            if invisible == 1 and gw > 0 and gh > 0:
                issues["invisible_but_valid_bbox"] += 1
                detail.append(
                    f"  [{seq_base}] v{vi} f{fi}: invisible=1 but bbox=[{gx},{gy},{gw},{gh}]"
                )

            # --- Check 6: invisible / visibility ratio consistency ---
            if invisible == 1 and vis_ratio >= 0.05:
                issues["invisible_vis_mismatch"] += 1
                detail.append(
                    f"  [{seq_base}] v{vi} f{fi}: invisible=1 but vis={vis_ratio:.4f}"
                )
            if invisible == 0 and vis_ratio < 0.05:
                issues["visible_vis_mismatch"] += 1
                detail.append(
                    f"  [{seq_base}] v{vi} f{fi}: invisible=0 but vis={vis_ratio:.4f}"
                )

    return {"issues": issues, "detail": detail}


# ---------------------------------------------------------------------------
# Batch / CLI
# ---------------------------------------------------------------------------

def check_batch(output_dir, seq_ids=None, verbose=False):
    if seq_ids is None:
        seq_ids = sorted([
            d for d in os.listdir(output_dir)
            if os.path.isdir(os.path.join(output_dir, d)) and d.startswith("seq_")
        ])

    total_issues = {}
    bad_seqs = []

    for seq_id in seq_ids:
        seq_dir = os.path.join(output_dir, seq_id)
        if not os.path.isfile(os.path.join(seq_dir, "attributes.json")):
            continue
        try:
            result = check_sequence(seq_dir, verbose=verbose)
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

    print(f"\n=== bbox/mask check: {len(seq_ids)} sequences ===")
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

    parser = argparse.ArgumentParser(description="Check bbox/mask consistency")
    parser.add_argument("--seq-dir", default=None)
    parser.add_argument("--output-dir", default="output/SynMVTrack")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.seq_dir:
        r = check_sequence(args.seq_dir, verbose=True)
        for line in r["detail"]:
            print(line)
        print(f"Issues: {r['issues']}")
    else:
        check_batch(args.output_dir, verbose=args.verbose)
