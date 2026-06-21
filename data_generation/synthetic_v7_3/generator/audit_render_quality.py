#!/usr/bin/env python3
"""Audit rendered SynMVTrack sequences for view consistency and mask quality."""
import argparse
import json
import math
from pathlib import Path

import numpy as np


def _load_image(path):
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"))


def _load_mask(path):
    from PIL import Image

    return np.asarray(Image.open(path).convert("L")) > 127


def _sample_indices(n_frames, frames_per_seq):
    if n_frames <= 0:
        return []
    if n_frames <= frames_per_seq:
        return list(range(n_frames))
    return sorted(set(int(round(x)) for x in np.linspace(0, n_frames - 1, frames_per_seq)))


def _masked_mean_rgb(img, mask, min_pixels):
    if mask.sum() < min_pixels:
        return None
    pixels = img[mask]
    if pixels.size == 0:
        return None
    return pixels.mean(axis=0)


def _pairwise_max_dist(vectors):
    if len(vectors) < 2:
        return 0.0
    max_dist = 0.0
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            max_dist = max(max_dist, float(np.linalg.norm(vectors[i] - vectors[j])))
    return max_dist


def _mask_bbox(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return (x1, y1, x2 - x1 + 1, y2 - y1 + 1)


def _bbox_area(box):
    return max(float(box[2]), 0.0) * max(float(box[3]), 0.0)


def _bbox_iou(a, b):
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(ix2 - ix1, 0.0), max(iy2 - iy1, 0.0)
    inter = iw * ih
    union = _bbox_area(a) + _bbox_area(b) - inter
    return inter / union if union > 0 else 0.0


def _load_bbox_rows(path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(errors="ignore").splitlines():
        view_boxes = []
        for item in line.strip().split("|"):
            try:
                vals = [float(x) for x in item.split(",")]
            except ValueError:
                vals = []
            view_boxes.append(tuple(vals[:4]) if len(vals) >= 4 else None)
        rows.append(view_boxes)
    return rows


def audit_sequence(seq_dir, frames_per_seq, min_pixels, color_max_rgb_dist, bbox_min_iou, bbox_max_area_ratio):
    attrs_path = seq_dir / "attributes.json"
    if not attrs_path.exists():
        return {"sequence": seq_dir.name, "issues": ["missing_attributes"], "warnings": []}
    attrs = json.loads(attrs_path.read_text())
    n_frames = int(attrs.get("num_frames", 0))
    n_views = int(attrs.get("num_views", 0))
    issues = []
    warnings = []
    sampled = _sample_indices(n_frames, frames_per_seq)
    color_dists = []
    partial_occ = 0
    all_blocked = 0
    checked_frames = 0
    missing_rgb = 0
    missing_mask = 0
    bbox_rows = _load_bbox_rows(seq_dir / "full_projected_bbox.txt")
    bbox_checked = 0
    bbox_bad = 0
    bbox_worst_iou = 1.0
    bbox_worst_area_ratio = 1.0

    for fi in sampled:
        colors = []
        visible_any = False
        blocked_views = 0
        valid_views = 0
        for vi in range(n_views):
            img_path = seq_dir / "img" / f"{vi:04d}" / f"{fi:06d}.jpg"
            mask_path = seq_dir / "masks" / f"{vi:04d}" / f"{fi:06d}.png"
            full_path = seq_dir / "full_masks" / f"{vi:04d}" / f"{fi:06d}.png"
            if not img_path.exists():
                missing_rgb += 1
                continue
            if not mask_path.exists() or not full_path.exists():
                missing_mask += 1
                continue

            img = _load_image(img_path)
            mask = _load_mask(mask_path)
            full = _load_mask(full_path)
            if full.sum() < min_pixels:
                continue
            valid_views += 1

            full_bbox = _mask_bbox(full)
            if bbox_rows and fi < len(bbox_rows) and vi < len(bbox_rows[fi]) and full_bbox is not None:
                anno_bbox = bbox_rows[fi][vi]
                if anno_bbox is not None and _bbox_area(anno_bbox) > 0:
                    iou = _bbox_iou(anno_bbox, full_bbox)
                    area_ratio = _bbox_area(anno_bbox) / max(_bbox_area(full_bbox), 1.0)
                    bbox_checked += 1
                    bbox_worst_iou = min(bbox_worst_iou, iou)
                    bbox_worst_area_ratio = max(bbox_worst_area_ratio, area_ratio)
                    if iou < bbox_min_iou or area_ratio > bbox_max_area_ratio or area_ratio < 1.0 / bbox_max_area_ratio:
                        bbox_bad += 1

            vis_ratio = float(mask.sum()) / float(max(full.sum(), 1))
            if vis_ratio < 0.85:
                blocked_views += 1
            if vis_ratio > 0.05:
                visible_any = True
                color = _masked_mean_rgb(img, mask, min_pixels)
                if color is not None:
                    colors.append(color)

        if valid_views:
            checked_frames += 1
            if 0 < blocked_views < valid_views:
                partial_occ += 1
            if blocked_views == valid_views:
                all_blocked += 1
        if visible_any and len(colors) >= 2:
            color_dists.append(_pairwise_max_dist(colors))

    if missing_rgb:
        issues.append(f"missing_rgb:{missing_rgb}")
    if missing_mask:
        issues.append(f"missing_mask:{missing_mask}")
    if sampled and checked_frames == 0:
        issues.append("no_valid_rendered_frames")
    if bbox_rows and bbox_checked == 0:
        issues.append("no_valid_bbox_checks")
    if bbox_checked:
        bad_ratio = bbox_bad / max(bbox_checked, 1)
        if bad_ratio > 0.10:
            issues.append(f"loose_or_misaligned_bbox:{bad_ratio:.3f}")
        elif bad_ratio > 0.02:
            warnings.append(f"some_loose_or_misaligned_bbox:{bad_ratio:.3f}")

    max_color_dist = max(color_dists) if color_dists else 0.0
    mean_color_dist = float(np.mean(color_dists)) if color_dists else 0.0
    if max_color_dist > color_max_rgb_dist:
        issues.append(f"cross_view_target_color_drift:{max_color_dist:.1f}")
    elif mean_color_dist > color_max_rgb_dist * 0.65:
        warnings.append(f"high_mean_target_color_drift:{mean_color_dist:.1f}")

    partial_ratio = partial_occ / max(checked_frames, 1)
    all_blocked_ratio = all_blocked / max(checked_frames, 1)
    if checked_frames and partial_ratio < 0.10:
        warnings.append(f"low_rendered_partial_occ:{partial_ratio:.3f}")
    if checked_frames and all_blocked_ratio > 0.35:
        issues.append(f"too_many_rendered_all_blocked:{all_blocked_ratio:.3f}")

    return {
        "sequence": seq_dir.name,
        "frames": n_frames,
        "views": n_views,
        "checked_frames": checked_frames,
        "partial_ratio": partial_ratio,
        "all_blocked_ratio": all_blocked_ratio,
        "mean_color_dist": mean_color_dist,
        "max_color_dist": max_color_dist,
        "bbox_checked": bbox_checked,
        "bbox_bad_ratio": bbox_bad / max(bbox_checked, 1),
        "bbox_worst_iou": bbox_worst_iou if bbox_checked else None,
        "bbox_worst_area_ratio": bbox_worst_area_ratio if bbox_checked else None,
        "issues": issues,
        "warnings": warnings,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--max-sequences", type=int, default=40)
    parser.add_argument("--frames-per-seq", type=int, default=12)
    parser.add_argument("--min-mask-pixels", type=int, default=40)
    parser.add_argument("--color-max-rgb-dist", type=float, default=90.0)
    parser.add_argument("--bbox-min-iou", type=float, default=0.45)
    parser.add_argument("--bbox-max-area-ratio", type=float, default=2.5)
    parser.add_argument("--show-bad", type=int, default=20)
    args = parser.parse_args()

    seq_dirs = sorted(p for p in args.dataset.glob("seq_*") if p.is_dir())
    if args.max_sequences > 0:
        seq_dirs = seq_dirs[: args.max_sequences]

    rendered = list(args.dataset.glob("seq_*/img/*/*.jpg"))
    if not rendered:
        print(f"dataset={args.dataset}")
        print("status=PENDING rendered_jpg=0")
        return 2

    rows = [
        audit_sequence(
            seq_dir,
            frames_per_seq=args.frames_per_seq,
            min_pixels=args.min_mask_pixels,
            color_max_rgb_dist=args.color_max_rgb_dist,
            bbox_min_iou=args.bbox_min_iou,
            bbox_max_area_ratio=args.bbox_max_area_ratio,
        )
        for seq_dir in seq_dirs
    ]
    issue_rows = [row for row in rows if row["issues"]]
    warn_rows = [row for row in rows if row["warnings"]]
    partials = [row["partial_ratio"] for row in rows if row.get("checked_frames")]
    color_means = [row["mean_color_dist"] for row in rows if row.get("checked_frames")]

    print(f"dataset={args.dataset}")
    print(f"rendered_jpg={len(rendered)} checked_sequences={len(rows)} issues={len(issue_rows)} warnings={len(warn_rows)}")
    if partials:
        print(f"rendered_partial_occ_mean={np.mean(partials):.3f} min={np.min(partials):.3f}")
    if color_means:
        print(f"target_cross_view_color_dist_mean={np.mean(color_means):.2f} max={max(row['max_color_dist'] for row in rows):.2f}")
    bbox_rows = [row for row in rows if row.get("bbox_checked")]
    if bbox_rows:
        print(
            "bbox_tightness "
            f"bad_ratio_mean={np.mean([row['bbox_bad_ratio'] for row in bbox_rows]):.3f} "
            f"worst_iou_min={min(row['bbox_worst_iou'] for row in bbox_rows):.3f} "
            f"worst_area_ratio_max={max(row['bbox_worst_area_ratio'] for row in bbox_rows):.2f}"
        )
    if issue_rows:
        print("bad_sequences:")
        for row in issue_rows[: args.show_bad]:
            print(
                f"{row['sequence']} frames={row.get('frames', 0)} views={row.get('views', 0)} "
                f"checked={row.get('checked_frames', 0)} partial={row.get('partial_ratio', 0.0):.3f} "
                f"all_blocked={row.get('all_blocked_ratio', 0.0):.3f} "
                f"color_mean={row.get('mean_color_dist', 0.0):.1f} color_max={row.get('max_color_dist', 0.0):.1f} "
                f"bbox_bad={row.get('bbox_bad_ratio', 0.0):.3f} "
                f"issues={','.join(row['issues'])} warnings={','.join(row['warnings'])}"
            )
    if warn_rows:
        print("warning_sequences:")
        for row in warn_rows[: args.show_bad]:
            print(
                f"{row['sequence']} partial={row['partial_ratio']:.3f} "
                f"color_mean={row['mean_color_dist']:.1f} warnings={','.join(row['warnings'])}"
            )

    return 1 if issue_rows else 0


if __name__ == "__main__":
    raise SystemExit(main())
