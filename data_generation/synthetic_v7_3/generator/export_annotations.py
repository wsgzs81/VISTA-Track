#!/usr/bin/env python3
"""
export_annotations.py - Generate bbox, invisible, and frame-level attributes
from rendered masks and visibility data.

Reads:
    {seq_dir}/masks/{view:04d}/{frame:06d}.png     visible masks
    {seq_dir}/full_masks/{view:04d}/{frame:06d}.png full projected masks
    {seq_dir}/visibility.txt                         per-frame visibility ratios
    {seq_dir}/full_projected_bbox.txt                full projected bboxes
    {seq_dir}/calibs.json                            camera calibration
    {seq_dir}/render_meta.json                       trajectory + challenge info

Writes:
    {seq_dir}/groundtruth_v{view:04d}.txt            x,y,w,h per frame
    {seq_dir}/invisible_v{view:04d}.txt              0/1 per frame
    {seq_dir}/attributes.json                        per-frame attributes per view
"""

import json
import math
import os

import numpy as np


# ---------------------------------------------------------------------------
# Mask utilities
# ---------------------------------------------------------------------------

def load_mask(mask_path):
    """Load a PNG mask as boolean array (True = target pixel)."""
    try:
        from PIL import Image
        img = Image.open(mask_path).convert('L')
        arr = np.array(img)
        return arr > 128
    except Exception:
        return None


def mask_to_bbox(mask):
    """Convert boolean mask to [x, y, w, h] tight bounding box.

    Returns [0, 0, 0, 0] for empty masks or None input.
    """
    if mask is None:
        return [0, 0, 0, 0]
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return [x1, y1, x2 - x1 + 1, y2 - y1 + 1]


def mask_to_bbox_safe(mask_path):
    """Load mask and convert to bbox. Returns [0,0,0,0] on failure."""
    mask = load_mask(mask_path)
    if mask is None:
        return [0, 0, 0, 0]
    return mask_to_bbox(mask)


# ---------------------------------------------------------------------------
# Visibility data loading
# ---------------------------------------------------------------------------

def load_visibility(seq_dir):
    """Load visibility.txt. Returns list[list[float]]: vis[view][frame]."""
    path = os.path.join(seq_dir, "visibility.txt")
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append([float(v) for v in line.split(",")])
    # Transpose: rows[frame][view] -> result[view][frame]
    if not rows:
        return []
    n_frames = len(rows)
    n_views = len(rows[0])
    result = [[rows[fi][vi] for fi in range(n_frames)] for vi in range(n_views)]
    return result


def load_full_bboxes(seq_dir):
    """Load full_projected_bbox.txt. Returns list[list]: bbox[view][frame]."""
    path = os.path.join(seq_dir, "full_projected_bbox.txt")
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            view_parts = line.split("|")
            frame_bboxes = []
            for vp in view_parts:
                nums = [int(x) for x in vp.split(",")]
                if nums[0] == -1:
                    frame_bboxes.append(None)
                else:
                    frame_bboxes.append(nums)
            rows.append(frame_bboxes)
    if not rows:
        return []
    n_frames = len(rows)
    n_views = len(rows[0])
    result = [[rows[fi][vi] for fi in range(n_frames)] for vi in range(n_views)]
    return result


# ---------------------------------------------------------------------------
# Frustum check
# ---------------------------------------------------------------------------

def target_inside_frustum(target_pos_3d, cam_data, resolution):
    """Check if a 3D point projects inside the camera image.

    Args:
        target_pos_3d: [x, y, z] in our coordinate system
        cam_data: dict with K, R, T
        resolution: [width, height]
    """
    K = np.array(cam_data["K"])
    R = np.array(cam_data["R"])  # world-to-camera
    T = np.array(cam_data["T"])

    pt = np.array(target_pos_3d)
    cam_pt = R @ pt + T

    if cam_pt[2] <= 0.1:
        return False

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u = fx * cam_pt[0] / cam_pt[2] + cx
    v = fy * cam_pt[1] / cam_pt[2] + cy

    w, h = resolution
    return 0 <= u < w and 0 <= v < h


# ---------------------------------------------------------------------------
# Per-frame attribute computation
# ---------------------------------------------------------------------------

def compute_frame_attributes(
    frame_idx, view_idx, vis_ratio, visible_bbox, full_bbox,
    target_pos_3d, cam_data, resolution, main_challenge,
    prev_full_bbox=None, ref_full_bbox=None,
):
    """Compute all frame-level attributes for one frame/view.

    Args:
        frame_idx: frame index
        view_idx: view index
        vis_ratio: visibility_ratio for this frame/view
        visible_bbox: [x,y,w,h] from visible mask, or [0,0,0,0]
        full_bbox: [x,y,w,h] from full projected mask, or None
        target_pos_3d: [x,y,z] target world position
        cam_data: camera calibration dict
        resolution: [w, h]
        main_challenge: trajectory's main challenge type
        prev_full_bbox: full_bbox from previous frame (for motion attrs)
        ref_full_bbox: reference full_bbox (median/first non-empty)

    Returns:
        dict with POC, FOC, OV, BC, MB, DEF, LR, ARC, SV, invisible
    """
    attrs = {}

    # --- Invisible ---
    attrs["invisible"] = 1 if vis_ratio < 0.05 else 0

    # --- POC: Partial Occlusion ---
    attrs["POC"] = 1 if 0.05 <= vis_ratio < 0.7 else 0

    # --- FOC: Full Occlusion ---
    # Target center is inside camera frustum but visibility near zero
    inside_frustum = target_inside_frustum(target_pos_3d, cam_data, resolution)
    attrs["FOC"] = 1 if (vis_ratio < 0.05 and inside_frustum) else 0

    # --- OV: Out of View ---
    # Target center projects outside image, or full bbox mostly outside
    if full_bbox is not None:
        fw, fh = full_bbox[2], full_bbox[3]
        # Check if full bbox has very little overlap with image
        overlap_x = max(0, min(full_bbox[0] + fw, resolution[0]) - max(full_bbox[0], 0))
        overlap_y = max(0, min(full_bbox[1] + fh, resolution[1]) - max(full_bbox[1], 0))
        overlap_area = overlap_x * overlap_y
        full_area = fw * fh
        mostly_outside = full_area > 0 and overlap_area < full_area * 0.2
    else:
        mostly_outside = not inside_frustum

    attrs["OV"] = 1 if (not inside_frustum or mostly_outside) else 0

    # --- LR: Low Resolution ---
    # Visible bbox area < 1000 pixels
    vb_area = visible_bbox[2] * visible_bbox[3] if visible_bbox else 0
    attrs["LR"] = 1 if vb_area > 0 and vb_area < 1000 else 0

    # --- ARC: Aspect Ratio Change ---
    # Aspect ratio outside [0.5, 2.0] compared to reference
    if ref_full_bbox is not None and full_bbox is not None:
        ref_w, ref_h = ref_full_bbox[2], ref_full_bbox[3]
        cur_w, cur_h = full_bbox[2], full_bbox[3]
        if ref_h > 0 and ref_w > 0 and cur_h > 0 and cur_w > 0:
            ref_ar = ref_w / ref_h
            cur_ar = cur_w / cur_h
            if ref_ar > 0:
                ar_ratio = cur_ar / ref_ar
                attrs["ARC"] = 1 if (ar_ratio < 0.5 or ar_ratio > 2.0) else 0
            else:
                attrs["ARC"] = 0
        else:
            attrs["ARC"] = 0
    else:
        attrs["ARC"] = 0

    # --- SV: Scale Variation ---
    # Bbox area ratio outside [0.5, 2.0] compared to reference
    if ref_full_bbox is not None and full_bbox is not None:
        ref_area = ref_full_bbox[2] * ref_full_bbox[3]
        cur_area = full_bbox[2] * full_bbox[3]
        if ref_area > 0:
            area_ratio = cur_area / ref_area
            attrs["SV"] = 1 if (area_ratio < 0.5 or area_ratio > 2.0) else 0
        else:
            attrs["SV"] = 0
    else:
        attrs["SV"] = 0

    # --- BC: Background Clutter ---
    # Driven by main_challenge from trajectory
    attrs["BC"] = 1 if main_challenge == "BC" else 0

    # --- MB: Motion Blur ---
    # Driven by main_challenge (fast motion segments)
    attrs["MB"] = 1 if main_challenge == "MB" else 0

    # --- DEF: Deformation ---
    attrs["DEF"] = 1 if main_challenge == "DEF" else 0

    return attrs


# ---------------------------------------------------------------------------
# Sequence attribute summary (MVTrack style)
# ---------------------------------------------------------------------------

def compute_sequence_attributes(view_attributes, n_frames):
    """Compute sequence-level attribute summary from per-frame attributes.

    Returns dict: {attr_name: {present: bool, frames: int, ratio: float}}
    """
    attr_names = ["POC", "FOC", "OV", "BC", "MB", "DEF", "LR", "ARC", "SV"]
    summary = {}

    total_frames = n_frames * len(view_attributes)

    for attr in attr_names:
        count = 0
        for vi in view_attributes:
            for frame_attrs in vi:
                if frame_attrs.get(attr, 0) == 1:
                    count += 1
        summary[attr] = {
            "present": count > 0,
            "frames": count,
            "ratio": round(count / max(total_frames, 1), 4),
        }

    return summary


# ---------------------------------------------------------------------------
# Main export
# ---------------------------------------------------------------------------

def export_annotations(seq_dir, cfg=None):
    """Full annotation export from rendered masks.

    Args:
        seq_dir: sequence directory path
        cfg: optional config dict

    Writes groundtruth_v*.txt, invisible_v*.txt, attributes.json
    """
    # Load metadata
    with open(os.path.join(seq_dir, "attributes.json")) as f:
        meta = json.load(f)
    with open(os.path.join(seq_dir, "calibs.json")) as f:
        calibs = json.load(f)

    n_frames = meta["num_frames"]
    resolution = meta.get("resolution", [640, 480])

    # Load trajectory data for 3D positions and challenge type
    render_meta_path = os.path.join(seq_dir, "render_meta.json")
    if os.path.isfile(render_meta_path):
        with open(render_meta_path) as f:
            render_meta = json.load(f)
    else:
        render_meta = {}

    trajectory = render_meta.get("trajectory", {})
    positions = trajectory.get("positions_m", [])
    main_challenge = trajectory.get("main_challenge", "POC")

    # Load visibility data
    visibility = load_visibility(seq_dir)  # vis[view][frame]
    full_bboxes = load_full_bboxes(seq_dir)  # bbox[view][frame]
    n_views = len(visibility) if visibility else len(calibs)

    print(f"Exporting annotations: {seq_dir} ({n_frames}f x {n_views}v)")

    # Process each view
    all_view_attributes = []

    for vi in range(n_views):
        view_key = f"cam{vi}"
        if view_key not in calibs:
            view_key = str(vi)
        cam_data = calibs[view_key]

        gt_lines = []
        inv_lines = []
        frame_attributes = []

        # Compute reference full bbox (median of non-empty bboxes)
        ref_bbox = None
        if vi < len(full_bboxes):
            valid_bboxes = [b for b in full_bboxes[vi] if b is not None]
            if valid_bboxes:
                ref_bbox = [
                    int(np.median([b[i] for b in valid_bboxes]))
                    for i in range(4)
                ]

        prev_full_bbox = None

        for fi in range(n_frames):
            # Get visibility ratio
            vis_ratio = 1.0
            if vi < len(visibility) and fi < len(visibility[vi]):
                vis_ratio = visibility[vi][fi]

            # Compute visible bbox from mask
            mask_path = os.path.join(seq_dir, "masks", f"{vi:04d}", f"{fi:06d}.png")
            visible_bbox = mask_to_bbox_safe(mask_path)

            # Get full projected bbox
            full_bbox = None
            if vi < len(full_bboxes) and fi < len(full_bboxes[vi]):
                full_bbox = full_bboxes[vi][fi]

            # Get target 3D position
            target_pos = positions[fi] if fi < len(positions) else [0, 0, 0]

            # Compute frame attributes
            attrs = compute_frame_attributes(
                fi, vi, vis_ratio, visible_bbox, full_bbox,
                target_pos, cam_data, resolution, main_challenge,
                prev_full_bbox=prev_full_bbox,
                ref_full_bbox=ref_bbox,
            )

            frame_attributes.append(attrs)

            # Write groundtruth line
            gt_lines.append(
                f"{visible_bbox[0]},{visible_bbox[1]},"
                f"{visible_bbox[2]},{visible_bbox[3]}"
            )

            # Write invisible line
            inv_lines.append(str(attrs["invisible"]))

            prev_full_bbox = full_bbox

        all_view_attributes.append(frame_attributes)

        # Save groundtruth.txt
        gt_path = os.path.join(seq_dir, f"groundtruth_v{vi:04d}.txt")
        with open(gt_path, "w") as f:
            f.write("\n".join(gt_lines) + "\n")

        # Save invisible.txt
        inv_path = os.path.join(seq_dir, f"invisible_v{vi:04d}.txt")
        with open(inv_path, "w") as f:
            f.write("\n".join(inv_lines) + "\n")

        # Per-view attribute stats
        occ_count = sum(1 for a in frame_attributes if a["POC"] or a["FOC"])
        print(f"  View {vi}: {occ_count}/{n_frames} occluded frames")

    # Compute sequence-level attributes
    seq_attrs = compute_sequence_attributes(all_view_attributes, n_frames)

    # Update attributes.json
    meta["frame_attributes"] = {
        f"view_{vi}": all_view_attributes[vi] for vi in range(n_views)
    }
    meta["sequence_attributes"] = seq_attrs
    meta["main_challenge"] = main_challenge

    with open(os.path.join(seq_dir, "attributes.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Print summary
    print("  Sequence attributes:")
    for attr, info in sorted(seq_attrs.items()):
        if info["present"]:
            print(f"    {attr}: {info['frames']} frames ({info['ratio']:.1%})")

    return seq_attrs


# ---------------------------------------------------------------------------
# Batch export
# ---------------------------------------------------------------------------

def export_batch(output_dir, seq_ids=None):
    """Export annotations for multiple sequences."""
    if seq_ids is None:
        seq_ids = sorted([
            d for d in os.listdir(output_dir)
            if os.path.isdir(os.path.join(output_dir, d)) and d.startswith("seq_")
        ])

    all_summaries = {}
    for seq_id in seq_ids:
        seq_dir = os.path.join(output_dir, seq_id)
        if not os.path.isdir(seq_dir):
            continue
        if not os.path.isfile(os.path.join(seq_dir, "visibility.txt")):
            print(f"Skipping {seq_id}: no visibility.txt")
            continue
        try:
            summary = export_annotations(seq_dir)
            all_summaries[seq_id] = summary
        except Exception as e:
            print(f"Error exporting {seq_id}: {e}")

    # Print dataset-wide attribute distribution
    if all_summaries:
        print(f"\nDataset attribute distribution ({len(all_summaries)} sequences):")
        attr_totals = {}
        for seq_id, summary in all_summaries.items():
            for attr, info in summary.items():
                if attr not in attr_totals:
                    attr_totals[attr] = {"sequences": 0, "total_frames": 0}
                if info["present"]:
                    attr_totals[attr]["sequences"] += 1
                    attr_totals[attr]["total_frames"] += info["frames"]

        for attr, totals in sorted(attr_totals.items()):
            print(f"  {attr}: {totals['sequences']}/{len(all_summaries)} sequences, "
                  f"{totals['total_frames']} total frames")

    return all_summaries


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export annotations from rendered masks")
    parser.add_argument("--seq-dir", default=None, help="Single sequence directory")
    parser.add_argument("--output-dir", default="output/SynMVTrack",
                        help="Batch output directory")
    parser.add_argument("--seq-ids", nargs="*", default=None)
    args = parser.parse_args()

    if args.seq_dir:
        export_annotations(args.seq_dir)
    else:
        export_batch(args.output_dir, seq_ids=args.seq_ids)
