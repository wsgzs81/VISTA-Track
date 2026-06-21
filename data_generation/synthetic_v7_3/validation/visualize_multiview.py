#!/usr/bin/env python3
"""
visualize_multiview.py — Generate 4-panel debug video for a sequence.

Layout:
  cam1 | cam2
  cam3 | cam4

Overlays per panel:
  - green bbox (groundtruth)
  - semi-transparent red mask
  - visibility ratio text
  - invisible flag (red INVISIBLE / green VISIBLE)
  - main challenge label
  - projected 3D center (yellow crosshair)

Output: MP4 video via OpenCV.
"""

import json
import math
import os
import sys

import numpy as np


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_mask(path):
    try:
        from PIL import Image
        return np.array(Image.open(path).convert("L"))
    except Exception:
        return None


def load_lines(path):
    if not os.path.isfile(path):
        return []
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


def load_visibility(seq_dir):
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


def parse_gt_line(line):
    return [int(v) for v in line.strip().split(",")]


def load_image(path):
    """Load RGB image."""
    try:
        from PIL import Image
        img = Image.open(path).convert("RGB")
        return np.array(img)
    except Exception:
        return None


def project_point(pt_3d, K, R, T):
    """Project 3D point → (u, v) or None."""
    pt = np.array(pt_3d, dtype=np.float64)
    cam = R @ pt + T
    if cam[2] <= 0.01:
        return None
    px = K @ cam
    return float(px[0] / px[2]), float(px[1] / px[2])


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_bbox(img, bbox, color=(0, 255, 0), thickness=2):
    """Draw bounding box on image (in-place)."""
    try:
        import cv2
        x, y, w, h = bbox
        cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
    except ImportError:
        pass


def draw_mask_overlay(img, mask, color=(0, 0, 255), alpha=0.3):
    """Overlay semi-transparent mask on image (in-place)."""
    if mask is None:
        return
    try:
        import cv2
        overlay = img.copy()
        mask_bool = mask > 128
        overlay[mask_bool] = color
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    except ImportError:
        pass


def draw_crosshair(img, cx, cy, size=10, color=(0, 255, 255), thickness=2):
    """Draw a crosshair at (cx, cy)."""
    try:
        import cv2
        cx, cy = int(cx), int(cy)
        cv2.line(img, (cx - size, cy), (cx + size, cy), color, thickness)
        cv2.line(img, (cx, cy - size), (cx, cy + size), color, thickness)
    except ImportError:
        pass


def draw_text(img, text, org, color=(255, 255, 255), scale=0.5, thickness=1):
    """Draw text with background."""
    try:
        import cv2
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        x, y = org
        cv2.rectangle(img, (x, y - th - 4), (x + tw + 4, y + baseline + 2), (0, 0, 0), -1)
        cv2.putText(img, text, (x + 2, y), font, scale, color, thickness)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Panel builder
# ---------------------------------------------------------------------------

def build_panel(seq_dir, vi, fi, attrs, calibs, positions, visibility,
                resolution, seq_base):
    """Build one camera panel as numpy array (H, W, 3) RGB."""
    res_w, res_h = resolution
    panel = np.zeros((res_h, res_w, 3), dtype=np.uint8)

    # Try to load image
    img = None
    img_path = os.path.join(seq_dir, f"{seq_base}-{vi + 1}", "img", f"{fi + 1:08d}.jpg")
    if not os.path.isfile(img_path):
        img_path = os.path.join(seq_dir, "img", f"{vi:04d}", f"{fi:06d}.jpg")
    if os.path.isfile(img_path):
        img = load_image(img_path)

    if img is not None:
        panel = img.copy()
        if panel.shape[:2] != (res_h, res_w):
            try:
                import cv2
                panel = cv2.resize(panel, (res_w, res_h))
            except ImportError:
                pass

    # Load mask
    mask_path = os.path.join(seq_dir, f"{seq_base}-{vi + 1}", "masks", f"{fi + 1:08d}.png")
    if not os.path.isfile(mask_path):
        mask_path = os.path.join(seq_dir, "masks", f"{vi:04d}", f"{fi:06d}.png")
    mask = load_mask(mask_path)

    # Groundtruth bbox
    view_dir = os.path.join(seq_dir, f"{seq_base}-{vi + 1}")
    gt_lines = load_lines(os.path.join(view_dir, "groundtruth.txt"))
    if fi < len(gt_lines):
        bbox = parse_gt_line(gt_lines[fi])
    else:
        bbox = [0, 0, 0, 0]

    # Invisible flag
    inv_lines = load_lines(os.path.join(view_dir, "invisible.txt"))
    invisible = 0
    if fi < len(inv_lines):
        invisible = int(inv_lines[fi])

    # Visibility ratio
    vis_ratio = 1.0
    if vi < len(visibility) and fi < len(visibility[vi]):
        vis_ratio = visibility[vi][fi]

    # --- Draw mask overlay ---
    draw_mask_overlay(panel, mask, color=(0, 0, 255), alpha=0.3)

    # --- Draw bbox ---
    gx, gy, gw, gh = bbox
    if gw > 0 and gh > 0:
        color = (0, 255, 0) if invisible == 0 else (0, 128, 255)
        draw_bbox(panel, bbox, color=color, thickness=2)

    # --- Draw projected 3D center ---
    if fi < len(positions):
        cam_key = sorted(calibs.keys(),
                         key=lambda k: int("".join(filter(str.isdigit, k)) or 0))[vi]
        cam = calibs[cam_key]
        K = np.array(cam["K"], dtype=np.float64)
        R = np.array(cam["R"], dtype=np.float64)
        T = np.array(cam["T"], dtype=np.float64)
        proj = project_point(positions[fi], K, R, T)
        if proj is not None:
            draw_crosshair(panel, proj[0], proj[1], size=8, color=(0, 255, 255))

    # --- Text overlays ---
    y_off = 20
    draw_text(panel, f"View {vi}", (5, y_off), color=(255, 255, 255), scale=0.5)
    y_off += 22

    main_challenge = attrs.get("main_challenge", "?")
    draw_text(panel, f"Challenge: {main_challenge}", (5, y_off),
              color=(255, 200, 0), scale=0.45)
    y_off += 18

    vis_str = f"Vis: {vis_ratio:.2f}"
    vis_color = (0, 255, 0) if vis_ratio >= 0.05 else (0, 0, 255)
    draw_text(panel, vis_str, (5, y_off), color=vis_color, scale=0.45)
    y_off += 18

    inv_text = "INVISIBLE" if invisible else "VISIBLE"
    inv_color = (0, 0, 255) if invisible else (0, 255, 0)
    draw_text(panel, inv_text, (5, y_off), color=inv_color, scale=0.5)

    # Frame number bottom-right
    draw_text(panel, f"F{fi:04d}", (res_w - 60, res_h - 10),
              color=(200, 200, 200), scale=0.4)

    return panel


# ---------------------------------------------------------------------------
# Video generation
# ---------------------------------------------------------------------------

def generate_video(seq_dir, output_path=None, max_frames=None, fps=10,
                   panel_scale=0.5):
    """Generate 4-panel debug video for a sequence.

    Args:
        seq_dir: sequence directory
        output_path: output video path (default: seq_dir/debug_video.mp4)
        max_frames: limit frames (None = all)
        fps: video frame rate
        panel_scale: scale factor for each panel (<1 to shrink)
    """
    try:
        import cv2
    except ImportError:
        print("ERROR: opencv-python required for video generation")
        print("  pip install opencv-python")
        return None

    seq_base = os.path.basename(seq_dir)

    with open(os.path.join(seq_dir, "attributes.json")) as f:
        attrs = json.load(f)
    with open(os.path.join(seq_dir, "calibs.json")) as f:
        calibs = json.load(f)

    n_frames = attrs["num_frames"]
    resolution = attrs.get("resolution", [640, 480])
    res_w, res_h = resolution

    if max_frames:
        n_frames = min(n_frames, max_frames)

    # Load trajectory for 3D center projection
    render_meta_path = os.path.join(seq_dir, "render_meta.json")
    if os.path.isfile(render_meta_path):
        with open(render_meta_path) as f:
            render_meta = json.load(f)
        positions = render_meta.get("trajectory", {}).get("positions_m", [])
    else:
        positions = []

    visibility = load_visibility(seq_dir)

    # Determine number of views
    n_views = attrs.get("num_views", 0)
    if n_views == 0:
        subdirs = [
            d for d in os.listdir(seq_dir)
            if os.path.isdir(os.path.join(seq_dir, d)) and d.startswith(seq_base + "-")
        ]
        n_views = min(len(subdirs), 4)

    n_views = min(n_views, 4)  # cap at 4 for grid

    # Panel size
    pw = int(res_w * panel_scale)
    ph = int(res_h * panel_scale)

    # Grid: 2x2
    grid_w = pw * 2
    grid_h = ph * 2

    # Output path
    if output_path is None:
        output_path = os.path.join(seq_dir, "debug_video.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (grid_w, grid_h))

    print(f"Generating video: {seq_base} ({n_frames} frames, {n_views} views)")
    print(f"  Output: {output_path}")
    print(f"  Grid: {grid_w}x{grid_h} ({pw}x{ph} per panel)")

    for fi in range(n_frames):
        grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

        for vi in range(n_views):
            panel = build_panel(
                seq_dir, vi, fi, attrs, calibs, positions,
                visibility, resolution, seq_base,
            )

            # Resize panel
            try:
                import cv2
                panel_resized = cv2.resize(panel, (pw, ph))
            except ImportError:
                panel_resized = panel[:ph, :pw]

            row = vi // 2
            col = vi % 2
            grid[row * ph:(row + 1) * ph, col * pw:(col + 1) * pw] = panel_resized

        # Add frame counter overlay on grid
        try:
            import cv2
            cv2.putText(grid, f"Frame {fi + 1}/{n_frames}", (10, grid_h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        except ImportError:
            pass

        # OpenCV uses BGR
        grid_bgr = grid[:, :, ::-1].copy()
        writer.write(grid_bgr)

        if (fi + 1) % 50 == 0 or fi == n_frames - 1:
            print(f"  {fi + 1}/{n_frames} frames")

    writer.release()
    print(f"  Video saved: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Batch / CLI
# ---------------------------------------------------------------------------

def generate_batch(output_dir, seq_ids=None, max_frames=None, fps=10):
    if seq_ids is None:
        seq_ids = sorted([
            d for d in os.listdir(output_dir)
            if os.path.isdir(os.path.join(output_dir, d)) and d.startswith("seq_")
        ])

    paths = []
    for seq_id in seq_ids:
        seq_dir = os.path.join(output_dir, seq_id)
        if not os.path.isfile(os.path.join(seq_dir, "attributes.json")):
            continue
        try:
            p = generate_video(seq_dir, max_frames=max_frames, fps=fps)
            if p:
                paths.append(p)
        except Exception as e:
            print(f"  ERROR {seq_id}: {e}")

    return paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate multiview debug videos")
    parser.add_argument("--seq-dir", default=None)
    parser.add_argument("--output-dir", default="output/SynMVTrack")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--output", default=None, help="Output video path")
    args = parser.parse_args()

    if args.seq_dir:
        generate_video(args.seq_dir, output_path=args.output,
                       max_frames=args.max_frames, fps=args.fps)
    else:
        generate_batch(args.output_dir, max_frames=args.max_frames, fps=args.fps)
