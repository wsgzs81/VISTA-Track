#!/usr/bin/env python3
"""
generate_dataset.py — V0 dataset generation pipeline.

Orchestrates:
  1. Metadata generation (scene, target, cameras, trajectory, occluders, distractors)
  2. Blender rendering (RGB + dual masks)
  3. Annotation export (bbox, invisible, attributes)
  4. MVTrack format restructure
  5. Validation

Usage:
    # Full pipeline (metadata only, no rendering)
    python generate_dataset.py --config configs/dataset_v0.yaml --output output/SynMVTrack

    # Full pipeline with Blender rendering
    python generate_dataset.py --config configs/dataset_v0.yaml --output output/SynMVTrack --render --blender /usr/bin/blender

    # Just validation
    python generate_dataset.py --config configs/dataset_v0.yaml --output output/SynMVTrack --validate-only

    # Single step: metadata
    python generate_dataset.py --config configs/dataset_v0.yaml --output output/SynMVTrack --step metadata
"""

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

def _setup_paths():
    """Add generator scripts and project root to sys.path."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # generator/ contains the pipeline scripts
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    # project root for validation/
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


_setup_paths()


# ---------------------------------------------------------------------------
# Imports from sibling scripts
# ---------------------------------------------------------------------------

def _import_pipeline():
    """Lazy import pipeline modules."""
    from sample_scene import sample_scene
    from sample_target import sample_target
    from sample_camera_rig import sample_camera_rig
    from sample_trajectory import sample_trajectory
    from sample_occluders import sample_occluders
    from sample_distractors import sample_distractors
    from export_annotations import export_annotations
    from write_mvtrack_format import write_mvtrack_format, generate_splits
    return {
        "sample_scene": sample_scene,
        "sample_target": sample_target,
        "sample_camera_rig": sample_camera_rig,
        "sample_trajectory": sample_trajectory,
        "sample_occluders": sample_occluders,
        "sample_distractors": sample_distractors,
        "export_annotations": export_annotations,
        "write_mvtrack_format": write_mvtrack_format,
        "generate_splits": generate_splits,
    }


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(cfg_path):
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def resolve_paths(cfg, cfg_path):
    """Resolve relative asset paths against project root."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(cfg_path)))

    for key in ["target_index", "scene_index", "material_index", "hdri_index"]:
        p = cfg["assets"][key]
        if not os.path.isabs(p):
            cfg["assets"][key] = os.path.join(project_root, p)

    out_root = cfg.get("output_dir", "output/SynMVTrack")
    if not os.path.isabs(out_root):
        out_root = os.path.join(project_root, out_root)
    cfg["output_dir"] = out_root

    return cfg


# ---------------------------------------------------------------------------
# Single sequence generation
# ---------------------------------------------------------------------------

def generate_metadata(seq_id, cfg, seed=None, pipes=None):
    """Generate scene, target, cameras, trajectory, occluders, distractors.

    Order: scene -> target -> trajectory -> cameras (based on trajectory bounds)
    This ensures the camera FOV covers the entire target trajectory.
    """
    if pipes is None:
        pipes = _import_pipeline()

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    max_attempts = 20
    for attempt in range(max_attempts):
        attempt_seed = (seed + attempt * 1000) if seed is not None else None
        rng = random.Random(attempt_seed)

        # Sample scene and target first
        scene = pipes["sample_scene"](cfg, rng=rng)
        target = pipes["sample_target"](cfg, rng=rng)

        # Generate trajectory BEFORE cameras so we know target bounds
        # We need a temporary camera rig for trajectory generation
        temp_cameras = pipes["sample_camera_rig"](scene, cfg, rng=rng)
        trajectory = pipes["sample_trajectory"](scene, target, temp_cameras, cfg, rng=rng)

        # Compute trajectory bounding box (target center positions)
        positions = trajectory["positions_m"]
        xs = [p[0] for p in positions]
        zs = [p[2] for p in positions]
        ys = [p[1] for p in positions]
        traj_center_xy = [(min(xs) + max(xs)) / 2, (min(zs) + max(zs)) / 2]
        traj_half_extent = max(max(xs) - min(xs), max(zs) - min(zs)) / 2
        traj_height = (min(ys) + max(ys)) / 2

        # Now generate cameras aimed at trajectory center with distance
        # sufficient to keep entire trajectory in FOV
        from sample_camera_rig import (
            compute_camera_pose, compute_intrinsics, project_to_image,
        )
        n_frames = len(positions)
        resolution = cfg["render"]["resolution"]
        cam_cfg = cfg["camera"]
        views_choices = cfg["dataset"]["views_choices"]

        num_views = int(rng.choices(
            list(views_choices.keys()),
            weights=list(views_choices.values()),
            k=1,
        )[0])

        # Distance must be large enough that FOV covers trajectory extent
        fov_min, fov_max = cam_cfg["fov_range_deg"]
        fov = rng.uniform(fov_min, fov_max)
        half_fov_rad = math.radians(fov / 2.0)
        # Min distance so trajectory fits in FOV with margin
        min_dist = (traj_half_extent * 1.3) / math.tan(half_fov_rad)
        min_dist = max(min_dist, cam_cfg["distance_range_m"][0])
        # Max distance: don't go too far (exits room, lighting weak)
        max_dist = min(cam_cfg["distance_range_m"][1], 4.0)
        if min_dist > max_dist:
            min_dist = max_dist

        # Height near target height so vertical FOV is well-used
        h_min, h_max = cam_cfg["height_range_m"]
        height = max(h_min, min(h_max, traj_height + rng.uniform(-0.3, 0.5)))

        # Elevation: look roughly at target height (small angle)
        el_min, el_max = cam_cfg.get("elevation_range_deg", [-25, 5])
        # Clamp to small range so target stays near image center vertically
        el_min = max(el_min, -10)
        el_max = min(el_max, 10)

        # Place cameras: for each view, try azimuths until ≥90% trajectory frames
        # have clear line of sight (no occluder bbox on ray to target).
        traj_samples = [positions[fi] for fi in range(0, n_frames, max(1, n_frames // 15))]
        static_bboxes = [o["bbox_3d"] for o in scene.get("static_occluders", []) if o.get("bbox_3d")]
        min_az_sep = float(cam_cfg.get("min_azimuth_separation_deg", 0.0))
        requested_inside_margin = float(cam_cfg.get("inside_room_margin_m", 0.0))

        def _ray_aabb(ray_o, ray_d, lo, hi):
            tmin, tmax = -1e9, 1e9
            for j in range(3):
                if abs(ray_d[j]) < 1e-9:
                    if ray_o[j] < lo[j] or ray_o[j] > hi[j]:
                        return None
                else:
                    t1 = (lo[j] - ray_o[j]) / ray_d[j]
                    t2 = (hi[j] - ray_o[j]) / ray_d[j]
                    if t1 > t2:
                        t1, t2 = t2, t1
                    tmin = max(tmin, t1)
                    tmax = min(tmax, t2)
                    if tmin > tmax:
                        return None
            return tmin if tmax >= 0 else None

        def _is_los_clear(cam_pos, target_pos, bboxes):
            ray = np.array(target_pos) - np.array(cam_pos)
            ray_len = np.linalg.norm(ray)
            if ray_len < 1e-6:
                return True
            ray_d = ray / ray_len
            for bbox in bboxes:
                lo, hi = np.array(bbox[0]), np.array(bbox[1])
                t = _ray_aabb(np.array(cam_pos), ray_d, lo, hi)
                if t is not None and t < ray_len - 0.1:
                    return False
            return True

        def _cam_los_score(cam_pos, sample_positions, bboxes):
            if not bboxes:
                return 1.0
            clear = 0
            for tpos in sample_positions:
                if _is_los_clear(cam_pos, tpos, bboxes):
                    clear += 1
            return clear / len(sample_positions)

        cameras = []
        camera_sampling_failed = False
        base_az = rng.uniform(0, 360)
        for i in range(num_views):
            best_cam = None
            best_score = 0
            room = scene.get("room_size_m", [5.0, 2.8, 5.0])
            room_x, room_z = room[0], room[2]
            inside_margin = 0.0
            if requested_inside_margin > 0:
                # Keep cameras physically inside closed rooms. Clamp the margin
                # for narrow 3D-FRONT rooms instead of making sampling impossible.
                max_feasible_margin = max(0.0, min(room_x, room_z) / 2.0 - 0.30)
                inside_margin = min(requested_inside_margin, max_feasible_margin)

            def _camera_inside_room(cam_pos):
                if inside_margin > 0:
                    return (
                        inside_margin <= cam_pos[0] <= room_x - inside_margin and
                        inside_margin <= cam_pos[2] <= room_z - inside_margin
                    )
                return (
                    -0.5 <= cam_pos[0] <= room_x + 0.5 and
                    -0.5 <= cam_pos[2] <= room_z + 0.5
                )

            for try_az in range(160):
                az = (base_az + 360 * i / num_views + try_az * 4.5) % 360
                if min_az_sep > 0 and cameras:
                    existing = [float(cam.get("azimuth_deg", 0.0)) for cam in cameras]
                    closest = min(min(abs(az - old), 360.0 - abs(az - old)) for old in existing)
                    if closest < min_az_sep:
                        continue
                el = rng.uniform(el_min, el_max)
                dist = rng.uniform(min_dist, max_dist)

                K, _, _, _, _ = compute_intrinsics(fov, resolution[0], resolution[1])
                cam_pos, R_c2w = compute_camera_pose(az, el, dist, traj_center_xy, height)

                if not _camera_inside_room(cam_pos):
                    continue

                score = _cam_los_score(cam_pos, traj_samples, static_bboxes)
                if score > best_score:
                    best_score = score
                    from sample_camera_rig import compute_extrinsics
                    R_ext, T_ext = compute_extrinsics(R_c2w, cam_pos)
                    best_cam = {
                        "view_id": i,
                        "azimuth_deg": round(az, 1),
                        "elevation_deg": round(el, 1),
                        "distance_m": round(dist, 2),
                        "height_m": round(height, 2),
                        "fov_deg": round(fov, 1),
                        "K": [[round(v, 4) for v in row] for row in K],
                        "R": [[round(v, 6) for v in row] for row in R_ext],
                        "T": [round(v, 4) for v in T_ext],
                        "image_size": resolution,
                        "unit": "m",
                    }
                    if score >= 0.95:
                        break  # good enough

            if best_cam is None:
                if inside_margin > 0:
                    camera_sampling_failed = True
                    break
                # Fallback
                az = (base_az + 360 * i / num_views) % 360
                K, _, _, _, _ = compute_intrinsics(fov, resolution[0], resolution[1])
                cam_pos, R_c2w = compute_camera_pose(az, 0, min_dist, traj_center_xy, height)
                from sample_camera_rig import compute_extrinsics
                R_ext, T_ext = compute_extrinsics(R_c2w, cam_pos)
                best_cam = {
                    "view_id": i, "azimuth_deg": round(az, 1), "elevation_deg": 0,
                    "distance_m": round(min_dist, 2), "height_m": round(height, 2),
                    "fov_deg": round(fov, 1),
                    "K": [[round(v, 4) for v in row] for row in K],
                    "R": [[round(v, 6) for v in row] for row in R_ext],
                    "T": [round(v, 4) for v in T_ext],
                    "image_size": resolution, "unit": "m",
                }
            cameras.append(best_cam)

        if camera_sampling_failed:
            print(f"  RETRY {seq_id} attempt {attempt+1}: "
                  "could not place all cameras inside room")
            continue

        if min_az_sep > 0 and len(cameras) > 1:
            min_pair_sep = 360.0
            for a in range(len(cameras)):
                for b in range(a + 1, len(cameras)):
                    da = abs(float(cameras[a]["azimuth_deg"]) - float(cameras[b]["azimuth_deg"]))
                    min_pair_sep = min(min_pair_sep, da, 360.0 - da)
            if min_pair_sep < min_az_sep - 0.5:
                print(f"  RETRY {seq_id} attempt {attempt+1}: "
                      f"camera min azimuth separation {min_pair_sep:.1f}deg "
                      f"< required {min_az_sep:.1f}deg")
                continue

        # Validate two conditions:
        # 1. Target must project within image bounds of ALL views (in FOV)
        # 2. At least 1 view must have clear LOS (not blocked by furniture)
        cam_poses = []
        for cam in cameras:
            pos, R = compute_camera_pose(
                cam["azimuth_deg"], cam["elevation_deg"],
                cam["distance_m"], traj_center_xy, cam["height_m"],
            )
            cam_poses.append((pos, R, cam["K"]))

        totally_bad = 0  # frames where target is outside ALL views' FOV OR all LOS blocked
        sample_count = 0
        for fi in range(0, n_frames, max(1, n_frames // 20)):
            sample_count += 1
            tpos = positions[fi]

            # Check FOV: is target in image bounds for each view?
            in_fov = []
            for pos, R, K in cam_poses:
                uv = project_to_image(tpos, pos, R, K, resolution)
                in_fov.append(uv is not None)

            # Check LOS: for views where target is in FOV, is LOS clear?
            any_visible = False
            for vi, (pos, R, K) in enumerate(cam_poses):
                if not in_fov[vi]:
                    continue
                if _is_los_clear(pos, tpos, static_bboxes):
                    any_visible = True
                    break

            # Frame is bad if target is outside ALL FOVs OR no clear LOS
            if not any(in_fov) or not any_visible:
                totally_bad += 1

        if totally_bad <= sample_count * 0.05:
            break  # success — ≤5% frames with no visible view

        if attempt < max_attempts - 1:
            print(f"  RETRY {seq_id} attempt {attempt+1}: "
                  f"{totally_bad}/{sample_count} frames with no visible view "
                  f"(outside FOV or blocked)")
    else:
        print(f"  WARN {seq_id}: could not ensure FOV after {max_attempts} attempts")

    # Sample occluders with final cameras. Treat weak occlusion as a failed
    # sample instead of silently accepting a "clean" sequence.
    from sample_occluders import compute_per_camera_occlusion

    def _partial_occlusion_stats(candidate_occluders):
        all_occ_bboxes = [o["bbox_3d"] for o in candidate_occluders if "bbox_3d" in o]
        if not all_occ_bboxes:
            return 0.0, 0, len(trajectory["positions_m"])
        positions = trajectory["positions_m"]
        xs = [p[0] for p in positions]
        zs = [p[2] for p in positions]
        check_target_xy = [(min(xs) + max(xs)) / 2, (min(zs) + max(zs)) / 2]
        n_frames = len(positions)
        n_views = len(cameras)
        frame_occ = compute_per_camera_occlusion(positions, cameras, all_occ_bboxes, check_target_xy)
        frames_with_occlusion = 0  # frames where ≥1 view is blocked
        frames_all_clear = 0       # frames where all views are clear
        for fi in range(n_frames):
            blocked = sum(1 for ci in range(n_views) if frame_occ[ci][fi])
            if blocked > 0 and blocked < n_views:
                frames_with_occlusion += 1
            if blocked == 0:
                frames_all_clear += 1
        return frames_with_occlusion / n_frames, frames_with_occlusion, n_frames

    min_occ_ratio = float(cfg.get("quality", {}).get("min_partial_occlusion_ratio", 0.18))
    best_occluders = None
    best_stats = (-1.0, 0, len(trajectory["positions_m"]))
    for occ_attempt in range(8):
        candidate_occluders = pipes["sample_occluders"](
            scene, target, trajectory, cameras, cfg, rng=rng,
        )
        occ_ratio, occ_frames, total_frames = _partial_occlusion_stats(candidate_occluders)
        if occ_ratio > best_stats[0]:
            best_occluders = candidate_occluders
            best_stats = (occ_ratio, occ_frames, total_frames)
        if occ_ratio >= min_occ_ratio:
            occluders = candidate_occluders
            break
    else:
        occluders = best_occluders or []
        occ_ratio, occ_frames, total_frames = best_stats
        print(f"  WARN {seq_id}: best partial occlusion only {occ_ratio:.0%} "
              f"({occ_frames}/{total_frames}) after resampling")

    distractors = pipes["sample_distractors"](scene, target, trajectory, cameras, cfg, rng=rng)

    return scene, target, cameras, trajectory, occluders, distractors


def write_sequence_output(seq_id, seq_dir, scene, target, cameras,
                          trajectory, occluders, distractors, cfg):
    """Write initial pipeline output (attributes.json, calibs.json, etc.)."""
    n_frames = trajectory["num_frames"]
    n_views = len(cameras)
    res = cfg["render"]["resolution"]

    # Create directory structure
    for vi in range(n_views):
        for sub in ["img", "masks", "full_masks", "depth"]:
            os.makedirs(os.path.join(seq_dir, sub, f"{vi:04d}"), exist_ok=True)
    os.makedirs(os.path.join(seq_dir, "BEV"), exist_ok=True)

    # attributes.json (initial)
    attributes = {
        "sequence_id": seq_id,
        "trajectory_type": trajectory.get("trajectory_type", "unknown"),
        "main_challenge": trajectory.get("main_challenge", "unknown"),
        "num_frames": n_frames,
        "num_views": n_views,
        "resolution": res,
    }
    with open(os.path.join(seq_dir, "attributes.json"), "w") as f:
        json.dump(attributes, f, indent=2)

    # calibs.json
    calibs = {}
    for vi, cam in enumerate(cameras):
        calibs[f"cam{vi}"] = {
            "K": cam["K"],
            "R": cam["R"],
            "T": cam["T"],
            "image_size": res,
            "unit": "mm",
        }
    with open(os.path.join(seq_dir, "calibs.json"), "w") as f:
        json.dump(calibs, f, indent=2)

    # Placeholder visibility + full_projected_bbox (overwritten by renderer)
    vis_path = os.path.join(seq_dir, "visibility.txt")
    with open(vis_path, "w") as f:
        for fi in range(n_frames):
            f.write(",".join(["1.0000"] * n_views) + "\n")

    bbox_path = os.path.join(seq_dir, "full_projected_bbox.txt")
    with open(bbox_path, "w") as f:
        for fi in range(n_frames):
            f.write("|".join(["-1,-1,-1,-1"] * n_views) + "\n")

    # render_meta.json (for Blender renderer)
    render_meta = {
        "scene": {
            "scene_id": scene.get("scene_id", ""),
            "scene_type": scene.get("scene_type", ""),
            "source": scene.get("source", ""),
            "room_size_m": scene.get("room_size_m", [5, 2.8, 5]),
            "surfaces": [
                {k: v for k, v in s.items() if k != "bounds"}
                for s in scene.get("surfaces", [])
            ],
            "static_occluders": scene.get("static_occluders", []),
        },
        "cameras": [
            {
                "view_id": c.get("view_id"),
                "azimuth_deg": c.get("azimuth_deg"),
                "elevation_deg": c.get("elevation_deg"),
                "distance_m": c.get("distance_m"),
                "height_m": c.get("height_m"),
                "fov_deg": c.get("fov_deg"),
            }
            for c in cameras
        ],
        "target": {
            "target_id": target.get("target_id", ""),
            "category": target.get("category", ""),
            "mesh_path": target.get("mesh_path", ""),
            "scale_m": target.get("scale_m", 0.3),
            "color": target.get("color", [170, 160, 145]),
            "source": target.get("source", "unknown"),
        },
        "trajectory": {
            "positions_m": trajectory["positions_m"],
            "rotations": trajectory.get("rotations", []),
            "scales": trajectory.get("scales", []),
            "main_challenge": trajectory.get("main_challenge", ""),
        },
        "occluders": [
            {k: o[k] for k in ["name", "type", "bbox_3d", "model_path", "size_m", "position_m", "rotation"]
             if k in o}
            for o in occluders
        ],
        "distractors": [
            {"name": d["name"], "position_m": d["position_m"],
             "bbox_3d": d.get("bbox_3d"), "color": d.get("color")}
            for d in distractors
        ],
    }
    with open(os.path.join(seq_dir, "render_meta.json"), "w") as f:
        json.dump(render_meta, f, indent=2)


# ---------------------------------------------------------------------------
# Blender rendering
# ---------------------------------------------------------------------------

def render_sequence(seq_dir, cfg, blender_path="blender"):
    """Run Blender renderer for one sequence."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    render_script = os.path.join(script_dir, "render_sequence.py")
    if not os.path.isfile(render_script):
        # Fallback: try project root
        render_script = os.path.join(
            os.path.dirname(script_dir), "render_sequence.py"
        )

    res = cfg["render"]["resolution"]
    cmd = [
        blender_path, "--background", "--python", render_script,
        "--", "--seq-dir", seq_dir,
        "--resolution", str(res[0]), str(res[1]),
        "--samples", str(cfg["render"]["samples"]),
    ]
    if cfg["render"].get("save_mask", True):
        cmd.append("--save-mask")
    if cfg["render"].get("save_depth", False):
        cmd.append("--save-depth")
    if not cfg["render"].get("use_view_dependent_backdrop", True):
        cmd.append("--disable-view-backdrop")
    if cfg["render"].get("identity_priority", False):
        cmd.append("--identity-priority")

    print(f"  Rendering: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def render_quality_gate(seq_dir, cfg):
    """Audit one rendered sequence and optionally reject low-quality samples."""
    quality_cfg = cfg.get("quality", {})
    if not quality_cfg.get("enable_render_gate", False):
        return True, {"sequence": os.path.basename(seq_dir), "gate": "disabled"}

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    try:
        from audit_render_quality import audit_sequence
    except Exception as exc:
        return False, {
            "sequence": os.path.basename(seq_dir),
            "issues": [f"audit_import_failed:{exc}"],
            "warnings": [],
        }

    row = audit_sequence(
        seq_dir=__import__("pathlib").Path(seq_dir),
        frames_per_seq=int(quality_cfg.get("gate_frames_per_seq", 18)),
        min_pixels=int(quality_cfg.get("gate_min_mask_pixels", 48)),
        color_max_rgb_dist=float(quality_cfg.get("gate_color_max_rgb_dist", 85.0)),
        bbox_min_iou=float(quality_cfg.get("gate_bbox_min_iou", 0.55)),
        bbox_max_area_ratio=float(quality_cfg.get("gate_bbox_max_area_ratio", 2.0)),
    )

    issues = list(row.get("issues", []))
    partial_ratio = float(row.get("partial_ratio", 0.0))
    all_blocked_ratio = float(row.get("all_blocked_ratio", 0.0))
    min_partial = float(quality_cfg.get("gate_min_rendered_partial_occ", 0.18))
    max_all_blocked = float(quality_cfg.get("gate_max_rendered_all_blocked", 0.28))
    if partial_ratio < min_partial:
        issues.append(f"gate_low_partial_occ:{partial_ratio:.3f}")
    if all_blocked_ratio > max_all_blocked:
        issues.append(f"gate_too_many_all_blocked:{all_blocked_ratio:.3f}")

    row["gate_issues"] = issues
    ok = len(issues) == 0
    return ok, row


def reject_sequence(seq_dir, output_dir, reason_row):
    """Move a failed sequence aside so it cannot enter training splits."""
    reject_root = os.path.join(output_dir, "_rejected_quality")
    os.makedirs(reject_root, exist_ok=True)
    dst = os.path.join(reject_root, os.path.basename(seq_dir))
    if os.path.isdir(dst):
        dst = f"{dst}_{int(time.time())}"
    shutil.move(seq_dir, dst)
    with open(os.path.join(dst, "quality_reject.json"), "w") as f:
        json.dump(reason_row, f, indent=2)
    return dst


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def run_validation(seq_dir, verbose=False):
    """Run all validation checks on a single sequence."""
    results = {}

    try:
        from check_bbox_mask import check_sequence as bbox_check
        results["bbox_mask"] = bbox_check(seq_dir, verbose=verbose)
    except Exception as e:
        results["bbox_mask"] = {"error": str(e)}

    try:
        from check_calibration import check_sequence as calib_check
        results["calibration"] = calib_check(seq_dir, verbose=verbose)
    except Exception as e:
        results["calibration"] = {"error": str(e)}

    try:
        from check_visibility import analyze_sequence as vis_check
        results["visibility"] = vis_check(seq_dir, verbose=verbose)
    except Exception as e:
        results["visibility"] = {"error": str(e)}

    return results


def check_acceptance(seq_dir, verbose=False):
    """Check V0 acceptance criteria for one sequence.

    Returns (pass: bool, failures: list[str])
    """
    failures = []
    seq_base = os.path.basename(seq_dir)

    # 1. Complete file set
    required_root = ["attributes.json", "calibs.json", "visibility.txt",
                     "full_projected_bbox.txt", "render_meta.json"]
    for f in required_root:
        if not os.path.isfile(os.path.join(seq_dir, f)):
            failures.append(f"missing {f}")

    # Check per-view files
    with open(os.path.join(seq_dir, "attributes.json")) as f:
        attrs = json.load(f)
    n_views = attrs.get("num_views", 0)
    for vi in range(n_views):
        view_dir = os.path.join(seq_dir, f"{seq_base}-{vi + 1}")
        if not os.path.isdir(view_dir):
            failures.append(f"missing view dir {seq_base}-{vi + 1}")
            continue
        for f in ["groundtruth.txt", "invisible.txt", "visibility.txt",
                  "attributes.json", "full_projected_bbox.txt"]:
            if not os.path.isfile(os.path.join(view_dir, f)):
                failures.append(f"view {vi}: missing {f}")
        # Check img/ and masks/ are non-empty
        img_dir = os.path.join(view_dir, "img")
        mask_dir = os.path.join(view_dir, "masks")
        if os.path.isdir(img_dir):
            n_img = len(os.listdir(img_dir))
            if n_img == 0:
                failures.append(f"view {vi}: no images")
        else:
            failures.append(f"view {vi}: no img/ dir")
        if os.path.isdir(mask_dir):
            n_mask = len(os.listdir(mask_dir))
            if n_mask == 0:
                failures.append(f"view {vi}: no masks")
        else:
            failures.append(f"view {vi}: no masks/ dir")

    # 3. Invisible label consistency
    try:
        from check_bbox_mask import check_sequence as bbox_check
        r = bbox_check(seq_dir)
        bm_issues = r["issues"]
        total_bm = sum(bm_issues.values())
        if total_bm > 0:
            # Allow small tolerance
            if bm_issues.get("invisible_vis_mismatch", 0) > 5 or \
               bm_issues.get("visible_vis_mismatch", 0) > 5:
                failures.append(
                    f"invisible/vis mismatches: "
                    f"inv_vis={bm_issues.get('invisible_vis_mismatch', 0)} "
                    f"vis_inv={bm_issues.get('visible_vis_mismatch', 0)}"
                )
    except Exception as e:
        failures.append(f"bbox_check error: {e}")

    # 4. Calibration projection
    try:
        from check_calibration import check_sequence as calib_check
        r = calib_check(seq_dir, max_reproj_px=80.0)
        proj_errs = r["issues"].get("projection_error", 0)
        if proj_errs > 10:
            failures.append(f"calibration projection errors: {proj_errs}")
        bad_det = r["issues"].get("bad_rotation_determinant", 0)
        if bad_det > 0:
            failures.append(f"bad rotation determinant: {bad_det}")
    except Exception as e:
        failures.append(f"calib_check error: {e}")

    # 5. BEV continuity
    bev_path = os.path.join(seq_dir, "BEV", "target_bev.txt")
    if os.path.isfile(bev_path):
        with open(bev_path) as f:
            bev_lines = [l.strip() for l in f if l.strip()]
        if len(bev_lines) < 2:
            failures.append("BEV has too few points")
        else:
            # Check for large jumps (> 50 grid cells = 1m)
            prev = None
            big_jumps = 0
            for line in bev_lines:
                gx, gy = [int(v) for v in line.split(",")]
                if prev is not None:
                    dist = ((gx - prev[0]) ** 2 + (gy - prev[1]) ** 2) ** 0.5
                    if dist > 50:
                        big_jumps += 1
                prev = (gx, gy)
            if big_jumps > len(bev_lines) * 0.05:
                failures.append(f"BEV discontinuous: {big_jumps} big jumps")
    else:
        failures.append("BEV/target_bev.txt missing")

    # 6. Visibility check (not single-view-only)
    try:
        from check_visibility import analyze_sequence as vis_check
        r = vis_check(seq_dir)
        if r["discard"]:
            failures.append(f"visibility discard: {r['reason']}")
    except Exception as e:
        failures.append(f"vis_check error: {e}")

    passed = len(failures) == 0
    return passed, failures


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def _sample_v1_params(cfg, i, rng):
    """Sample per-sequence parameters for V1 (variable frames, resolution, samples).

    Returns dict with frames, resolution, samples.
    """
    ds_cfg = cfg["dataset"]
    render_cfg = cfg["render"]

    # Frames
    frames_weighted_choices = ds_cfg.get("frames_per_sequence_weighted_choices")
    frames_choices = ds_cfg.get("frames_per_sequence_choices")
    if frames_weighted_choices:
        if isinstance(frames_weighted_choices, dict):
            values = list(frames_weighted_choices.keys())
            weights = list(frames_weighted_choices.values())
        else:
            values = []
            weights = []
            for item in frames_weighted_choices:
                if isinstance(item, dict):
                    value = item.get("frames", item.get("value"))
                    weight = item.get("weight", 1.0)
                else:
                    value, weight = item
                values.append(value)
                weights.append(weight)
        frames = int(rng.choices(values, weights=weights, k=1)[0])
    elif frames_choices:
        frames = rng.choice(frames_choices)
    else:
        frames = ds_cfg.get("frames_per_sequence", 300)

    # Resolution
    res_choices = render_cfg.get("resolution_choices")
    if res_choices:
        resolution = rng.choice(res_choices)
    else:
        resolution = render_cfg.get("resolution", [640, 480])

    # Samples
    samp_choices = render_cfg.get("samples_choices")
    if samp_choices:
        samples = rng.choice(samp_choices)
    else:
        samples = render_cfg.get("samples", 32)

    return {
        "frames": frames,
        "resolution": list(resolution) if not isinstance(resolution, list) else resolution,
        "samples": samples,
    }


def generate_dataset(cfg, output_dir, steps=None, blender_path="blender",
                     validate_only=False, verbose=False):
    """Run the full dataset generation pipeline.

    Args:
        cfg: config dict
        output_dir: output directory
        steps: list of steps to run, or None for all
            ["metadata", "render", "export", "mvtrack", "validate"]
        blender_path: path to Blender executable
        validate_only: only run validation
        verbose: verbose output
    """
    pipes = _import_pipeline()

    n_seqs = cfg["dataset"]["num_sequences"]
    seed = cfg.get("seed", 42)
    os.makedirs(output_dir, exist_ok=True)

    seq_ids = [f"seq_{i:04d}" for i in range(n_seqs)]

    # Per-sequence params for V1
    rng_params = random.Random(seed + 9999)  # separate RNG for params
    seq_params = [_sample_v1_params(cfg, i, rng_params) for i in range(n_seqs)]

    if validate_only:
        return validate_all(output_dir, seq_ids, verbose=verbose)

    if steps is None:
        steps = ["metadata", "render", "export", "mvtrack", "validate"]

    # --- Step 1: Metadata ---
    if "metadata" in steps:
        print("=" * 60)
        print("  Step 1: Generating metadata")
        print("=" * 60)
        t0 = time.time()

        for i, seq_id in enumerate(seq_ids):
            seq_seed = seed + i if seed is not None else None
            seq_dir = os.path.join(output_dir, seq_id)
            sp = seq_params[i]

            # Per-sequence config override (V1 variable params)
            seq_cfg = dict(cfg)
            seq_cfg["dataset"] = dict(cfg["dataset"])
            seq_cfg["dataset"]["frames_per_sequence"] = sp["frames"]
            seq_cfg["render"] = dict(cfg["render"])
            seq_cfg["render"]["resolution"] = sp["resolution"]
            seq_cfg["render"]["samples"] = sp["samples"]

            print(f"\n[{seq_id}] Generating metadata "
                  f"(frames={sp['frames']} res={sp['resolution']} samples={sp['samples']})...")
            t_seq = time.time()

            try:
                scene, target, cameras, trajectory, occluders, distractors = \
                    generate_metadata(seq_id, seq_cfg, seed=seq_seed, pipes=pipes)
            except (RecursionError, RuntimeError, MemoryError) as e:
                print(f"  SKIP {seq_id}: {type(e).__name__}: {str(e)[:100]}")
                continue

            # Defensive check: the trajectory generator must produce exactly the
            # per-sequence frame count used by audits and render queues.
            for key in ["positions_m", "rotations", "scales"]:
                if key in trajectory and len(trajectory[key]) != sp["frames"]:
                    raise RuntimeError(
                        f"{seq_id} {key} length {len(trajectory[key])} "
                        f"!= requested frames {sp['frames']}"
                    )

            write_sequence_output(
                seq_id, seq_dir, scene, target, cameras,
                trajectory, occluders, distractors, seq_cfg,
            )

            elapsed = time.time() - t_seq
            az_str = ",".join(f"{c['azimuth_deg']:.0f}" for c in cameras)
            print(f"  scene={scene['scene_id']} target={target['target_id']} "
                  f"views={len(cameras)} az=[{az_str}] "
                  f"challenge={trajectory['main_challenge']} "
                  f"({elapsed:.1f}s)")

        print(f"\nMetadata done: {n_seqs} sequences ({time.time() - t0:.1f}s)")

    # --- Step 2: Render ---
    if "render" in steps:
        print("\n" + "=" * 60)
        print("  Step 2: Blender rendering")
        print("=" * 60)
        t0 = time.time()

        for i, seq_id in enumerate(seq_ids):
            seq_dir = os.path.join(output_dir, seq_id)
            if not os.path.isdir(seq_dir):
                continue
            sp = seq_params[i]
            # Per-sequence render config
            seq_cfg = dict(cfg)
            seq_cfg["dataset"] = dict(cfg["dataset"])
            seq_cfg["dataset"]["frames_per_sequence"] = sp["frames"]
            seq_cfg["render"] = dict(cfg["render"])
            seq_cfg["render"]["resolution"] = sp["resolution"]
            seq_cfg["render"]["samples"] = sp["samples"]

            print(f"\n[{seq_id}] Rendering "
                  f"(res={sp['resolution']} samples={sp['samples']})...")
            try:
                render_sequence(seq_dir, seq_cfg, blender_path=blender_path)
                ok, quality_row = render_quality_gate(seq_dir, seq_cfg)
                if not ok:
                    rejected = reject_sequence(seq_dir, output_dir, quality_row)
                    print(
                        f"  QUALITY REJECTED {seq_id}: "
                        f"{','.join(quality_row.get('gate_issues', []))} -> {rejected}"
                    )
                else:
                    print(
                        f"  QUALITY PASS {seq_id}: "
                        f"partial={quality_row.get('partial_ratio', 0.0):.3f} "
                        f"all_blocked={quality_row.get('all_blocked_ratio', 0.0):.3f} "
                        f"color_max={quality_row.get('max_color_dist', 0.0):.1f}"
                    )
            except Exception as e:
                print(f"  RENDER FAILED: {e}")
                if verbose:
                    import traceback
                    traceback.print_exc()

        print(f"\nRendering done ({time.time() - t0:.1f}s)")

    # --- Step 3: Export annotations ---
    if "export" in steps:
        print("\n" + "=" * 60)
        print("  Step 3: Exporting annotations")
        print("=" * 60)
        t0 = time.time()

        for seq_id in seq_ids:
            seq_dir = os.path.join(output_dir, seq_id)
            vis_path = os.path.join(seq_dir, "visibility.txt")
            if not os.path.isfile(vis_path):
                print(f"  Skipping {seq_id}: no visibility.txt")
                continue
            try:
                print(f"\n[{seq_id}] Exporting annotations...")
                pipes["export_annotations"](seq_dir)
            except Exception as e:
                print(f"  EXPORT FAILED: {e}")
                if verbose:
                    import traceback
                    traceback.print_exc()

        print(f"\nAnnotation export done ({time.time() - t0:.1f}s)")

    # --- Step 4: MVTrack format ---
    if "mvtrack" in steps:
        print("\n" + "=" * 60)
        print("  Step 4: MVTrack format restructure")
        print("=" * 60)
        t0 = time.time()

        for seq_id in seq_ids:
            seq_dir = os.path.join(output_dir, seq_id)
            if not os.path.isfile(os.path.join(seq_dir, "calibs.json")):
                continue
            try:
                print(f"\n[{seq_id}] Writing MVTrack format...")
                pipes["write_mvtrack_format"](seq_dir, cfg=cfg)
            except Exception as e:
                print(f"  MVTRACK FAILED: {e}")
                if verbose:
                    import traceback
                    traceback.print_exc()

        # Generate splits (supports V1 custom counts + challenge bias)
        print("\nGenerating splits...")
        split_cfg = cfg.get("split", None)
        pipes["generate_splits"](output_dir, split_cfg=split_cfg)

        print(f"\nMVTrack format done ({time.time() - t0:.1f}s)")

    # --- Step 5: Validate ---
    if "validate" in steps:
        validate_all(output_dir, seq_ids, verbose=verbose)

    return output_dir


def validate_all(output_dir, seq_ids=None, verbose=False):
    """Run acceptance checks on all sequences."""
    if seq_ids is None:
        seq_ids = sorted([
            d for d in os.listdir(output_dir)
            if os.path.isdir(os.path.join(output_dir, d)) and d.startswith("seq_")
        ])

    print("\n" + "=" * 60)
    print("  V0 Acceptance Validation")
    print("=" * 60)

    passed = []
    failed = []

    for seq_id in seq_ids:
        seq_dir = os.path.join(output_dir, seq_id)
        if not os.path.isdir(seq_dir):
            continue

        ok, reasons = check_acceptance(seq_dir, verbose=verbose)
        if ok:
            passed.append(seq_id)
            if verbose:
                print(f"  PASS  {seq_id}")
        else:
            failed.append((seq_id, reasons))
            print(f"  FAIL  {seq_id}: {'; '.join(reasons)}")

    print(f"\n{'=' * 60}")
    print(f"  Results: {len(passed)}/{len(passed) + len(failed)} passed")
    print(f"{'=' * 60}")

    if failed:
        print(f"\n  Failed sequences ({len(failed)}):")
        for seq_id, reasons in failed:
            print(f"    {seq_id}:")
            for r in reasons:
                print(f"      - {r}")
    else:
        print("\n  All sequences passed V0 acceptance!")

    return passed, failed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SynMVTrack V0 dataset generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Steps:
  metadata  - Sample scene, target, cameras, trajectory, occluders
  render    - Blender rendering (RGB + masks)
  export    - Annotation export from masks
  mvtrack   - MVTrack format restructure
  validate  - Acceptance checks

Examples:
  # Full pipeline without rendering
  python generate_dataset.py --config configs/dataset_v0.yaml

  # Full pipeline with rendering
  python generate_dataset.py --config configs/dataset_v0.yaml --render

  # Just metadata
  python generate_dataset.py --config configs/dataset_v0.yaml --step metadata

  # Validate existing output
  python generate_dataset.py --config configs/dataset_v0.yaml --validate-only
        """,
    )
    parser.add_argument("--config", required=True, help="Config YAML path")
    parser.add_argument("--output", default=None, help="Output directory")
    parser.add_argument("--render", action="store_true",
                        help="Include Blender rendering step")
    parser.add_argument("--blender", default="blender",
                        help="Path to Blender executable")
    parser.add_argument("--step", nargs="+", default=None,
                        choices=["metadata", "render", "export", "mvtrack", "validate"],
                        help="Run specific steps only")
    parser.add_argument("--validate-only", action="store_true",
                        help="Only run validation on existing output")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (overrides config)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Load config
    cfg = load_config(os.path.abspath(args.config))
    cfg = resolve_paths(cfg, args.config)

    if args.seed is not None:
        cfg["seed"] = args.seed
    elif "seed" not in cfg:
        cfg["seed"] = 42

    # Output directory
    output_dir = args.output or cfg.get("output_dir", "output/SynMVTrack")
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(args.config))),
            output_dir,
        )

    # Steps
    steps = args.step
    if steps is None and not args.validate_only:
        steps = ["metadata", "export", "mvtrack", "validate"]
        if args.render:
            steps.insert(1, "render")

    print(f"SynMVTrack V0 Dataset Generation")
    print(f"  Config: {args.config}")
    print(f"  Output: {output_dir}")
    print(f"  Sequences: {cfg['dataset']['num_sequences']}")
    print(f"  Frames: {cfg['dataset']['frames_per_sequence']}")
    print(f"  Views: {cfg['dataset']['views_choices']}")
    print(f"  Steps: {steps or ['validate']}")
    print()

    t0 = time.time()
    result = generate_dataset(
        cfg, output_dir,
        steps=steps,
        blender_path=args.blender,
        validate_only=args.validate_only,
        verbose=args.verbose,
    )
    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed / 60:.1f}min)")


if __name__ == "__main__":
    main()
