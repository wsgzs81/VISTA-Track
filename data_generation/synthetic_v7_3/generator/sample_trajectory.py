#!/usr/bin/env python3
"""
sample_trajectory.py - Challenge-aware trajectory sampling with validation.

Generates target trajectories conditioned on a main challenge attribute,
then validates that the difficulty is actually visible from multiple views.
"""

import math
import random

import numpy as np


# ---------------------------------------------------------------------------
# Challenge distribution
# ---------------------------------------------------------------------------

CHALLENGE_DIST = {
    "POC": 0.20,   # Partial Occlusion
    "FOC": 0.15,   # Full Occlusion
    "OV": 0.15,    # Out of View
    "BC": 0.15,    # Background Clutter
    "MB": 0.10,    # Motion Blur
    "SV": 0.10,    # Scale Variation
    "LR": 0.05,    # Low Resolution
    "ARC": 0.05,   # Aspect Ratio Change
    "DEF": 0.05,   # Deformable
}

# Map challenges to trajectory generation strategies
_CHALLENGE_STRATEGY = {
    "POC": "crossing",       # Cross behind static occluders
    "FOC": "leave_return",   # Leave entirely then return
    "OV": "edge_drift",      # Drift toward room edges
    "BC": "smooth_curve",    # Normal motion in clutter
    "MB": "fast_motion",     # High-speed segments
    "SV": "approach_recede", # Move toward/away from cameras
    "LR": "far_away",        # Stay distant
    "ARC": "rotation_heavy", # Lots of rotation
    "DEF": "depth_oscillate", # Oscillating depth motion
}


# ---------------------------------------------------------------------------
# Smooth motion helpers
# ---------------------------------------------------------------------------

def _smooth_component(n_frames, rng, speed_scale=1.0, freq_scale=1.0):
    """Generate a smooth 1D trajectory via superimposed sinusoids + drift."""
    t = np.arange(n_frames, dtype=np.float64)
    val = np.zeros(n_frames)

    # 2-3 sinusoidal components at different frequencies
    for _ in range(rng.randint(2, 3)):
        freq = rng.uniform(0.5, 3.0) * freq_scale / n_frames
        amp = rng.uniform(0.5, 1.0) * speed_scale
        phase = rng.uniform(0, 2 * np.pi)
        val += amp * np.sin(2 * np.pi * freq * t + phase)

    # Small random walk perturbation
    noise = np.cumsum(np.array([rng.gauss(0, 0.03 * speed_scale) for _ in range(n_frames)]))
    val += noise

    return val


def _apply_room_clamp(x, z, room_x, room_z, margin=0.8):
    """Clamp positions to stay inside room bounds."""
    x = np.clip(x, margin, room_x - margin)
    z = np.clip(z, margin, room_z - margin)
    return x, z


# ---------------------------------------------------------------------------
# Trajectory generators (one per strategy)
# ---------------------------------------------------------------------------

def _gen_smooth_curve(n_frames, fps, room_x, room_z, room_h, rng):
    """Basic smooth random walk — fast, covering most of the room."""
    cx, cz = room_x / 2, room_z / 2
    x = cx + _smooth_component(n_frames, rng, speed_scale=room_x * 0.6)
    z = cz + _smooth_component(n_frames, rng, speed_scale=room_z * 0.6)
    x, z = _apply_room_clamp(x, z, room_x, room_z)
    y = np.full(n_frames, room_h * 0.3)
    yaw = np.cumsum([rng.gauss(0, 2.0) for _ in range(n_frames)]) % 360
    return x, y, z, yaw


def _gen_crossing(n_frames, fps, room_x, room_z, room_h, rng, occluders=None):
    """Cross back and forth through occluder zones."""
    cx, cz = room_x / 2, room_z / 2
    # Determine crossing direction
    if occluders:
        # Use occluder positions to define crossing path
        all_occ_xy = []
        for occ in occluders:
            bbox = occ.get("bbox_3d", [[0, 0, 0], [1, 1, 1]])
            ox = (bbox[0][0] + bbox[1][0]) / 2
            oz = (bbox[0][1] + bbox[1][1]) / 2
            all_occ_xy.append((ox, oz))
        if all_occ_xy:
            # Cross through a random occluder
            ox, oz = rng.choice(all_occ_xy)
            cx, cz = ox, oz

    # Generate crossing path: sweep across room through occluder zone
    n_crossings = rng.randint(3, 5)
    x = np.zeros(n_frames)
    z = np.zeros(n_frames)
    seg_len = n_frames // n_crossings

    for i in range(n_crossings):
        s = i * seg_len
        e = min(s + seg_len, n_frames)
        t = np.linspace(0, 1, e - s)
        if i % 2 == 0:
            x[s:e] = cx + (room_x * 0.7) * (2 * t - 1)
        else:
            x[s:e] = cx - (room_x * 0.7) * (2 * t - 1)
        z[s:e] = cz + _smooth_component(e - s, rng, speed_scale=room_z * 0.4)

    x, z = _apply_room_clamp(x, z, room_x, room_z)
    y = np.full(n_frames, room_h * 0.3)
    yaw = np.linspace(0, 360 * n_crossings, n_frames) % 360
    return x, y, z, yaw


def _gen_leave_return(n_frames, fps, room_x, room_z, room_h, rng):
    """Leave toward a wall, pause, return — faster."""
    cx, cz = room_x / 2, room_z / 2

    # Phase 1: move to center (10%)
    # Phase 2: leave toward wall (25%)
    # Phase 3: stay at wall (15%)
    # Phase 4: return to center (35%)
    # Phase 5: stay in center (15%)
    phases = [0.10, 0.25, 0.15, 0.35, 0.15]
    wall_side = rng.choice(["left", "right", "near", "far"])
    if wall_side == "left":
        wall_x, wall_z = 0.4, cz
    elif wall_side == "right":
        wall_x, wall_z = room_x - 0.4, cz
    elif wall_side == "near":
        wall_x, wall_z = cx, 0.4
    else:
        wall_x, wall_z = cx, room_z - 0.4

    x = np.zeros(n_frames)
    z = np.zeros(n_frames)
    idx = 0
    waypoints = [(cx, cz), (wall_x, wall_z), (wall_x, wall_z), (cx, cz), (cx, cz)]

    for phase_i, frac in enumerate(phases):
        seg_len = int(n_frames * frac)
        s = idx
        e = min(s + seg_len, n_frames)
        if e <= s:
            break
        wx, wz = waypoints[phase_i]
        px, pz = waypoints[max(0, phase_i - 1)]
        t = np.linspace(0, 1, e - s)
        # Smooth interpolation
        t_smooth = t * t * (3 - 2 * t)  # smoothstep
        x[s:e] = px + (wx - px) * t_smooth
        z[s:e] = pz + (wz - pz) * t_smooth
        idx = e

    # Fill remaining
    if idx < n_frames:
        x[idx:] = x[idx - 1]
        z[idx:] = z[idx - 1]

    x, z = _apply_room_clamp(x, z, room_x, room_z)
    y = np.full(n_frames, room_h * 0.3)
    yaw = np.full(n_frames, rng.uniform(0, 360))
    return x, y, z, yaw


def _gen_edge_drift(n_frames, fps, room_x, room_z, room_h, rng):
    """Drift around room — fast movement covering wide area."""
    cx, cz = room_x / 2, room_z / 2
    x = cx + _smooth_component(n_frames, rng, speed_scale=room_x * 0.5)
    z = cz + _smooth_component(n_frames, rng, speed_scale=room_z * 0.5)

    # Slight bias toward a side but NOT to the very edge
    edge_x = rng.choice([room_x * 0.3, room_x * 0.7])
    edge_z = rng.choice([room_z * 0.3, room_z * 0.7])
    edge_bias_start = n_frames // 2
    for i in range(edge_bias_start, n_frames):
        t = (i - edge_bias_start) / (n_frames - edge_bias_start)
        bias = min(t * 0.3, 0.4)
        x[i] = x[i] * (1 - bias) + edge_x * bias
        z[i] = z[i] * (1 - bias) + edge_z * bias

    x, z = _apply_room_clamp(x, z, room_x, room_z)
    y = np.full(n_frames, room_h * 0.3)
    yaw = np.cumsum([rng.gauss(0, 2.0) for _ in range(n_frames)]) % 360
    return x, y, z, yaw


def _gen_fast_motion(n_frames, fps, room_x, room_z, room_h, rng):
    """Alternating fast/slow segments for motion blur."""
    cx, cz = room_x / 2, room_z / 2
    x = np.zeros(n_frames)
    z = np.zeros(n_frames)

    # Alternate between fast (speed=3x) and slow segments
    seg_len = rng.randint(20, 60)
    fast = False
    pos_x, pos_z = cx, cz
    yaw_val = rng.uniform(0, 360)

    for s in range(0, n_frames, seg_len):
        e = min(s + seg_len, n_frames)
        length = e - s
        speed = rng.uniform(2.0, 4.0) if fast else rng.uniform(0.2, 0.5)
        dx = speed * np.cos(np.radians(yaw_val)) / fps
        dz = speed * np.sin(np.radians(yaw_val)) / fps
        for i in range(s, e):
            pos_x += dx + rng.gauss(0, 0.01)
            pos_z += dz + rng.gauss(0, 0.01)
            x[i] = pos_x
            z[i] = pos_z
        if fast:
            yaw_val += rng.uniform(-60, 60)
        fast = not fast

    x, z = _apply_room_clamp(x, z, room_x, room_z)
    y = np.full(n_frames, room_h * 0.3)
    yaw = np.full(n_frames, yaw_val % 360)
    return x, y, z, yaw


def _gen_approach_recede(n_frames, fps, room_x, room_z, room_h, rng, cameras=None):
    """Move toward and away from camera positions to vary scale."""
    cx, cz = room_x / 2, room_z / 2

    # Pick a camera to approach
    if cameras:
        cam = rng.choice(cameras)
        # Camera target position (approximate from config)
        cam_dist = cam.get("distance_m", 3.0)
        cam_az = math.radians(cam.get("azimuth_deg", 0))
        cam_target_x = cx + cam_dist * math.cos(cam_az)
        cam_target_z = cz + cam_dist * math.sin(cam_az)
    else:
        cam_target_x = rng.uniform(0.5, room_x - 0.5)
        cam_target_z = rng.uniform(0.5, room_z - 0.5)

    # Oscillate between center and camera direction
    x = np.zeros(n_frames)
    z = np.zeros(n_frames)
    n_cycles = rng.randint(2, 4)
    t = np.linspace(0, 2 * np.pi * n_cycles, n_frames)
    osc = 0.5 * (1 + np.sin(t))  # 0 to 1

    for i in range(n_frames):
        x[i] = cx + (cam_target_x - cx) * osc[i] * 0.6
        z[i] = cz + (cam_target_z - cz) * osc[i] * 0.6
        x[i] += rng.gauss(0, 0.05)
        z[i] += rng.gauss(0, 0.05)

    x, z = _apply_room_clamp(x, z, room_x, room_z)
    y = np.full(n_frames, room_h * 0.3)
    yaw = np.linspace(0, 360 * n_cycles, n_frames) % 360
    return x, y, z, yaw


def _gen_far_away(n_frames, fps, room_x, room_z, room_h, rng):
    """Stay away from cameras but NOT in corners (avoid wall blocking)."""
    # Pick a position at moderate distance from center, not in a corner
    cx, cz = room_x / 2, room_z / 2
    # Offset from center by 30-50% of room size
    dx = rng.uniform(-0.3, 0.3) * room_x
    dz = rng.uniform(-0.3, 0.3) * room_z
    base_x = cx + dx
    base_z = cz + dz

    x = base_x + _smooth_component(n_frames, rng, speed_scale=0.5)
    z = base_z + _smooth_component(n_frames, rng, speed_scale=0.5)
    x, z = _apply_room_clamp(x, z, room_x, room_z)
    y = np.full(n_frames, room_h * 0.25)
    yaw = np.cumsum([rng.gauss(0, 0.5) for _ in range(n_frames)]) % 360
    return x, y, z, yaw


def _gen_rotation_heavy(n_frames, fps, room_x, room_z, room_h, rng):
    """Stay mostly in place but rotate heavily."""
    cx, cz = room_x / 2, room_z / 2
    x = cx + _smooth_component(n_frames, rng, speed_scale=0.2)
    z = cz + _smooth_component(n_frames, rng, speed_scale=0.2)
    x, z = _apply_room_clamp(x, z, room_x, room_z)
    y = np.full(n_frames, room_h * 0.3)

    # Heavy continuous rotation with variable speed
    yaw = np.cumsum(np.abs([rng.gauss(3.0, 2.0) for _ in range(n_frames)])) % 360

    return x, y, z, yaw


def _gen_depth_oscillate(n_frames, fps, room_x, room_z, room_h, rng):
    """Oscillating motion along depth axis (z)."""
    cx, cz = room_x / 2, room_z / 2

    x = cx + _smooth_component(n_frames, rng, speed_scale=0.3)
    # Strong z oscillation
    n_cycles = rng.randint(3, 6)
    t = np.linspace(0, 2 * np.pi * n_cycles, n_frames)
    z = cz + (room_z * 0.25) * np.sin(t) + np.array([rng.gauss(0, 0.05) for _ in range(n_frames)])
    x, z = _apply_room_clamp(x, z, room_x, room_z)
    y = np.full(n_frames, room_h * 0.3)
    yaw = np.cumsum([rng.gauss(0, 0.8) for _ in range(n_frames)]) % 360
    return x, y, z, yaw


_GENERATORS = {
    "smooth_curve": _gen_smooth_curve,
    "crossing": _gen_crossing,
    "leave_return": _gen_leave_return,
    "edge_drift": _gen_edge_drift,
    "fast_motion": _gen_fast_motion,
    "approach_recede": _gen_approach_recede,
    "far_away": _gen_far_away,
    "rotation_heavy": _gen_rotation_heavy,
    "depth_oscillate": _gen_depth_oscillate,
}


# ---------------------------------------------------------------------------
# Rotation and scale generation
# ---------------------------------------------------------------------------

def _generate_rotations(yaw, n_frames, rng, challenge):
    """Generate per-frame [rx, ry, rz] rotations in degrees."""
    ry = yaw  # primary yaw rotation

    rx = np.zeros(n_frames)
    rz = np.zeros(n_frames)

    if challenge == "ARC":
        # Heavy roll and pitch variation
        rx = 15 * np.sin(np.linspace(0, 4 * np.pi, n_frames)) + np.array([rng.gauss(0, 3) for _ in range(n_frames)])
        rz = 10 * np.cos(np.linspace(0, 6 * np.pi, n_frames)) + np.array([rng.gauss(0, 2) for _ in range(n_frames)])
    elif challenge == "DEF":
        # Gentle rocking
        rx = 5 * np.sin(np.linspace(0, 8 * np.pi, n_frames))
        rz = 3 * np.cos(np.linspace(0, 10 * np.pi, n_frames))
    else:
        # Minimal perturbation
        rx = np.array([rng.gauss(0, 1.0) for _ in range(n_frames)])
        rz = np.array([rng.gauss(0, 0.5) for _ in range(n_frames)])

    rotations = np.column_stack([rx, ry, rz])
    return rotations


def _generate_scales(n_frames, rng, challenge):
    """Generate per-frame [sx, sy, sz] scale factors."""
    if challenge == "SV":
        # Oscillating scale
        n_cycles = rng.randint(2, 4)
        t = np.linspace(0, 2 * np.pi * n_cycles, n_frames)
        base = 0.7 + 0.6 * (0.5 + 0.5 * np.sin(t))  # range [0.7, 1.3]
        scales = np.column_stack([base, base, base])
    elif challenge == "DEF":
        # Slight deformation (non-uniform scale)
        t = np.linspace(0, 6 * np.pi, n_frames)
        sx = 1.0 + 0.1 * np.sin(t)
        sy = 1.0 + 0.05 * np.cos(t * 1.3)
        sz = 1.0 + 0.08 * np.sin(t * 0.7)
        scales = np.column_stack([sx, sy, sz])
    else:
        # Constant scale with tiny noise
        base = np.ones(n_frames) + np.array([rng.gauss(0, 0.01) for _ in range(n_frames)])
        scales = np.column_stack([base, base, base])

    return scales


# ---------------------------------------------------------------------------
# Visibility projection (for validation)
# ---------------------------------------------------------------------------

def _project_target(target_pos, camera, resolution, target_xy=None):
    """Project target position to image coords. Returns (u, v) or None."""
    from sample_camera_rig import compute_camera_pose, compute_intrinsics, project_to_image

    K, _, _, _, _ = compute_intrinsics(camera["fov_deg"], resolution[0], resolution[1])
    if target_xy is None:
        target_xy = [0, 0]
    cam_pos, R_c2w = compute_camera_pose(
        camera["azimuth_deg"], camera["elevation_deg"],
        camera["distance_m"], target_xy, camera["height_m"],
    )
    return project_to_image(target_pos, cam_pos, R_c2w, K, resolution)


def _estimate_apparent_size(target_pos, target_scale, camera, resolution, target_xy=None):
    """Estimate target apparent size in pixels. Returns pixel area or 0."""
    from sample_camera_rig import compute_camera_pose, compute_intrinsics

    K, fx, fy, _, _ = compute_intrinsics(camera["fov_deg"], resolution[0], resolution[1])
    if target_xy is None:
        target_xy = [0, 0]
    cam_pos, _ = compute_camera_pose(
        camera["azimuth_deg"], camera["elevation_deg"],
        camera["distance_m"], target_xy, camera["height_m"],
    )

    dist = np.linalg.norm(np.array(target_pos) - np.array(cam_pos))
    if dist < 0.1:
        return resolution[0] * resolution[1]

    # Approximate: target longest side * scale projected at distance
    longest = 0.3 * (target_scale if isinstance(target_scale, (int, float)) else max(target_scale))
    pixel_size = fx * longest / dist
    return pixel_size * pixel_size


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _find_difficulty_segments(frames_data, challenge, cameras, resolution, target_xy=None):
    """Identify frames that should be 'difficult' based on challenge type.

    Returns set of frame indices that are challenging.
    """
    n_frames = len(frames_data["positions_m"])
    difficult_frames = set()

    if challenge in ("POC", "FOC"):
        # Frames near occluders or fully hidden
        for i in range(n_frames):
            pos = frames_data["positions_m"][i]
            # Check if any camera can't see this position
            hidden_count = 0
            for cam in cameras:
                uv = _project_target(pos, cam, resolution, target_xy)
                if uv is None:
                    hidden_count += 1
            if challenge == "POC" and hidden_count >= 1:
                difficult_frames.add(i)
            elif challenge == "FOC" and hidden_count == len(cameras):
                difficult_frames.add(i)

    elif challenge == "OV":
        # Frames where target is near/outside image border
        for i in range(n_frames):
            pos = frames_data["positions_m"][i]
            border_count = 0
            for cam in cameras:
                uv = _project_target(pos, cam, resolution, target_xy)
                if uv is None:
                    border_count += 1
                else:
                    margin = 30  # pixels
                    if (uv[0] < margin or uv[0] > resolution[0] - margin or
                            uv[1] < margin or uv[1] > resolution[1] - margin):
                        border_count += 1
            if border_count >= 1:
                difficult_frames.add(i)

    elif challenge == "MB":
        # Frames with high velocity (consecutive position change)
        for i in range(1, n_frames):
            p0 = np.array(frames_data["positions_m"][i - 1])
            p1 = np.array(frames_data["positions_m"][i])
            vel = np.linalg.norm(p1 - p0) * 30  # *fps to get m/s
            if vel > 2.0:  # > 2 m/s
                difficult_frames.add(i)

    elif challenge == "SV":
        # Frames where target is far from all cameras (small apparent size)
        for i in range(n_frames):
            pos = frames_data["positions_m"][i]
            scale = frames_data["scales"][i][0]
            min_area = float("inf")
            for cam in cameras:
                area = _estimate_apparent_size(pos, scale, cam, resolution, target_xy)
                min_area = min(min_area, area)
            if min_area < 400:  # < 20x20 pixels
                difficult_frames.add(i)

    elif challenge == "LR":
        # Almost all frames are hard (target is far away)
        for i in range(n_frames):
            pos = frames_data["positions_m"][i]
            scale = frames_data["scales"][i][0]
            max_visible_area = 0
            for cam in cameras:
                area = _estimate_apparent_size(pos, scale, cam, resolution, target_xy)
                max_visible_area = max(max_visible_area, area)
            if max_visible_area < 900:  # < 30x30
                difficult_frames.add(i)

    elif challenge == "ARC":
        # Frames where bbox aspect ratio is extreme
        for i in range(n_frames):
            rot = frames_data["rotations"][i]
            pitch = abs(rot[0])
            roll = abs(rot[2])
            if pitch > 20 or roll > 15:
                difficult_frames.add(i)

    elif challenge in ("BC", "DEF"):
        # For these, use a random 30% segment as "difficult"
        seg_start = int(n_frames * 0.3)
        seg_end = int(n_frames * 0.6)
        for i in range(seg_start, seg_end):
            difficult_frames.add(i)

    return difficult_frames


def validate_trajectory(frames_data, cameras, scene, cfg, target_xy=None):
    """Validate trajectory meets quality requirements.

    Returns (is_valid, list_of_failure_reasons).
    """
    resolution = cfg["render"]["resolution"]
    n_frames = len(frames_data["positions_m"])
    challenge = frames_data["main_challenge"]
    failures = []

    # Find difficulty frames
    diff_frames = _find_difficulty_segments(
        frames_data, challenge, cameras, resolution, target_xy,
    )

    # Check 1: At least one view has difficulty segment
    if len(diff_frames) == 0:
        failures.append(f"challenge={challenge} but no difficulty frames found")

    # Check 2: During difficulty, at least one other view is visible
    if diff_frames:
        sample_diff = list(diff_frames)[::max(1, len(diff_frames) // 10)]
        for fi in sample_diff:
            pos = frames_data["positions_m"][fi]
            visible_count = 0
            for cam in cameras:
                uv = _project_target(pos, cam, resolution, target_xy)
                if uv is not None:
                    visible_count += 1
            if visible_count < 1:
                failures.append(
                    f"frame {fi}: target invisible from all views during difficulty"
                )
                break

    # Check 3: Target must be visible from ALL views in majority of non-difficulty frames
    non_diff_indices = [i for i in range(n_frames) if i not in diff_frames]
    if non_diff_indices:
        sample_nd = non_diff_indices[::max(1, len(non_diff_indices) // 20)]
        n_views = len(cameras)
        bad_frames = 0
        for fi in sample_nd:
            pos = frames_data["positions_m"][fi]
            visible_count = 0
            for cam in cameras:
                uv = _project_target(pos, cam, resolution, target_xy)
                if uv is not None:
                    visible_count += 1
            if visible_count < n_views:
                bad_frames += 1
        if bad_frames > len(sample_nd) * 0.3:
            failures.append(
                f"target not visible from all views in {bad_frames}/{len(sample_nd)} "
                f"sampled non-difficulty frames"
            )

    # Check 4: Target not too small the entire sequence
    sample_count = len(range(0, n_frames, max(1, n_frames // 20)))
    tiny_count = 0
    for i in range(0, n_frames, max(1, n_frames // 20)):
        pos = frames_data["positions_m"][i]
        scale = frames_data["scales"][i][0]
        max_area = 0
        for cam in cameras:
            area = _estimate_apparent_size(pos, scale, cam, resolution, target_xy)
            max_area = max(max_area, area)
        if max_area < 100:  # < 10x10 pixels
            tiny_count += 1
    if tiny_count >= sample_count * 0.8:
        failures.append("target too small (<10x10px) in >80% of frames")

    return len(failures) == 0, failures


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def sample_trajectory(scene, target, cameras, cfg, rng=None):
    """Sample a challenge-aware trajectory with validation.

    Args:
        scene: scene dict (needs room_size_m, static_occluders)
        target: target dict
        cameras: list of camera dicts
        cfg: dataset config dict
        rng: random.Random instance

    Returns:
        trajectory dict with keys:
            positions_m: list of [x, y, z] per frame
            rotations: list of [rx, ry, rz] degrees per frame
            scales: list of [sx, sy, sz] per frame
            main_challenge: str
            trajectory_type: str (for compatibility)
            num_frames: int
            fps: int
    """
    if rng is None:
        rng = random.Random()

    n_frames = cfg["dataset"]["frames_per_sequence"]
    fps = cfg["dataset"]["fps"]
    room = scene.get("room_size_m", [5.0, 2.8, 5.0])
    room_x, room_h, room_z = room[0], room[1], room[2]
    occluders = scene.get("static_occluders", [])

    # Sample challenge
    challenges = list(CHALLENGE_DIST.keys())
    weights = list(CHALLENGE_DIST.values())
    main_challenge = rng.choices(challenges, weights=weights, k=1)[0]

    # Map challenge to trajectory strategy
    strategy = _CHALLENGE_STRATEGY[main_challenge]
    gen_func = _GENERATORS[strategy]

    max_attempts = 30
    resolution = cfg["render"]["resolution"]
    target_xy = [room_x / 2, room_z / 2]  # room center

    for attempt in range(max_attempts):
        # Generate positions
        if strategy == "crossing":
            x, y, z, yaw = gen_func(n_frames, fps, room_x, room_z, room_h, rng,
                                     occluders=occluders)
        elif strategy == "approach_recede":
            x, y, z, yaw = gen_func(n_frames, fps, room_x, room_z, room_h, rng,
                                     cameras=cameras)
        else:
            x, y, z, yaw = gen_func(n_frames, fps, room_x, room_z, room_h, rng)

        # Generate rotations and scales
        rotations = _generate_rotations(yaw, n_frames, rng, main_challenge)
        scales = _generate_scales(n_frames, rng, main_challenge)

        # Build frames data
        positions = np.column_stack([x, y, z])
        frames_data = {
            "positions_m": positions.tolist(),
            "rotations": rotations.tolist(),
            "scales": scales.tolist(),
            "main_challenge": main_challenge,
        }

        # Validate
        valid, reasons = validate_trajectory(frames_data, cameras, scene, cfg, target_xy)
        if valid:
            break
    else:
        # Exhausted attempts; proceed with last result
        pass

    # Build final output
    trajectory = {
        "positions_m": [[round(p, 4) for p in pos] for pos in frames_data["positions_m"]],
        "rotations": [[round(r, 2) for r in rot] for rot in frames_data["rotations"]],
        "scales": [[round(s, 4) for s in sc] for sc in frames_data["scales"]],
        "main_challenge": main_challenge,
        "trajectory_type": strategy,
        "num_frames": n_frames,
        "fps": fps,
    }

    return trajectory
