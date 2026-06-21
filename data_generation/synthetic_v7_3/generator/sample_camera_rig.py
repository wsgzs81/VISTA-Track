#!/usr/bin/env python3
"""
sample_camera_rig.py - Multi-view camera rig sampling with validation.

Generates camera poses around a target activity area, validates coverage
and angular diversity, and exports calibs.json with intrinsics/extrinsics.
"""

import math
import random

import numpy as np


# ---------------------------------------------------------------------------
# Core camera math
# ---------------------------------------------------------------------------

def compute_intrinsics(fov_deg, width, height):
    """Compute 3x3 intrinsic matrix from horizontal FOV (degrees).

    Returns K as list-of-lists, (fx, fy, cx, cy).
    """
    fx = width / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    fy = fx  # square pixels
    cx = width / 2.0
    cy = height / 2.0
    K = [
        [round(fx, 4), 0.0, round(cx, 4)],
        [0.0, round(fy, 4), round(cy, 4)],
        [0.0, 0.0, 1.0],
    ]
    return K, fx, fy, cx, cy


def compute_camera_pose(azimuth_deg, elevation_deg, distance_m, target_xy, height_m):
    """Compute camera world position and rotation matrix.

    Camera looks toward target from azimuth/elevation/distance.

    Coordinate system: x=east, y=up, z=south (y is vertical).
    Azimuth 0 = +x direction, increases CCW when viewed from above.

    Returns:
        position: [x, y, z] camera center in world coords
        R: 3x3 rotation matrix (camera-to-world)
    """
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)

    # Look direction (from camera toward target)
    dx = math.cos(el) * math.cos(az)
    dy = math.sin(el)
    dz = math.cos(el) * math.sin(az)
    look = np.array([dx, dy, dz])

    # Camera position: offset from target along negative look direction
    cam_pos = np.array([
        target_xy[0] - distance_m * dx,
        height_m - distance_m * dy,  # y is up
        target_xy[1] - distance_m * dz,
    ])

    # Build camera frame
    # Camera z-axis = look direction (camera looks along +z in camera coords)
    z_axis = look / np.linalg.norm(look)

    # World up
    world_up = np.array([0.0, 1.0, 0.0])

    # Camera x-axis = right (cross of look and up)
    x_axis = np.cross(world_up, z_axis)
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-6:
        # Degenerate: looking straight up/down
        x_axis = np.array([1.0, 0.0, 0.0])
        x_norm = 1.0
    x_axis = x_axis / x_norm

    # Camera y-axis = down in camera frame
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)

    # R maps camera coords -> world coords
    R = np.column_stack([x_axis, y_axis, z_axis])

    return cam_pos.tolist(), R


def compute_extrinsics(R_c2w, cam_pos):
    """Compute world-to-camera extrinsic matrix.

    Returns:
        R_ext: 3x3 rotation (world to camera) as list-of-lists
        T_ext: 3x1 translation as list
    """
    R_ext = R_c2w.T  # camera-to-world -> world-to-camera
    T_ext = -R_ext @ np.array(cam_pos)
    return R_ext.tolist(), T_ext.tolist()


# ---------------------------------------------------------------------------
# Visibility projection
# ---------------------------------------------------------------------------

def project_to_image(world_pt, cam_pos, R_c2w, K, resolution):
    """Project a 3D world point to 2D image coordinates.

    Returns (u, v) in pixel coords, or None if behind camera.
    """
    pt = np.array(world_pt)
    pos = np.array(cam_pos)

    # World to camera
    R_ext = R_c2w.T
    T_ext = -R_ext @ pos
    cam_pt = R_ext @ pt + T_ext

    # Behind camera
    if cam_pt[2] <= 0.1:
        return None

    # Camera to image
    fx, fy = K[0][0], K[1][1]
    cx, cy = K[0][2], K[1][2]
    u = fx * cam_pt[0] / cam_pt[2] + cx
    v = fy * cam_pt[1] / cam_pt[2] + cy

    w, h = resolution
    if 0 <= u < w and 0 <= v < h:
        return (round(u, 1), round(v, 1))
    return None


# ---------------------------------------------------------------------------
# Validation checks
# ---------------------------------------------------------------------------

def validate_rig(cameras, target_xy, resolution):
    """Validate camera rig against coverage and diversity requirements.

    Returns (is_valid, list_of_failure_reasons).
    """
    failures = []

    # Check 1: ALL views must see target activity area
    target_3d = [target_xy[0], 0.0, target_xy[1]]  # ground-level target
    for cam in cameras:
        uv = project_to_image(
            target_3d, cam["_pos"], cam["_R"], cam["K"], resolution,
        )
        if uv is None:
            failures.append(f"cam{cam.get('view_id', '?')} cannot see target center")

    # Also check a point slightly above ground (where target actually is)
    target_3d_elevated = [target_xy[0], 0.5, target_xy[1]]
    elevated_fail = 0
    for cam in cameras:
        uv = project_to_image(
            target_3d_elevated, cam["_pos"], cam["_R"], cam["K"], resolution,
        )
        if uv is None:
            elevated_fail += 1
    if elevated_fail > 0:
        failures.append(f"{elevated_fail} views cannot see elevated target (0.5m)")

    # Check 2: Camera overlap (pairwise angular separation < 90 deg)
    # Use camera positions relative to target center
    for i in range(len(cameras)):
        for j in range(i + 1, len(cameras)):
            pi = np.array(cameras[i]["_pos"])
            pj = np.array(cameras[j]["_pos"])
            target_3d_np = np.array([target_xy[0], 0.0, target_xy[1]])
            di = pi - target_3d_np
            dj = pj - target_3d_np
            ni, nj = np.linalg.norm(di), np.linalg.norm(dj)
            if ni < 1e-6 or nj < 1e-6:
                continue
            cos_a = np.dot(di, dj) / (ni * nj)
            angle = math.degrees(math.acos(max(-1, min(1, cos_a))))
            if angle > 90:
                failures.append(
                    f"cam{i}-cam{j} angle={angle:.0f}deg (too far, no overlap)"
                )

    # Check 3: Not all cameras in same direction (handle wrap-around)
    azimuths = sorted(cam["_azimuth"] for cam in cameras)
    # Max gap between consecutive azimuths (with wrap-around)
    gaps = [azimuths[i + 1] - azimuths[i] for i in range(len(azimuths) - 1)]
    gaps.append(360 - azimuths[-1] + azimuths[0])
    max_gap = max(gaps)
    # If the largest gap is > 300 deg, cameras are clustered in one direction
    if max_gap > 300:
        failures.append(f"max azimuth gap={max_gap:.0f}deg (cameras too clustered)")

    # Check 4: Not all top-down
    elevations = [cam["_elevation"] for cam in cameras]
    if all(e > 20 for e in elevations):
        failures.append("all cameras are top-down (>20deg)")

    # Check 5: Not all low-angle
    if all(e < -30 for e in elevations):
        failures.append("all cameras are low-angle (<-30deg)")

    return len(failures) == 0, failures


# ---------------------------------------------------------------------------
# Main sampling
# ---------------------------------------------------------------------------

def sample_camera_rig(scene, cfg, rng=None):
    """Sample a multi-view camera rig with validation.

    Args:
        scene: scene dict (needs room_size_m or world_bounds_m)
        cfg: dataset config dict (needs camera, render, dataset sections)
        rng: random.Random instance

    Returns:
        list of camera dicts, each with:
            view_id, K, R, T, image_size, unit,
            azimuth_deg, elevation_deg, distance_m, height_m, fov_deg
    """
    if rng is None:
        rng = random.Random()

    cam_cfg = cfg["camera"]
    res = cfg["render"]["resolution"]
    views_choices = cfg["dataset"]["views_choices"]

    # Number of views (weighted choice)
    num_views = int(rng.choices(
        list(views_choices.keys()),
        weights=list(views_choices.values()),
        k=1,
    )[0])

    # Target activity area: center of room (xy plane)
    room = scene.get("room_size_m", [5.0, 2.8, 5.0])
    target_xy = [room[0] / 2, room[2] / 2]

    # Ranges
    h_min, h_max = cam_cfg["height_range_m"]
    d_min, d_max = cam_cfg["distance_range_m"]
    fov_min, fov_max = cam_cfg["fov_range_deg"]
    el_min, el_max = cam_cfg.get("elevation_range_deg", [-45, 10])

    max_attempts = 50
    for attempt in range(max_attempts):
        cameras = []

        # Spread azimuths evenly with jitter
        base_az = rng.uniform(0, 360)
        for i in range(num_views):
            az = (base_az + 360 * i / num_views + rng.uniform(-30, 30)) % 360
            el = rng.uniform(el_min, el_max)
            dist = rng.uniform(d_min, d_max)
            height = rng.uniform(h_min, h_max)
            fov = rng.uniform(fov_min, fov_max)

            K, fx, fy, cx, cy = compute_intrinsics(fov, res[0], res[1])
            cam_pos, R_c2w = compute_camera_pose(az, el, dist, target_xy, height)
            R_ext, T_ext = compute_extrinsics(R_c2w, cam_pos)

            cameras.append({
                "view_id": i,
                "azimuth_deg": round(az, 1),
                "elevation_deg": round(el, 1),
                "distance_m": round(dist, 2),
                "height_m": round(height, 2),
                "fov_deg": round(fov, 1),
                "K": K,
                "R": [[round(v, 6) for v in row] for row in R_ext],
                "T": [round(v, 4) for v in T_ext],
                "image_size": res,
                "unit": "m",
                # Internal fields for validation
                "_pos": cam_pos,
                "_R": R_c2w,
                "_azimuth": az,
                "_elevation": el,
            })

        valid, reasons = validate_rig(cameras, target_xy, res)
        if valid:
            break
    else:
        # Exhausted attempts; use last generated set
        pass

    # Strip internal fields before returning
    for cam in cameras:
        del cam["_pos"]
        del cam["_R"]
        del cam["_azimuth"]
        del cam["_elevation"]

    return cameras
