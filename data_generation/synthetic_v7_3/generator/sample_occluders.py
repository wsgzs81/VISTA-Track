#!/usr/bin/env python3
"""
sample_occluders.py - Challenge-aware occluder sampling with per-camera visibility.

Places occluders (furniture, pillars, walls, human proxies, similar objects)
along the target path to produce POC/FOC/OV frame-level occlusion labels.
"""

import math
import random

import numpy as np


# ---------------------------------------------------------------------------
# Line-of-sight occlusion checking
# ---------------------------------------------------------------------------

def _ray_aabb_intersect(ray_o, ray_d, box_lo, box_hi):
    """Check ray-AABB intersection. Returns t_min or None."""
    t_min = -float("inf")
    t_max = float("inf")

    for i in range(3):
        if abs(ray_d[i]) < 1e-9:
            if ray_o[i] < box_lo[i] or ray_o[i] > box_hi[i]:
                return None
        else:
            t1 = (box_lo[i] - ray_o[i]) / ray_d[i]
            t2 = (box_hi[i] - ray_o[i]) / ray_d[i]
            if t1 > t2:
                t1, t2 = t2, t1
            t_min = max(t_min, t1)
            t_max = min(t_max, t2)
            if t_min > t_max:
                return None

    if t_max < 0:
        return None
    return t_min if t_min > 0 else t_max


def compute_per_camera_occlusion(positions, cameras, occluder_bboxes, target_xy):
    """Compute per-frame, per-camera occlusion mask.

    Args:
        positions: list of [x, y, z] target positions
        cameras: list of camera dicts (with K, R, T, azimuth_deg, etc.)
        occluder_bboxes: list of [[lox, loy, loz], [hix, hiy, hiz]]
        target_xy: [room_x/2, room_z/2] for camera pose computation

    Returns:
        list of list: frame_occlusion[cam_idx][frame_idx] = True if occluded
    """
    from sample_camera_rig import compute_camera_pose

    n_frames = len(positions)
    n_cams = len(cameras)
    result = [[False] * n_frames for _ in range(n_cams)]

    if not occluder_bboxes:
        return result

    # Pre-compute camera positions
    cam_positions = []
    for cam in cameras:
        pos, _ = compute_camera_pose(
            cam["azimuth_deg"], cam["elevation_deg"],
            cam["distance_m"], target_xy, cam["height_m"],
        )
        cam_positions.append(pos)

    for ci in range(n_cams):
        cam_pos = np.array(cam_positions[ci])
        for fi in range(n_frames):
            target_pos = np.array(positions[fi])
            ray = target_pos - cam_pos
            ray_len = np.linalg.norm(ray)
            if ray_len < 1e-6:
                continue
            ray_dir = ray / ray_len

            for bbox in occluder_bboxes:
                lo = np.array(bbox[0])
                hi = np.array(bbox[1])
                t = _ray_aabb_intersect(cam_pos, ray_dir, lo, hi)
                # Occluder blocks if intersection is between camera and target
                if t is not None and t < ray_len - 0.05:
                    result[ci][fi] = True
                    break  # one occluder is enough

    return result


# ---------------------------------------------------------------------------
# Occluder mesh builders
# ---------------------------------------------------------------------------

def _make_box_mesh(extents, center, color=(150, 150, 150)):
    """Create a trimesh box with color."""
    import trimesh
    mesh = trimesh.creation.box(extents=extents)
    mesh.apply_translation(center)
    mesh.visual = trimesh.visual.ColorVisuals()
    mesh.visual.vertex_colors = np.full(
        (len(mesh.vertices), 4), list(color) + [255], dtype=np.uint8,
    )
    return mesh


def _make_cylinder_mesh(radius, height, center, color=(140, 130, 120)):
    """Create a trimesh cylinder."""
    import trimesh
    mesh = trimesh.creation.cylinder(radius=radius, height=height, sections=12)
    mesh.apply_translation(center)
    mesh.visual = trimesh.visual.ColorVisuals()
    mesh.visual.vertex_colors = np.full(
        (len(mesh.vertices), 4), list(color) + [255], dtype=np.uint8,
    )
    return mesh


def _make_human_proxy(rng):
    """Create a simple humanoid proxy (stacked boxes).

    Returns (mesh, bbox_3d, height).
    """
    import trimesh

    height = rng.uniform(1.6, 1.9)
    shoulder_w = rng.uniform(0.35, 0.50)
    torso_d = rng.uniform(0.15, 0.25)

    parts = []
    # Legs
    leg_h = height * 0.45
    leg_w = shoulder_w * 0.35
    for dx in [-shoulder_w / 4, shoulder_w / 4]:
        leg = trimesh.creation.box(extents=[leg_w, torso_d * 0.8, leg_h])
        leg.apply_translation([dx, 0, leg_h / 2])
        parts.append(leg)

    # Torso
    torso_h = height * 0.35
    torso = trimesh.creation.box(extents=[shoulder_w, torso_d, torso_h])
    torso.apply_translation([0, 0, leg_h + torso_h / 2])
    parts.append(torso)

    # Head
    head_r = 0.1
    head = trimesh.creation.icosphere(radius=head_r, subdivisions=2)
    head.apply_translation([0, 0, leg_h + torso_h + head_r + 0.02])
    parts.append(head)

    # Color: skin/clothing tones
    color = (
        rng.randint(100, 200),
        rng.randint(80, 160),
        rng.randint(70, 140),
    )
    for p in parts:
        p.visual = trimesh.visual.ColorVisuals()
        p.visual.vertex_colors = np.full(
            (len(p.vertices), 4), list(color) + [255], dtype=np.uint8,
        )

    mesh = trimesh.util.concatenate(parts)
    return mesh, height


def _make_similar_object(target_category, rng):
    """Create an object similar in shape to the target category.

    For BC distractor-like occluders that happen to also block views.
    """
    import trimesh

    if target_category == "container":
        r = rng.uniform(0.03, 0.06)
        h = rng.uniform(0.10, 0.25)
        mesh = trimesh.creation.cylinder(radius=r, height=h, sections=10)
    elif target_category == "toy_sport":
        r = rng.uniform(0.04, 0.10)
        mesh = trimesh.creation.icosphere(radius=r, subdivisions=2)
    elif target_category == "electronics_tool":
        sx = rng.uniform(0.08, 0.20)
        sy = rng.uniform(0.03, 0.08)
        sz = rng.uniform(0.02, 0.05)
        mesh = trimesh.creation.box(extents=[sx, sy, sz])
    else:
        s = rng.uniform(0.05, 0.15)
        mesh = trimesh.creation.box(extents=[s, s, s])

    color = (rng.randint(80, 220), rng.randint(80, 220), rng.randint(80, 220))
    mesh.visual = trimesh.visual.ColorVisuals()
    mesh.visual.vertex_colors = np.full(
        (len(mesh.vertices), 4), list(color) + [255], dtype=np.uint8,
    )
    return mesh


# ---------------------------------------------------------------------------
# Occluder placement strategies
# ---------------------------------------------------------------------------

def _place_along_path(positions, rng, room_x, room_z, margin=0.2):
    """Pick a position near the target path for occluder placement."""
    # Sample a frame from the middle 60% of the sequence
    n = len(positions)
    fi = rng.randint(int(n * 0.2), int(n * 0.8))
    pos = positions[fi]

    # Offset perpendicular to a random direction
    angle = rng.uniform(0, 2 * math.pi)
    offset = rng.uniform(0.3, 0.8)
    ox = pos[0] + offset * math.cos(angle)
    oz = pos[2] + offset * math.sin(angle)

    # Clamp to room
    ox = max(margin, min(room_x - margin, ox))
    oz = max(margin, min(room_z - margin, oz))
    return ox, oz, fi


def _place_between_target_and_cam(positions, cam_idx, cameras, rng, target_xy, room_x, room_z):
    """Place occluder between target and a specific camera."""
    from sample_camera_rig import compute_camera_pose

    cam = cameras[cam_idx]
    cam_pos, _ = compute_camera_pose(
        cam["azimuth_deg"], cam["elevation_deg"],
        cam["distance_m"], target_xy, cam["height_m"],
    )

    # Pick a frame from the middle
    n = len(positions)
    fi = rng.randint(int(n * 0.15), int(n * 0.85))
    target_pos = np.array(positions[fi])

    # Place occluder at 40-70% of the way from target to camera
    t = rng.uniform(0.4, 0.7)
    interp = target_pos + t * (np.array(cam_pos) - target_pos)
    ox, oz = float(interp[0]), float(interp[2])

    margin = 0.2
    ox = max(margin, min(room_x - margin, ox))
    oz = max(margin, min(room_z - margin, oz))
    return ox, oz, fi


def _world_bbox_from_ground_rect(cx, cz, sx, depth, y0, height):
    """Return bbox in world [x, y(height), z] coordinates."""
    return [
        [round(cx - sx / 2, 3), round(y0, 3), round(cz - depth / 2, 3)],
        [round(cx + sx / 2, 3), round(y0 + height, 3), round(cz + depth / 2, 3)],
    ]


# ---------------------------------------------------------------------------
# Main sampling functions
# ---------------------------------------------------------------------------

def sample_occluders_for_challenge(
    challenge, scene, target, trajectory, cameras, cfg, rng=None,
):
    """Sample occluders appropriate for the main challenge.

    Returns list of occluder dicts with:
        name, type, bbox_3d, mesh, frame_occlusion (per-camera list of per-frame bool)
    """
    if rng is None:
        rng = random.Random()

    room = scene.get("room_size_m", [5.0, 2.8, 5.0])
    room_x, room_h, room_z = room[0], room[1], room[2]
    target_xy = [room_x / 2, room_z / 2]
    positions = trajectory["positions_m"]
    n_frames = len(positions)
    category = target.get("category", "misc")

    occluders = []

    # Include scene's static occluders (furniture) — but limit to 5 and
    # filter out any that block ALL views of the target for the ENTIRE sequence.
    static_occluders = scene.get("static_occluders", [])
    if static_occluders:
        # Sort by distance from room center (keep closest furniture)
        room_cx, room_cz = room_x / 2, room_z / 2
        scored = []
        for soc in static_occluders:
            bbox = soc.get("bbox_3d")
            if not bbox:
                continue
            lo, hi = bbox
            extents = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]]
            if any(e <= 0 for e in extents):
                continue
            cx = (lo[0] + hi[0]) / 2
            cz = (lo[2] + hi[2]) / 2
            dist = math.sqrt((cx - room_cx)**2 + (cz - room_cz)**2)
            scored.append((dist, soc, bbox, extents))

        # Keep at most 2 closest to center (leave room for challenge-specific occluders)
        scored.sort(key=lambda x: x[0])
        for _, soc, bbox, extents in scored[:2]:
            lo, hi = bbox
            mesh = _make_box_mesh(
                extents,
                [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2],
            )
            occ = {
                "name": soc.get("name", "furniture"),
                "type": "static",
                "bbox_3d": bbox,
                "mesh": mesh,
            }
            for key in ["model_path", "size_m", "position_m", "rotation", "jid"]:
                if key in soc:
                    occ[key] = soc[key]
            occluders.append(occ)

    # Add challenge-specific occluders
    if challenge == "POC":
        occluders.extend(_sample_poc_occluders(
            positions, cameras, rng, room_x, room_z, room_h,
            target_xy, category, n_frames,
        ))
    elif challenge == "FOC":
        occluders.extend(_sample_foc_occluders(
            positions, cameras, rng, room_x, room_z, room_h,
            target_xy, category, n_frames,
        ))
    elif challenge == "OV":
        occluders.extend(_sample_ov_occluders(
            positions, cameras, rng, room_x, room_z, room_h,
            target_xy, n_frames,
        ))
    elif challenge == "crossing":
        # Generic crossing occluder
        occluders.extend(_sample_poc_occluders(
            positions, cameras, rng, room_x, room_z, room_h,
            target_xy, category, n_frames,
        ))
    else:
        # All other challenges (BC, MB, SV, LR, ARC, DEF): add at least 1 occluder
        # so that some views have partial occlusion during motion
        occluders.extend(_sample_poc_occluders(
            positions, cameras, rng, room_x, room_z, room_h,
            target_xy, category, n_frames,
        ))

    # Compute per-occluder per-camera frame occlusion
    for occ in occluders:
        occ_frame_occ = compute_per_camera_occlusion(
            positions, cameras, [occ["bbox_3d"]], target_xy,
        )
        occ["frame_occlusion"] = {
            f"cam{ci}": [occ_frame_occ[ci][fi] for fi in range(n_frames)]
            for ci in range(len(cameras))
        }

    # Compute combined occlusion (any occluder blocks)
    all_bboxes = [occ["bbox_3d"] for occ in occluders]
    combined_frame_occ = compute_per_camera_occlusion(positions, cameras, all_bboxes, target_xy)

    # Also compute aggregate: any-camera occlusion per frame
    for occ in occluders:
        agg = [False] * n_frames
        for fi in range(n_frames):
            for ci in range(len(cameras)):
                if occ["frame_occlusion"][f"cam{ci}"][fi]:
                    agg[fi] = True
                    break
        occ["frame_occlusion_any"] = agg

    # Filter: remove static occluders that make target invisible from ALL views
    # for more than 50% of frames. This prevents furniture walls blocking everything.
    n_views = len(cameras)
    occluders_filtered = []
    for occ in occluders:
        if occ.get("type") != "static":
            # Keep challenge-specific occluders (POC boxes, FOC walls, etc.)
            occluders_filtered.append(occ)
            continue

        # Recompute combined occlusion WITHOUT this occluder
        other_bboxes = [o["bbox_3d"] for o in occluders if o is not occ and o.get("type") == "static"]
        # Add non-static occluders
        other_bboxes += [o["bbox_3d"] for o in occluders if o.get("type") != "static"]
        if other_bboxes:
            without_occ = compute_per_camera_occlusion(positions, cameras, other_bboxes, target_xy)
        else:
            without_occ = [[False] * n_frames for _ in range(n_views)]

        # Count frames where removing this occluder restores visibility in at least one view
        restored = 0
        all_blocked = 0
        for fi in range(n_frames):
            currently_blocked_all = all(combined_frame_occ[ci][fi] for ci in range(n_views))
            if currently_blocked_all:
                all_blocked += 1
                still_blocked_all = all(without_occ[ci][fi] for ci in range(n_views))
                if not still_blocked_all:
                    restored += 1

        # If this occluder is the sole cause of all-views-blocking for many frames, drop it
        if all_blocked > 0 and restored > all_blocked * 0.5:
            print(f"  Dropping occluder {occ['name']}: causes all-view block in {all_blocked} frames "
                  f"(removing restores {restored})")
            continue

        occluders_filtered.append(occ)

    occluders = occluders_filtered

    return occluders


def _sample_poc_occluders(positions, cameras, rng, room_x, room_z, room_h,
                           target_xy, category, n_frames):
    """Place 1-3 occluders for partial occlusion (POC)."""
    occluders = []
    num = rng.randint(1, 3)

    for i in range(num):
        # Choose type
        occ_type = rng.choice(["box", "pillar", "human_proxy", "similar_object"])

        if occ_type == "box":
            sx = rng.uniform(0.3, 1.2)
            sy = rng.uniform(0.2, 0.6)
            sz = rng.uniform(0.6, 2.0)
            ox, oz, _ = _place_along_path(positions, rng, room_x, room_z)
            center = [ox, oz, sz / 2]
            bbox = _world_bbox_from_ground_rect(ox, oz, sx, sy, 0.0, sz)
            mesh = _make_box_mesh([sx, sy, sz], center)
            name = f"poc_box_{i}"

        elif occ_type == "pillar":
            r = rng.uniform(0.08, 0.20)
            h = room_h * rng.uniform(0.8, 1.0)
            ox, oz, _ = _place_along_path(positions, rng, room_x, room_z)
            center = [ox, oz, h / 2]
            bbox = _world_bbox_from_ground_rect(ox, oz, 2 * r, 2 * r, 0.0, h)
            mesh = _make_cylinder_mesh(r, h, center)
            name = f"poc_pillar_{i}"

        elif occ_type == "human_proxy":
            proxy_mesh, height = _make_human_proxy(rng)
            ox, oz, _ = _place_along_path(positions, rng, room_x, room_z)
            proxy_mesh.apply_translation([ox, oz, 0])
            w = 0.5
            bbox = _world_bbox_from_ground_rect(ox, oz, w, 0.30, 0.0, height)
            mesh = proxy_mesh
            name = f"poc_human_{i}"

        else:  # similar_object
            mesh = _make_similar_object(category, rng)
            ext = mesh.extents
            ox, oz, _ = _place_between_target_and_cam(
                positions, rng.randint(0, len(cameras) - 1),
                cameras, rng, target_xy, room_x, room_z,
            )
            mesh.apply_translation([ox, oz, ext[2] / 2 + 0.5])
            bbox = _world_bbox_from_ground_rect(ox, oz, ext[0], ext[1], 0.5, ext[2])
            name = f"poc_similar_{i}"

        occluders.append({
            "name": name,
            "type": "static",
            "bbox_3d": bbox,
            "mesh": mesh,
        })

    return occluders


def _sample_foc_occluders(positions, cameras, rng, room_x, room_z, room_h,
                           target_xy, category, n_frames):
    """Place a large wall-like occluder that fully blocks one camera for 30-80 frames.

    Strategy: place a wall between the target mid-segment and one camera.
    """
    occluders = []

    # Pick which camera to fully occlude
    cam_idx = rng.randint(0, len(cameras) - 1)

    # Find a segment of 30-80 frames where we want full occlusion
    seg_len = rng.randint(30, min(80, n_frames // 3))
    seg_start = rng.randint(int(n_frames * 0.1), int(n_frames * 0.6))
    seg_end = min(seg_start + seg_len, n_frames)

    # Place a large wall between target segment midpoint and the camera
    mid_fi = (seg_start + seg_end) // 2
    target_mid = np.array(positions[mid_fi])

    from sample_camera_rig import compute_camera_pose
    cam = cameras[cam_idx]
    cam_pos, _ = compute_camera_pose(
        cam["azimuth_deg"], cam["elevation_deg"],
        cam["distance_m"], target_xy, cam["height_m"],
    )

    # Wall at 30-60% of the way from target to camera
    t = rng.uniform(0.3, 0.6)
    wall_center = target_mid + t * (np.array(cam_pos) - target_mid)
    wx, wz = float(wall_center[0]), float(wall_center[2])

    # Wall dimensions: wide enough to block view
    sx = rng.uniform(1.0, 2.5)  # width
    sy = rng.uniform(0.08, 0.15)  # thin
    sz = room_h * rng.uniform(0.7, 1.0)  # tall

    # Clamp to room
    margin = 0.2
    wx = max(margin + sx / 2, min(room_x - margin - sx / 2, wx))
    wz = max(margin + sy / 2, min(room_z - margin - sy / 2, wz))

    center = [wx, wz, sz / 2]
    bbox = _world_bbox_from_ground_rect(wx, wz, sx, sy, 0.0, sz)

    # Rotate wall to face perpendicular to camera-target line
    mesh = _make_box_mesh([sx, sy, sz], center, color=(130, 125, 120))

    occluders.append({
        "name": f"foc_wall_cam{cam_idx}",
        "type": "static",
        "bbox_3d": bbox,
        "mesh": mesh,
        "target_cam_idx": cam_idx,
        "target_frames": (seg_start, seg_end),
    })

    # Add a secondary smaller occluder for partial blocking in other views
    if rng.random() > 0.4:
        ox2, oz2, _ = _place_along_path(positions, rng, room_x, room_z)
        s2 = rng.uniform(0.2, 0.5)
        h2 = rng.uniform(0.5, 1.2)
        center2 = [ox2, oz2, h2 / 2]
        bbox2 = _world_bbox_from_ground_rect(ox2, oz2, s2, s2, 0.0, h2)
        mesh2 = _make_box_mesh([s2, s2, h2], center2)
        occluders.append({
            "name": "foc_secondary_box",
            "type": "static",
            "bbox_3d": bbox2,
            "mesh": mesh2,
        })

    return occluders


def _sample_ov_occluders(positions, cameras, rng, room_x, room_z, room_h,
                          target_xy, n_frames):
    """Place wall/partition occluders near room edge for OV challenge.

    The target trajectory (edge_drift) moves toward the edge, these
    occluders help block line of sight as target approaches boundary.
    """
    occluders = []

    # Place walls along 1-2 edges
    edges = rng.sample(["left", "right", "near", "far"], k=rng.randint(1, 2))
    for edge in edges:
        if edge == "left":
            ox, oz = 0.15, room_z / 2
            sx, sy = 0.1, room_z * 0.6
        elif edge == "right":
            ox, oz = room_x - 0.15, room_z / 2
            sx, sy = 0.1, room_z * 0.6
        elif edge == "near":
            ox, oz = room_x / 2, 0.15
            sx, sy = room_x * 0.6, 0.1
        else:
            ox, oz = room_x / 2, room_z - 0.15
            sx, sy = room_x * 0.6, 0.1

        sz = room_h * 0.6
        center = [ox, oz, sz / 2]
        bbox = _world_bbox_from_ground_rect(ox, oz, sx, sy, 0.0, sz)
        mesh = _make_box_mesh([sx, sy, sz], center, color=(160, 155, 150))
        occluders.append({
            "name": f"ov_partition_{edge}",
            "type": "static",
            "bbox_3d": bbox,
            "mesh": mesh,
        })

    return occluders


# ---------------------------------------------------------------------------
# Convenience wrapper matching generate_sequence.py interface
# ---------------------------------------------------------------------------

def sample_occluders(scene, target, trajectory, cameras, cfg, rng=None):
    """Top-level occluder sampling. Delegates to challenge-aware sampler.

    Returns list of occluder dicts.
    """
    challenge = trajectory.get("main_challenge", "POC")
    return sample_occluders_for_challenge(
        challenge, scene, target, trajectory, cameras, cfg, rng=rng,
    )
