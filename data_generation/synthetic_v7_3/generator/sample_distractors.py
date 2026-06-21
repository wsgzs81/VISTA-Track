#!/usr/bin/env python3
"""
sample_distractors.py - Distractor sampling for BC (Background Clutter) challenge.

Places objects that visually resemble the target (similar color, shape, size)
to create background confusion. BC is defined as background/target appearance
similarity, NOT random clutter.
"""

import math
import random

import numpy as np
import trimesh


# ---------------------------------------------------------------------------
# Shape generators per target category
# ---------------------------------------------------------------------------

def _make_container_distractor(rng):
    """Cup/bottle/can-like shape."""
    import trimesh

    shape = rng.choice(["cylinder", "box", "tapered"])
    if shape == "cylinder":
        r = rng.uniform(0.025, 0.055)
        h = rng.uniform(0.08, 0.22)
        mesh = trimesh.creation.cylinder(radius=r, height=h, sections=10)
    elif shape == "box":
        sx = rng.uniform(0.04, 0.08)
        sy = rng.uniform(0.04, 0.08)
        sz = rng.uniform(0.08, 0.20)
        mesh = trimesh.creation.box(extents=[sx, sy, sz])
    else:
        # Tapered (cup-like)
        r_bot = rng.uniform(0.03, 0.05)
        r_top = rng.uniform(0.04, 0.06)
        h = rng.uniform(0.08, 0.15)
        mesh = trimesh.creation.cylinder(radius=r_bot, height=h, sections=10)

    return mesh


def _make_toy_sport_distractor(rng):
    """Ball/block/figure-like shape."""
    import trimesh

    shape = rng.choice(["sphere", "box", "capsule"])
    if shape == "sphere":
        r = rng.uniform(0.03, 0.10)
        mesh = trimesh.creation.icosphere(radius=r, subdivisions=2)
    elif shape == "box":
        s = rng.uniform(0.04, 0.12)
        mesh = trimesh.creation.box(extents=[s, s * rng.uniform(0.8, 1.2), s])
    else:
        r = rng.uniform(0.02, 0.04)
        h = rng.uniform(0.06, 0.15)
        mesh = trimesh.creation.capsule(radius=r, height=h)

    return mesh


def _make_electronics_distractor(rng):
    """Rectangular slab/box like phone/remote/charger."""
    import trimesh

    sx = rng.uniform(0.06, 0.20)
    sy = rng.uniform(0.03, 0.08)
    sz = rng.uniform(0.01, 0.04)
    mesh = trimesh.creation.box(extents=[sx, sy, sz])
    return mesh


def _make_household_distractor(rng):
    """Misc household shape - box, cylinder, or irregular."""
    import trimesh

    shape = rng.choice(["box", "cylinder", "disc"])
    if shape == "box":
        sx = rng.uniform(0.05, 0.20)
        sy = rng.uniform(0.05, 0.15)
        sz = rng.uniform(0.05, 0.25)
        mesh = trimesh.creation.box(extents=[sx, sy, sz])
    elif shape == "cylinder":
        r = rng.uniform(0.03, 0.08)
        h = rng.uniform(0.10, 0.30)
        mesh = trimesh.creation.cylinder(radius=r, height=h, sections=10)
    else:
        r = rng.uniform(0.05, 0.12)
        mesh = trimesh.creation.cylinder(radius=r, height=0.01, sections=12)

    return mesh


def _make_misc_distractor(rng):
    """Generic small object."""
    import trimesh

    s = rng.uniform(0.04, 0.15)
    mesh = trimesh.creation.box(extents=[s, s * rng.uniform(0.7, 1.3), s * rng.uniform(0.5, 1.0)])
    return mesh


CATEGORY_BUILDERS = {
    "container": _make_container_distractor,
    "toy_sport": _make_toy_sport_distractor,
    "electronics_tool": _make_electronics_distractor,
    "household": _make_household_distractor,
    "misc": _make_misc_distractor,
}


# ---------------------------------------------------------------------------
# Color matching
# ---------------------------------------------------------------------------

def _extract_target_color(target):
    """Get target color from mesh vertex colors or defaults."""
    mesh = target.get("mesh")
    if mesh is not None:
        vis = mesh.visual
        if hasattr(vis, "vertex_colors") and vis.vertex_colors is not None:
            colors = vis.vertex_colors[:, :3].astype(np.float32) / 255.0
            # Median color (robust to outliers)
            median_color = np.median(colors, axis=0)
            return (median_color * 255).astype(np.uint8).tolist()

    # Default colors by category
    defaults = {
        "container": [180, 180, 180],
        "toy_sport": [200, 100, 80],
        "electronics_tool": [60, 60, 65],
        "household": [160, 150, 140],
        "misc": [150, 150, 150],
    }
    return defaults.get(target.get("category", "misc"), [150, 150, 150])


def _perturb_color(base_color, rng, hue_range=30, brightness_range=0.2):
    """Create a similar but not identical color."""
    color = np.array(base_color, dtype=np.float32)

    # Random shift per channel
    shift = np.random.uniform(-hue_range, hue_range, 3)
    color = color + shift

    # Brightness variation
    brightness = rng.uniform(1.0 - brightness_range, 1.0 + brightness_range)
    color = color * brightness

    return tuple(int(max(0, min(255, c))) for c in color)


# ---------------------------------------------------------------------------
# Placement
# ---------------------------------------------------------------------------

def _place_distractors_around_room(n_distractors, room_x, room_z, rng, positions=None):
    """Place distractors around the room, biased near target path if available."""
    placements = []
    margin = 0.3

    for i in range(n_distractors):
        if positions and rng.random() > 0.3:
            # Place near target path (60cm-150cm from path)
            fi = rng.randint(0, len(positions) - 1)
            pos = positions[fi]
            angle = rng.uniform(0, 2 * math.pi)
            dist = rng.uniform(0.6, 1.5)
            x = pos[0] + dist * math.cos(angle)
            z = pos[2] + dist * math.sin(angle)
            x = max(margin, min(room_x - margin, x))
            z = max(margin, min(room_z - margin, z))
        else:
            # Random room position
            x = rng.uniform(margin, room_x - margin)
            z = rng.uniform(margin, room_z - margin)

        # Place on a surface (table or floor)
        surface_h = rng.choice([0.0, 0.4, 0.75])  # floor, low table, desk
        placements.append((x, z, surface_h))

    return placements


def _world_bbox_from_ground_rect(cx, cz, sx, depth, y0, height):
    """Return bbox in world [x, y(height), z] coordinates."""
    return [
        [round(cx - sx / 2, 3), round(y0, 3), round(cz - depth / 2, 3)],
        [round(cx + sx / 2, 3), round(y0 + height, 3), round(cz + depth / 2, 3)],
    ]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def sample_distractors(scene, target, trajectory, cameras, cfg, rng=None):
    """Sample distractor objects for BC (Background Clutter) challenge.

    Only generates distractors when main_challenge == "BC".
    Distractors visually resemble the target in shape and color.

    Args:
        scene: scene dict
        target: target dict with category, mesh
        trajectory: trajectory dict with main_challenge, positions_m
        cameras: camera dicts
        cfg: dataset config
        rng: random.Random instance

    Returns:
        list of distractor dicts with:
            name, position_m, mesh, bbox_3d, color, similar_to
    """
    if rng is None:
        rng = random.Random()

    challenge = trajectory.get("main_challenge", "")

    # Only generate distractors for BC challenge
    if challenge != "BC":
        return []

    room = scene.get("room_size_m", [5.0, 2.8, 5.0])
    room_x, room_h, room_z = room[0], room[1], room[2]
    category = target.get("category", "misc")
    positions = trajectory["positions_m"]

    # Number of distractors: enough to create confusion but not fill the room
    n_distractors = rng.randint(3, 8)

    # Get target appearance
    target_color = _extract_target_color(target)
    builder = CATEGORY_BUILDERS.get(category, _make_misc_distractor)

    # Place distractors
    placements = _place_distractors_around_room(
        n_distractors, room_x, room_z, rng, positions,
    )

    distractors = []
    for i, (x, z, surface_h) in enumerate(placements):
        # Build similar-shaped mesh
        mesh = builder(rng)

        # Apply similar color
        color = _perturb_color(target_color, rng, hue_range=25, brightness_range=0.15)
        mesh.visual = trimesh.visual.ColorVisuals()
        mesh.visual.vertex_colors = np.full(
            (len(mesh.vertices), 4), list(color) + [255], dtype=np.uint8,
        )

        # Position
        ext = mesh.extents
        mesh.apply_translation([x, z, surface_h + ext[2] / 2])

        bbox = _world_bbox_from_ground_rect(x, z, ext[0], ext[1], surface_h, ext[2])

        distractors.append({
            "name": f"bc_distractor_{i}",
            "position_m": [round(x, 3), round(surface_h + ext[2] / 2, 3), round(z, 3)],
            "mesh": mesh,
            "bbox_3d": bbox,
            "color": color,
            "similar_to": category,
        })

    return distractors
