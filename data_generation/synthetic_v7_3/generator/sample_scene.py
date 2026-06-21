#!/usr/bin/env python3
"""
sample_scene.py - Scene sampling and procedural generation.

Produces a scene dict with geometry, walkable area, surfaces, and
occluders. Supports procedural rooms/corridors/tabletops and
3D-FRONT scenes (when data is available).
"""

import json
import math
import os
import random

import numpy as np
import trimesh


# ---------------------------------------------------------------------------
# Material helpers
# ---------------------------------------------------------------------------

def load_material_index(index_path):
    """Load material index or return empty list."""
    if os.path.isfile(index_path):
        with open(index_path) as f:
            return json.load(f)
    return []


def pick_random_material(materials, rng):
    """Pick a random material path from index."""
    if not materials:
        return None
    mat = rng.choice(materials)
    # Each material dir typically has *_Color.jpg
    mat_dir = mat.get("path", "")
    if os.path.isdir(mat_dir):
        for f in os.listdir(mat_dir):
            if "Color" in f and f.endswith((".jpg", ".png")):
                return os.path.join(mat_dir, f)
            if "BaseColor" in f and f.endswith((".jpg", ".png")):
                return os.path.join(mat_dir, f)
    return None


def make_color_material(color, alpha=None):
    """Create a simple colored material."""
    mat = trimesh.visual.material.SimpleMaterial()
    mat.diffuse = color if len(color) == 4 else list(color[:3]) + [255]
    return mat


# ---------------------------------------------------------------------------
# Procedural geometry builders
# ---------------------------------------------------------------------------

def build_floor(size_x, size_z, color=(180, 170, 155)):
    """Create floor plane centered at origin."""
    mesh = trimesh.creation.box(extents=[size_x, size_z, 0.02])
    mesh.apply_translation([size_x / 2, size_z / 2, -0.01])
    mesh.visual = make_color_material(color)
    return mesh


def build_wall(size_x, size_z, wall_axis, position, color=(220, 215, 210)):
    """Create a wall plane.

    wall_axis: 'x' for walls along x-axis (constant z), 'z' for walls along z-axis (constant x)
    position: the constant coordinate value
    """
    height = size_z  # reuse param as height
    if wall_axis == "x":
        mesh = trimesh.creation.box(extents=[size_x, 0.1, height])
        mesh.apply_translation([size_x / 2, position, height / 2])
    else:
        mesh = trimesh.creation.box(extents=[0.1, size_z, height])
        mesh.apply_translation([position, size_z / 2, height / 2])
    mesh.visual = make_color_material(color)
    return mesh


def build_ceiling(size_x, size_z, height, color=(240, 240, 240)):
    mesh = trimesh.creation.box(extents=[size_x, size_z, 0.02])
    mesh.apply_translation([size_x / 2, size_z / 2, height + 0.01])
    mesh.visual = make_color_material(color)
    return mesh


def build_furniture_box(size, position, color=None, rng=None):
    """Create a box representing furniture."""
    if color is None:
        _r = rng or random
        color = (_r.randint(80, 200), _r.randint(80, 200), _r.randint(80, 200))
    mesh = trimesh.creation.box(extents=size)
    mesh.apply_translation([position[0], position[1], size[2] / 2])
    mesh.visual = make_color_material(color)
    return mesh


# ---------------------------------------------------------------------------
# Procedural room generator
# ---------------------------------------------------------------------------

FURNITURE_DEFS = {
    "living_room": [
        {"name": "sofa", "size": [2.0, 0.8, 0.8], "color": (100, 80, 60)},
        {"name": "coffee_table", "size": [1.0, 0.6, 0.4], "color": (140, 100, 60)},
        {"name": "tv_stand", "size": [1.5, 0.4, 0.5], "color": (60, 60, 60)},
        {"name": "shelf", "size": [0.8, 0.3, 1.8], "color": (120, 90, 50)},
    ],
    "bedroom": [
        {"name": "bed", "size": [2.0, 1.6, 0.5], "color": (200, 200, 210)},
        {"name": "nightstand", "size": [0.5, 0.4, 0.5], "color": (120, 80, 50)},
        {"name": "wardrobe", "size": [1.2, 0.6, 2.0], "color": (160, 130, 90)},
        {"name": "desk", "size": [1.2, 0.6, 0.75], "color": (140, 110, 70)},
    ],
    "study": [
        {"name": "desk", "size": [1.4, 0.7, 0.75], "color": (100, 70, 40)},
        {"name": "chair", "size": [0.5, 0.5, 0.9], "color": (50, 50, 50)},
        {"name": "bookshelf", "size": [1.0, 0.3, 2.0], "color": (130, 100, 60)},
        {"name": "cabinet", "size": [0.8, 0.4, 1.0], "color": (150, 120, 80)},
    ],
    "dining_room": [
        {"name": "dining_table", "size": [1.6, 0.9, 0.75], "color": (120, 80, 40)},
        {"name": "chair", "size": [0.45, 0.45, 0.9], "color": (100, 70, 40)},
        {"name": "sideboard", "size": [1.4, 0.4, 0.9], "color": (140, 110, 70)},
    ],
}


def generate_procedural_room(scene_cfg, material_index, rng):
    """Generate a procedural room with walls, floor, furniture."""
    params = scene_cfg.get("procedural_params", {})
    room_type = params.get("room_subtype", "living_room")
    size_x = params.get("size_x", 5.0)
    size_z = params.get("size_z", 4.0)
    height = params.get("height", 2.8)

    meshes = []
    surfaces = []
    occluders = []

    # Floor
    floor = build_floor(size_x, size_z)
    meshes.append(floor)
    surfaces.append({"name": "floor", "type": "plane", "bounds": [[0, 0], [size_x, size_z]]})

    # Walls (4 walls)
    wall_colors = [(220, 215, 210), (210, 205, 200), (215, 210, 205), (225, 220, 215)]
    wall_positions = [
        ("x", 0.0, size_x, wall_colors[0]),       # wall at y=0
        ("x", size_z, size_x, wall_colors[1]),     # wall at y=size_z
        ("z", 0.0, size_z, wall_colors[2]),        # wall at x=0
        ("z", size_x, size_z, wall_colors[3]),     # wall at x=size_x
    ]
    for axis, pos, span, color in wall_positions:
        if axis == "x":
            w = build_wall(span, height, "x", pos, color)
        else:
            w = build_wall(span, height, "z", pos, color)
        meshes.append(w)
        surfaces.append({"name": f"wall_{axis}_{pos}", "type": "wall"})

    # Ceiling
    ceiling = build_ceiling(size_x, size_z, height)
    meshes.append(ceiling)

    # Furniture
    furn_defs = FURNITURE_DEFS.get(room_type, FURNITURE_DEFS["living_room"])
    clutter = scene_cfg.get("clutter_level", "medium")
    max_furn = {"low": 2, "medium": 4, "high": 6}.get(clutter, 4)
    num_furn = rng.randint(max(1, max_furn - 1), max_furn)

    placed_boxes = []  # (cx, cy, half_x, half_y) for collision check
    for i in range(num_furn):
        furn = rng.choice(furn_defs)
        sx, sy, sz = furn["size"]

        # Find non-overlapping position
        for _ in range(20):
            cx = rng.uniform(0.5 + sx / 2, size_x - 0.5 - sx / 2)
            cy = rng.uniform(0.5 + sy / 2, size_z - 0.5 - sy / 2)
            overlap = False
            for (px, py, phx, phy) in placed_boxes:
                if abs(cx - px) < (sx / 2 + phx + 0.1) and abs(cy - py) < (sy / 2 + phy + 0.1):
                    overlap = True
                    break
            if not overlap:
                break

        placed_boxes.append((cx, cy, sx / 2, sy / 2))
        color = furn["color"]
        # Add slight color variation
        color = tuple(max(0, min(255, c + rng.randint(-15, 15))) for c in color)

        box = build_furniture_box([sx, sy, sz], [cx, cy], color, rng=rng)
        meshes.append(box)

        is_occluder = sz > 0.6
        surfaces.append({
            "name": furn["name"],
            "type": "furniture",
            "position_m": [round(cx, 2), round(cy, 2)],
            "size_m": [sx, sy, sz],
        })
        if is_occluder:
            occluders.append({
                "name": furn["name"],
                "bbox_3d": [
                    [round(cx - sx / 2, 2), round(cy - sy / 2, 2), 0],
                    [round(cx + sx / 2, 2), round(cy + sy / 2, 2), round(sz, 2)],
                ],
            })

    # Random clutter (small boxes on surfaces)
    clutter_count = {"low": rng.randint(0, 2), "medium": rng.randint(2, 5), "high": rng.randint(5, 10)}.get(clutter, 3)
    for _ in range(clutter_count):
        cs = rng.uniform(0.05, 0.2)
        # Place on top of a furniture piece
        if placed_boxes:
            base = rng.choice(placed_boxes)
            cx = base[0] + rng.uniform(-base[2] * 0.6, base[2] * 0.6)
            cy = base[1] + rng.uniform(-base[3] * 0.6, base[3] * 0.6)
            # Find base height
            for surf in surfaces:
                if surf.get("type") == "furniture":
                    fx, fy = surf.get("position_m", [0, 0])
                    if abs(fx - base[0]) < 0.01 and abs(fy - base[1]) < 0.01:
                        fz = surf.get("size_m", [0, 0, 0])[2]
                        break
            else:
                fz = 0.75
            clutter_box = build_furniture_box(
                [cs, cs, cs], [cx, cy],
                color=(rng.randint(50, 220), rng.randint(50, 220), rng.randint(50, 220)),
            )
            clutter_box.apply_translation([0, 0, fz])
            meshes.append(clutter_box)

    # Walkable area: grid with obstacles removed
    walkable = build_walkable_grid(size_x, size_z, placed_boxes, resolution=0.1)

    return meshes, surfaces, occluders, walkable


# ---------------------------------------------------------------------------
# Procedural corridor generator
# ---------------------------------------------------------------------------

def generate_procedural_corridor(scene_cfg, material_index, rng):
    params = scene_cfg.get("procedural_params", {})
    length = params.get("length", 8.0)
    width = params.get("width", 1.5)
    height = params.get("height", 2.8)

    meshes = []
    surfaces = []
    occluders = []

    # Floor (corridor along x-axis)
    floor = build_floor(length, width)
    meshes.append(floor)
    surfaces.append({"name": "floor", "type": "plane", "bounds": [[0, 0], [length, width]]})

    # Two side walls
    for pos, label in [(0.0, "left"), (width, "right")]:
        w = build_wall(length, height, "x", pos, color=(210, 205, 200))
        meshes.append(w)
        surfaces.append({"name": f"wall_{label}", "type": "wall"})

    # Ceiling
    ceiling = build_ceiling(length, width, height)
    meshes.append(ceiling)

    # Optional doors along the corridor
    if params.get("has_doors", False):
        num_doors = params.get("num_doors", 2)
        for i in range(num_doors):
            dx = (i + 1) * length / (num_doors + 1)
            # Door frame box on each wall
            for side in [0, width]:
                door = build_furniture_box(
                    [0.8, 0.1, 2.0], [dx, side],
                    color=(120, 90, 50),
                )
                meshes.append(door)
                occluders.append({
                    "name": f"door_{i}_{'left' if side == 0 else 'right'}",
                    "bbox_3d": [
                        [round(dx - 0.4, 2), round(side - 0.05, 2), 0],
                        [round(dx + 0.4, 2), round(side + 0.05, 2), 2.0],
                    ],
                })

    walkable = build_walkable_grid(length, width, [], resolution=0.1)

    return meshes, surfaces, occluders, walkable


# ---------------------------------------------------------------------------
# Procedural open area generator
# ---------------------------------------------------------------------------

def generate_procedural_open_area(scene_cfg, material_index, rng):
    params = scene_cfg.get("procedural_params", {})
    size_x = params.get("size_x", 10.0)
    size_z = params.get("size_z", 10.0)
    height = params.get("height", 3.0)

    meshes = []
    surfaces = []
    occluders = []

    # Ground plane
    ground_colors = {
        "grass": (80, 130, 60),
        "concrete": (160, 160, 155),
        "dirt": (130, 100, 70),
        "asphalt": (80, 80, 85),
    }
    ground_type = params.get("ground_material", "concrete")
    floor = build_floor(size_x, size_z, color=ground_colors.get(ground_type, (160, 160, 155)))
    meshes.append(floor)
    surfaces.append({"name": "ground", "type": "plane", "bounds": [[0, 0], [size_x, size_z]]})

    # Boundary
    boundary = params.get("boundary_type", "open")
    if boundary == "wall":
        for axis, pos, span in [("x", 0.0, size_x), ("x", size_z, size_x),
                                 ("z", 0.0, size_z), ("z", size_x, size_z)]:
            w = build_wall(span, height, axis, pos, color=(150, 145, 140))
            meshes.append(w)
            surfaces.append({"name": f"boundary_{axis}_{pos}", "type": "wall"})
    elif boundary == "fence":
        fence_h = 1.2
        for axis, pos, span in [("x", 0.0, size_x), ("x", size_z, size_x),
                                 ("z", 0.0, size_z), ("z", size_x, size_z)]:
            if axis == "x":
                f = trimesh.creation.box(extents=[span, 0.05, fence_h])
                f.apply_translation([span / 2, pos, fence_h / 2])
            else:
                f = trimesh.creation.box(extents=[0.05, span, fence_h])
                f.apply_translation([pos, span / 2, fence_h / 2])
            f.visual = make_color_material((100, 100, 100))
            meshes.append(f)
            occluders.append({
                "name": f"fence_{axis}_{pos}",
                "bbox_3d": [
                    [0 if axis == "z" else 0, 0 if axis == "x" else 0, 0],
                    [size_x if axis == "x" else pos + 0.05,
                     size_z if axis == "z" else pos + 0.05, fence_h],
                ],
            })

    walkable = build_walkable_grid(size_x, size_z, [], resolution=0.2)

    return meshes, surfaces, occluders, walkable


# ---------------------------------------------------------------------------
# Procedural tabletop generator
# ---------------------------------------------------------------------------

def generate_procedural_tabletop(scene_cfg, material_index, rng):
    params = scene_cfg.get("procedural_params", {})
    tx = params.get("table_size", [1.2, 0.04, 0.8])
    th = params.get("table_height", 0.75)

    meshes = []
    surfaces = []
    occluders = []

    # Tabletop
    table = trimesh.creation.box(extents=[tx[0], tx[2], tx[1]])
    table.apply_translation([0, 0, th])
    table.visual = make_color_material((140, 100, 60))
    meshes.append(table)
    surfaces.append({
        "name": "table",
        "type": "furniture",
        "position_m": [0, 0],
        "size_m": [tx[0], tx[2], th],
    })

    # Table legs (4)
    leg_r = 0.03
    for lx in [-tx[0] / 2 + 0.05, tx[0] / 2 - 0.05]:
        for ly in [-tx[2] / 2 + 0.05, tx[2] / 2 - 0.05]:
            leg = trimesh.creation.cylinder(radius=leg_r, height=th, sections=8)
            leg.apply_translation([lx, ly, th / 2])
            leg.visual = make_color_material((100, 70, 40))
            meshes.append(leg)

    # Floor (large enough to be background)
    floor = build_floor(4.0, 4.0)
    meshes.append(floor)
    surfaces.append({"name": "floor", "type": "plane"})

    walkable = build_walkable_grid(4.0, 4.0, [(0, 0, tx[0] / 2, tx[2] / 2)], resolution=0.1)

    return meshes, surfaces, occluders, walkable


# ---------------------------------------------------------------------------
# Walkable area grid
# ---------------------------------------------------------------------------

def build_walkable_grid(size_x, size_z, obstacles, resolution=0.1):
    """Build a 2D walkability grid. Obstacles: list of (cx, cy, half_x, half_y)."""
    nx = int(size_x / resolution)
    nz = int(size_z / resolution)
    grid = np.ones((nx, nz), dtype=np.uint8)

    for (cx, cy, hx, hy) in obstacles:
        x0 = max(0, int((cx - hx) / resolution))
        x1 = min(nx, int((cx + hx) / resolution) + 1)
        z0 = max(0, int((cy - hy) / resolution))
        z1 = min(nz, int((cy + hy) / resolution) + 1)
        grid[x0:x1, z0:z1] = 0

    # Border margin
    margin = max(1, int(0.3 / resolution))
    grid[:margin, :] = 0
    grid[-margin:, :] = 0
    grid[:, :margin] = 0
    grid[:, -margin:] = 0

    return grid


# ---------------------------------------------------------------------------
# 3D-FRONT scene generator
# ---------------------------------------------------------------------------

def _build_front_room_meshes(front_json, room_size, rng, offset=None):
    """Build floor + wall meshes from 3D-FRONT JSON data.

    Args:
        offset: [x, y, z] offset to subtract from room coords (shifts to origin)

    Returns (meshes, surfaces, occluders, walkable).
    """
    meshes = []
    surfaces = []
    occluders = []

    if offset is None:
        offset = [0, 0, 0]

    ox, oy, oz = offset
    sx, sy, sz = room_size  # width, height, depth

    # Floor - positioned at origin (0,0) after offset
    floor = trimesh.creation.box(extents=[sx, sz, 0.02])
    floor.apply_translation([sx / 2, sz / 2, -0.01])
    floor.visual = make_color_material([180, 170, 160])
    meshes.append(floor)
    surfaces.append({
        "name": "floor",
        "center_m": [sx / 2, 0, sz / 2],
        "size_m": [sx, sz],
        "normal": [0, 1, 0],
    })

    # Walls (4 sides)
    wall_color = [210, 200, 190]
    wall_thickness = 0.1

    # North wall (z=0)
    wall_n = trimesh.creation.box(extents=[sx, sy, wall_thickness])
    wall_n.apply_translation([sx / 2, sy / 2, -wall_thickness / 2])
    wall_n.visual = make_color_material(wall_color)
    meshes.append(wall_n)

    # South wall (z=sz)
    wall_s = trimesh.creation.box(extents=[sx, sy, wall_thickness])
    wall_s.apply_translation([sx / 2, sy / 2, sz + wall_thickness / 2])
    wall_s.visual = make_color_material(wall_color)
    meshes.append(wall_s)

    # West wall (x=0)
    wall_w = trimesh.creation.box(extents=[wall_thickness, sy, sz])
    wall_w.apply_translation([-wall_thickness / 2, sy / 2, sz / 2])
    wall_w.visual = make_color_material(wall_color)
    meshes.append(wall_w)

    # East wall (x=sx)
    wall_e = trimesh.creation.box(extents=[wall_thickness, sy, sz])
    wall_e.apply_translation([sx + wall_thickness / 2, sy / 2, sz / 2])
    wall_e.visual = make_color_material(wall_color)
    meshes.append(wall_e)

    surfaces.append({"name": "wall_north", "center_m": [sx / 2, sy / 2, 0], "size_m": [sx, sy], "normal": [0, 0, 1]})
    surfaces.append({"name": "wall_south", "center_m": [sx / 2, sy / 2, sz], "size_m": [sx, sy], "normal": [0, 0, -1]})

    # Walkable area
    walkable = build_walkable_grid(sx, sz, [], resolution=0.1)

    return meshes, surfaces, occluders, walkable


def _load_front_furniture(front_json, rng, future_model_dir=None, offset=None):
    """Load furniture placements from 3D-FRONT JSON.

    Maps scene tree children to furniture entries via uid matching,
    then resolves 3D-FUTURE model paths.

    Args:
        offset: [x, 0, z] to subtract from positions (shifts to origin)

    Returns list of occluder dicts with pos, size, jid, title, bbox_3d, model_path.
    """
    furniture_list = front_json.get("furniture", [])
    scene = front_json.get("scene", {})

    if offset is None:
        offset = [0, 0, 0]
    ox, _, oz = offset

    # Build uid → furniture data map
    furn_by_uid = {}
    for f in furniture_list:
        uid = f.get("uid", "")
        if uid:
            furn_by_uid[uid] = f

    # Build jid → model path lookup
    available_models = set()
    if future_model_dir and os.path.isdir(future_model_dir):
        available_models = set(os.listdir(future_model_dir))

    occluders = []
    seen_jids = set()

    # Iterate rooms and their children
    rooms = scene.get("room", []) if isinstance(scene, dict) else []
    for room in rooms:
        if room.get("empty", 0) == 1:
            continue
        for child in room.get("children", []):
            ref = child.get("ref", "")
            if not ref:
                continue

            # Match child ref to furniture uid
            furn = furn_by_uid.get(ref)
            if furn is None:
                continue

            jid = furn.get("jid", "")
            title = furn.get("title", "")
            size = furn.get("size", [0, 0, 0])

            if not jid or not size or len(size) < 3:
                continue
            if furn.get("type") in ("Window", "Door", "Empty"):
                continue
            if jid in seen_jids:
                continue
            seen_jids.add(jid)

            w, h, d = size[0], size[1], size[2]
            if w * d < 0.05 or h < 0.1:
                continue

            # Position from child (shift by offset to normalize to origin)
            pos = child.get("pos", [0, 0, 0])
            rot = child.get("rot", [0, 0, 0, 1])
            cx, cy, cz = pos[0] - ox, pos[1], pos[2] - oz

            # Bbox in trimesh z-up coords
            bbox_3d = [
                [round(cx - w / 2, 3), round(cy - d / 2, 3), round(cz, 3)],
                [round(cx + w / 2, 3), round(cy + d / 2, 3), round(cz + h, 3)],
            ]

            # Resolve 3D-FUTURE model path
            model_path = None
            if jid in available_models:
                obj_path = os.path.join(future_model_dir, jid, "normalized_model.obj")
                if os.path.isfile(obj_path):
                    model_path = obj_path

            occluders.append({
                "name": f"{title.replace('/', '_')}_{jid[:8]}",
                "jid": jid,
                "title": title,
                "category": furn.get("category", ""),
                "position_m": [round(cx, 3), round(cy, 3), round(cz, 3)],
                "size_m": [round(w, 3), round(h, 3), round(d, 3)],
                "bbox_3d": bbox_3d,
                "rotation": rot,
                "model_path": model_path,
            })

    return occluders


def _collect_positions(node, pos_map, parent_pos=None):
    """Recursively collect furniture positions from 3D-FRONT scene tree."""
    if parent_pos is None:
        parent_pos = [0, 0, 0]

    pos = node.get("pos", [0, 0, 0])
    rot = node.get("rot", [0, 0, 0, 1])
    scale = node.get("scale", [1, 1, 1])
    ref = node.get("ref", "")

    # World position = parent + local
    world_pos = [
        parent_pos[0] + pos[0],
        parent_pos[1] + pos[1],
        parent_pos[2] + pos[2],
    ]

    # If ref looks like a furniture jid (has /model suffix), store position
    if ref and "/" in ref:
        jid = ref.split("/")[0]
        if jid not in pos_map:
            # Convert rotation quaternion to euler if needed
            pos_map[jid] = world_pos + [rot]

    # Recurse into children
    for child in node.get("children", []):
        if isinstance(child, dict):
            _collect_positions(child, pos_map, world_pos)


def generate_3dfront_room(scene_cfg, material_index, rng):
    """Generate a room from 3D-FRONT data.

    Args:
        scene_cfg: scene entry from scene_index.json
        material_index: material index (unused for 3D-FRONT)
        rng: random.Random

    Returns:
        (meshes, surfaces, occluders, walkable_grid)
    """
    json_path = scene_cfg.get("scene_path", scene_cfg.get("json_path", ""))
    room_size = scene_cfg.get("room_size_m", [5, 2.8, 5])
    future_model_dir = scene_cfg.get("future_model_dir", "")

    # Resolve relative paths (scene_path is relative to scene_index.json)
    if json_path and not os.path.isabs(json_path):
        scene_idx = cfg.get("assets", {}).get("scene_index", "")
        if scene_idx:
            base_dir = os.path.dirname(os.path.abspath(scene_idx))
            json_path = os.path.normpath(os.path.join(base_dir, json_path))
        else:
            json_path = os.path.abspath(json_path)

    # Load 3D-FRONT JSON
    front_json = {}
    if os.path.isfile(json_path):
        with open(json_path) as f:
            front_json = json.load(f)

    # Use room_size from index directly (already filtered for reasonable size)
    # Only compute offset from THIS room's furniture children
    all_rooms = front_json.get("scene", {}).get("room", [])

    # Find the non-empty room with most furniture (likely the one in the index)
    best_room = None
    best_count = 0
    for room in all_rooms:
        if room.get("empty", 0) == 1:
            continue
        children = room.get("children", [])
        if len(children) > best_count:
            best_count = len(children)
            best_room = room

    all_children = best_room.get("children", []) if best_room else []

    xs, zs = [], []
    for ch in all_children:
        p = ch.get("pos", [0, 0, 0])
        xs.append(p[0])
        zs.append(p[2])

    if xs and zs:
        room_x_min, room_x_max = min(xs), max(xs)
        room_z_min, room_z_max = min(zs), max(zs)
        room_offset = [room_x_min, 0, room_z_min]
    else:
        room_offset = [0, 0, 0]

    # Use the room_size from the scene index (already computed and filtered)
    actual_room_size = room_size

    # Build room geometry (floor + walls) with actual bounds
    meshes, surfaces, _, walkable = _build_front_room_meshes(
        front_json, actual_room_size, rng, offset=room_offset
    )

    # Load furniture as occluders (shifted to origin)
    occluders = _load_front_furniture(front_json, rng, future_model_dir=future_model_dir,
                                       offset=room_offset)

    # Also add furniture as meshes (simple boxes)
    for occ in occluders:
        pos = occ["position_m"]
        sz = occ["size_m"]
        if sz[0] <= 0 or sz[1] <= 0 or sz[2] <= 0:
            continue

        box = trimesh.creation.box(extents=[sz[0], sz[2], sz[1]])
        box.apply_translation([pos[0], pos[2], pos[1] + sz[1] / 2])

        # Random wood/fabric color
        colors = [
            [139, 90, 43],   # wood brown
            [160, 140, 120], # light wood
            [100, 100, 110], # gray
            [80, 60, 50],    # dark wood
            [180, 170, 150], # fabric
        ]
        box.visual = make_color_material(rng.choice(colors))
        meshes.append(box)

    # Clamp occluder count — too many furniture items block target visibility
    # Keep at most 3 closest to room center to avoid permanent blocking
    max_occ = 3
    if len(occluders) > max_occ:
        room_cx = actual_room_size[0] / 2
        room_cz = actual_room_size[2] / 2
        occluders.sort(key=lambda o: (
            (o["position_m"][0] - room_cx) ** 2 + (o["position_m"][2] - room_cz) ** 2
        ))
        occluders = occluders[:max_occ]

    # Update scene_cfg so sample_scene uses the actual room size
    scene_cfg["room_size_m"] = actual_room_size

    return meshes, surfaces, occluders, walkable


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

GENERATORS = {
    "room": generate_procedural_room,
    "corridor": generate_procedural_corridor,
    "open_area": generate_procedural_open_area,
    "tabletop": generate_procedural_tabletop,
    "3d-front": generate_3dfront_room,
}


def sample_scene(cfg, rng=None):
    """Sample and generate a scene. Returns scene dict with geometry.

    Args:
        cfg: full dataset config dict
        rng: random.Random instance (created if None)

    Returns:
        scene dict with keys:
            scene_id, scene_type, world_bounds_m,
            meshes (list of trimesh.Trimesh),
            surfaces, static_occluders, walkable_area (2D numpy grid)
    """
    if rng is None:
        rng = random.Random()

    # Load scene index
    scene_idx_path = cfg["assets"]["scene_index"]
    with open(scene_idx_path) as f:
        scenes = json.load(f)

    # Load material index (optional)
    mat_idx_path = cfg["assets"].get("material_index", "")
    material_index = load_material_index(mat_idx_path) if mat_idx_path else []

    # Pick a scene (retry if room too large)
    for _ in range(20):
        scene_cfg = rng.choice(scenes)
        rm = scene_cfg.get("room_size_m", [5, 2.8, 5])
        if rm[0] <= 6.5 and rm[2] <= 6.5:
            break
    scene_type = scene_cfg["type"]

    print(f"  Scene: {scene_cfg['scene_id']} type={scene_type}")

    # Generate geometry — use 3D-FRONT loader for 3D-FRONT sources
    if scene_cfg.get("source") == "3D-FRONT":
        generator = generate_3dfront_room
    else:
        generator = GENERATORS.get(scene_type)
    if generator is None:
        raise ValueError(f"Unknown scene type: {scene_type}")

    meshes, surfaces, occluders, walkable = generator(scene_cfg, material_index, rng)

    # Compute world bounds from geometry (avoid trimesh.util.concatenate to prevent
    # RecursionError during deepcopy of complex mesh visuals)
    if meshes:
        all_mins = []
        all_maxs = []
        for m in meshes:
            if len(m.vertices) > 0:
                all_mins.append(m.vertices.min(axis=0))
                all_maxs.append(m.vertices.max(axis=0))
        if all_mins:
            bounds = [np.min(all_mins, axis=0).tolist(),
                       np.max(all_maxs, axis=0).tolist()]
        else:
            room = scene_cfg.get("room_size_m", [5.0, 2.8, 5.0])
            bounds = [[0, 0, 0], [room[0], room[2], room[1]]]
    else:
        room = scene_cfg.get("room_size_m", [5.0, 2.8, 5.0])
        bounds = [[0, 0, 0], [room[0], room[2], room[1]]]

    scene = {
        "scene_id": scene_cfg["scene_id"],
        "scene_type": scene_type,
        "source": scene_cfg.get("source", "procedural"),
        "world_bounds_m": [[round(b, 3) for b in row] for row in bounds],
        "room_size_m": scene_cfg.get("room_size_m", [5, 2.8, 5]),
        "meshes": meshes,
        "surfaces": surfaces,
        "static_occluders": occluders,
        "walkable_area": walkable,
        "config": scene_cfg,
    }

    return scene
