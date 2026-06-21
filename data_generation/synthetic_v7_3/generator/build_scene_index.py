#!/usr/bin/env python3
"""
build_scene_index.py - Build scene index from 3D-FRONT + procedural generators.

Scans 3D-FRONT JSON scenes, classifies room types, and also generates
procedural corridor/open_area scenes when 3D-FRONT data is unavailable.
Outputs scene_index.json.
"""

import argparse
import json
import math
import os
import random
from pathlib import Path


# ---------------------------------------------------------------------------
# 3D-FRONT JSON parsing
# ---------------------------------------------------------------------------

FURNITURE_CATEGORIES = {
    "table": ["dining table", "coffee table", "desk", "side table", "table"],
    "sofa": ["sofa", "loveseat", "couch", "armchair"],
    "bed": ["bed", "bunk bed"],
    "chair": ["chair", "dining chair", "stool", "bench"],
    "shelf": ["shelf", "bookshelf", "cabinet", "tv stand", "shelf unit"],
    "lamp": ["lamp", "chandelier", "pendant light", "floor lamp"],
}


def classify_room(furniture_jids, room_type_str):
    """Classify a room into our 4 categories."""
    rt = (room_type_str or "").lower()

    if "dining" in rt or "kitchen" in rt:
        return "tabletop"
    if "bedroom" in rt or "living" in rt or "study" in rt or "bathroom" in rt:
        return "room"
    if "corridor" in rt or "hallway" in rt or "entrance" in rt:
        return "corridor"

    # Fallback: check furniture
    has_table = False
    has_sofa = False
    for jid in (furniture_jids or []):
        j = jid.lower()
        if any(t in j for t in FURNITURE_CATEGORIES["table"]):
            has_table = True
        if any(t in j for t in FURNITURE_CATEGORIES["sofa"]):
            has_sofa = True
    if has_table and not has_sofa:
        return "tabletop"
    return "room"


def estimate_clutter(furniture_count):
    if furniture_count <= 3:
        return "low"
    if furniture_count <= 8:
        return "medium"
    return "high"


def parse_3dfront_scene(json_path, future_model_dir):
    """Parse a single 3D-FRONT JSON file into scene entries."""
    try:
        with open(json_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return []

    scenes = []
    scene_basename = os.path.splitext(os.path.basename(json_path))[0]

    # Build furniture lookup
    furniture_map = {}
    for furn in data.get("furniture", []):
        jid = furn.get("jid", "")
        uid = furn.get("uid", "")
        category = furn.get("category", "")
        title = furn.get("title", "")
        furniture_map[uid] = {
            "jid": jid,
            "category": category,
            "title": title,
        }

    # Parse rooms
    for room in data.get("scene", {}).get("room", []):
        room_type = room.get("type", "")
        room_area = room.get("size", 0)
        children = room.get("children", [])

        # Skip empty rooms
        if room.get("empty", 0) == 1:
            continue

        # Collect furniture in this room
        room_furniture = []
        for child in children:
            ref = child.get("ref", "")
            if ref in furniture_map:
                room_furniture.append(furniture_map[ref])

        # Estimate room dimensions from children positions
        xs, ys, zs = [], [], []
        for ch in children:
            pos = ch.get("pos", [0, 0, 0])
            xs.append(pos[0])
            ys.append(pos[1])
            zs.append(pos[2])

        if xs:
            x_range = max(xs) - min(xs)
            y_range = max(ys) - min(ys)
            z_range = max(zs) - min(zs)

            # Skip overly large rooms (target too hard to see)
            if x_range > 6.0 or z_range > 6.0:
                continue

            size_x = max(2.0, x_range)
            size_z = max(2.0, z_range)
            size_y = max(2.4, min(4.0, y_range)) if y_range > 0.5 else 2.8
        else:
            # Fallback: sqrt of area
            side = max(2.0, math.sqrt(max(1.0, float(room_area)))) if room_area else 4.0
            size_x = side
            size_y = 2.8
            size_z = side

        # Clamp to reasonable values
        size_x = max(1.5, min(15.0, size_x))
        size_y = max(2.4, min(4.0, size_y))
        size_z = max(1.5, min(15.0, size_z))

        scene_type = classify_room(
            [f.get("jid", "") for f in room_furniture],
            room_type,
        )

        has_table = any(
            any(t in f.get("title", "").lower() for t in FURNITURE_CATEGORIES["table"])
            for f in room_furniture
        )
        has_sofa = any(
            any(t in f.get("title", "").lower() for t in FURNITURE_CATEGORIES["sofa"])
            for f in room_furniture
        )

        scene_id = f"front_{scene_type}_{scene_basename}_{len(scenes):03d}"

        scenes.append({
            "scene_id": scene_id,
            "source": "3D-FRONT",
            "scene_path": os.path.abspath(json_path),
            "future_model_dir": future_model_dir,
            "type": scene_type,
            "room_size_m": [round(size_x, 2), round(size_y, 2), round(size_z, 2)],
            "has_table": has_table,
            "has_sofa": has_sofa,
            "furniture_count": len(room_furniture),
            "clutter_level": estimate_clutter(len(room_furniture)),
            "allowed_for_train": True,
        })

    return scenes


def scan_3dfront(front_dir, future_dir):
    """Scan all 3D-FRONT JSON files."""
    if not os.path.isdir(front_dir):
        return []

    scenes = []
    for fname in sorted(os.listdir(front_dir)):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(front_dir, fname)
        scenes.extend(parse_3dfront_scene(fpath, future_dir))
    return scenes


# ---------------------------------------------------------------------------
# Procedural scene generators
# ---------------------------------------------------------------------------

def generate_procedural_corridor(index, seed=42):
    """Generate a procedural corridor scene."""
    rng = random.Random(seed + index)
    length = round(rng.uniform(4.0, 12.0), 1)
    width = round(rng.uniform(1.2, 2.5), 1)
    height = round(rng.uniform(2.6, 3.2), 1)
    return {
        "scene_id": f"proc_corridor_{index:04d}",
        "source": "procedural",
        "scene_path": None,
        "future_model_dir": None,
        "type": "corridor",
        "room_size_m": [length, height, width],
        "has_table": False,
        "has_sofa": False,
        "furniture_count": 0,
        "clutter_level": "low",
        "procedural_params": {
            "length": length,
            "width": width,
            "height": height,
            "wall_material": rng.choice(["concrete", "plaster", "brick"]),
            "floor_material": rng.choice(["tile", "wood", "concrete"]),
            "has_doors": rng.choice([True, False]),
            "num_doors": rng.randint(0, 4) if rng.random() > 0.3 else 0,
        },
        "allowed_for_train": True,
    }


def generate_procedural_open_area(index, seed=42):
    """Generate a procedural open area scene."""
    rng = random.Random(seed + index + 10000)
    size_x = round(rng.uniform(6.0, 20.0), 1)
    size_z = round(rng.uniform(6.0, 20.0), 1)
    return {
        "scene_id": f"proc_open_{index:04d}",
        "source": "procedural",
        "scene_path": None,
        "future_model_dir": None,
        "type": "open_area",
        "room_size_m": [size_x, 3.0, size_z],
        "has_table": False,
        "has_sofa": False,
        "furniture_count": 0,
        "clutter_level": "low",
        "procedural_params": {
            "size_x": size_x,
            "size_z": size_z,
            "height": 3.0,
            "ground_material": rng.choice(["grass", "concrete", "dirt", "asphalt"]),
            "sky_hdri": True,
            "has_walls": rng.choice([True, False]),
            "boundary_type": rng.choice(["fence", "wall", "open", "hedge"]),
        },
        "allowed_for_train": True,
    }


def generate_procedural_room(index, seed=42):
    """Generate a procedural room scene."""
    rng = random.Random(seed + index + 30000)
    size_x = round(rng.uniform(3.0, 7.0), 1)
    size_z = round(rng.uniform(3.0, 7.0), 1)
    height = round(rng.uniform(2.6, 3.2), 1)
    room_types = ["living_room", "bedroom", "study", "dining_room"]
    room_type = rng.choice(room_types)
    clutter = rng.choice(["low", "medium", "high"])
    furniture_count = {"low": rng.randint(1, 3), "medium": rng.randint(4, 7), "high": rng.randint(8, 14)}[clutter]
    has_table = rng.random() > 0.3
    has_sofa = room_type in ("living_room",) or rng.random() > 0.6
    return {
        "scene_id": f"proc_room_{index:04d}",
        "source": "procedural",
        "scene_path": None,
        "future_model_dir": None,
        "type": "room",
        "room_size_m": [size_x, height, size_z],
        "has_table": has_table,
        "has_sofa": has_sofa,
        "furniture_count": furniture_count,
        "clutter_level": clutter,
        "procedural_params": {
            "size_x": size_x,
            "size_z": size_z,
            "height": height,
            "room_subtype": room_type,
            "wall_material": rng.choice(["paint_white", "paint_beige", "wallpaper", "wood_panel"]),
            "floor_material": rng.choice(["wood_oak", "wood_walnut", "carpet", "tile", "laminate"]),
            "has_windows": rng.choice([True, False]),
            "num_windows": rng.randint(0, 3),
            "has_ceiling_light": True,
        },
        "allowed_for_train": True,
    }


def generate_procedural_tabletop(index, seed=42):
    """Generate a procedural tabletop scene."""
    rng = random.Random(seed + index + 20000)
    table_x = round(rng.uniform(0.6, 2.0), 2)
    table_z = round(rng.uniform(0.6, 1.5), 2)
    table_h = round(rng.uniform(0.7, 0.85), 2)
    return {
        "scene_id": f"proc_tabletop_{index:04d}",
        "source": "procedural",
        "scene_path": None,
        "future_model_dir": None,
        "type": "tabletop",
        "room_size_m": [table_x, table_h, table_z],
        "has_table": True,
        "has_sofa": False,
        "furniture_count": 1,
        "clutter_level": "low",
        "procedural_params": {
            "table_size": [table_x, 0.04, table_z],
            "table_height": table_h,
            "table_material": rng.choice(["wood_oak", "wood_walnut", "white_laminate", "glass"]),
            "surface_items": rng.randint(0, 5),
            "background": rng.choice(["studio_hdri", "room_wall", "outdoor"]),
        },
        "allowed_for_train": True,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build scene index")
    parser.add_argument("--front-dir", default="assets/raw/3dfront",
                        help="Path to 3D-FRONT JSON directory")
    parser.add_argument("--future-dir", default="assets/raw/3dfuture",
                        help="Path to 3D-FUTURE model directory")
    parser.add_argument("--output", default="assets/index/scene_index.json")
    parser.add_argument("--num-rooms", type=int, default=20)
    parser.add_argument("--num-corridors", type=int, default=20)
    parser.add_argument("--num-open", type=int, default=15)
    parser.add_argument("--num-tabletop", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    scenes = []

    # 1. Scan 3D-FRONT
    front_scenes = scan_3dfront(
        os.path.abspath(args.front_dir),
        os.path.abspath(args.future_dir),
    )
    print(f"3D-FRONT scenes found: {len(front_scenes)}")
    scenes.extend(front_scenes)

    # 2. Generate procedural scenes to fill gaps
    types_needed = {"room": args.num_rooms,
                    "corridor": args.num_corridors,
                    "open_area": args.num_open,
                    "tabletop": args.num_tabletop}

    # Count what we already have
    for s in scenes:
        t = s["type"]
        if t in types_needed:
            types_needed[t] = max(0, types_needed[t] - 1)

    gen_funcs = {
        "room": generate_procedural_room,
        "corridor": generate_procedural_corridor,
        "open_area": generate_procedural_open_area,
        "tabletop": generate_procedural_tabletop,
    }

    offset = len(scenes)
    for scene_type, count in types_needed.items():
        gen_func = gen_funcs[scene_type]
        for i in range(count):
            scenes.append(gen_func(offset + i, seed=args.seed))
        offset += count
        if count > 0:
            print(f"Generated {count} procedural {scene_type} scenes")

    # 3. Classify 3D-FRONT rooms that are tabletop
    # (already done in parse_3dfront_scene)

    # Sort: 3D-FRONT first, then procedural
    scenes.sort(key=lambda x: (0 if x["source"] == "3D-FRONT" else 1, x["scene_id"]))

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(os.path.abspath(args.output), "w") as f:
        json.dump(scenes, f, indent=2)

    print(f"\nTotal scenes: {len(scenes)}")
    types = {}
    sources = {}
    for s in scenes:
        types[s["type"]] = types.get(s["type"], 0) + 1
        sources[s["source"]] = sources.get(s["source"], 0) + 1
    print("By type:")
    for t, c in sorted(types.items()):
        print(f"  {t}: {c}")
    print("By source:")
    for s, c in sorted(sources.items()):
        print(f"  {s}: {c}")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
