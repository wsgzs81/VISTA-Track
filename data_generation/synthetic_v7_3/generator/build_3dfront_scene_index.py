#!/usr/bin/env python3
"""
build_3dfront_scene_index.py — Build scene_index.json from 3D-FRONT JSONs.

Each 3D-FRONT JSON describes a house with:
    furniture: [{jid, title, category, size, bbox, pos, rot, scale}, ...]
    scene: tree of mesh instances with boundingBox
    boundingBox: room dimensions

Output: scene_index.json with entries:
    {
      "scene_id": "front_03c8c59c",
      "source": "3d-front",
      "uid": "03c8c59c-ba14-4486-ba57-3cd79b7f3eb9",
      "json_path": ".../03c8...json",
      "room_type": "LivingRoom",
      "room_size_m": [8.4, 2.8, 12.2],
      "num_furniture": 25,
      "furniture_jids": ["93a6de7f-...", ...],
      "occluder_candidates": [...],
      "material_ids": [...]
    }

Usage:
    python build_3dfront_scene_index.py --front-dir /path/to/3D-FRONT
    python build_3dfront_scene_index.py --front-dir /path/to/3D-FRONT --merge
"""

import json
import math
import os
import sys


# Room type inference from furniture composition
ROOM_SIGNATURES = {
    "LivingRoom": ["sofa", "tv", "coffee table"],
    "Bedroom": ["bed", "nightstand", "wardrobe"],
    "DiningRoom": ["dining table", "chair"],
    "Kitchen": ["cabinet", "kitchen"],
    "Bathroom": ["toilet", "sink", "bathtub"],
    "Office": ["desk", "chair", "bookcase"],
}


def infer_room_type(furniture_list):
    """Infer room type from furniture titles."""
    titles = " ".join(
        f.get("title", "").lower() for f in furniture_list
    )

    scores = {}
    for room_type, keywords in ROOM_SIGNATURES.items():
        score = sum(1 for kw in keywords if kw in titles)
        if score > 0:
            scores[room_type] = score

    if scores:
        return max(scores, key=scores.get)
    return "LivingRoom"  # default


def compute_room_size(bounding_box):
    """Compute room size [w, h, d] from boundingBox min/max."""
    if not bounding_box:
        return [5.0, 2.8, 5.0]

    mn = bounding_box.get("min", [0, 0, 0])
    mx = bounding_box.get("max", [5, 2.8, 5])

    # 3D-FRONT z-axis can be negative
    w = abs(mx[0] - mn[0])
    h = abs(mx[1] - mn[1])
    d = abs(mx[2] - mn[2])

    # Clamp to reasonable values
    w = max(2.0, min(w, 20.0))
    h = max(2.0, min(h, 5.0))
    d = max(2.0, min(d, 20.0))

    return [round(w, 2), round(h, 2), round(d, 2)]


def get_occluder_candidates(furniture_list):
    """Identify furniture items good for occlusion (tall, wide objects).

    Returns list of {jid, title, size, category}.
    """
    candidates = []
    occluder_categories = {"Cabinet/Shelf/Desk", "Sofa", "Bed", "Table"}

    for f in furniture_list:
        jid = f.get("jid", "")
        title = f.get("title", "")
        category = f.get("category", "")
        size = f.get("size", [0, 0, 0])

        if not jid or not size or len(size) < 3:
            continue

        w, h, d = size[0], size[1], size[2]

        # Good occluder: at least 0.5m tall, has decent footprint
        is_tall = h >= 0.5
        has_footprint = w * d >= 0.1
        is_furniture = category in occluder_categories or any(
            kw in title.lower() for kw in ["table", "sofa", "bed", "cabinet",
                                             "shelf", "desk", "chair", "storage"]
        )

        if is_tall and has_footprint and is_furniture:
            candidates.append({
                "jid": jid,
                "title": title,
                "size": [round(w, 3), round(h, 3), round(d, 3)],
                "category": category,
            })

    return candidates


def build_index(front_dir, texture_dir=None):
    """Build scene index from 3D-FRONT directory.

    Args:
        front_dir: path to extracted 3D-FRONT/ (containing .json files)
        texture_dir: path to 3D-FRONT-texture/

    Returns:
        list of scene entries
    """
    json_files = sorted([
        f for f in os.listdir(front_dir)
        if f.endswith(".json")
    ])

    print(f"Scanning {len(json_files)} house files in {front_dir}...")

    entries = []
    for i, jf in enumerate(json_files):
        jpath = os.path.join(front_dir, jf)
        try:
            with open(jpath) as f:
                data = json.load(f)
        except Exception as e:
            continue

        uid = data.get("uid", jf.replace(".json", ""))
        furniture = data.get("furniture", [])

        # Only keep houses with enough furniture for interesting scenes
        valid_furniture = [
            f for f in furniture
            if f.get("size") and len(f.get("size", [])) >= 3
        ]

        if len(valid_furniture) < 3:
            continue

        # Room geometry
        scene = data.get("scene", {})
        bbox = None
        if isinstance(scene, dict):
            bbox = scene.get("boundingBox")
        room_size = compute_room_size(bbox)

        # Room type
        room_type = infer_room_type(valid_furniture)

        # Occluder candidates
        occluders = get_occluder_candidates(valid_furniture)

        # Material IDs
        material_ids = data.get("materialList", [])

        # Furniture jids (link to 3D-FUTURE models)
        jids = [f["jid"] for f in valid_furniture if f.get("jid")]

        entry = {
            "scene_id": f"front_{uid[:8]}",
            "source": "3d-front",
            "uid": uid,
            "json_path": jpath,
            "room_type": room_type,
            "room_size_m": room_size,
            "num_furniture": len(valid_furniture),
            "furniture_jids": jids,
            "occluder_candidates": occluders,
            "material_ids": material_ids,
        }

        entries.append(entry)

        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(json_files)} scanned, {len(entries)} valid")

    print(f"  Total: {len(entries)} scenes indexed")
    return entries


def merge_with_existing(new_entries, existing_index_path):
    """Merge new entries, avoiding duplicates by uid."""
    existing = []
    if os.path.isfile(existing_index_path):
        with open(existing_index_path) as f:
            existing = json.load(f)

    existing_uids = {e.get("uid", e.get("scene_id", "")) for e in existing}
    added = 0
    for entry in new_entries:
        if entry["uid"] not in existing_uids:
            existing.append(entry)
            existing_uids.add(entry["uid"])
            added += 1

    return existing, added


def print_stats(entries):
    """Print dataset statistics."""
    room_types = {}
    sizes = []
    occluder_counts = []

    for e in entries:
        rt = e.get("room_type", "unknown")
        room_types[rt] = room_types.get(rt, 0) + 1
        sizes.append(e.get("room_size_m", [0, 0, 0]))
        occluder_counts.append(len(e.get("occluder_candidates", [])))

    print(f"\nRoom type distribution:")
    for rt, n in sorted(room_types.items(), key=lambda x: -x[1]):
        print(f"  {rt:15s}: {n:5d}")

    if sizes:
        import statistics
        areas = [s[0] * s[2] for s in sizes]
        print(f"\nRoom area (m²):")
        print(f"  mean={statistics.mean(areas):.1f}  "
              f"median={statistics.median(areas):.1f}  "
              f"min={min(areas):.1f}  max={max(areas):.1f}")

    if occluder_counts:
        print(f"\nOccluder candidates per scene:")
        print(f"  mean={statistics.mean(occluder_counts):.1f}  "
              f"max={max(occluder_counts)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build scene index from 3D-FRONT")
    parser.add_argument("--front-dir", required=True,
                        help="Path to extracted 3D-FRONT/")
    parser.add_argument("--texture-dir", default=None,
                        help="Path to 3D-FRONT-texture/ (optional)")
    parser.add_argument("--output", default="assets/index/scene_index.json")
    parser.add_argument("--merge", action="store_true",
                        help="Merge with existing index")
    args = parser.parse_args()

    entries = build_index(args.front_dir, texture_dir=args.texture_dir)
    print_stats(entries)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    if args.merge:
        entries, added = merge_with_existing(entries, args.output)
        print(f"\nMerged: {added} new entries, {len(entries)} total")
    else:
        print(f"\nWriting {len(entries)} entries to {args.output}")

    with open(args.output, "w") as f:
        json.dump(entries, f, indent=2)
    print(f"  Saved: {args.output}")
