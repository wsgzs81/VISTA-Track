#!/usr/bin/env python3
"""
build_3dfront_target_index.py — Build target_index.json from 3D-FUTURE-model.

Each 3D-FUTURE model directory has:
    raw_model.obj   — original mesh
    normalized_model.obj — normalized mesh (fits unit cube)
    model.mtl       — material file
    texture.png     — diffuse texture
    image.jpg       — rendered preview

Output: target_index.json with entries:
    {
      "target_id": "3dft_0fee7a5a",
      "source": "3d-future",
      "jid": "0fee7a5a-5478-4c0d-af5f-e6dd0159bf6d",
      "mesh_path": ".../raw_model.obj",
      "texture_path": ".../texture.png",
      "preview_path": ".../image.jpg",
      "category": "Table",
      "subcategory": "side table",
      "bbox_m": [w, h, d],
      "vertices": 0,
      "faces": 0
    }

Usage:
    python build_3dfront_target_index.py --model-dir /path/to/3D-FUTURE-model
    python build_3dfront_target_index.py --model-dir /path/to/3D-FUTURE-model --merge
"""

import json
import os
import sys


# Category mapping from 3D-FUTURE titles to our categories
CATEGORY_MAP = {
    "table": "Table",
    "chair": "Chair",
    "sofa": "Sofa",
    "cabinet": "Cabinet",
    "shelf": "Cabinet",
    "desk": "Cabinet",
    "bed": "Bed",
    "lamp": "Lamp",
    "light": "Lamp",
    "ottoman": "Sofa",
    "storage": "Cabinet",
    "armoire": "Cabinet",
    "dresser": "Cabinet",
    "bookcase": "Cabinet",
    "nightstand": "Table",
    "wardrobe": "Cabinet",
    "tv": "Electronics",
    "monitor": "Electronics",
    "screen": "Electronics",
    "rug": "Decor",
    "curtain": "Decor",
    "mirror": "Decor",
    "plant": "Decor",
}


def infer_category(title, category_hint=""):
    """Infer our category from 3D-FUTURE title or category hint."""
    title_lower = title.lower() if title else ""
    cat_lower = category_hint.lower() if category_hint else ""

    for keyword, cat in CATEGORY_MAP.items():
        if keyword in title_lower or keyword in cat_lower:
            return cat
    return "misc"


def parse_subcategory(title):
    """Extract subcategory from title like 'table/side table' → 'side table'."""
    if not title:
        return ""
    parts = title.split("/")
    return parts[-1].strip() if len(parts) > 1 else title.strip()


def extract_bbox_from_obj(obj_path):
    """Read OBJ file and extract bounding box [w, h, d] in meters.

    Returns (bbox, vertex_count, face_count) or ([0,0,0], 0, 0) on failure.
    """
    min_v = [float("inf")] * 3
    max_v = [float("-inf")] * 3
    v_count = 0
    f_count = 0

    try:
        with open(obj_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("v "):
                    parts = line.split()
                    if len(parts) >= 4:
                        v = [float(parts[1]), float(parts[2]), float(parts[3])]
                        for i in range(3):
                            min_v[i] = min(min_v[i], v[i])
                            max_v[i] = max(max_v[i], v[i])
                        v_count += 1
                elif line.startswith("f "):
                    f_count += 1
    except Exception:
        return [0, 0, 0], 0, 0

    if v_count == 0:
        return [0, 0, 0], 0, 0

    bbox = [max_v[i] - min_v[i] for i in range(3)]
    return bbox, v_count, f_count


def build_index(model_dir, metadata_path=None):
    """Build target index from 3D-FUTURE-model directory.

    Args:
        model_dir: path to extracted 3D-FUTURE-model/
        metadata_path: optional path to 3D-FUTURE model_info.json

    Returns:
        list of target entries
    """
    # Load metadata if available
    metadata = {}
    if metadata_path and os.path.isfile(metadata_path):
        with open(metadata_path) as f:
            for item in json.load(f):
                jid = item.get("model_id", item.get("jid", ""))
                if jid:
                    metadata[jid] = item

    entries = []
    model_dirs = sorted([
        d for d in os.listdir(model_dir)
        if os.path.isdir(os.path.join(model_dir, d))
    ])

    print(f"Scanning {len(model_dirs)} model directories in {model_dir}...")

    for i, mid in enumerate(model_dirs):
        mdir = os.path.join(model_dir, mid)
        obj_path = os.path.join(mdir, "raw_model.obj")
        norm_path = os.path.join(mdir, "normalized_model.obj")
        tex_path = os.path.join(mdir, "texture.png")
        preview_path = os.path.join(mdir, "image.jpg")

        # Use normalized model if available, else raw
        mesh_path = norm_path if os.path.isfile(norm_path) else obj_path
        if not os.path.isfile(mesh_path):
            continue

        # Extract bbox from mesh
        bbox, v_count, f_count = extract_bbox_from_obj(mesh_path)

        # Get category from metadata
        meta = metadata.get(mid, {})
        title = meta.get("title", "")
        category_hint = meta.get("super-category", meta.get("category", ""))
        category = infer_category(title, category_hint)
        subcategory = parse_subcategory(title)

        entry = {
            "target_id": f"3dft_{mid[:8]}",
            "source": "3d-future",
            "jid": mid,
            "mesh_path": mesh_path,
            "texture_path": tex_path if os.path.isfile(tex_path) else "",
            "preview_path": preview_path if os.path.isfile(preview_path) else "",
            "category": category,
            "subcategory": subcategory,
            "bbox_m": [round(b, 4) for b in bbox],
            "vertices": v_count,
            "faces": f_count,
        }

        entries.append(entry)

        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(model_dirs)} scanned, {len(entries)} valid")

    print(f"  Total: {len(entries)} models indexed")
    return entries


def merge_with_existing(new_entries, existing_index_path):
    """Merge new entries with existing target_index.json.

    Avoids duplicates by jid.
    """
    existing = []
    if os.path.isfile(existing_index_path):
        with open(existing_index_path) as f:
            existing = json.load(f)

    existing_jids = {e.get("jid", e.get("target_id", "")) for e in existing}
    added = 0
    for entry in new_entries:
        if entry["jid"] not in existing_jids:
            existing.append(entry)
            existing_jids.add(entry["jid"])
            added += 1

    return existing, added


def print_category_stats(entries):
    """Print category distribution."""
    cats = {}
    for e in entries:
        c = e.get("category", "unknown")
        cats[c] = cats.get(c, 0) + 1

    print("\nCategory distribution:")
    for c, n in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  {c:15s}: {n:5d}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build target index from 3D-FUTURE-model")
    parser.add_argument("--model-dir", required=True,
                        help="Path to extracted 3D-FUTURE-model/")
    parser.add_argument("--metadata", default=None,
                        help="Path to 3D-FUTURE model_info.json (optional)")
    parser.add_argument("--output", default="assets/index/target_index.json")
    parser.add_argument("--merge", action="store_true",
                        help="Merge with existing index")
    args = parser.parse_args()

    entries = build_index(args.model_dir, metadata_path=args.metadata)
    print_category_stats(entries)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    if args.merge:
        entries, added = merge_with_existing(entries, args.output)
        print(f"\nMerged: {added} new entries, {len(entries)} total")
    else:
        print(f"\nWriting {len(entries)} entries to {args.output}")

    with open(args.output, "w") as f:
        json.dump(entries, f, indent=2)
    print(f"  Saved: {args.output}")
