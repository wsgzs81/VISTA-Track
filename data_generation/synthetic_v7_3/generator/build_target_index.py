#!/usr/bin/env python3
"""
build_target_index.py - Rebuild target_index.json from cleaned assets.

Scans assets/cleaned/targets/, reads mesh metadata, and writes the
consolidated index. Can also filter / re-score without reprocessing.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import trimesh


def inspect_cleaned_asset(asset_dir: str, asset_id: str) -> dict | None:
    """Read metadata from a cleaned target directory."""
    glb_path = os.path.join(asset_dir, "model.glb")
    if not os.path.isfile(glb_path):
        return None

    try:
        mesh = trimesh.load(glb_path, force="mesh")
    except Exception:
        return None

    has_tex = os.path.isfile(os.path.join(asset_dir, "texture.png"))

    # Also check for embedded textures in GLB
    if not has_tex:
        try:
            scene = trimesh.load(glb_path, force="scene")
            if isinstance(scene, trimesh.Scene):
                for geom in scene.geometry.values():
                    if hasattr(geom, "visual") and hasattr(geom.visual, "material"):
                        mat = geom.visual.material
                        if hasattr(mat, "image") and mat.image is not None:
                            has_tex = True
                            break
        except Exception:
            pass

    bbox_m = [round(float(x), 4) for x in mesh.extents]

    return {
        "asset_id": asset_id,
        "source": asset_id.split("_")[0].upper() if "_" in asset_id else "unknown",
        "category": "unknown",
        "mesh_path": os.path.relpath(glb_path, os.path.dirname(os.path.dirname(asset_dir))),
        "license": "see_source",
        "bbox_3d_m": bbox_m,
        "num_vertices": int(mesh.vertices.shape[0]),
        "num_faces": int(mesh.faces.shape[0]),
        "has_texture": has_tex,
        "connected_components": len(mesh.split()) if hasattr(mesh, "split") else 1,
        "quality_score": 1.0,
        "allowed_for_train": True,
    }


def main():
    parser = argparse.ArgumentParser(description="Build target index from cleaned assets")
    parser.add_argument("--cleaned-dir", default="assets/cleaned/targets")
    parser.add_argument("--output", default="assets/index/target_index.json")
    args = parser.parse_args()

    cleaned_dir = os.path.abspath(args.cleaned_dir)
    output_path = os.path.abspath(args.output)

    if not os.path.isdir(cleaned_dir):
        print(f"Error: {cleaned_dir} does not exist")
        sys.exit(1)

    entries = []
    dirs = sorted(os.listdir(cleaned_dir))
    print(f"Scanning {len(dirs)} directories in {cleaned_dir}")

    for i, name in enumerate(dirs):
        asset_dir = os.path.join(cleaned_dir, name)
        if not os.path.isdir(asset_dir):
            continue
        rec = inspect_cleaned_asset(asset_dir, name)
        if rec is not None:
            entries.append(rec)
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(dirs)}] indexed={len(entries)}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(entries, f, indent=2)

    print(f"Index: {len(entries)} entries -> {output_path}")

    # Summary
    sources = {}
    for e in entries:
        s = e["source"]
        sources[s] = sources.get(s, 0) + 1
    for s, c in sorted(sources.items()):
        print(f"  {s}: {c}")


if __name__ == "__main__":
    import sys
    main()
