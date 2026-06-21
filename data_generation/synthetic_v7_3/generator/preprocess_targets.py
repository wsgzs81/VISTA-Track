#!/usr/bin/env python3
"""
preprocess_targets.py - Clean and validate 3D target assets.

Discards assets that fail quality checks (empty mesh, too many faces,
no texture, bad bounding box, etc.) and copies passing assets to
assets/cleaned/targets/.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import trimesh

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
MAX_FACES = 400_000
MAX_VERTICES = 200_000
MIN_BBOX_SIDE_M = 0.01      # 1 cm - reject micro meshes
MAX_BBOX_SIDE_M = 1.5       # 1.5 m - reject room-scale meshes
MAX_CONNECTED_COMPONENTS = 5
MIN_FACES = 20               # reject degenerate / trivial meshes
MIN_VERTICES = 10


# ---------------------------------------------------------------------------
# Mesh loading
# ---------------------------------------------------------------------------
def load_gso_mesh(obj_dir: str) -> trimesh.Trimesh | None:
    """Load the main model.obj from a GSO object directory."""
    obj_path = os.path.join(obj_dir, "model.obj")
    if not os.path.isfile(obj_path):
        return None
    try:
        mesh = trimesh.load(obj_path, force="mesh", skip_materials=True)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(mesh.dump())
        return mesh
    except Exception:
        return None


def load_objaverse_mesh(glb_path: str) -> trimesh.Trimesh | None:
    """Load a .glb file (Objaverse)."""
    try:
        loaded = trimesh.load(glb_path, force="mesh")
        if isinstance(loaded, trimesh.Scene):
            parts = []
            for geom in loaded.geometry.values():
                if isinstance(geom, trimesh.Trimesh):
                    parts.append(geom)
            if not parts:
                return None
            mesh = trimesh.util.concatenate(parts)
        else:
            mesh = loaded
        return mesh
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Texture check
# ---------------------------------------------------------------------------
def has_texture_gso(obj_dir: str) -> bool:
    tex = os.path.join(obj_dir, "texture.png")
    if os.path.isfile(tex) and os.path.getsize(tex) > 100:
        return True
    # Also check for any image in materials/textures/
    mat_dir = os.path.join(obj_dir, "materials", "textures")
    if os.path.isdir(mat_dir):
        for f in os.listdir(mat_dir):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                return True
    return False


def has_texture_objaverse(glb_path: str) -> bool:
    """GLB embeds textures; check via trimesh visual."""
    try:
        loaded = trimesh.load(glb_path, force="scene")
        if isinstance(loaded, trimesh.Scene):
            for geom in loaded.geometry.values():
                if hasattr(geom, "visual") and hasattr(geom.visual, "material"):
                    mat = geom.visual.material
                    if hasattr(mat, "image") and mat.image is not None:
                        return True
        return False
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Quality checks
# ---------------------------------------------------------------------------
def compute_quality_score(
    mesh: trimesh.Trimesh,
    has_tex: bool,
    bbox_ok: bool,
) -> float:
    score = 1.0

    # Penalize missing texture
    if not has_tex:
        score -= 0.2

    # Penalize high connected components
    try:
        cc = len(mesh.split())
    except Exception:
        cc = 1
    if cc > 3:
        score -= 0.05 * (cc - 3)

    # Penalize non-manifold edges
    try:
        non_manifold = np.sum(mesh.get_face_adjacency_edges() == -1)
        if non_manifold > 0:
            score -= 0.1
    except Exception:
        pass

    # Penalize bad bbox
    if not bbox_ok:
        score -= 0.3

    return round(max(0.0, min(1.0, score)), 2)


def check_mesh(mesh: trimesh.Trimesh):
    """Run all quality checks. Returns (passed: bool, info: dict, reason: str)."""
    info = {}

    # Non-empty
    if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
        return False, info, "empty mesh"

    nv = mesh.vertices.shape[0]
    nf = mesh.faces.shape[0]
    info["num_vertices"] = int(nv)
    info["num_faces"] = int(nf)

    # Face / vertex limits
    if nf < MIN_FACES or nv < MIN_VERTICES:
        return False, info, f"too few geometry (v={nv} f={nf})"
    if nf > MAX_FACES:
        return False, info, f"too many faces ({nf} > {MAX_FACES})"
    if nv > MAX_VERTICES:
        return False, info, f"too many vertices ({nv} > {MAX_VERTICES})"

    # Bounding box
    extents = mesh.extents  # [dx, dy, dz] in mesh units
    info["bbox_raw"] = [round(float(x), 6) for x in extents]
    longest = float(np.max(extents))
    if longest < 1e-6:
        return False, info, "degenerate bbox"

    # Connected components
    try:
        components = mesh.split()
        n_cc = len(components)
    except Exception:
        n_cc = 1
    info["connected_components"] = n_cc
    if n_cc > MAX_CONNECTED_COMPONENTS:
        return False, info, f"too many components ({n_cc})"

    return True, info, "ok"


def normalize_and_export(
    mesh: trimesh.Trimesh,
    target_longest_m: float = 0.3,
):
    """Normalize mesh so its longest side == target_longest_m, center at origin."""
    longest = float(np.max(mesh.extents))
    if longest < 1e-6:
        return None, None
    scale = target_longest_m / longest
    mesh.apply_scale(scale)
    mesh.vertices -= mesh.centroid
    bbox_m = [round(float(x), 4) for x in mesh.extents]
    return mesh, bbox_m


# ---------------------------------------------------------------------------
# Process single asset
# ---------------------------------------------------------------------------
def process_gso_object(
    obj_dir: str,
    asset_id: str,
    out_dir: str,
) -> dict | None:
    mesh = load_gso_mesh(obj_dir)
    if mesh is None:
        return None

    passed, info, reason = check_mesh(mesh)
    if not passed:
        return None

    has_tex = has_texture_gso(obj_dir)

    # Normalize
    mesh_norm, bbox_m = normalize_and_export(mesh)
    if mesh_norm is None:
        return None

    # Bbox sanity after normalization
    bbox_ok = all(MIN_BBOX_SIDE_M <= s <= MAX_BBOX_SIDE_M for s in bbox_m)

    quality = compute_quality_score(mesh_norm, has_tex, bbox_ok)

    # Export
    target_subdir = os.path.join(out_dir, asset_id)
    os.makedirs(target_subdir, exist_ok=True)
    glb_path = os.path.join(target_subdir, "model.glb")
    mesh_norm.export(glb_path)

    # Copy texture if exists
    tex_src = os.path.join(obj_dir, "texture.png")
    if os.path.isfile(tex_src):
        shutil.copy2(tex_src, os.path.join(target_subdir, "texture.png"))

    return {
        "asset_id": asset_id,
        "source": "GSO",
        "category": "unknown",
        "mesh_path": os.path.relpath(glb_path, os.path.dirname(out_dir)),
        "license": "CC-BY-4.0",
        "bbox_3d_m": bbox_m,
        "num_vertices": info["num_vertices"],
        "num_faces": info["num_faces"],
        "has_texture": has_tex,
        "connected_components": info.get("connected_components", 1),
        "quality_score": quality,
        "allowed_for_train": quality >= 0.5,
    }


def process_objaverse_object(
    glb_path: str,
    asset_id: str,
    out_dir: str,
) -> dict | None:
    mesh = load_objaverse_mesh(glb_path)
    if mesh is None:
        return None

    passed, info, reason = check_mesh(mesh)
    if not passed:
        return None

    has_tex = has_texture_objaverse(glb_path)

    mesh_norm, bbox_m = normalize_and_export(mesh)
    if mesh_norm is None:
        return None

    bbox_ok = all(MIN_BBOX_SIDE_M <= s <= MAX_BBOX_SIDE_M for s in bbox_m)
    quality = compute_quality_score(mesh_norm, has_tex, bbox_ok)

    target_subdir = os.path.join(out_dir, asset_id)
    os.makedirs(target_subdir, exist_ok=True)
    out_glb = os.path.join(target_subdir, "model.glb")
    mesh_norm.export(out_glb)

    return {
        "asset_id": asset_id,
        "source": "Objaverse",
        "category": "unknown",
        "mesh_path": os.path.relpath(out_glb, os.path.dirname(out_dir)),
        "license": "Objaverse-1.0",
        "bbox_3d_m": bbox_m,
        "num_vertices": info["num_vertices"],
        "num_faces": info["num_faces"],
        "has_texture": has_tex,
        "connected_components": info.get("connected_components", 1),
        "quality_score": quality,
        "allowed_for_train": quality >= 0.5,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def discover_assets(raw_dir: str):
    """Yield (source, asset_id, path) for every raw asset."""
    gso_dir = os.path.join(raw_dir, "gso")
    if os.path.isdir(gso_dir):
        for name in sorted(os.listdir(gso_dir)):
            obj_dir = os.path.join(gso_dir, name)
            if os.path.isdir(obj_dir) and os.path.isfile(os.path.join(obj_dir, "model.obj")):
                yield ("GSO", f"gso_{name}", obj_dir)

    obj_dir = os.path.join(raw_dir, "objaverse", "glbs")
    if os.path.isdir(obj_dir):
        for sub in sorted(os.listdir(obj_dir)):
            sub_path = os.path.join(obj_dir, sub)
            if os.path.isdir(sub_path):
                for f in sorted(os.listdir(sub_path)):
                    if f.endswith(".glb"):
                        uid = f.replace(".glb", "")
                        yield ("Objaverse", f"obj_{uid}", os.path.join(sub_path, f))


def main():
    parser = argparse.ArgumentParser(description="Preprocess target assets")
    parser.add_argument("--raw-dir", default="assets/raw", help="Root of raw assets")
    parser.add_argument("--out-dir", default="assets/cleaned/targets", help="Output directory")
    parser.add_argument("--max-assets", type=int, default=0, help="Max assets to process (0=all)")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers (future use)")
    args = parser.parse_args()

    raw_dir = os.path.abspath(args.raw_dir)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    assets = list(discover_assets(raw_dir))
    print(f"Discovered {len(assets)} raw assets")

    if args.max_assets > 0:
        assets = assets[: args.max_assets]

    results = []
    stats = {"total": 0, "passed": 0, "failed": 0, "fail_reasons": {}}

    for i, (source, asset_id, path) in enumerate(assets):
        stats["total"] += 1
        if source == "GSO":
            rec = process_gso_object(path, asset_id, out_dir)
        else:
            rec = process_objaverse_object(path, asset_id, out_dir)

        if rec is None:
            stats["failed"] += 1
        else:
            results.append(rec)
            stats["passed"] += 1

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(assets)}] passed={stats['passed']} failed={stats['failed']}")

    print(f"\nDone: {stats['passed']}/{stats['total']} passed, {stats['failed']} failed")

    # Save index
    index_path = os.path.join(os.path.dirname(out_dir), "index", "target_index.json")
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    with open(index_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Index saved to {index_path} ({len(results)} entries)")


if __name__ == "__main__":
    main()
