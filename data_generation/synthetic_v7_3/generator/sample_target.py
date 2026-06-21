#!/usr/bin/env python3
"""
sample_target.py - Target asset sampling with category quotas.

Loads meshes from target_index.json, applies scale normalization
and material perturbation.
"""

import json
import os
import random
import re

import numpy as np
import trimesh


# ---------------------------------------------------------------------------
# Category inference
# ---------------------------------------------------------------------------

CATEGORY_RULES = {
    "container": [
        "bottle", "cup", "mug", "jar", "box", "bin", "basket", "bowl",
        "container", "pot", "vase", "flask", "can", "tin", "jug",
        "pitcher", "tub", "bucket", "pail", "crate", "drawer",
        "storage", "organizer", "tote", "bag", "pack",
    ],
    "household": [
        "mop", "broom", "towel", "soap", "detergent", "cleaner",
        "sponge", "brush", "cloth", "mat", "rug", "curtain",
        "pillow", "blanket", "lamp", "clock", "mirror", "candle",
        "vase", "plant", "flower", "frame", "decoration", "hook",
        "hanger", "iron", "dryer", "fan", "heater", "humidifier",
    ],
    "toy_sport": [
        "ball", "bat", "racket", "toy", "game", "block", "lego",
        "doll", "car", "train", "puzzle", "dice", "card",
        "football", "basketball", "soccer", "tennis", "golf",
        "frisbee", "skateboard", "bike", "swing", "kite",
        "dinosaur", "animal", "robot", "figure", "action",
        "jenga", "monopoly", "scrabble", "chess", "checker",
        "play", "fun", "kid", "child", "baby",
    ],
    "electronics_tool": [
        "phone", "tablet", "laptop", "computer", "keyboard", "mouse",
        "monitor", "screen", "camera", "speaker", "headphone",
        "charger", "cable", "adapter", "remote", "controller",
        "drill", "hammer", "screwdriver", "wrench", "saw",
        "tool", "knife", "scissors", "tape", "glue",
        "router", "modem", "drive", "disk", "usb", "battery",
        "flashlight", "bulb", "light", "power",
    ],
}

# Compiled regex patterns for faster matching
_CATEGORY_PATTERNS = {}
for cat, keywords in CATEGORY_RULES.items():
    _CATEGORY_PATTERNS[cat] = re.compile(
        r"\b(" + "|".join(keywords) + r")\b", re.IGNORECASE
    )


def infer_category(asset_id, source):
    """Infer category from asset ID/name."""
    name = asset_id.lower()

    # Strip prefix
    name = re.sub(r"^(gso_|obj_)", "", name)

    # Normalize: replace underscores/separators with spaces for matching
    name_norm = re.sub(r"[_\-\.]+", " ", name)

    # Try each category on normalized name
    for cat, pattern in _CATEGORY_PATTERNS.items():
        if pattern.search(name_norm):
            return cat

    # Extra heuristics for common GSO patterns
    extra = {
        "container": ["mug", "bottle", "thermos", "tumbler", "glass", "pitcher",
                       "kettle", "lunch", "box", "pack", "set"],
        "toy_sport": ["game", "toy", "play", "ball", "nintendo", "lego", "figure",
                       "jenga", "doll", "dinosaur", "kid"],
        "household": ["plant", "flower", "candle", "clock", "lamp", "frame",
                       "decor", "ornament", "mop", "broom", "towel"],
        "electronics_tool": ["ink", "cartridge", "drive", "router", "mouse",
                              "keyboard", "speaker", "adapter", "charger",
                              "canon", "hp", "epson", "office", "depot"],
    }
    for cat, keywords in extra.items():
        for kw in keywords:
            if kw in name_norm:
                return cat

    return "misc"


# ---------------------------------------------------------------------------
# Category quotas
# ---------------------------------------------------------------------------

DEFAULT_QUOTAS = {
    "container": 0.30,
    "household": 0.30,
    "toy_sport": 0.20,
    "electronics_tool": 0.15,
    "misc": 0.05,
}


def _get_asset_id(target):
    """Get asset ID from target record (handles both asset_id and target_id)."""
    return target.get("asset_id", target.get("target_id", "unknown"))


def build_category_buckets(targets):
    """Group targets by inferred category."""
    buckets = {cat: [] for cat in DEFAULT_QUOTAS}
    for t in targets:
        aid = _get_asset_id(t)
        cat = infer_category(aid, t.get("source", ""))
        if cat in buckets:
            buckets[cat].append(t)
        else:
            buckets["misc"].append(t)
    return buckets


def sample_by_quota(buckets, quotas, rng):
    """Sample one target respecting category quotas."""
    # Filter to categories that have available targets
    available = {cat: items for cat, items in buckets.items() if items}
    if not available:
        return None

    # Renormalize quotas for available categories
    total_weight = sum(quotas.get(c, 0) for c in available)
    if total_weight == 0:
        # Fallback: uniform
        all_items = []
        for items in available.values():
            all_items.extend(items)
        return rng.choice(all_items)

    weights = [quotas.get(c, 0) / total_weight for c in available]
    chosen_cat = rng.choices(list(available.keys()), weights=weights, k=1)[0]
    return rng.choice(available[chosen_cat])


# ---------------------------------------------------------------------------
# Mesh loading and normalization
# ---------------------------------------------------------------------------

def load_target_mesh(target):
    """Load mesh from target record."""
    # The mesh_path in index is relative to project root
    mesh_path = target.get("mesh_path", "")
    if not mesh_path or not os.path.isfile(mesh_path):
        # Try alternative paths (GLB and OBJ for 3D-FUTURE compatibility)
        asset_id = target.get("asset_id", target.get("target_id", ""))
        alt_paths = [
            f"assets/cleaned/targets/{asset_id}/model.glb",
            f"assets/cleaned/targets/{asset_id}/model.obj",
            os.path.join(os.path.dirname(mesh_path) if mesh_path else "", "model.glb"),
            os.path.join(os.path.dirname(mesh_path) if mesh_path else "", "raw_model.obj"),
            os.path.join(os.path.dirname(mesh_path) if mesh_path else "", "normalized_model.obj"),
        ]
        for p in alt_paths:
            if os.path.isfile(p):
                mesh_path = p
                break
        else:
            return None

    try:
        mesh = trimesh.load(mesh_path, force="mesh")
        if isinstance(mesh, trimesh.Scene):
            parts = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
            if not parts:
                return None
            mesh = trimesh.util.concatenate(parts)
        return mesh
    except Exception:
        return None


def normalize_target_scale(mesh, target_longest=0.3):
    """Normalize mesh so longest bbox side == target_longest meters."""
    longest = float(np.max(mesh.extents))
    if longest < 1e-6:
        return mesh, target_longest
    scale = target_longest / longest
    mesh.apply_scale(scale)
    mesh.vertices -= mesh.centroid
    return mesh, round(scale * longest, 4)


def perturb_material(mesh, rng, intensity=0.3):
    """Apply random color/material perturbation."""
    if mesh.visual is None:
        mesh.visual = trimesh.visual.ColorVisuals()

    # Get current color or default to gray
    if hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None:
        colors = mesh.visual.vertex_colors.copy()
    else:
        colors = np.full((len(mesh.vertices), 4), [180, 180, 180, 255], dtype=np.uint8)

    # Random hue shift and brightness variation
    shift = rng.uniform(-intensity, intensity)
    brightness = rng.uniform(1.0 - intensity * 0.5, 1.0 + intensity * 0.5)

    colors_float = colors[:, :3].astype(np.float32) / 255.0

    # Brightness
    colors_float *= brightness

    # Slight hue shift by rotating RGB channels
    if abs(shift) > 0.1:
        # Convert to simple hue rotation
        r, g, b = colors_float[:, 0], colors_float[:, 1], colors_float[:, 2]
        if shift > 0:
            colors_float[:, 0] = r * (1 - shift) + g * shift
            colors_float[:, 2] = b * (1 - shift) + r * shift
        else:
            shift = -shift
            colors_float[:, 1] = g * (1 - shift) + b * shift
            colors_float[:, 0] = r * (1 - shift) + b * shift

    colors_float = np.clip(colors_float, 0, 1)
    colors[:, :3] = (colors_float * 255).astype(np.uint8)
    mesh.visual.vertex_colors = colors

    return mesh


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def sample_target(cfg, rng=None, quotas=None, instance_id=1):
    """Sample and prepare a target asset.

    Args:
        cfg: dataset config dict
        rng: random.Random instance
        quotas: category quota dict (uses DEFAULT_QUOTAS if None)
        instance_id: instance identifier for this target

    Returns:
        target dict with keys:
            target_id, category, mesh_path, mesh (trimesh.Trimesh),
            scale_m, instance_id, is_deformable, source
    """
    if rng is None:
        rng = random.Random()
    if quotas is None:
        quotas = DEFAULT_QUOTAS

    # Load target index
    idx_path = cfg["assets"]["target_index"]
    with open(idx_path) as f:
        targets = json.load(f)

    # Filter to allowed
    allowed = [t for t in targets if t.get("allowed_for_train", True)]
    if not allowed:
        allowed = targets

    # Build category buckets and sample
    buckets = build_category_buckets(allowed)
    chosen = sample_by_quota(buckets, quotas, rng)
    if chosen is None:
        raise RuntimeError("No targets available for sampling")

    category = infer_category(_get_asset_id(chosen), chosen.get("source", ""))

    # Load mesh
    mesh = load_target_mesh(chosen)
    if mesh is None:
        # Retry with a different target
        for _ in range(10):
            chosen = rng.choice(allowed)
            category = infer_category(_get_asset_id(chosen), chosen.get("source", ""))
            mesh = load_target_mesh(chosen)
            if mesh is not None:
                break
        if mesh is None:
            raise RuntimeError("Failed to load any target mesh")

    # Normalize scale
    mesh, actual_scale = normalize_target_scale(mesh, target_longest=rng.uniform(0.30, 0.60))

    # Material perturbation
    perturb_material(mesh, rng, intensity=rng.uniform(0.05, 0.25))

    # Strip visual to prevent RecursionError during deepcopy of complex materials
    mesh.visual = None

    # Check if mesh is likely deformable (cloth, rope, etc.)
    deformable_keywords = ["cloth", "rope", "chain", "fabric", "string", "wire", "tube"]
    aid = _get_asset_id(chosen)
    is_deformable = any(kw in aid.lower() for kw in deformable_keywords)

    print(f"  Target: {aid} cat={category} scale={actual_scale}m "
          f"v={mesh.vertices.shape[0]} f={mesh.faces.shape[0]}")

    return {
        "target_id": aid,
        "category": category,
        "mesh_path": chosen.get("mesh_path", ""),
        "mesh": mesh,
        "scale_m": actual_scale,
        "instance_id": instance_id,
        "is_deformable": is_deformable,
        "source": chosen.get("source", "unknown"),
    }
