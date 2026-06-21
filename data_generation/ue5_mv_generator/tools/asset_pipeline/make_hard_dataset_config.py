#!/usr/bin/env python3
"""Create MVTrackSynth-Hard-v1 configs from the current realistic asset cache."""
import argparse
import json
import pathlib
from typing import Dict, List

import yaml


ROOT = pathlib.Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "tools" / "cache"
REGISTRY_PATH = CACHE_DIR / "asset_registry.json"
CONFIGS_DIR = ROOT / "configs"


def load_registry() -> List[Dict]:
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(f"Missing registry: {REGISTRY_PATH}")
    return json.loads(REGISTRY_PATH.read_text())


def asset_scale(uid: str) -> List[float]:
    key = uid.lower()
    if any(k in key for k in ["pomegranate", "ginger", "hamburger", "bun"]):
        return [0.18, 0.38]
    if any(k in key for k in ["television", "sofa", "cabinet", "picnic_table", "street_seating"]):
        return [0.55, 1.25]
    if any(k in key for k in ["chair", "stool", "basket", "table", "bench", "commode"]):
        return [0.28, 0.85]
    return [0.25, 0.75]


def motion_type(uid: str) -> str:
    key = uid.lower()
    if any(k in key for k in ["table", "sofa", "cabinet", "bench", "television"]):
        return "controlled_orbit_slow"
    return "controlled_orbit"


def build_categories(registry: List[Dict]) -> List[Dict]:
    categories = []
    for meta in sorted(registry, key=lambda x: x.get("uid", "")):
        uid = meta.get("uid", "")
        if not uid:
            continue
        mesh = f"/Game/Assets/Realistic/{uid}/model"
        categories.append({
            "name": uid,
            "mesh": mesh,
            "scale_range_m": asset_scale(uid),
            "ground_z_cm": 0.0,
            "motion_type": motion_type(uid),
        })
    return categories


def filter_registry(registry: List[Dict], include_uids: List[str]) -> List[Dict]:
    if not include_uids:
        return registry
    include = set(include_uids)
    filtered = [meta for meta in registry if meta.get("uid") in include]
    missing = sorted(include - {meta.get("uid") for meta in filtered})
    if missing:
        raise RuntimeError(f"Requested assets are missing from registry: {missing}")
    return filtered


def build_config(registry: List[Dict], num_sequences: int, frames: int, output_root: str, seed: int,
                 profile: str) -> Dict:
    categories = build_categories(registry)
    if not categories:
        raise RuntimeError("No usable realistic assets in registry")
    balanced = profile == "balanced"

    return {
        "dataset": {
            "name": "MVTrackSynth-Balanced-v1" if balanced else "MVTrackSynth-Hard-v1",
            "version": "1.0",
            "seed": seed,
            "fps": 30,
            "frames_per_sequence": frames,
            "num_sequences": num_sequences,
            "num_cameras_min": 4,
            "num_cameras_max": 6,
            "resolution": {"width": 1280, "height": 720},
            "design_goal": "fixed-camera multi-view SOT with readable targets, controlled occlusion, distractors, scale/viewpoint variation" if balanced else "multi-view single-object tracking hard cases: cross-view identity, occlusion, distractors, scale/viewpoint variation",
        },
        "visibility": {
            "min_visible_cameras": 2,
            "min_cameras_with_occlusion": 1,
            "max_consecutive_invisible": 45,
            "min_visibility_ratio": 0.03,
        },
        "split": {"train": 0.8, "val": 0.1, "test": 0.1, "split_by_asset": True},
        "output": {
            "root": output_root,
            "image_format": "png",
            "depth_format": "png",
            "mask_format": "png",
            "annotation_format": "json",
            "shard_size_sequences": 50,
        },
        "targets": {
            "selection_mode": "sequential",
            "categories": categories,
        },
        "occluders": {
            "enabled": True,
            "count_range": [1, 3] if balanced else [2, 5],
            "policy": "balanced_partial_occlusion_plus_near_target_clutter" if balanced else "view_specific_plus_near_target_clutter",
            "categories": [
                {"name": "view_specific_pillar", "mesh": "builtin_cylinder", "placement": "between_selected_camera_and_target"},
                {"name": "view_specific_panel", "mesh": "builtin_cube", "placement": "between_selected_camera_and_target"},
                {"name": "near_target_clutter", "mesh": "builtin_cube", "placement": "near_target"},
            ],
        },
        "distractors": {
            "enabled": True,
            "similar_target_like_range": [1, 3] if balanced else [2, 5],
            "background_furniture_range": [3, 7] if balanced else [5, 11],
            "purpose": "identity-preserving tracking under same-class/similar-shape clutter",
        },
        "rendering": {
            "engine": "ue5_vulkan",
            "lighting": "per-sequence synchronized randomized indoor lighting",
            "sim2real_effects": ["cast_shadows", "fog_density_jitter", "focal_length_jitter", "viewpoint_baseline_jitter"],
        },
        "training_notes": {
            "stage_use": "Use as second-stage hard multi-view fusion data after real single-view pretraining.",
            "avoid": "Do not replace real first-stage tracking data with this synthetic set until GMOT/GMTD/GOT-like validation improves.",
        },
        "quality_gate": {
            "min_sequence_mean_visibility": 0.50 if balanced else 0.35,
            "max_invisible_annotation_fraction": 0.05,
            "target_full_partial_heavy_mix": "Prefer a balanced mix; reject sequences where most views are heavy/invisible.",
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-sequences", type=int, default=12)
    parser.add_argument("--frames", type=int, default=90)
    parser.add_argument("--seed", type=int, default=20260616)
    parser.add_argument("--output-root", default=str(ROOT / "output_hard_v1_preview_20260616"))
    parser.add_argument("--config-name", default="dataset_hard_v1_preview_20260616.yaml")
    parser.add_argument("--include-uids", nargs="*", default=[],
                        help="Optional asset UID allow-list for texture-verified previews.")
    parser.add_argument("--profile", choices=["hard", "balanced"], default="hard",
                        help="Generation difficulty profile. Use balanced for training/QC previews.")
    args = parser.parse_args()

    cfg = build_config(filter_registry(load_registry(), args.include_uids),
                       args.num_sequences, args.frames, args.output_root, args.seed, args.profile)
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    out = CONFIGS_DIR / args.config_name
    out.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False))
    print(out)


if __name__ == "__main__":
    main()
