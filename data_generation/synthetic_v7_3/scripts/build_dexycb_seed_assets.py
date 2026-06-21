#!/usr/bin/env python3
"""Build minimal SynMVTrack asset indexes from an MV3DPT DexYCB backup.

This is a recovery helper, not the original v7.3 asset pack. It creates:

- assets/index/target_index.json from DexYCB first-frame OBJ meshes
- assets/index/scene_index.json from procedural scene records
- empty material_index.json and hdri_index.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def _scene_record(scene_id: str, scene_type: str, rng: random.Random) -> dict:
    if scene_type == "room":
        size = [round(rng.uniform(3.2, 5.6), 2), round(rng.uniform(2.6, 3.2), 2), round(rng.uniform(3.2, 5.6), 2)]
        clutter = rng.choice(["medium", "high"])
        furniture_count = rng.randint(4, 10)
    elif scene_type == "corridor":
        size = [round(rng.uniform(4.5, 7.5), 2), round(rng.uniform(2.6, 3.1), 2), round(rng.uniform(1.4, 2.2), 2)]
        clutter = "low"
        furniture_count = rng.randint(0, 2)
    elif scene_type == "open_area":
        size = [round(rng.uniform(5.0, 6.5), 2), 3.0, round(rng.uniform(5.0, 6.5), 2)]
        clutter = "low"
        furniture_count = rng.randint(0, 3)
    else:
        scene_type = "tabletop"
        size = [round(rng.uniform(1.2, 2.0), 2), round(rng.uniform(0.72, 0.86), 2), round(rng.uniform(0.8, 1.6), 2)]
        clutter = "medium"
        furniture_count = rng.randint(1, 5)

    return {
        "scene_id": scene_id,
        "source": "procedural_recovery",
        "scene_path": None,
        "future_model_dir": None,
        "type": scene_type,
        "room_size_m": size,
        "has_table": scene_type in {"room", "tabletop"},
        "has_sofa": scene_type == "room" and rng.random() > 0.45,
        "furniture_count": furniture_count,
        "clutter_level": clutter,
        "procedural_params": {
            "recovery_seed_scene": True,
            "wall_material": rng.choice(["paint_white", "paint_gray", "wood_panel", "concrete"]),
            "floor_material": rng.choice(["wood_oak", "tile", "concrete", "carpet"]),
            "has_windows": rng.random() > 0.35,
            "num_windows": rng.randint(0, 3),
            "has_ceiling_light": True,
        },
        "allowed_for_train": True,
    }


def build_targets(mv3dpt_root: Path, max_targets: int) -> list[dict]:
    dex_root = mv3dpt_root / "dex-ycb-multiview"
    meshes = sorted(dex_root.glob("*/first_frame_mesh.obj"))
    if max_targets > 0:
        meshes = meshes[:max_targets]

    entries = []
    for idx, mesh_path in enumerate(meshes):
        seq_name = mesh_path.parent.name
        entries.append(
            {
                "asset_id": f"dexycb_{idx:04d}_{seq_name}",
                "source": "DEXYCB_MV3DPT_BACKUP",
                "category": "dexycb_hand_object",
                "mesh_path": str(mesh_path.resolve()),
                "license": "see_mv3dpt_source",
                "has_texture": False,
                "quality_score": 0.5,
                "allowed_for_train": True,
            }
        )
    return entries


def build_scenes(num_scenes: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    scene_types = ["room", "room", "room", "corridor", "open_area", "tabletop"]
    scenes = []
    for i in range(num_scenes):
        scene_type = scene_types[i % len(scene_types)]
        scenes.append(_scene_record(f"recovery_{scene_type}_{i:04d}", scene_type, rng))
    return scenes


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mv3dpt-root", required=True, type=Path, help="Directory containing dex-ycb-multiview")
    parser.add_argument("--output-assets", default=Path("assets"), type=Path)
    parser.add_argument("--max-targets", default=64, type=int)
    parser.add_argument("--num-scenes", default=120, type=int)
    parser.add_argument("--seed", default=74606, type=int)
    args = parser.parse_args()

    mv3dpt_root = args.mv3dpt_root.expanduser().resolve()
    if not (mv3dpt_root / "dex-ycb-multiview").is_dir():
        raise SystemExit(f"missing dex-ycb-multiview under {mv3dpt_root}")

    out_index = args.output_assets / "index"
    targets = build_targets(mv3dpt_root, args.max_targets)
    if not targets:
        raise SystemExit(f"no first_frame_mesh.obj files found under {mv3dpt_root / 'dex-ycb-multiview'}")

    scenes = build_scenes(args.num_scenes, args.seed)
    write_json(out_index / "target_index.json", targets)
    write_json(out_index / "scene_index.json", scenes)
    write_json(out_index / "material_index.json", [])
    write_json(out_index / "hdri_index.json", [])

    print(f"target_index: {len(targets)} entries -> {out_index / 'target_index.json'}")
    print(f"scene_index: {len(scenes)} entries -> {out_index / 'scene_index.json'}")
    print("material_index: 0 entries")
    print("hdri_index: 0 entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

