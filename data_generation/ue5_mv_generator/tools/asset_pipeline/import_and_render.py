#!/usr/bin/env python3
"""
Import downloaded assets into UE5 and generate a realistic dataset.

This script:
1. Reads the asset registry
2. Creates a UE5 Python import script
3. Updates the job manifest to use real assets
4. Runs the orchestrator with realistic scenes
"""
import json
import pathlib
import random

CACHE_DIR = pathlib.Path(__file__).resolve().parent.parent / "cache"
CONFIGS_DIR = pathlib.Path(__file__).resolve().parents[2] / "configs"


def load_registry():
    reg_path = CACHE_DIR / "asset_registry.json"
    if not reg_path.exists():
        print("[ERROR] No asset registry found. Run download_assets.py first.")
        return []
    return json.loads(reg_path.read_text())


def categorize_assets(registry):
    """Categorize downloaded assets into tracking target categories."""
    categories = {
        "furniture": [],
        "food": [],
        "electronics": [],
        "container": [],
        "decoration": [],
        "other": []
    }

    furniture_kw = ["chair", "table", "sofa", "bench", "stool", "cabinet", "shelf", "desk", "nightstand", "commode"]
    food_kw = ["food", "apple", "lime", "ginger", "pomegranate", "hamburger", "bun", "bread"]
    electronics_kw = ["television", "tv", "monitor", "laptop", "phone", "computer"]
    container_kw = ["basket", "box", "pot", "vase", "bowl", "bucket"]
    decoration_kw = ["lamp", "light", "plant", "flower", "clock", "frame", "mirror"]

    for asset in registry:
        name = asset.get("uid", "").lower()
        tags = " ".join(asset.get("tags", [])).lower()
        combined = name + " " + tags

        if any(k in combined for k in furniture_kw):
            categories["furniture"].append(asset)
        elif any(k in combined for k in food_kw):
            categories["food"].append(asset)
        elif any(k in combined for k in electronics_kw):
            categories["electronics"].append(asset)
        elif any(k in combined for k in container_kw):
            categories["container"].append(asset)
        elif any(k in combined for k in decoration_kw):
            categories["decoration"].append(asset)
        else:
            categories["other"].append(asset)

    return categories


def generate_realistic_dataset_config(registry, num_sequences=100):
    """Generate a dataset config that uses real assets."""
    cats = categorize_assets(registry)

    print("\n=== Asset Categories ===")
    for cat_name, assets in cats.items():
        print(f"  {cat_name}: {len(assets)} assets")
        for a in assets[:3]:
            print(f"    - {a['uid']}")
        if len(assets) > 3:
            print(f"    ... and {len(assets)-3} more")

    # Build target categories from available assets
    target_categories = []
    for cat_name, assets in cats.items():
        if not assets:
            continue
        for asset in assets:
            target_categories.append({
                "name": asset["uid"],
                "mesh": asset.get("path", ""),
                "mesh_format": asset.get("format", "fbx"),
                "source": asset.get("source", "unknown"),
                "license": asset.get("license", "unknown"),
                "scale_range_m": [0.1, 0.8],
                "motion_type": "physics_static" if cat_name == "furniture" else "physics_impulse"
            })

    if not target_categories:
        print("[ERROR] No categorized assets found")
        return None

    # Create the config
    config = {
        "dataset": {
            "name": "MVTrackSynth-Realistic-v1",
            "version": "1.0",
            "seed": 20260615,
            "fps": 30,
            "frames_per_sequence": 300,
            "num_sequences": num_sequences,
            "num_cameras_min": 4,
            "num_cameras_max": 6,
            "resolution": {"width": 1280, "height": 720}
        },
        "split": {"train": 0.8, "val": 0.1, "test": 0.1, "split_by_asset": True},
        "output": {
            "root": str(pathlib.Path(__file__).resolve().parent.parent / "output"),
            "image_format": "png",
            "depth_format": "png",
            "mask_format": "png",
            "annotation_format": "json",
            "shard_size_sequences": 50
        },
        "targets": {
            "categories": target_categories
        },
        "occluders": {
            "enabled": True,
            "count_range": [2, 6],
            "categories": [
                {"name": "wall_segment", "mesh": "builtin_cube", "scale_range_m": [0.5, 3.0],
                 "placement": "between_camera_and_target"},
                {"name": "pillar", "mesh": "builtin_cylinder", "scale_range_m": [0.2, 0.8],
                 "placement": "random_near_target"},
                {"name": "box_obstacle", "mesh": "builtin_cube", "scale_range_m": [0.3, 1.5],
                 "placement": "random_in_scene"}
            ]
        },
        "physics": {
            "gravity": -9.81,
            "settlement_time_s": 3.0,
            "settlement_linear_vel_threshold": 0.02,
            "settlement_angular_vel_threshold": 2.0,
            "max_settlement_attempts": 5
        },
        "motion": {
            "default_type": "physics_impulse",
            "impulse_force_range": [0.5, 5.0],
            "trajectory_radius_m": 2.0
        },
        "rendering": {
            "engine": "ue5_vulkan",
            "use_xvfb": True,
            "environment": {
                "sun_intensity": 10.0,
                "sun_color": [1.0, 0.95, 0.85],
                "sky_intensity": 1.0,
                "fog_density": 0.02,
                "ground_color": [0.4, 0.4, 0.4],
                "add_background_walls": True
            }
        }
    }

    # Save config
    config_path = CONFIGS_DIR / "dataset_realistic.yaml"
    import yaml
    config_path.write_text(yaml.dump(config, default_flow_style=False, allow_unicode=True))
    print(f"\n[Config] Saved to {config_path}")

    # Save asset mapping for UE5 import
    asset_map = {}
    for cat in target_categories:
        asset_map[cat["name"]] = {
            "model_path": cat["mesh"],
            "format": cat["mesh_format"],
            "source": cat["source"],
            "license": cat["license"],
            "scale_m": cat["scale_range_m"][1]
        }

    map_path = CACHE_DIR / "asset_import_map.json"
    map_path.write_text(json.dumps(asset_map, indent=2))
    print(f"[Asset Map] Saved to {map_path}")

    return config


def generate_ue5_import_script(registry):
    """Generate a UE5 Python script to import all assets."""
    script_lines = [
        '# Auto-generated UE5 asset import script',
        '# Run inside UE5 Editor: exec(open(path).read())',
        'import unreal',
        'import os',
        '',
        'ASSET_BASE = "/Game/Assets/Realistic"',
        'CACHE_DIR = os.environ.get("VISTA_ASSET_CACHE", "tools/cache")',
        '',
        'def import_fbx(fbx_path, dest_path, asset_name):',
        '    """Import FBX into UE content browser."""',
        '    task = unreal.AssetImportTask()',
        '    task.set_editor_property("filename", fbx_path)',
        '    task.set_editor_property("destination_path", dest_path)',
        '    task.set_editor_property("replace_existing", True)',
        '    task.set_editor_property("automated", True)',
        '    task.set_editor_property("save", True)',
        '    task.set_editor_property("original_filename", asset_name)',
        '    options = unreal.FbxImportUI()',
        '    options.set_editor_property("import_mesh", True)',
        '    options.set_editor_property("import_materials", True)',
        '    options.set_editor_property("import_textures", True)',
        '    options.set_editor_property("automated_import_should_detect_type", True)',
        '    task.set_editor_property("options", options)',
        '    unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])',
        '    return task.get_editor_property("result")',
        '',
        'def import_texture(texture_path, dest_path):',
        '    task = unreal.AssetImportTask()',
        '    task.set_editor_property("filename", texture_path)',
        '    task.set_editor_property("destination_path", dest_path)',
        '    task.set_editor_property("replace_existing", True)',
        '    task.set_editor_property("automated", True)',
        '    task.set_editor_property("save", True)',
        '    unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])',
        '    res = task.get_editor_property("result")',
        '    return res[0] if res else None',
        '',
        'def import_textures(meta, dest_path):',
        '    imported = {}',
        '    for role, tex_path in (meta.get("textures") or {}).items():',
        '        if not tex_path or not os.path.exists(tex_path):',
        '            continue',
        '        tex = import_texture(tex_path, f"{dest_path}/Textures")',
        '        if tex:',
        '            imported[role] = tex',
        '    return imported',
        '',
        'assets = [',
    ]

    for asset in registry:
        path = asset.get("path", "")
        uid = asset.get("uid", "unknown")
        fmt = asset.get("format", "fbx")
        cat = "Objects"
        textures = json.dumps(asset.get("textures", {}))
        script_lines.append(f'    {{"uid": "{uid}", "path": "{path}", "format": "{fmt}", "cat": "{cat}", "textures": {textures}}},')

    script_lines.extend([
        ']',
        '',
        'imported = 0',
        'for a in assets:',
        '    dest = f"{ASSET_BASE}/{a[\'cat\']}"',
        '    if os.path.exists(a["path"]):',
        '        import_textures(a, dest)',
        '        result = import_fbx(a["path"], dest, a["uid"])',
        '        if result:',
        '            imported += 1',
        '            unreal.log(f"Imported: {a[\'uid\']}")',
        '        else:',
        '            unreal.log_warning(f"Failed: {a[\'uid\']}")',
        '',
        'unreal.log(f"Total imported: {imported}/{len(assets)}")',
    ])

    script_path = pathlib.Path(__file__).resolve().parent.parent / "ue_project" / "Content" / "Python" / "import_realistic_assets.py"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text("\n".join(script_lines))
    print(f"[UE5 Import Script] Saved to {script_path}")
    return script_path


def main():
    registry = load_registry()
    if not registry:
        print("[ERROR] No assets found. Run: python download_assets.py --source polyhaven --count 30")
        return

    print(f"[Registry] {len(registry)} assets loaded")

    config = generate_realistic_dataset_config(registry, num_sequences=10)
    if not config:
        return

    generate_ue5_import_script(registry)

    print("\n=== Next Steps ===")
    print("1. Import assets into UE5:")
    print("   ssh server, run UE Editor with import_realistic_assets.py")
    print("2. Generate dataset:")
    print("   python tools/orchestrator/orchestrator.py --config configs/dataset_realistic.yaml --mode generate")
    print("3. Download results:")
    print("   scp -r server:/path/to/output ./mvtrack_realistic_dataset")


if __name__ == "__main__":
    main()
