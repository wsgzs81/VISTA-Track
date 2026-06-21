import json
import os
from pathlib import Path

import unreal

ASSET_BASE = "/Game/Assets/Realistic"
CACHE_DIR = os.environ.get("VISTA_ASSET_CACHE", "tools/cache")


def expected_asset_path(filename, dest_path):
    return f"{dest_path}/{Path(filename).stem}"


def import_asset(filename, dest_path, replace=True):
    task = unreal.AssetImportTask()
    task.set_editor_property("filename", filename)
    task.set_editor_property("destination_path", dest_path)
    task.set_editor_property("replace_existing", replace)
    task.set_editor_property("automated", True)
    task.set_editor_property("save", True)
    unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
    asset_path = expected_asset_path(filename, dest_path)
    if unreal.EditorAssetLibrary.does_asset_exist(asset_path):
        return unreal.EditorAssetLibrary.load_asset(asset_path)
    return None


def import_fbx(fbx_path, dest_path):
    task = unreal.AssetImportTask()
    task.set_editor_property("filename", fbx_path)
    task.set_editor_property("destination_path", dest_path)
    task.set_editor_property("replace_existing", True)
    task.set_editor_property("automated", True)
    task.set_editor_property("save", True)

    options = unreal.FbxImportUI()
    options.set_editor_property("import_mesh", True)
    options.set_editor_property("import_materials", True)
    options.set_editor_property("import_textures", True)
    options.set_editor_property("automated_import_should_detect_type", True)
    task.set_editor_property("options", options)

    unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
    model_asset = f"{dest_path}/{Path(fbx_path).stem}"
    if unreal.EditorAssetLibrary.does_asset_exist(model_asset):
        return [model_asset]
    return []


def find_first_static_mesh(asset_paths):
    for asset_path in asset_paths or []:
        asset = unreal.EditorAssetLibrary.load_asset(asset_path)
        if isinstance(asset, unreal.StaticMesh):
            return asset
    return None


def import_textures(uid, meta, dest_path):
    textures = meta.get("textures", {}) or {}
    imported = {}
    role_order = ["diffuse", "rough", "metallic", "ao", "normal_gl", "normal_dx", "normal"]
    for role in role_order:
        src = textures.get(role)
        if not src or not os.path.exists(src):
            continue
        tex_asset = import_asset(src, f"{dest_path}/Textures")
        if tex_asset:
            imported[role] = tex_asset
            unreal.log(f"Imported texture: {uid}:{role}")
    return imported


reg_path = os.path.join(CACHE_DIR, "asset_registry.json")
with open(reg_path) as f:
    assets = json.load(f)

only_uids = {
    x.strip() for x in os.environ.get("MVTRACK_IMPORT_UIDS", "").split(",") if x.strip()
}
skip_uids = {
    x.strip() for x in os.environ.get("MVTRACK_SKIP_UIDS", "").split(",") if x.strip()
}
skip_uids.update({
    # This asset currently triggers a UE 5.6 FBX importer crash in headless mode.
    "modern_wooden_cabinet",
})

imported = 0
for meta in assets:
    path = meta.get("path", "")
    uid = meta.get("uid", "unknown")
    if only_uids and uid not in only_uids:
        continue
    if uid in skip_uids:
        unreal.log_warning(f"Skipping known-problem asset: {uid}")
        continue
    if not os.path.exists(path):
        unreal.log_warning(f"Missing model: {uid}: {path}")
        continue

    dest = f"{ASSET_BASE}/{uid}"
    try:
        texture_assets = import_textures(uid, meta, dest)
        result = import_fbx(path, dest)
        static_mesh = find_first_static_mesh(result)
        if static_mesh:
            unreal.EditorAssetLibrary.save_loaded_asset(static_mesh)
        if texture_assets:
            unreal.log(f"Imported textures for {uid}: {', '.join(sorted(texture_assets.keys()))}")
        imported += 1
        unreal.log(f"Imported realistic asset: {uid}")
    except Exception as exc:
        unreal.log_warning(f"Failed: {uid}: {exc}")

unreal.log(f"Total imported: {imported}/{len(assets)}")
