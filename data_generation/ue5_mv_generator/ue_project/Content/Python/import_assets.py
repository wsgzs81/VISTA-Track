"""
UE Editor Python script for asset import.
Run inside UE Editor: exec(open(path).read())
Or via: UnrealEditor-Cmd -ExecutePythonScript=<path>
"""
import unreal
import os
import json


def import_glb_mesh(glb_path, dest_path):
    """Import a GLB/FBX mesh into UE content browser."""
    task = unreal.AssetImportTask()
    task.set_editor_property("filename", glb_path)
    task.set_editor_property("destination_path", dest_path)
    task.set_editor_property("replace_existing", True)
    task.set_editor_property("automated", True)
    task.set_editor_property("save", True)

    unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
    return task.get_editor_property("result")


def generate_convex_collision(asset_path):
    """Generate convex hull collision for a static mesh."""
    mesh = unreal.EditorAssetLibrary.load_asset(asset_path)
    if not mesh:
        unreal.log_warning("Cannot load asset: " + asset_path)
        return False

    # Set collision complexity to use simple collision
    mesh.set_editor_property("collision_trace_flag",
                              unreal.CollisionTraceFlag.CTF_USE_DEFAULT_CONVEX)
    return True


def setup_physics_asset(asset_path):
    """Enable physics simulation on a static mesh."""
    mesh = unreal.EditorAssetLibrary.load_asset(asset_path)
    if not mesh:
        return False

    # Enable simulation
    body_setup = mesh.get_editor_property("body_setup")
    if body_setup:
        body_setup.set_editor_property("collision_trace_flag",
                                        unreal.CollisionTraceFlag.CTF_USE_DEFAULT_CONVEX)
    return True


def batch_import_assets(source_dir, dest_base, category):
    """Import all mesh files from a directory."""
    imported = []
    for fname in os.listdir(source_dir):
        if fname.endswith((".glb", ".fbx", ".obj")):
            src = os.path.join(source_dir, fname)
            dest = dest_base + "/" + category
            result = import_glb_mesh(src, dest)
            if result:
                imported.append(fname)
                unreal.log("Imported: " + fname)
            else:
                unreal.log_warning("Failed: " + fname)
    return imported


if __name__ == "__main__":
    unreal.log("MVTrack Asset Import Script loaded")
