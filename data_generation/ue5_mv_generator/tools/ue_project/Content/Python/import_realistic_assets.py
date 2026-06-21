# Auto-generated UE5 asset import script
# Run inside UE5 Editor: exec(open(path).read())
import unreal
import os

ASSET_BASE = "/Game/Assets/Realistic"
CACHE_DIR = os.environ.get("VISTA_ASSET_CACHE", "tools/cache")

def import_fbx(fbx_path, dest_path, asset_name):
    """Import FBX into UE content browser."""
    task = unreal.AssetImportTask()
    task.set_editor_property("filename", fbx_path)
    task.set_editor_property("destination_path", dest_path)
    task.set_editor_property("replace_existing", True)
    task.set_editor_property("automated", True)
    task.set_editor_property("save", True)
    task.set_editor_property("original_filename", asset_name)
    unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
    return task.get_editor_property("result")

assets = [
    {"uid": "painted_wooden_table", "path": "tools/cache/polyhaven/painted_wooden_table/model.fbx", "format": "fbx", "cat": "Objects"},
    {"uid": "round_wooden_table_01", "path": "tools/cache/polyhaven/round_wooden_table_01/model.fbx", "format": "fbx", "cat": "Objects"},
    {"uid": "food_pomegranate_01", "path": "tools/cache/polyhaven/food_pomegranate_01/model.fbx", "format": "fbx", "cat": "Objects"},
    {"uid": "chinese_stool", "path": "tools/cache/polyhaven/chinese_stool/model.fbx", "format": "fbx", "cat": "Objects"},
    {"uid": "Television_01", "path": "tools/cache/polyhaven/Television_01/model.fbx", "format": "fbx", "cat": "Objects"},
    {"uid": "coffee_table_round_01", "path": "tools/cache/polyhaven/coffee_table_round_01/model.fbx", "format": "fbx", "cat": "Objects"},
    {"uid": "small_wooden_table_01", "path": "tools/cache/polyhaven/small_wooden_table_01/model.fbx", "format": "fbx", "cat": "Objects"},
    {"uid": "modular_street_seating", "path": "tools/cache/polyhaven/modular_street_seating/model.fbx", "format": "fbx", "cat": "Objects"},
    {"uid": "wicker_basket_01", "path": "tools/cache/polyhaven/wicker_basket_01/model.fbx", "format": "fbx", "cat": "Objects"},
    {"uid": "wooden_picnic_table", "path": "tools/cache/polyhaven/wooden_picnic_table/model.fbx", "format": "fbx", "cat": "Objects"},
    {"uid": "painted_wooden_cabinet_02", "path": "tools/cache/polyhaven/painted_wooden_cabinet_02/model.fbx", "format": "fbx", "cat": "Objects"},
    {"uid": "modern_wooden_cabinet", "path": "tools/cache/polyhaven/modern_wooden_cabinet/model.fbx", "format": "fbx", "cat": "Objects"},
    {"uid": "hamburger_buns", "path": "tools/cache/polyhaven/hamburger_buns/model.fbx", "format": "fbx", "cat": "Objects"},
    {"uid": "stationery_supplies", "path": "tools/cache/polyhaven/stationery_supplies/model.fbx", "format": "fbx", "cat": "Objects"},
    {"uid": "BarberShopChair_01", "path": "tools/cache/polyhaven/BarberShopChair_01/model.fbx", "format": "fbx", "cat": "Objects"},
    {"uid": "Sofa_01", "path": "tools/cache/polyhaven/Sofa_01/model.fbx", "format": "fbx", "cat": "Objects"},
    {"uid": "gallinera_chair", "path": "tools/cache/polyhaven/gallinera_chair/model.fbx", "format": "fbx", "cat": "Objects"},
    {"uid": "chinese_commode", "path": "tools/cache/polyhaven/chinese_commode/model.fbx", "format": "fbx", "cat": "Objects"},
    {"uid": "gothic_coffee_table", "path": "tools/cache/polyhaven/gothic_coffee_table/model.fbx", "format": "fbx", "cat": "Objects"},
    {"uid": "painted_wooden_bench", "path": "tools/cache/polyhaven/painted_wooden_bench/model.fbx", "format": "fbx", "cat": "Objects"},
    {"uid": "modern_arm_chair_01", "path": "tools/cache/polyhaven/modern_arm_chair_01/model.fbx", "format": "fbx", "cat": "Objects"},
    {"uid": "WoodenTable_03", "path": "tools/cache/polyhaven/WoodenTable_03/model.fbx", "format": "fbx", "cat": "Objects"},
    {"uid": "sofa_03", "path": "tools/cache/polyhaven/sofa_03/model.fbx", "format": "fbx", "cat": "Objects"},
    {"uid": "folding_wooden_stool", "path": "tools/cache/polyhaven/folding_wooden_stool/model.fbx", "format": "fbx", "cat": "Objects"},
    {"uid": "GreenChair_01", "path": "tools/cache/polyhaven/GreenChair_01/model.fbx", "format": "fbx", "cat": "Objects"},
    {"uid": "GothicCommode_01", "path": "tools/cache/polyhaven/GothicCommode_01/model.fbx", "format": "fbx", "cat": "Objects"},
    {"uid": "food_ginger_01", "path": "tools/cache/polyhaven/food_ginger_01/model.fbx", "format": "fbx", "cat": "Objects"},
    {"uid": "mid_century_lounge_chair", "path": "tools/cache/polyhaven/mid_century_lounge_chair/model.fbx", "format": "fbx", "cat": "Objects"},
    {"uid": "painted_wooden_chair_01", "path": "tools/cache/polyhaven/painted_wooden_chair_01/model.fbx", "format": "fbx", "cat": "Objects"},
    {"uid": "painted_wooden_stool", "path": "tools/cache/polyhaven/painted_wooden_stool/model.fbx", "format": "fbx", "cat": "Objects"},
]

imported = 0
for a in assets:
    dest = f"{ASSET_BASE}/{a['cat']}"
    if os.path.exists(a["path"]):
        result = import_fbx(a["path"], dest, a["uid"])
        if result:
            imported += 1
            unreal.log(f"Imported: {a['uid']}")
        else:
            unreal.log_warning(f"Failed: {a['uid']}")

unreal.log(f"Total imported: {imported}/{len(assets)}")
