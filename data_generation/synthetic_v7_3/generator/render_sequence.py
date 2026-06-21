#!/usr/bin/env python3
"""
render_sequence.py - Blender-based sequence renderer for SynMVTrack.

Runs inside Blender:  blender --background --python render_sequence.py -- --seq-dir ...

Reads the sequence directory produced by generate_sequence.py, imports geometry,
places cameras from calibs.json, and renders RGB + mask + depth per view per frame.

Output:
    {seq_dir}/img/{view:04d}/{frame:06d}.jpg
    {seq_dir}/masks/{view:04d}/{frame:06d}.png
    {seq_dir}/depth/{view:04d}/{frame:06d}.npy   (optional)
"""

import argparse
import colorsys
import hashlib
import json
import math
import os
import sys

import bpy
import numpy as np


# ---------------------------------------------------------------------------
# Coordinate system conversion
# Our system:  x=east, y=up, z=south
# Blender:     x=right, y=forward, z=up
# Mapping:     our (x,y,z) -> blender (x, z, y)
# ---------------------------------------------------------------------------

def our_to_blender(pos):
    """Convert position from our coords to Blender coords."""
    return [pos[0], pos[2], pos[1]]


def our_rotation_to_blender(R_our):
    """Convert 3x3 camera-to-world rotation to Blender.

    R_our maps camera frame (right, up, forward) to our world coords.
    Blender camera frame: (right, up, backward) — looks along -Z.

    R_bl = C @ R_our @ M
    where C converts our world -> Blender world, M converts our cam -> Blender cam.
    """
    C = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.float64)
    M = np.diag([1.0, 1.0, -1.0])  # our +Z (forward) -> Blender -Z (forward)
    R_bl = C @ np.array(R_our) @ M
    return R_bl


def our_translation_to_blender(T_our):
    """Convert translation from our system to Blender."""
    return [T_our[0], T_our[2], T_our[1]]


def _stable_seed(*parts):
    """Derive a repeatable uint32 seed from sequence metadata."""
    key = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


# ---------------------------------------------------------------------------
# Scene setup
# ---------------------------------------------------------------------------

def clear_scene():
    """Remove all default objects."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    # Remove orphan data
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)


def setup_render(resolution, samples, output_dir, save_depth=False, identity_priority=False):
    """Configure render settings."""
    scene = bpy.context.scene
    scene.render.resolution_x = resolution[0]
    scene.render.resolution_y = resolution[1]
    scene.render.resolution_percentage = 100
    scene.render.image_settings.quality = 90

    # Cycles for realistic rendering
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.cycles.preview_samples = 16
    try:
        if identity_priority:
            scene.view_settings.view_transform = "Standard"
            scene.view_settings.look = "None"
            scene.view_settings.exposure = -0.45
            scene.view_settings.gamma = 1.0
        else:
            scene.view_settings.view_transform = "Filmic"
            scene.view_settings.look = "Medium High Contrast"
            scene.view_settings.exposure = float(np.random.uniform(-0.25, 0.2))
            scene.view_settings.gamma = float(np.random.uniform(0.95, 1.08))
    except Exception:
        pass

    # Try GPU if available — prefer OPTIX for RTX cards
    try:
        cprefs = bpy.context.preferences.addons['cycles'].preferences
        cprefs.compute_device_type = 'OPTIX'
        cprefs.get_devices()
        force_cpu = os.environ.get("SYNMVTRACK_RENDER_DEVICE", "GPU").strip().upper() == "CPU"
        for device in cprefs.devices:
            device.use = False if force_cpu else (device.type != 'CPU')
        if force_cpu:
            scene.cycles.device = 'CPU'
            scene.cycles.use_denoising = True
            print("  CPU render requested by SYNMVTRACK_RENDER_DEVICE=CPU")
        else:
            scene.cycles.device = 'GPU'
            scene.cycles.use_denoising = False
            print(f"  GPU: {cprefs.compute_device_type}, denoising off")
    except Exception as e:
        print(f"  WARNING: OPTIX failed ({e}), trying CUDA")
        try:
            cprefs.compute_device_type = 'CUDA'
            cprefs.get_devices()
            force_cpu = os.environ.get("SYNMVTRACK_RENDER_DEVICE", "GPU").strip().upper() == "CPU"
            for device in cprefs.devices:
                device.use = False if force_cpu else (device.type != 'CPU')
            scene.cycles.device = 'CPU' if force_cpu else 'GPU'
            scene.cycles.use_denoising = bool(force_cpu)
        except Exception:
            scene.cycles.device = 'CPU'

    # Transparent background for masks
    scene.render.film_transparent = True

    # Enable depth pass
    if save_depth:
        scene.view_layers["ViewLayer"].use_pass_z = True

    # Output settings
    scene.render.filepath = output_dir
    scene.render.use_file_extension = True


def _jitter_color(color, strength=0.28):
    """Return a bounded RGB color variant to avoid flat synthetic appearance."""
    base = np.array(color[:3], dtype=np.float32)
    scale = np.random.uniform(1.0 - strength, 1.0 + strength, size=3)
    shift = np.random.uniform(-28, 28, size=3)
    return np.clip(base * scale + shift, 12, 245).astype(np.float32)


def _shade(color, factor):
    return np.clip(np.array(color[:3], dtype=np.float32) * factor, 0, 255)


def _random_rich_color(value_range=(0.38, 0.92), sat_range=(0.45, 0.95)):
    """Sample a saturated RGB color so targets do not collapse to gray/black."""
    h = float(np.random.uniform(0.0, 1.0))
    s = float(np.random.uniform(*sat_range))
    v = float(np.random.uniform(*value_range))
    rgb = colorsys.hsv_to_rgb(h, s, v)
    return tuple(int(np.clip(c * 255.0, 20, 245)) for c in rgb)


def _color_luma(color):
    c = np.array(color[:3], dtype=np.float32) / 255.0
    return float(0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2])


def _color_saturation(color):
    c = np.array(color[:3], dtype=np.float32) / 255.0
    return float(max(c) - min(c))


def _tune_matte_bsdf(mat, roughness=0.82, specular=0.18):
    if mat is None or not mat.use_nodes:
        return
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if not bsdf:
        return
    if "Roughness" in bsdf.inputs:
        bsdf.inputs["Roughness"].default_value = float(roughness)
    if "Metallic" in bsdf.inputs:
        bsdf.inputs["Metallic"].default_value = 0.0
    for spec_name in ("Specular IOR Level", "Specular"):
        if spec_name in bsdf.inputs:
            bsdf.inputs[spec_name].default_value = float(specular)


def create_material(name, color, alpha=1.0, procedural=True, jitter_strength=0.28):
    """Create a varied PBR-ish material instead of a flat diffuse color.

    The synthetic data is used for tracker robustness, so we intentionally add
    domain randomization through color ramps, roughness, and subtle bump maps.
    """
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    base = _jitter_color(color, strength=jitter_strength)
    bsdf.inputs['Base Color'].default_value = (*[c / 255.0 for c in base], alpha)
    bsdf.inputs['Roughness'].default_value = float(np.random.uniform(0.35, 0.95))
    bsdf.inputs['Metallic'].default_value = float(np.random.choice([0.0, 0.0, 0.0, 0.15, 0.35]))
    _tune_matte_bsdf(mat, roughness=bsdf.inputs['Roughness'].default_value, specular=0.22)

    if procedural:
        noise = nodes.new(type="ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = float(np.random.uniform(8.0, 38.0))
        noise.inputs["Detail"].default_value = float(np.random.uniform(5.0, 14.0))
        noise.inputs["Roughness"].default_value = float(np.random.uniform(0.35, 0.75))

        ramp = nodes.new(type="ShaderNodeValToRGB")
        ramp.color_ramp.elements[0].position = float(np.random.uniform(0.18, 0.38))
        ramp.color_ramp.elements[1].position = float(np.random.uniform(0.62, 0.88))
        dark = _shade(base, np.random.uniform(0.45, 0.85))
        light = np.clip(_shade(base, np.random.uniform(1.05, 1.55)) + np.random.uniform(0, 35, 3), 0, 255)
        ramp.color_ramp.elements[0].color = (*[c / 255.0 for c in dark], alpha)
        ramp.color_ramp.elements[1].color = (*[c / 255.0 for c in light], alpha)

        mat.node_tree.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
        mat.node_tree.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])

        bump_noise = nodes.new(type="ShaderNodeTexNoise")
        bump_noise.inputs["Scale"].default_value = float(np.random.uniform(20.0, 80.0))
        bump_noise.inputs["Detail"].default_value = float(np.random.uniform(4.0, 10.0))
        bump = nodes.new(type="ShaderNodeBump")
        bump.inputs["Strength"].default_value = float(np.random.uniform(0.015, 0.08))
        bump.inputs["Distance"].default_value = float(np.random.uniform(0.015, 0.06))
        mat.node_tree.links.new(bump_noise.outputs["Fac"], bump.inputs["Height"])
        mat.node_tree.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def _blend_rgb(a, b, weight):
    a = np.array(a[:3], dtype=np.float32)
    b = np.array(b[:3], dtype=np.float32)
    return tuple(int(np.clip(v, 0, 255)) for v in (a * (1.0 - weight) + b * weight))


def _scene_palette(scene_data):
    """Return a stable indoor palette so closed rooms do not collapse to gray."""
    palettes = [
        {"floor": (142, 112, 78), "wall": (202, 192, 176), "ceiling": (232, 228, 220)},
        {"floor": (118, 125, 118), "wall": (190, 202, 198), "ceiling": (232, 236, 234)},
        {"floor": (156, 145, 126), "wall": (214, 205, 190), "ceiling": (236, 232, 224)},
        {"floor": (126, 102, 86), "wall": (196, 188, 198), "ceiling": (230, 228, 234)},
    ]
    key = scene_data.get("scene_id") or scene_data.get("scene_type") or scene_data.get("source") or "scene"
    return palettes[_stable_seed(key) % len(palettes)]


def create_room_material(name, color, kind="wall"):
    """Create low-noise indoor materials for view-consistent tracking data."""
    mat = create_material(
        name,
        color,
        procedural=False,
        jitter_strength=0.025,
    )
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    if not bsdf:
        return mat
    bsdf.inputs["Metallic"].default_value = 0.0
    bsdf.inputs["Roughness"].default_value = 0.82 if kind != "floor" else 0.68

    # Add broad, low-contrast variation. This avoids the noisy speckle that made
    # v7.1 look like a gray procedural texture instead of a shared room.
    if kind in {"floor", "wall"}:
        base = np.array(color[:3], dtype=np.float32)
        dark = _blend_rgb(base, (45, 42, 38), 0.14 if kind == "wall" else 0.22)
        light = _blend_rgb(base, (230, 220, 205), 0.06 if kind == "wall" else 0.10)
        noise = nodes.new(type="ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = 2.2 if kind == "wall" else 4.5
        noise.inputs["Detail"].default_value = 3.0
        noise.inputs["Roughness"].default_value = 0.42
        ramp = nodes.new(type="ShaderNodeValToRGB")
        ramp.color_ramp.elements[0].position = 0.28
        ramp.color_ramp.elements[1].position = 0.78
        ramp.color_ramp.elements[0].color = (*[c / 255.0 for c in dark], 1)
        ramp.color_ramp.elements[1].color = (*[c / 255.0 for c in light], 1)
        mat.node_tree.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
        mat.node_tree.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    return mat


def _material_has_image_texture(mat):
    if mat is None or not mat.use_nodes:
        return False
    return any(node.type == "TEX_IMAGE" for node in mat.node_tree.nodes)


def _material_is_low_information(mat):
    """Approximate whether a material is likely dark/monochrome from its BSDF."""
    if mat is None or not mat.use_nodes:
        return True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if not bsdf or "Base Color" not in bsdf.inputs:
        return True
    color = [float(x) * 255.0 for x in bsdf.inputs["Base Color"].default_value[:3]]
    return _color_luma(color) < 0.22 or _color_saturation(color) < 0.08


def _material_is_overbright(mat):
    if mat is None or not mat.use_nodes:
        return True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if not bsdf or "Base Color" not in bsdf.inputs:
        return True
    color = [float(x) * 255.0 for x in bsdf.inputs["Base Color"].default_value[:3]]
    return _color_luma(color) > 0.82 and _color_saturation(color) < 0.12


def _stabilize_non_target_materials(obj, fallback_color, name_prefix="object"):
    """Replace white/no-info furniture materials with neutral matte colors."""
    mesh_objs = []
    if obj is not None and obj.type == "MESH":
        mesh_objs.append(obj)
    if obj is not None:
        try:
            children = list(obj.children_recursive)
        except Exception:
            children = []
            stack = list(obj.children)
            while stack:
                child = stack.pop()
                children.append(child)
                stack.extend(list(child.children))
        mesh_objs.extend([c for c in children if c.type == "MESH"])

    for idx, mesh_obj in enumerate(mesh_objs):
        mats = list(mesh_obj.data.materials) if mesh_obj.data else []
        replace = not mats or all(
            (not _material_has_image_texture(m)) and
            (_material_is_low_information(m) or _material_is_overbright(m))
            for m in mats
        )
        if replace:
            mesh_obj.data.materials.clear()
            mat = create_material(
                f"mat_{name_prefix}_{idx}_matte",
                fallback_color,
                procedural=False,
                jitter_strength=0.07,
            )
            _tune_matte_bsdf(mat, roughness=0.86, specular=0.12)
            mesh_obj.data.materials.append(mat)
        else:
            for mat in mats:
                _tune_matte_bsdf(mat, roughness=0.82, specular=0.15)


def _stable_target_color(target_data):
    """Return a muted fallback color for untextured targets."""
    fallback = target_data.get("color", [170, 160, 145])
    base = np.array(fallback[:3], dtype=np.float32)
    if _color_luma(base) < 0.25:
        base = np.array([170, 160, 145], dtype=np.float32)

    gray = np.full(3, float(np.mean(base)), dtype=np.float32)
    base = base * 0.82 + gray * 0.18
    return tuple(int(np.clip(v, 35, 225)) for v in base)


def _replace_target_with_domain_materials(obj, target_data, min_slots=1, max_slots=1):
    """Give untextured targets one stable material, not random patchwork."""
    if obj is None or obj.type != "MESH" or obj.data is None:
        return

    obj.data.materials.clear()
    mat = create_material(
        "mat_target_stable_fallback",
        _stable_target_color(target_data),
        procedural=False,
        jitter_strength=0.04,
    )
    obj.data.materials.append(mat)

    if obj.data.polygons:
        for poly in obj.data.polygons:
            poly.material_index = 0
        obj.data.update()


def _apply_target_appearance_randomization(obj, target_data):
    """Preserve object identity while adding only mild reflectance variation."""
    if obj is None or obj.type != "MESH" or obj.data is None:
        return

    fallback = tuple(target_data.get("color", [180, 90, 70]))
    if not obj.data.materials:
        _replace_target_with_domain_materials(obj, target_data)
        return

    has_texture = any(_material_has_image_texture(mat) for mat in obj.data.materials)
    if not has_texture:
        _replace_target_with_domain_materials(obj, target_data)
        return

    for i, mat in enumerate(list(obj.data.materials)):
        if mat is None:
            obj.data.materials[i] = create_material(
                f"mat_target_{i}",
                fallback,
                procedural=False,
                jitter_strength=0.04,
            )
            continue
        if _material_has_image_texture(mat):
            # Keep dataset texture; only make reflectance slightly less toy-like.
            mat.use_nodes = True
            _tune_matte_bsdf(
                mat,
                roughness=float(np.random.uniform(0.78, 0.92)),
                specular=0.12,
            )
            continue
        obj.data.materials[i] = create_material(
            f"{mat.name}_stable_fallback",
            fallback,
            procedural=False,
            jitter_strength=0.04,
        )


def add_mesh_from_data(name, vertices, faces, color=(150, 150, 150), location=None):
    """Add a mesh object from vertex/face data."""
    mesh = bpy.data.meshes.new(name)
    # Convert vertices to Blender coords
    bl_verts = [our_to_blender(v) for v in vertices]
    mesh.from_pydata(bl_verts, [], faces)
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)

    mat = create_material(f"mat_{name}", color)
    obj.data.materials.append(mat)

    if location:
        obj.location = our_to_blender(location)

    return obj


def _join_imported_meshes(mesh_objects):
    """Join imported target parts so masks and animation cover the full object."""
    if not mesh_objects:
        return None
    if len(mesh_objects) == 1:
        obj = mesh_objects[0]
        obj.name = "target"
        return obj

    bpy.ops.object.select_all(action='DESELECT')
    for obj in mesh_objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_objects[0]
    bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    obj.name = "target"
    return obj


def _normalize_imported_target(obj, target_data):
    """Center imported mesh data and scale longest side to target scale_m."""
    scale_m = float(target_data.get("scale_m", 0.3) or 0.3)
    if obj is None or obj.type != 'MESH' or not obj.data.vertices:
        return obj

    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    try:
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    except Exception:
        pass

    coords = [v.co for v in obj.data.vertices]
    mins = [min(v[i] for v in coords) for i in range(3)]
    maxs = [max(v[i] for v in coords) for i in range(3)]
    center = [(mins[i] + maxs[i]) / 2.0 for i in range(3)]
    longest = max(maxs[i] - mins[i] for i in range(3))
    factor = scale_m / longest if longest > 1e-6 else 1.0

    for vert in obj.data.vertices:
        for i in range(3):
            vert.co[i] = (vert.co[i] - center[i]) * factor
    obj.data.update()
    return obj


def _create_placeholder_target(target_data):
    """Create a procedural target when no mesh is available."""
    shape = str(target_data.get("shape") or target_data.get("category") or "cube").lower()
    scale_m = float(target_data.get("scale_m", 0.3) or 0.3)
    color = tuple(target_data.get("color", [200, 80, 80]))

    if "sphere" in shape or "ball" in shape:
        bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=scale_m / 2)
        obj = bpy.context.active_object
    elif "cylinder" in shape:
        bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=scale_m * 0.4, depth=scale_m * 1.2)
        obj = bpy.context.active_object
    else:
        dims = {
            "box_tall": [scale_m * 0.75, scale_m * 1.6, scale_m * 0.75],
            "box_flat": [scale_m * 1.5, scale_m * 0.55, scale_m * 1.0],
            "cube_large": [scale_m * 1.25, scale_m * 1.25, scale_m * 1.25],
        }.get(shape, [scale_m, scale_m, scale_m])
        bpy.ops.mesh.primitive_cube_add(size=1.0)
        obj = bpy.context.active_object
        # Bake the base dimensions into mesh data; trajectory scales remain relative.
        obj.scale = [dims[0] / 2, dims[2] / 2, dims[1] / 2]
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    obj.name = "target"
    mat = create_material("mat_target", color, procedural=False, jitter_strength=0.04)
    obj.data.materials.append(mat)
    return obj


def load_target_with_animation(target_data, trajectory, frame_start=1, project_root=None):
    """Load target mesh and keyframe its trajectory.

    Args:
        target_data: dict with mesh_path or vertices/faces
        trajectory: dict with positions_m, rotations, scales
        frame_start: first Blender frame number
        project_root: absolute path to project root (for resolving relative mesh paths)

    Returns:
        Blender object or None
    """
    mesh_path = target_data.get("mesh_path", "")
    obj = None

    # Resolve relative paths using project_root
    if mesh_path and not os.path.isabs(mesh_path):
        if project_root:
            mesh_path = os.path.join(project_root, mesh_path)
        else:
            # Fallback: derive from seq_dir two levels up (output/SynMVTrack_v1/seq_xxxx -> root)
            mesh_path = os.path.abspath(mesh_path)

    if mesh_path and os.path.isfile(mesh_path):
        # Try to import GLB/OBJ
        ext = os.path.splitext(mesh_path)[1].lower()
        try:
            # Track objects before import to identify new ones
            objs_before = set(bpy.data.objects)

            if ext == ".glb":
                bpy.ops.import_scene.gltf(filepath=mesh_path)
            elif ext in (".obj", ".OBJ"):
                bpy.ops.import_scene.obj(filepath=mesh_path)
            elif ext in (".fbx", ".FBX"):
                bpy.ops.import_scene.fbx(filepath=mesh_path)

            # Find newly imported objects (not in scene before import)
            new_objs = [o for o in bpy.data.objects if o not in objs_before and o.type == 'MESH']
            if new_objs:
                obj = _join_imported_meshes(new_objs)
                obj = _normalize_imported_target(obj, target_data)
                _apply_target_appearance_randomization(obj, target_data)
        except Exception as e:
            print(f"  Warning: failed to import {mesh_path}: {e}")

    if obj is None:
        obj = _create_placeholder_target(target_data)

    # Set up animation
    positions = trajectory["positions_m"]
    rotations = trajectory["rotations"]
    scales = trajectory["scales"]
    n_frames = len(positions)

    scene = bpy.context.scene
    scene.frame_start = frame_start
    scene.frame_end = frame_start + n_frames - 1

    for fi in range(n_frames):
        frame = frame_start + fi
        scene.frame_set(frame)

        # Position
        pos = positions[fi]
        obj.location = our_to_blender(pos)

        # Rotation (degrees -> radians, convert to Blender euler)
        rot = rotations[fi]
        # Our rotation: [rx, ry, rz] in degrees
        # Convert through coordinate system
        rx = math.radians(rot[0])
        ry = math.radians(rot[1])
        rz = math.radians(rot[2])
        obj.rotation_euler = [rx, rz, ry]  # our (x,y,z) euler -> Blender (x,z,y)

        # Scale
        sc = scales[fi]
        obj.scale = [sc[0], sc[2], sc[1]]

        # Keyframe
        obj.keyframe_insert(data_path="location", frame=frame)
        obj.keyframe_insert(data_path="rotation_euler", frame=frame)
        obj.keyframe_insert(data_path="scale", frame=frame)

    return obj


def add_occluders(occluders_data, project_root=None):
    """Add occluder objects to the scene. Loads real 3D models when available."""
    objects = []
    for i, occ in enumerate(occluders_data):
        name = occ.get("name", f"occluder_{i}")
        model_path = occ.get("model_path", "")

        # Resolve model path
        if model_path and not os.path.isabs(model_path) and project_root:
            model_path = os.path.join(project_root, model_path)

        loaded = False
        if model_path and os.path.isfile(model_path):
            ext = os.path.splitext(model_path)[1].lower()
            try:
                if ext == ".obj":
                    # Blender 4.2 has no OBJ importer — use trimesh to load
                    # and create mesh from raw data
                    import trimesh as _trimesh
                    tmesh = _trimesh.load(model_path, force="mesh")
                    if isinstance(tmesh, _trimesh.Scene):
                        parts = [g for g in tmesh.geometry.values() if isinstance(g, _trimesh.Trimesh)]
                        if parts:
                            tmesh = _trimesh.util.concatenate(parts)
                        else:
                            raise ValueError("empty scene")

                    # Convert y-up to z-up: [x,y,z] -> [x,z,y]
                    raw_verts = tmesh.vertices
                    verts = [(v[0], v[2], v[1]) for v in raw_verts]
                    faces = [tuple(f) for f in tmesh.faces]

                    mesh_data = bpy.data.meshes.new(f"mesh_{name}")
                    mesh_data.from_pydata(verts, [], faces)
                    mesh_data.update()

                    obj = bpy.data.objects.new(name, mesh_data)
                    bpy.context.collection.objects.link(obj)
                    imported = [obj]
                elif ext == ".glb":
                    bpy.ops.import_scene.gltf(filepath=model_path)
                    imported = [o for o in bpy.context.selected_objects if o.type == 'MESH']
                elif ext == ".fbx":
                    bpy.ops.import_scene.fbx(filepath=model_path)
                    imported = [o for o in bpy.context.selected_objects if o.type == 'MESH']
                else:
                    imported = []

                if ext == ".obj":
                    pass  # already handled above, imported is already set
                else:
                    imported = [o for o in bpy.context.selected_objects if o.type == 'MESH']
                if imported:
                    # Parent all imported meshes under first one
                    parent = imported[0]
                    parent.name = name
                    for child in imported[1:]:
                        child.parent = parent

                    # Add material if missing
                    if parent.data and not parent.data.materials:
                        mat = create_material(
                            f"mat_{name}",
                            (np.random.randint(100, 220),
                             np.random.randint(100, 220),
                             np.random.randint(100, 220)),
                            procedural=False,
                            jitter_strength=0.08,
                        )
                        parent.data.materials.append(mat)

                    furniture_color = (
                        int(np.random.randint(95, 165)),
                        int(np.random.randint(85, 150)),
                        int(np.random.randint(75, 140)),
                    )
                    _stabilize_non_target_materials(parent, furniture_color, name_prefix=name)

                    # Scale to match bbox dimensions
                    bbox = occ.get("bbox_3d")
                    size_m = occ.get("size_m", [1, 1, 1])
                    if bbox:
                        lo, hi = bbox
                        # target_extents in canonical coords: [x, y_up, z_depth]
                        target_extents = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]]
                        center = [(lo[0]+hi[0])/2, (lo[1]+hi[1])/2, (lo[2]+hi[2])/2]
                    else:
                        target_extents = size_m
                        pos = occ.get("position_m", [0, 0, 0])
                        center = pos

                    # Get current model bounding box in Blender coords
                    parent.scale = (1, 1, 1)
                    bpy.context.view_layer.update()
                    bb = parent.bound_box
                    xs = [v[0] for v in bb]
                    ys = [v[1] for v in bb]
                    zs = [v[2] for v in bb]
                    model_size = [max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs)]

                    # Scale canonical SynMVTrack [x, y_up, z_depth] extents to
                    # Blender [x, y_depth, z_up] extents.
                    dx, dy, dz = target_extents
                    if model_size[0] > 1e-6:
                        parent.scale.x *= dx / model_size[0]
                    if model_size[1] > 1e-6:
                        parent.scale.y *= dz / model_size[1]
                    if model_size[2] > 1e-6:
                        parent.scale.z *= dy / model_size[2]

                    # Position
                    parent.location = our_to_blender(center)

                    # Rotation from quaternion
                    rotation = occ.get("rotation", [0, 0, 0, 1])
                    if len(rotation) == 4 and any(abs(r) > 1e-6 for r in rotation[:3]):
                        from mathutils import Quaternion
                        # 3D-FRONT quaternion: [w, x, y, z] or [x, y, z, w]?
                        # Check if it's scalar-first or scalar-last
                        q = Quaternion([rotation[3], rotation[0], rotation[1], rotation[2]])
                        parent.rotation_mode = 'QUATERNION'
                        parent.rotation_quaternion = q

                    loaded = True
                    objects.append(parent)
            except Exception as e:
                print(f"  Warning: failed to load {model_path}: {e}")

        if not loaded:
            # Fallback: create box from bbox
            bbox = occ.get("bbox_3d")
            if not bbox:
                continue
            lo, hi = bbox
            extents = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]]
            if any(e <= 0 for e in extents):
                continue
            center = [(lo[0]+hi[0])/2, (lo[1]+hi[1])/2, (lo[2]+hi[2])/2]

            bpy.ops.mesh.primitive_cube_add(size=1.0)
            obj = bpy.context.active_object
            obj.name = name
            obj.scale = [extents[0] / 2, extents[2] / 2, extents[1] / 2]
            obj.location = our_to_blender(center)
            color = (130, 125, 120)
            mat = create_material(f"mat_{name}", color, procedural=False, jitter_strength=0.08)
            obj.data.materials.append(mat)
            objects.append(obj)

    return objects


def add_distractors(distractors_data):
    """Add distractor objects to the scene."""
    objects = []
    for i, dist in enumerate(distractors_data):
        bbox = dist.get("bbox_3d")
        name = dist.get("name", f"distractor_{i}")
        color = dist.get("color", (150, 150, 150))
        pos = dist.get("position_m", [0, 0, 0])

        bpy.ops.mesh.primitive_cube_add(size=0.1)
        obj = bpy.context.active_object
        obj.name = name
        obj.location = our_to_blender(pos)

        if bbox:
            lo, hi = bbox
            extents = [(hi[0] - lo[0]) / 2, (hi[2] - lo[2]) / 2, (hi[1] - lo[1]) / 2]
            obj.scale = [max(e, 0.01) for e in extents]

        mat = create_material(f"mat_{name}", color, procedural=False, jitter_strength=0.08)
        obj.data.materials.append(mat)
        objects.append(obj)

    return objects


def add_scene_geometry(scene_data):
    """Add scene floor, walls, furniture from scene metadata."""
    objects = []

    room = scene_data.get("room_size_m", [5.0, 2.8, 5.0])
    colors = scene_data.get("colors", {})
    palette = _scene_palette(scene_data)
    floor_color = tuple(colors.get("floor", palette["floor"]))
    wall_color = tuple(colors.get("wall", palette["wall"]))
    ceiling_color = tuple(colors.get("ceiling", palette["ceiling"]))

    # Floor
    bpy.ops.mesh.primitive_plane_add(size=1.0)
    floor = bpy.context.active_object
    floor.name = "floor"
    floor.scale = [room[0] / 2, room[2] / 2, 1]
    floor.location = our_to_blender([room[0] / 2, 0, room[2] / 2])
    mat = create_room_material("mat_floor", floor_color, kind="floor")
    floor.data.materials.append(mat)
    objects.append(floor)

    if scene_data.get("demo_mode") == "cube_room" or scene_data.get("closed_room", False):
        # Only procedural cube rooms get opaque synthetic walls. Real 3D-FRONT
        # cameras may sit outside the room bounds, so fake walls would hide the target.
        wall_h = room[1]
        wall_specs = [
            ("wall_back", [room[0] / 2, wall_h / 2, 0], [room[0], 0.05, wall_h]),
            ("wall_front", [room[0] / 2, wall_h / 2, room[2]], [room[0], 0.05, wall_h]),
            ("wall_left", [0, wall_h / 2, room[2] / 2], [0.05, room[2], wall_h]),
            ("wall_right", [room[0], wall_h / 2, room[2] / 2], [0.05, room[2], wall_h]),
        ]
        for name, center, size in wall_specs:
            bpy.ops.mesh.primitive_cube_add(size=1.0)
            wall = bpy.context.active_object
            wall.name = name
            wall.scale = [size[0] / 2, size[2] / 2, size[1] / 2]
            wall.location = our_to_blender(center)
            mat = create_room_material(f"mat_{name}", wall_color, kind="wall")
            wall.data.materials.append(mat)
            objects.append(wall)

        # Ceiling to close the cube room
        bpy.ops.mesh.primitive_cube_add(size=1.0)
        ceiling = bpy.context.active_object
        ceiling.name = "ceiling"
        ceiling.scale = [room[0] / 2, room[2] / 2, 0.04]
        ceiling.location = our_to_blender([room[0] / 2, room[1], room[2] / 2])
        mat = create_room_material("mat_ceiling", ceiling_color, kind="ceiling")
        ceiling.data.materials.append(mat)
        objects.append(ceiling)

    # Furniture surfaces
    for surf in scene_data.get("surfaces", []):
        if surf.get("type") != "furniture":
            continue
        name = surf.get("name", "furniture")
        pos = surf.get("position_m", [0, 0])
        size = surf.get("size_m", [1, 1, 0.5])

        bpy.ops.mesh.primitive_cube_add(size=1.0)
        furn = bpy.context.active_object
        furn.name = name
        furn.scale = [size[0] / 2, size[2] / 2, size[1] / 2]
        furn.location = our_to_blender([pos[0], size[2] / 2, pos[1]])
        color = (
            np.random.randint(80, 200),
            np.random.randint(80, 200),
            np.random.randint(80, 200),
        )
        mat = create_material(f"mat_{name}", color, procedural=False, jitter_strength=0.08)
        furn.data.materials.append(mat)
        objects.append(furn)

    return objects


def create_backdrop_material(name):
    """Create a camera-facing textured background plane material."""
    base = np.array(_random_rich_color(value_range=(0.45, 0.78), sat_range=(0.18, 0.55)), dtype=np.float32)
    accent = np.array(_random_rich_color(value_range=(0.35, 0.88), sat_range=(0.25, 0.75)), dtype=np.float32)

    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    bsdf.inputs["Roughness"].default_value = float(np.random.uniform(0.65, 0.98))
    bsdf.inputs["Metallic"].default_value = 0.0

    if np.random.random() < 0.50:
        tex = nodes.new(type="ShaderNodeTexChecker")
        tex.inputs["Scale"].default_value = float(np.random.uniform(5.0, 16.0))
        tex.inputs["Color1"].default_value = (*[c / 255.0 for c in base], 1)
        tex.inputs["Color2"].default_value = (*[c / 255.0 for c in accent], 1)
        mat.node_tree.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    else:
        noise = nodes.new(type="ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = float(np.random.uniform(3.0, 18.0))
        noise.inputs["Detail"].default_value = float(np.random.uniform(7.0, 15.0))
        ramp = nodes.new(type="ShaderNodeValToRGB")
        ramp.color_ramp.elements[0].position = float(np.random.uniform(0.20, 0.42))
        ramp.color_ramp.elements[1].position = float(np.random.uniform(0.58, 0.88))
        ramp.color_ramp.elements[0].color = (*[c / 255.0 for c in base], 1)
        ramp.color_ramp.elements[1].color = (*[c / 255.0 for c in accent], 1)
        mat.node_tree.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
        mat.node_tree.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    return mat


def add_camera_backdrop(cam_obj, trajectory, scene_data):
    """Add a large textured plane behind the target for non-empty backgrounds."""
    if cam_obj is None or not trajectory or not trajectory.get("positions_m"):
        return None
    try:
        from mathutils import Vector

        positions = np.array(trajectory["positions_m"], dtype=np.float32)
        center_our = positions.mean(axis=0).tolist()
        center_bl = Vector(our_to_blender(center_our))
        cam_pos = Vector(cam_obj.location)
        view_dir = center_bl - cam_pos
        if view_dir.length < 1e-6:
            return None
        view_dir.normalize()

        room = scene_data.get("room_size_m", [5.0, 2.8, 5.0])
        room_extent = max(float(room[0]), float(room[2]), 4.0)
        loc = center_bl + view_dir * float(np.random.uniform(room_extent * 0.65, room_extent * 1.15))

        bpy.ops.mesh.primitive_plane_add(size=1.0, location=loc)
        plane = bpy.context.active_object
        plane.name = "camera_backdrop"
        plane.scale = [room_extent * 0.85, room_extent * 0.55, 1.0]
        plane.rotation_euler = (-view_dir).to_track_quat("Z", "Y").to_euler()
        plane.data.materials.append(create_backdrop_material("mat_camera_backdrop"))
        return plane
    except Exception as exc:
        print(f"  Warning: failed to add camera backdrop: {exc}")
        return None


# ---------------------------------------------------------------------------
# Camera setup
# ---------------------------------------------------------------------------

def setup_camera(cam_data, resolution):
    """Create and configure a Blender camera from calibration data.

    Args:
        cam_data: dict with K, R, T, image_size
        resolution: [width, height]
    """
    K = np.array(cam_data["K"])
    R = np.array(cam_data["R"])  # world-to-camera rotation
    T = np.array(cam_data["T"])  # world-to-camera translation

    # Compute camera-to-world: cam_pos = -R^T @ T
    R_c2w = R.T
    cam_pos_our = -R_c2w @ T

    # Create camera
    cam = bpy.data.cameras.new("Camera")
    cam_obj = bpy.data.objects.new("Camera", cam)
    bpy.context.collection.objects.link(cam_obj)

    # Set intrinsics from K
    fx = K[0, 0]
    sensor_width = 36.0  # mm (default 35mm)
    cam.lens = fx * sensor_width / resolution[0]
    cam.sensor_width = sensor_width
    cam.clip_start = 0.01
    cam.clip_end = 50.0

    # Build camera-to-world 4x4 in Blender coords
    R_bl = our_rotation_to_blender(R_c2w)
    pos_bl = our_to_blender(cam_pos_our.tolist())

    mat = np.eye(4)
    mat[:3, :3] = R_bl
    mat[:3, 3] = pos_bl

    from mathutils import Matrix as _Mat
    cam_obj.matrix_world = _Mat([list(row) for row in mat])
    if os.environ.get("SYNMVTRACK_CAMERA_TRACK_QUAT", "0").strip() == "1":
        from mathutils import Vector
        forward_bl = Vector(our_to_blender(R_c2w[:, 2].tolist()))
        if forward_bl.length > 1e-8:
            cam_obj.location = Vector(pos_bl)
            cam_obj.rotation_euler = forward_bl.to_track_quat('-Z', 'Y').to_euler()

    return cam_obj


def animate_camera(cam_obj, n_frames, frame_start=0):
    """Hold camera static for all frames (multi-view = multiple cameras)."""
    scene = bpy.context.scene
    for fi in range(n_frames):
        scene.frame_set(frame_start + fi)
        cam_obj.keyframe_insert(data_path="location", frame=frame_start + fi)
        cam_obj.keyframe_insert(data_path="rotation_euler", frame=frame_start + fi)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_rgb(output_path, resolution, samples):
    """Render RGB image."""
    scene = bpy.context.scene
    scene.render.image_settings.file_format = 'JPEG'
    scene.render.image_settings.quality = 90
    scene.render.filepath = output_path
    scene.render.film_transparent = False

    # Ensure world background is bright enough (may have been darkened by mask render)
    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg_color = scene.get("rgb_bg_color", [0.58, 0.60, 0.62])
        bg.inputs['Color'].default_value = (*[float(c) for c in bg_color], 1)
        bg.inputs['Strength'].default_value = float(scene.get("rgb_bg_strength", 0.9))

    bpy.ops.render.render(write_still=True)


def render_mask(output_path, target_obj, resolution, samples):
    """Render binary mask of target (occluders visible = normal occlusion)."""
    _render_target_mask(output_path, target_obj, hide_occluders=False)


def render_full_mask(output_path, target_obj):
    """Render full projected target mask (occluders hidden)."""
    _render_target_mask(output_path, target_obj, hide_occluders=True)


def _create_mask_emission_material(name, color, strength=1.0):
    """Create a pure emission material for version-stable binary masks."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    for node in list(nodes):
        nodes.remove(node)
    emission = nodes.new(type="ShaderNodeEmission")
    emission.inputs["Color"].default_value = color
    emission.inputs["Strength"].default_value = float(strength)
    output = nodes.new(type="ShaderNodeOutputMaterial")
    mat.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return mat


def _render_target_mask(output_path, target_obj, hide_occluders=False):
    """Core mask renderer. Renders target as white on black background.

    Temporarily gives target a white emission material so it renders as
    pure white regardless of original material or scene lighting.

    Args:
        hide_occluders: if True, also hide objects named 'occluder_*' or 'foc_*'
    """
    scene = bpy.context.scene

    # Hide all non-target mesh objects
    all_objs = [o for o in scene.objects if o.type == 'MESH']
    hidden = []
    for obj in all_objs:
        if obj == target_obj or obj.name == "Camera":
            continue
        if obj.hide_render is False:
            obj.hide_render = True
            hidden.append(obj)

    # Give target a white emission material
    orig_mats = list(target_obj.data.materials) if target_obj.data else []
    emit_mat = _create_mask_emission_material("mask_white_emit", (1, 1, 1, 1), strength=10.0)

    target_obj.data.materials.clear()
    target_obj.data.materials.append(emit_mat)

    # Black background
    scene.render.image_settings.file_format = 'PNG'
    scene.render.film_transparent = False

    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs['Color'].default_value = (0, 0, 0, 1)

    # Disable lights for pure emission render
    light_states = {}
    for obj in scene.objects:
        if obj.type == 'LIGHT':
            light_states[obj.name] = obj.hide_render
            obj.hide_render = True

    scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)

    # Restore lights
    for name, state in light_states.items():
        scene.objects[name].hide_render = state

    # Restore target materials
    target_obj.data.materials.clear()
    for m in orig_mats:
        target_obj.data.materials.append(m)
    bpy.data.materials.remove(emit_mat)

    # Restore hidden objects
    for obj in hidden:
        obj.hide_render = False
    if bg:
        bg.inputs['Color'].default_value = (0.05, 0.05, 0.05, 1)


def render_depth_npy(output_path, resolution):
    """Render depth and save as .npy."""
    scene = bpy.context.scene
    scene.view_layers["ViewLayer"].use_pass_z = True

    # Use compositor to save depth
    scene.use_nodes = True
    tree = scene.node_tree
    for node in tree.nodes:
        tree.nodes.remove(node)

    rl = tree.nodes.new('CompositorNodeRLayers')
    depth_out = tree.nodes.new('CompositorNodeOutputFile')
    depth_out.base_path = os.path.dirname(output_path)
    depth_out.file_slots[0].path = os.path.basename(output_path).replace('.npy', '')
    depth_out.format.file_format = 'OPEN_EXR'
    depth_out.format.color_depth = '32'

    tree.links.new(rl.outputs['Depth'], depth_out.inputs[0])

    scene.render.filepath = output_path.replace('.npy', '_placeholder')
    bpy.ops.render.render(write_still=True)

    # Note: The EXR depth needs post-processing to .npy
    # This is done in post_render_cleanup()


# ---------------------------------------------------------------------------
# Mask analysis (visibility, bbox)
# ---------------------------------------------------------------------------

def _load_mask_pixels(png_path):
    """Load a rendered PNG mask as a boolean array (True = target pixel)."""
    try:
        from PIL import Image
        img = Image.open(png_path).convert('L')
        arr = np.array(img)
        if arr.size and int(arr.max()) <= 8:
            return arr > 0
        return arr > 128
    except ImportError:
        # Fallback: use Blender's built-in
        return None


def _dilate_mask(mask, radius=1):
    """Small binary dilation used only to tolerate one-pixel raster edges."""
    if mask is None or radius <= 0:
        return mask
    out = mask.copy()
    for _ in range(radius):
        padded = np.pad(out, ((1, 1), (1, 1)), mode="constant", constant_values=False)
        expanded = np.zeros_like(out, dtype=bool)
        for dy in range(3):
            for dx in range(3):
                expanded |= padded[dy:dy + out.shape[0], dx:dx + out.shape[1]]
        out = expanded
    return out


def _save_mask_pixels(png_path, mask):
    """Write a clean binary PNG mask after leakage cleanup."""
    if mask is None:
        return
    try:
        from PIL import Image
        arr = (mask.astype(np.uint8) * 255)
        Image.fromarray(arr, mode='L').save(png_path)
    except ImportError:
        pass


def compute_mask_area_and_bbox(mask_path):
    """Compute mask pixel area and tight bounding box.

    Returns:
        area: int (number of white pixels)
        bbox_xywh: [x, y, w, h] or None if mask is empty
    """
    mask = _load_mask_pixels(mask_path)
    if mask is None:
        return 0, None

    area = int(np.sum(mask))
    if area == 0:
        return 0, None

    ys, xs = np.where(mask)
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    bbox = [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]
    return area, bbox


def render_visible_and_full(seq_dir, vi, fi, target_obj, occluder_objs,
                             resolution, samples):
    """Render both visible mask and full projected mask for one frame.

    Returns:
        visibility_ratio: float in [0, 1]
        full_bbox_xywh: [x, y, w, h] or None
    """
    visible_path = os.path.join(seq_dir, "masks", f"{vi:04d}", f"{fi:06d}.png")
    full_path = os.path.join(seq_dir, "full_masks", f"{vi:04d}", f"{fi:06d}.png")

    # Pass 1: render visible mask (occluders in scene block the target)
    # render_mask hides all non-target objects anyway, so this gives the
    # full projected mask. For visible mask, we need occluders to actually
    # block — we achieve this by NOT hiding them in the render.
    #
    # Correct approach:
    #   visible_mask = render with occluders as opaque (they block target)
    #   full_mask    = render with occluders hidden (target fully visible)

    # --- Full mask: hide occluders, render target only ---
    for obj in occluder_objs:
        obj.hide_render = True
    _render_target_mask(full_path, target_obj, hide_occluders=True)

    # --- Visible mask: keep occluders visible, render target ---
    # Occluders are physical objects; the mask render already hides all
    # non-target meshes. To get occlusion, we need to render differently:
    # render the scene normally on black background, extract target pixels.
    for obj in occluder_objs:
        obj.hide_render = False
    _render_visible_mask(visible_path, target_obj, occluder_objs)

    # Remove render-pass leakage: visible target pixels must be a subset of
    # the full projected target mask, with one-pixel tolerance for raster edges.
    full_mask = _load_mask_pixels(full_path)
    visible_mask = _load_mask_pixels(visible_path)
    if full_mask is not None and visible_mask is not None:
        clean_visible = np.logical_and(visible_mask, _dilate_mask(full_mask, radius=1))
        _save_mask_pixels(full_path, full_mask)
        _save_mask_pixels(visible_path, clean_visible)

    # Analyze masks
    full_area, full_bbox = compute_mask_area_and_bbox(full_path)
    visible_area, _ = compute_mask_area_and_bbox(visible_path)

    visibility_ratio = visible_area / max(full_area, 1)
    visibility_ratio = round(min(visibility_ratio, 1.0), 4)

    return visibility_ratio, full_bbox


def _render_visible_mask(output_path, target_obj, occluder_objs):
    """Render visible mask: target pixels NOT blocked by occluders.

    Strategy: render target as white on black, but keep occluders opaque.
    The occluders physically block the camera from seeing target pixels.
    """
    scene = bpy.context.scene

    # Make target white emission, everything else black
    # Temporarily give target an emission material
    orig_mats = list(target_obj.data.materials) if target_obj.data else []
    emit_mat = _create_mask_emission_material("mask_emit", (1, 1, 1, 1), strength=10.0)

    target_obj.data.materials.clear()
    target_obj.data.materials.append(emit_mat)

    # Make everything non-target black
    all_mesh_objs = [o for o in scene.objects if o.type == 'MESH']
    saved_mats = {}
    black_mat = _create_mask_emission_material("mask_black", (0, 0, 0, 1), strength=1.0)

    for obj in all_mesh_objs:
        if obj == target_obj or obj.name == "Camera":
            continue
        if obj.hide_render:
            continue
        saved_mats[obj.name] = list(obj.data.materials)
        obj.data.materials.clear()
        obj.data.materials.append(black_mat)

    # Black background
    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs['Color'].default_value = (0, 0, 0, 1)

    scene.render.image_settings.file_format = 'PNG'
    scene.render.film_transparent = False
    scene.render.filepath = output_path

    # Disable all lights for pure emission render.
    light_states = {}
    for obj in scene.objects:
        if obj.type == 'LIGHT':
            light_states[obj.name] = obj.hide_render
            obj.hide_render = True

    # The white target is emissive; with normal Cycles bounces it can illuminate
    # black floors/occluders and leak noisy non-target pixels into masks.
    cycle_bounce_state = {}
    if hasattr(scene, 'cycles'):
        for attr in [
            'max_bounces', 'diffuse_bounces', 'glossy_bounces',
            'transmission_bounces', 'transparent_max_bounces', 'volume_bounces',
        ]:
            if hasattr(scene.cycles, attr):
                cycle_bounce_state[attr] = getattr(scene.cycles, attr)
                setattr(scene.cycles, attr, 0)

    bpy.ops.render.render(write_still=True)

    for attr, value in cycle_bounce_state.items():
        setattr(scene.cycles, attr, value)

    # Restore lights
    for name, state in light_states.items():
        scene.objects[name].hide_render = state

    # Restore materials
    target_obj.data.materials.clear()
    for mat in orig_mats:
        target_obj.data.materials.append(mat)
    bpy.data.materials.remove(emit_mat)

    for obj in all_mesh_objs:
        if obj.name in saved_mats:
            obj.data.materials.clear()
            for m in saved_mats[obj.name]:
                obj.data.materials.append(m)

    bpy.data.materials.remove(black_mat)


def post_render_cleanup(seq_dir):
    """Convert EXR depth files to .npy after rendering."""
    depth_dir = os.path.join(seq_dir, "depth")
    if not os.path.isdir(depth_dir):
        return

    try:
        import OpenEXR
        import Imath
        has_openexr = True
    except ImportError:
        has_openexr = False

    for root, dirs, files in os.walk(depth_dir):
        for f in files:
            if f.endswith(".exr"):
                exr_path = os.path.join(root, f)
                npy_path = exr_path.replace(".exr", ".npy")
                if has_openexr:
                    try:
                        exr = OpenEXR.InputFile(exr_path)
                        dw = exr.header()['dataWindow']
                        w = dw.max.x - dw.min.x + 1
                        h = dw.max.y - dw.min.y + 1
                        depth_str = exr.channel('R', Imath.PixelType(Imath.PixelType.FLOAT))
                        depth = np.frombuffer(depth_str, dtype=np.float32).reshape(h, w)
                        np.save(npy_path, depth)
                        os.remove(exr_path)
                    except Exception as e:
                        print(f"  Warning: EXR->npy failed for {exr_path}: {e}")
                else:
                    # Rename EXR to npy (caller can load with imageio)
                    os.rename(exr_path, npy_path)


# ---------------------------------------------------------------------------
# Visibility data export
# ---------------------------------------------------------------------------

def _save_visibility_data(seq_dir, visibility_data, full_bbox_data, n_views, n_frames):
    """Save visibility.txt and full_projected_bbox.txt.

    visibility.txt format (one row per frame, one column per view):
        vis_v0_f0,vis_v1_f0,vis_v2_f0
        vis_v0_f1,vis_v1_f1,vis_v2_f1
        ...

    full_projected_bbox.txt format (one row per frame, views separated by |):
        x0,y0,w0,h0|x1,y1,w1,h1|x2,y2,w2,h2
        ...
    """
    # visibility.txt
    vis_path = os.path.join(seq_dir, "visibility.txt")
    with open(vis_path, "w") as f:
        for fi in range(n_frames):
            vals = []
            for vi in range(n_views):
                vals.append(f"{visibility_data[vi][fi]:.4f}")
            f.write(",".join(vals) + "\n")

    # full_projected_bbox.txt
    bbox_path = os.path.join(seq_dir, "full_projected_bbox.txt")
    with open(bbox_path, "w") as f:
        for fi in range(n_frames):
            parts = []
            for vi in range(n_views):
                bbox = full_bbox_data[vi][fi]
                if bbox is not None:
                    parts.append(f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}")
                else:
                    parts.append("-1,-1,-1,-1")
            f.write("|".join(parts) + "\n")

    print(f"  Saved {vis_path}")
    print(f"  Saved {bbox_path}")

    # Print visibility stats
    all_vis = []
    for vi in range(n_views):
        all_vis.extend(visibility_data[vi])
    if all_vis:
        mean_vis = sum(all_vis) / len(all_vis)
        min_vis = min(all_vis)
        max_vis = max(all_vis)
        fully_occ = sum(1 for v in all_vis if v < 0.01)
        print(f"  Visibility stats: mean={mean_vis:.3f} min={min_vis:.3f} "
              f"max={max_vis:.3f} fully_occluded={fully_occ}/{len(all_vis)}")


# ---------------------------------------------------------------------------
# Main render loop
# ---------------------------------------------------------------------------

def render_sequence(seq_dir, cfg):
    """Render all frames for all views in a sequence.

    Args:
        seq_dir: path to sequence directory
        cfg: render config dict (resolution, samples, save_depth, etc.)
    """
    resolution = cfg.get("resolution", [640, 480])
    samples = cfg.get("samples", 32)
    save_mask = cfg.get("save_mask", True)
    save_depth = cfg.get("save_depth", False)
    max_frames = cfg.get("max_frames", None)
    identity_priority = bool(cfg.get("identity_priority", False))
    use_view_dependent_backdrop = bool(cfg.get("use_view_dependent_backdrop", True))
    force_closed_room = bool(cfg.get("force_closed_room", False))

    # Load sequence metadata
    with open(os.path.join(seq_dir, "attributes.json")) as f:
        attributes = json.load(f)
    with open(os.path.join(seq_dir, "calibs.json")) as f:
        calibs = json.load(f)

    n_frames = attributes["num_frames"]
    n_views = attributes.get("num_views", len(calibs))

    # Try to load scene/target/trajectory data
    meta_path = os.path.join(seq_dir, "render_meta.json")
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    else:
        meta = {}

    scene_data = meta.get("scene", {})
    if force_closed_room:
        scene_data = dict(scene_data)
        scene_data["closed_room"] = True
    target_data = meta.get("target", {})
    trajectory = meta.get("trajectory", {})
    occluders_data = meta.get("occluders", [])
    distractors_data = meta.get("distractors", [])

    # Override n_frames from trajectory if available
    if trajectory:
        n_frames = len(trajectory.get("positions_m", []))
    n_views = len(calibs)

    # Cap frames if requested
    if max_frames is not None and n_frames > max_frames:
        n_frames = max_frames
        # Also trim trajectory data
        if trajectory:
            trajectory = {k: v[:n_frames] if isinstance(v, list) else v
                          for k, v in trajectory.items()}

    # Derive project root from seq_dir: .../project_root/output/SynMVTrack_v1/seq_xxxx -> root
    _parts = os.path.normpath(seq_dir).split(os.sep)
    if "output" in _parts:
        _idx = _parts.index("output")
        project_root = os.sep.join(_parts[:_idx])
    else:
        project_root = os.path.dirname(seq_dir)

    print(f"Rendering {seq_dir}: {n_frames} frames x {n_views} views")
    print(f"  Resolution: {resolution}, Samples: {samples}")
    print(f"  Project root: {project_root}")
    seq_id = attributes.get("sequence_id", os.path.basename(seq_dir))
    target_id = target_data.get("target_id", "target")
    appearance_seed = int(meta.get(
        "appearance_seed",
        _stable_seed(seq_id, target_id, n_frames, n_views),
    ))
    print(f"  Appearance seed: {appearance_seed}")

    # Create output directories
    for vi in range(n_views):
        for subdir in ["img", "masks", "full_masks", "depth"]:
            os.makedirs(os.path.join(seq_dir, subdir, f"{vi:04d}"), exist_ok=True)

    # Render each view
    # Track visibility data across all views
    visibility_data = {vi: [] for vi in range(n_views)}   # vis[vi][fi] = ratio
    full_bbox_data = {vi: [] for vi in range(n_views)}    # bbox[vi][fi] = [x,y,w,h]

    for vi in range(n_views):
        view_key = f"cam{vi}"
        if view_key not in calibs:
            view_key = str(vi)
        cam_data = calibs[view_key]

        print(f"  View {vi}: setting up camera")

        # Clear and rebuild scene for each view (memory efficiency)
        # Reuse the same sequence seed for every view so multi-view colors match.
        np.random.seed(appearance_seed)
        clear_scene()

        # Setup render
        setup_render(
            resolution,
            samples,
            seq_dir,
            save_depth=save_depth,
            identity_priority=identity_priority,
        )

        # Add scene geometry
        add_scene_geometry(scene_data)

        # Add occluders and collect references
        occluder_objs = add_occluders(occluders_data, project_root=project_root)

        # Add distractors
        add_distractors(distractors_data)

        # Load and animate target
        target_obj = None
        if trajectory and target_data:
            target_obj = load_target_with_animation(target_data, trajectory,
                                                     project_root=project_root)

        # Setup camera
        cam_obj = setup_camera(cam_data, resolution)
        bpy.context.scene.camera = cam_obj
        animate_camera(cam_obj, n_frames)
        if use_view_dependent_backdrop:
            add_camera_backdrop(cam_obj, trajectory, scene_data)

        if os.environ.get("SYNMVTRACK_DEBUG_CAMERA", "0").strip() == "1" and target_obj:
            try:
                from bpy_extras.object_utils import world_to_camera_view
                bpy.context.scene.frame_set(1)
                bpy.context.view_layer.update()
                center = target_obj.matrix_world.translation
                ndc = world_to_camera_view(bpy.context.scene, cam_obj, center)
                print(
                    "  DEBUG_CAMERA "
                    f"view={vi} cam_loc={tuple(round(v, 3) for v in cam_obj.location)} "
                    f"target_loc={tuple(round(v, 3) for v in center)} "
                    f"ndc=({ndc.x:.3f},{ndc.y:.3f},{ndc.z:.3f})"
                )
            except Exception as exc:
                print(f"  DEBUG_CAMERA failed: {exc}")

        # Add lighting
        if identity_priority:
            bpy.context.scene["rgb_bg_color"] = [0.42, 0.43, 0.44]
            bpy.context.scene["rgb_bg_strength"] = 0.28
        else:
            bpy.context.scene["rgb_bg_color"] = [float(x) for x in np.random.uniform(0.30, 0.78, size=3)]
            bpy.context.scene["rgb_bg_strength"] = float(np.random.uniform(0.35, 1.05))
        if scene_data.get("demo_mode") == "cube_room" or scene_data.get("closed_room", False):
            room = scene_data.get("room_size_m", [4.0, 4.0, 4.0])
            bpy.ops.object.light_add(
                type='AREA',
                location=our_to_blender([room[0] / 2.0, room[1] - 0.18, room[2] / 2.0]),
            )
            key_light = bpy.context.active_object
            key_light.name = "room_key_area"
            key_light.data.energy = 150.0
            key_light.data.size = max(1.2, min(room[0], room[2]) * 0.55)

            for x, z in [
                (room[0] * 0.25, room[2] * 0.25),
                (room[0] * 0.75, room[2] * 0.25),
                (room[0] * 0.50, room[2] * 0.75),
            ]:
                bpy.ops.object.light_add(type='POINT', location=our_to_blender([x, room[1] * 0.62, z]))
                fill = bpy.context.active_object
                fill.name = "room_fill_point"
                fill.data.energy = 16.0
                fill.data.shadow_soft_size = 1.2
        else:
            sun_pos = [float(np.random.uniform(1.5, 4.0)),
                       float(np.random.uniform(2.0, 4.5)),
                       float(np.random.uniform(1.5, 4.0))]
            bpy.ops.object.light_add(type='SUN', location=our_to_blender(sun_pos))
            sun = bpy.context.active_object
            sun.data.energy = float(np.random.uniform(2.5, 7.5))
            sun.data.angle = math.radians(float(np.random.uniform(12, 55)))

            area_pos = [float(np.random.uniform(0.8, 4.2)),
                        float(np.random.uniform(1.5, 3.2)),
                        float(np.random.uniform(0.8, 4.2))]
            bpy.ops.object.light_add(type='AREA', location=our_to_blender(area_pos))
            area_light = bpy.context.active_object
            area_light.data.energy = float(np.random.uniform(120.0, 420.0))
            area_light.data.size = float(np.random.uniform(2.0, 6.5))

        # Render frame by frame
        for fi in range(n_frames):
            frame = fi + 1
            bpy.context.scene.frame_set(frame)

            # RGB
            rgb_path = os.path.join(seq_dir, "img", f"{vi:04d}", f"{fi:06d}.jpg")
            render_rgb(rgb_path, resolution, samples)

            # Dual mask: visible + full projected
            if save_mask and target_obj:
                vis_ratio, full_bbox = render_visible_and_full(
                    seq_dir, vi, fi, target_obj, occluder_objs,
                    resolution, samples,
                )
                visibility_data[vi].append(vis_ratio)
                full_bbox_data[vi].append(full_bbox)
            else:
                visibility_data[vi].append(1.0)
                full_bbox_data[vi].append(None)

            # Depth
            if save_depth:
                depth_path = os.path.join(seq_dir, "depth", f"{vi:04d}", f"{fi:06d}.npy")
                render_depth_npy(depth_path, resolution)

            if (fi + 1) % 10 == 0:
                print(f"    View {vi}: frame {fi + 1}/{n_frames}")

    # --- Save visibility and full bbox data ---
    _save_visibility_data(seq_dir, visibility_data, full_bbox_data, n_views, n_frames)

    # Post-process depth EXR -> npy
    if save_depth:
        post_render_cleanup(seq_dir)

    print(f"  Done: {seq_dir}")


# ---------------------------------------------------------------------------
# Batch rendering
# ---------------------------------------------------------------------------

def render_batch(output_dir, cfg, seq_ids=None):
    """Render multiple sequences."""
    if seq_ids is None:
        seq_ids = sorted([
            d for d in os.listdir(output_dir)
            if os.path.isdir(os.path.join(output_dir, d)) and d.startswith("seq_")
        ])

    for seq_id in seq_ids:
        seq_dir = os.path.join(output_dir, seq_id)
        if not os.path.isdir(seq_dir):
            print(f"Skipping {seq_id}: not found")
            continue
        try:
            render_sequence(seq_dir, cfg)
        except Exception as e:
            print(f"Error rendering {seq_id}: {e}")
            import traceback
            traceback.print_exc()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    # Parse args (Blender passes everything after --)
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]

    parser = argparse.ArgumentParser(description="Render SynMVTrack sequences with Blender")
    parser.add_argument("--seq-dir", default=None, help="Single sequence directory")
    parser.add_argument("--output-dir", default="output/SynMVTrack", help="Batch output directory")
    parser.add_argument("--seq-ids", nargs="*", default=None, help="Specific sequence IDs")
    parser.add_argument("--resolution", nargs=2, type=int, default=[640, 480])
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--save-mask", action="store_true", default=True)
    parser.add_argument("--save-depth", action="store_true", default=False)
    parser.add_argument("--max-frames", type=int, default=None, help="Cap number of frames to render")
    parser.add_argument("--disable-view-backdrop", action="store_true",
                        help="Do not add camera-facing per-view backdrop planes")
    parser.add_argument("--identity-priority", action="store_true",
                        help="Reduce color-management randomization for cross-view target identity")
    parser.add_argument("--force-closed-room", action="store_true",
                        help="Render procedural walls/ceiling for a shared physical background")
    args = parser.parse_args(argv)

    cfg = {
        "resolution": args.resolution,
        "samples": args.samples,
        "save_mask": args.save_mask,
        "save_depth": args.save_depth,
        "max_frames": args.max_frames,
        "use_view_dependent_backdrop": not args.disable_view_backdrop,
        "identity_priority": args.identity_priority,
        "force_closed_room": args.force_closed_room,
    }

    if args.seq_dir:
        render_sequence(args.seq_dir, cfg)
    else:
        render_batch(args.output_dir, cfg, seq_ids=args.seq_ids)


if __name__ == "__main__":
    main()
