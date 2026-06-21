#!/usr/bin/env python3
"""Download realistic 3D assets from Poly Haven (CC0)."""
import json
import pathlib
import random
import time
import urllib.request
from urllib.parse import urlparse
from typing import Dict, Optional

CACHE_DIR = pathlib.Path(__file__).resolve().parent.parent / "cache" / "polyhaven"


def fetch_json(url: str) -> Dict:
    req = urllib.request.Request(url, headers={"User-Agent": "MVTrackGen/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def pick_file(entry: Dict, preferred_quality: str, fallback_quality: str = "1k") -> Optional[Dict]:
    for q in [preferred_quality, "2k", fallback_quality, "4k", "8k"]:
        if q in entry:
            quality_entry = entry[q]
            if isinstance(quality_entry, dict):
                if "url" in quality_entry:
                    return {"quality": q, "url": quality_entry["url"]}
                for fmt in ["fbx", "png", "jpg", "exr"]:
                    if fmt in quality_entry and isinstance(quality_entry[fmt], dict) and "url" in quality_entry[fmt]:
                        out = dict(quality_entry[fmt])
                        out["format"] = fmt
                        out["quality"] = q
                        return out
    return None


def safe_download(url: str, path: pathlib.Path):
    req = urllib.request.Request(url, headers={"User-Agent": "MVTrackGen/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        path.write_bytes(resp.read())


def texture_role_from_name(name: str) -> Optional[str]:
    stem = name.lower()
    if any(k in stem for k in ["diff", "basecolor", "base_color", "albedo"]):
        return "diffuse"
    if any(k in stem for k in ["rough", "roughness"]):
        return "rough"
    if "metal" in stem:
        return "metallic"
    if stem.endswith("_ao") or "_ao_" in stem or "ambientocclusion" in stem:
        return "ao"
    if "nor_gl" in stem or "normal_gl" in stem:
        return "normal_gl"
    if "nor_dx" in stem or "normal_dx" in stem:
        return "normal_dx"
    if "normal" in stem:
        return "normal"
    return None


def extension_from_url(url: str, default: str = ".bin") -> str:
    suffix = pathlib.Path(urlparse(url).path).suffix
    return suffix if suffix else default


def download_texture(tex_dir: pathlib.Path, tex_name: str, url: str) -> pathlib.Path:
    tex_path = tex_dir / tex_name
    if not tex_path.exists() or tex_path.stat().st_size == 0:
        safe_download(url, tex_path)
    return tex_path


def collect_fbx_included_textures(files: Dict, quality: str, tex_dir: pathlib.Path) -> Dict[str, str]:
    """Download FBX sidecar textures while preserving their semantic filenames."""
    tex_map = {}
    fbx_choice = pick_file(files.get("fbx", {}), quality)
    if not fbx_choice:
        return tex_map

    includes = files.get("fbx", {}).get(fbx_choice["quality"], {}).get("fbx", {}).get("include", {})
    if not isinstance(includes, dict):
        return tex_map

    for rel_path, finfo in includes.items():
        if not isinstance(finfo, dict) or "url" not in finfo:
            continue
        tex_name = pathlib.Path(rel_path).name
        tex_path = download_texture(tex_dir, tex_name, finfo["url"])
        role = texture_role_from_name(tex_name)
        if role and role not in tex_map:
            tex_map[role] = str(tex_path)
    return tex_map


def collect_channel_textures(files: Dict, uid: str, quality: str, tex_dir: pathlib.Path) -> Dict[str, str]:
    """Fallback for assets whose FBX include list is missing or incomplete."""
    tex_map = {}
    channel_to_role = {
        "Diffuse": "diffuse",
        "Rough": "rough",
        "AO": "ao",
        "Metal": "metallic",
        "nor_gl": "normal_gl",
        "nor_dx": "normal_dx",
    }
    for channel, role in channel_to_role.items():
        if role in tex_map or channel not in files:
            continue
        choice = pick_file(files[channel], quality)
        if not choice or "url" not in choice:
            continue
        ext = extension_from_url(choice["url"], ".jpg" if role == "diffuse" else ".png")
        tex_name = f"{uid}_{role}_{choice.get('quality', quality)}{ext}"
        tex_path = download_texture(tex_dir, tex_name, choice["url"])
        tex_map[role] = str(tex_path)
    return tex_map


def download_polyhaven(count=30, quality="1k", uids=None):
    """Download FBX models from Poly Haven. CC0 license."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("[Poly Haven] Fetching asset list...")
    all_assets = fetch_json("https://api.polyhaven.com/assets?t=models")

    # Prefer useful categories
    preferred = []
    fallback = []
    for uid, info in all_assets.items():
        tags = " ".join(info.get("tags", []) + info.get("categories", [])).lower()
        if any(k in tags for k in ["food", "furniture", "manmade", "indoor", "kitchen", "office"]):
            preferred.append(uid)
        else:
            fallback.append(uid)

    if uids:
        selected = [uid for uid in uids if uid in all_assets]
        missing = [uid for uid in uids if uid not in all_assets]
        for uid in missing:
            print(f"[Poly Haven] Missing asset uid: {uid}")
    else:
        rng = random.Random(42)
        rng.shuffle(preferred)
        rng.shuffle(fallback)
        selected = (preferred + fallback)[:count]

    print(f"[Poly Haven] Downloading {len(selected)} assets (quality={quality})...")
    downloaded = []

    for i, uid in enumerate(selected):
        asset_dir = CACHE_DIR / uid
        fbx_path = asset_dir / "model.fbx"

        # Get file URLs
        try:
            files = fetch_json(f"https://api.polyhaven.com/files/{uid}")
        except Exception as e:
            print(f"  [{i+1}/{len(selected)}] {uid} API FAIL: {e}")
            continue

        # Get FBX URL
        fbx_url = None
        if "fbx" in files:
            fbx_choice = pick_file(files["fbx"], quality)
            if fbx_choice:
                fbx_url = fbx_choice.get("url")
        if not fbx_url and "gltf" in files:
            gltf_choice = pick_file(files["gltf"], quality)
            if gltf_choice:
                fbx_url = gltf_choice.get("url")

        if not fbx_url:
            print(f"  [{i+1}/{len(selected)}] {uid} no FBX")
            continue

        # Download FBX
        asset_dir.mkdir(parents=True, exist_ok=True)
        try:
            with urllib.request.urlopen(urllib.request.Request(fbx_url, headers={"User-Agent": "MVTrackGen/1.0"}), timeout=120) as resp:
                data = resp.read()
            fbx_path.write_bytes(data)

            # Download textures with stable semantic names. Poly Haven FBX packages
            # reference sidecar files under textures/<semantic_name>; preserving
            # those names lets UE's FBX importer rebuild material links.
            tex_dir = asset_dir / "textures"
            tex_dir.mkdir(exist_ok=True)
            tex_map = collect_fbx_included_textures(files, quality, tex_dir)
            fallback_map = collect_channel_textures(files, uid, quality, tex_dir)
            for role, tex_path in fallback_map.items():
                tex_map.setdefault(role, tex_path)

            meta = {
                "uid": uid, "source": "polyhaven", "license": "CC0",
                "path": str(fbx_path), "format": "fbx",
                "textures_dir": str(tex_dir),
                "textures": tex_map
            }
            (asset_dir / "meta.json").write_text(json.dumps(meta, indent=2))

            size_mb = len(data) / 1024 / 1024
            downloaded.append(str(fbx_path))
            print(f"  [{i+1}/{len(selected)}] {uid} ({size_mb:.1f}MB)")
        except Exception as e:
            print(f"  [{i+1}/{len(selected)}] {uid} FAIL: {e}")

        time.sleep(0.3)

    print(f"\n[Poly Haven] Downloaded {len(downloaded)} assets to {CACHE_DIR}")
    return downloaded


def build_registry():
    """Build unified asset registry."""
    registry = []
    for asset_dir in CACHE_DIR.iterdir():
        meta_path = asset_dir / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            registry.append(meta)

    reg_path = CACHE_DIR.parent / "asset_registry.json"
    reg_path.write_text(json.dumps(registry, indent=2))
    print(f"[Registry] {len(registry)} assets -> {reg_path}")
    return registry


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--quality", default="1k", choices=["1k", "2k", "4k"])
    parser.add_argument("--uids", nargs="*", default=None,
                        help="Optional explicit Poly Haven asset UIDs to redownload.")
    args = parser.parse_args()

    download_polyhaven(count=args.count, quality=args.quality, uids=args.uids)
    build_registry()
