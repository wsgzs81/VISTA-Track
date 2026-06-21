#!/usr/bin/env python3
"""Composite rendered SynMVTrack targets onto real video backgrounds.

This keeps the synthetic target geometry, masks, bboxes, visibility and camera
metadata intact, but replaces flat Blender backgrounds with frames sampled from
real tracking videos such as GOT-10k. Run after rendering and before export /
MVTrack formatting, or run on a copied output directory for visual probes.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def list_background_videos(root: Path) -> list[list[Path]]:
    videos: list[list[Path]] = []
    for dirpath, _, filenames in os.walk(root):
        frames = [
            Path(dirpath) / name
            for name in filenames
            if Path(name).suffix.lower() in IMG_EXTS
        ]
        if len(frames) >= 8:
            videos.append(sorted(frames))
    if not videos:
        raise RuntimeError(f"No background videos with >=8 frames under {root}")
    return videos


def copy_input(input_root: Path, output_root: Path) -> Path:
    if output_root.exists():
        raise FileExistsError(f"Output already exists: {output_root}")
    shutil.copytree(input_root, output_root)
    return output_root


def cover_crop(img: Image.Image, size: tuple[int, int], rng: random.Random) -> Image.Image:
    width, height = size
    img = img.convert("RGB")
    scale = max(width / img.width, height / img.height)
    new_size = (max(width, int(round(img.width * scale))), max(height, int(round(img.height * scale))))
    img = img.resize(new_size, Image.Resampling.BICUBIC)
    max_x = max(0, img.width - width)
    max_y = max(0, img.height - height)
    x = rng.randint(0, max_x) if max_x else 0
    y = rng.randint(0, max_y) if max_y else 0
    return img.crop((x, y, x + width, y + height))


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0.05)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def match_target_to_background(target: np.ndarray, bg: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Lightweight per-channel illumination matching inside the target bbox."""
    box = bbox_from_mask(alpha)
    if box is None:
        return target
    x0, y0, x1, y1 = box
    m = alpha[y0:y1, x0:x1] > 0.45
    if not np.any(m):
        return target

    tgt_crop = target[y0:y1, x0:x1]
    bg_crop = bg[y0:y1, x0:x1]
    tgt_mean = tgt_crop[m].mean(axis=0)
    bg_mean = bg_crop.reshape(-1, 3).mean(axis=0)
    scale = np.clip((bg_mean + 10.0) / (tgt_mean + 10.0), 0.72, 1.32)

    out = target.copy()
    adjusted = np.clip((target.astype(np.float32) - 127.5) * 0.94 + 127.5, 0, 255)
    adjusted = np.clip(adjusted * scale.reshape(1, 1, 3), 0, 255)
    out = adjusted.astype(np.uint8)
    return out


def composite_one(rgb_path: Path, mask_path: Path, bg_path: Path, rng: random.Random, feather: float) -> bool:
    if not rgb_path.is_file() or not mask_path.is_file():
        return False

    src = Image.open(rgb_path).convert("RGB")
    mask_img = Image.open(mask_path).convert("L").resize(src.size, Image.Resampling.NEAREST)
    if feather > 0:
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=feather))
    bg = cover_crop(Image.open(bg_path), src.size, rng)

    alpha = np.asarray(mask_img, dtype=np.float32) / 255.0
    if alpha.max() <= 0.01:
        bg.save(rgb_path, quality=92)
        return True

    src_arr = np.asarray(src, dtype=np.uint8)
    bg_arr = np.asarray(bg, dtype=np.uint8)
    tgt_arr = match_target_to_background(src_arr, bg_arr, alpha)
    a = alpha[..., None]
    comp = np.clip(tgt_arr.astype(np.float32) * a + bg_arr.astype(np.float32) * (1.0 - a), 0, 255)
    Image.fromarray(comp.astype(np.uint8), "RGB").save(rgb_path, quality=92)
    return True


def sync_mvtrack_view_dirs(seq_dir: Path) -> None:
    """If MVTrack per-view dirs already exist, mirror native img/ into them."""
    attrs_path = seq_dir / "attributes.json"
    if not attrs_path.is_file():
        return
    attrs = json.loads(attrs_path.read_text())
    seq_name = seq_dir.name
    for vi in range(int(attrs.get("num_views", 0))):
        native = seq_dir / "img" / f"{vi:04d}"
        view_img = seq_dir / f"{seq_name}-{vi + 1}" / "img"
        if not native.is_dir() or not view_img.is_dir():
            continue
        for idx, src in enumerate(sorted(native.glob("*.jpg")), start=1):
            dst = view_img / f"{idx:05d}.jpg"
            shutil.copy2(src, dst)


def process_sequence(seq_dir: Path, bg_videos: list[list[Path]], rng: random.Random,
                     max_frames: int | None, feather: float) -> tuple[int, int]:
    attrs = json.loads((seq_dir / "attributes.json").read_text())
    n_views = int(attrs.get("num_views", 0))
    n_frames = int(attrs.get("num_frames", 0))
    if max_frames is not None:
        n_frames = min(n_frames, max_frames)

    ok = 0
    total = 0
    for vi in range(n_views):
        video = rng.choice(bg_videos)
        start = rng.randint(0, max(0, len(video) - n_frames - 1))
        for fi in range(n_frames):
            rgb_path = seq_dir / "img" / f"{vi:04d}" / f"{fi:06d}.jpg"
            mask_path = seq_dir / "masks" / f"{vi:04d}" / f"{fi:06d}.png"
            bg_path = video[(start + fi) % len(video)]
            total += 1
            if composite_one(rgb_path, mask_path, bg_path, rng, feather):
                ok += 1

    sync_mvtrack_view_dirs(seq_dir)
    return ok, total


def main() -> None:
    parser = argparse.ArgumentParser(description="Composite SynMVTrack targets onto real video backgrounds")
    parser.add_argument("--input-root", required=True, help="Rendered SynMVTrack output root or one seq dir")
    parser.add_argument("--output-root", default=None, help="Copy input here before compositing")
    parser.add_argument("--background-root", required=True, help="Root containing real video frame folders")
    parser.add_argument("--seq-ids", nargs="*", default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--feather", type=float, default=0.9)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    input_root = Path(args.input_root)
    work_root = Path(args.output_root) if args.output_root else input_root
    if args.output_root:
        copy_input(input_root, work_root)

    bg_videos = list_background_videos(Path(args.background_root))
    if (work_root / "attributes.json").is_file():
        seq_dirs = [work_root]
    else:
        seq_ids = args.seq_ids or [p.name for p in sorted(work_root.glob("seq_*")) if p.is_dir()]
        seq_dirs = [work_root / sid for sid in seq_ids]

    grand_ok = 0
    grand_total = 0
    for seq_dir in seq_dirs:
        if not (seq_dir / "attributes.json").is_file():
            continue
        ok, total = process_sequence(seq_dir, bg_videos, rng, args.max_frames, args.feather)
        grand_ok += ok
        grand_total += total
        print(f"{seq_dir.name}: composited {ok}/{total}")

    print(f"Done: composited {grand_ok}/{grand_total} images using {len(bg_videos)} background videos")


if __name__ == "__main__":
    main()
