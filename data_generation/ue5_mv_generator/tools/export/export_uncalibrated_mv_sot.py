#!/usr/bin/env python3
"""Export UE5 MV-SOT renders into the public uncalibrated VISTA-MV-SOT format.

The UE renderer may keep camera intrinsics/extrinsics internally for annotation
and quality control. This exporter intentionally strips all calibration fields
from the released benchmark protocol.
"""

import argparse
import json
import math
import pathlib
import random
import shutil
from typing import Dict, List, Optional, Tuple


CALIBRATION_KEYS = {
    "camera_to_world",
    "world_to_camera",
    "intrinsic",
    "intrinsics",
    "extrinsic",
    "extrinsics",
    "K",
    "K_matrix_row_major",
    "focal_length_mm",
    "sensor_width_mm",
    "sensor_height_mm",
    "principal_point",
    "pose_world",
    "bbox_3d_world",
}


def read_json(path: pathlib.Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def find_sequences(root: pathlib.Path) -> List[pathlib.Path]:
    if (root / "seq_meta.json").exists():
        return [root]
    candidates = []
    for seq_meta in root.rglob("seq_meta.json"):
        seq_dir = seq_meta.parent
        if (seq_dir / "frames").exists():
            candidates.append(seq_dir)
    return sorted(set(candidates))


def split_name(idx: int, total: int, train: float, val: float) -> str:
    if total <= 0:
        return "train"
    r = idx / float(total)
    if r < train:
        return "train"
    if r < train + val:
        return "val"
    return "test"


def load_ann(seq_dir: pathlib.Path, cam_id: str, frame_idx: int) -> Optional[Dict]:
    path = seq_dir / "frames" / cam_id / "ann" / f"{frame_idx:06d}.json"
    if not path.exists():
        return None
    try:
        return read_json(path)
    except Exception:
        return None


def bbox_from_ann(ann: Optional[Dict]) -> Tuple[float, float, float, float]:
    if not ann:
        return (0.0, 0.0, 0.0, 0.0)
    if not ann.get("valid", False):
        return (0.0, 0.0, 0.0, 0.0)
    bbox = ann.get("bbox_2d_amodal_xywh") or ann.get("bbox_2d_visible_xywh") or [0, 0, 0, 0]
    if len(bbox) != 4:
        return (0.0, 0.0, 0.0, 0.0)
    try:
        x, y, w, h = [float(v) for v in bbox]
    except Exception:
        return (0.0, 0.0, 0.0, 0.0)
    if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0.0 or h <= 0.0:
        return (0.0, 0.0, 0.0, 0.0)
    return (x, y, w, h)


def visible_from_ann(ann: Optional[Dict], bbox: Tuple[float, float, float, float]) -> int:
    if not ann or not ann.get("valid", False):
        return 0
    vis = ann.get("visibility", {}) or {}
    ratio = float(vis.get("visibility_ratio", 0.0) or 0.0)
    pixels = int(vis.get("visible_pixels", 0) or 0)
    return int((pixels > 0 or ratio > 0.0) and bbox[2] > 0.0 and bbox[3] > 0.0)


def find_start_frame(seq_dir: pathlib.Path, frame_count: int, view_count: int, min_visible_views: int) -> int:
    for frame_idx in range(frame_count):
        visible_views = 0
        for view_idx in range(view_count):
            cam_id = f"cam_{view_idx:03d}"
            ann = load_ann(seq_dir, cam_id, frame_idx)
            bbox = bbox_from_ann(ann)
            visible_views += visible_from_ann(ann, bbox)
        if visible_views >= min_visible_views:
            return frame_idx
    return 0


def copy_or_link(src: pathlib.Path, dst: pathlib.Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if mode == "symlink":
        dst.symlink_to(src)
    elif mode == "hardlink":
        try:
            dst.hardlink_to(src)
        except OSError:
            shutil.copy2(src, dst)
    else:
        shutil.copy2(src, dst)


def derive_challenges(view_stats: List[Dict]) -> List[str]:
    labels = set()
    for stats in view_stats:
        if stats["occluded_frames"] > 0:
            labels.add("partial_occlusion")
        if stats["heavy_occlusion_frames"] > 0:
            labels.add("heavy_occlusion")
        if stats["invisible_frames"] > 0:
            labels.add("out_of_view_or_full_occlusion")
        if stats["mean_motion_px"] >= 18.0:
            labels.add("fast_motion")
        if stats["area_ratio_max"] > 0 and stats["area_ratio_min"] > 0:
            if stats["area_ratio_max"] / max(stats["area_ratio_min"], 1e-6) >= 1.8:
                labels.add("scale_variation")
    if len(view_stats) >= 4:
        labels.add("wide_baseline_multiview")
    return sorted(labels)


def assert_no_calibration(obj, path: str = "root") -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            lower = str(key).lower()
            if key in CALIBRATION_KEYS or "intrinsic" in lower or "extrinsic" in lower or lower == "k_matrix_row_major":
                raise ValueError(f"calibration key leaked into public metadata: {path}.{key}")
            assert_no_calibration(value, f"{path}.{key}")
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            assert_no_calibration(value, f"{path}[{i}]")


def export_sequence(
    seq_dir: pathlib.Path,
    out_split_dir: pathlib.Path,
    public_seq_id: str,
    split: str,
    link_mode: str,
    include_masks: bool,
    min_initial_visible_views: int,
) -> Dict:
    meta = read_json(seq_dir / "seq_meta.json")
    frame_count = int(meta.get("num_frames", 0) or 0)
    view_count = int(meta.get("num_cameras", 0) or 0)
    fps = int(meta.get("fps", 30) or 30)
    category = str(meta.get("target_category", "unknown"))

    public_seq_dir = out_split_dir / public_seq_id
    public_seq_dir.mkdir(parents=True, exist_ok=True)
    start_frame = find_start_frame(seq_dir, frame_count, view_count, min_initial_visible_views)
    exported_frame_count = max(0, frame_count - start_frame)

    view_stats = []
    views = []
    for view_idx in range(view_count):
        cam_id = f"cam_{view_idx:03d}"
        view_id = f"view_{view_idx:03d}"
        src_rgb = seq_dir / "frames" / cam_id / "rgb"
        src_mask = seq_dir / "frames" / cam_id / "mask"
        dst_view = public_seq_dir / view_id
        dst_img = dst_view / "img"
        dst_mask = dst_view / "mask"
        dst_img.mkdir(parents=True, exist_ok=True)

        gt_lines = []
        visible_lines = []
        occlusion_lines = []
        centers = []
        areas = []
        occluded_frames = 0
        heavy_occlusion_frames = 0
        invisible_frames = 0

        for export_idx, frame_idx in enumerate(range(start_frame, frame_count)):
            src_img = src_rgb / f"{frame_idx:06d}.png"
            if src_img.exists():
                copy_or_link(src_img, dst_img / f"{export_idx:06d}.png", link_mode)
            if include_masks:
                src_m = src_mask / f"{frame_idx:06d}.png"
                if src_m.exists():
                    copy_or_link(src_m, dst_mask / f"{export_idx:06d}.png", link_mode)

            ann = load_ann(seq_dir, cam_id, frame_idx)
            bbox = bbox_from_ann(ann)
            visible = visible_from_ann(ann, bbox)
            gt_lines.append("{:.3f},{:.3f},{:.3f},{:.3f}".format(*bbox))
            visible_lines.append(str(visible))

            vis = (ann or {}).get("visibility", {}) or {}
            state = str(vis.get("occlusion_state", "unknown")).lower()
            occlusion_lines.append(state)
            if "partial" in state or "heavy" in state or "tiny" in state:
                occluded_frames += 1
            if "heavy" in state or "tiny" in state:
                heavy_occlusion_frames += 1
            if visible == 0:
                invisible_frames += 1
            if visible:
                x, y, w, h = bbox
                centers.append((x + w / 2.0, y + h / 2.0))
                areas.append(max(0.0, w * h))

        motion = []
        for a, b in zip(centers[:-1], centers[1:]):
            motion.append(math.hypot(a[0] - b[0], a[1] - b[1]))
        area_ratio = [a / (1280.0 * 720.0) for a in areas]

        (dst_view / "groundtruth.txt").write_text("\n".join(gt_lines) + "\n", encoding="utf-8")
        (dst_view / "visible.txt").write_text("\n".join(visible_lines) + "\n", encoding="utf-8")
        (dst_view / "occlusion.txt").write_text("\n".join(occlusion_lines) + "\n", encoding="utf-8")

        stats = {
            "view_id": view_id,
            "visible_frames": sum(int(v) for v in visible_lines),
            "invisible_frames": invisible_frames,
            "occluded_frames": occluded_frames,
            "heavy_occlusion_frames": heavy_occlusion_frames,
            "mean_motion_px": sum(motion) / len(motion) if motion else 0.0,
            "area_ratio_min": min(area_ratio) if area_ratio else 0.0,
            "area_ratio_max": max(area_ratio) if area_ratio else 0.0,
        }
        view_stats.append(stats)
        views.append({
            "view_id": view_id,
            "image_dir": f"{view_id}/img",
            "groundtruth": f"{view_id}/groundtruth.txt",
            "visible": f"{view_id}/visible.txt",
            "occlusion": f"{view_id}/occlusion.txt",
            "mask_dir": f"{view_id}/mask" if include_masks else None,
        })

    challenges = derive_challenges(view_stats)
    public_meta = {
        "dataset": "VISTA-MV-SOT",
        "protocol_version": "0.1-uncalibrated",
        "split": split,
        "sequence_id": public_seq_id,
        "source_sequence_id": str(meta.get("sequence_id", seq_dir.name)),
        "category": category,
        "fps": fps,
        "frame_count": exported_frame_count,
        "view_count": view_count,
        "source_start_frame": start_frame,
        "annotation": "2d_bbox_xywh_pixels",
        "calibration_available_to_methods": False,
        "challenge_labels": challenges,
        "views": views,
        "quality_summary": {
            "mean_visible_frames_per_view": sum(s["visible_frames"] for s in view_stats) / max(1, len(view_stats)),
            "mean_motion_px": sum(s["mean_motion_px"] for s in view_stats) / max(1, len(view_stats)),
            "views_with_occlusion": sum(1 for s in view_stats if s["occluded_frames"] > 0),
        },
    }
    assert_no_calibration(public_meta)
    (public_seq_dir / "meta.json").write_text(json.dumps(public_meta, indent=2) + "\n", encoding="utf-8")
    return {
        "sequence_id": public_seq_id,
        "split": split,
        "category": category,
        "frame_count": frame_count,
        "exported_frame_count": exported_frame_count,
        "source_start_frame": start_frame,
        "view_count": view_count,
        "challenge_labels": challenges,
        "path": str(public_seq_dir.relative_to(out_split_dir.parent)),
    }


def write_dataset_card(out_root: pathlib.Path, index: List[Dict]) -> None:
    lines = [
        "# VISTA-MV-SOT",
        "",
        "Uncalibrated multi-view single-object tracking benchmark export.",
        "",
        "Public protocol:",
        "- Methods may use synchronized RGB frames, 2D bounding boxes, visibility flags, and optional masks.",
        "- Methods must not use camera intrinsics, extrinsics, or any private renderer pose.",
        "- Per-view files follow GOT-style `groundtruth.txt` with `x,y,w,h` in pixels.",
        "",
        "Splits:",
    ]
    for split in ("train", "val", "test"):
        n = sum(1 for item in index if item["split"] == split)
        lines.append(f"- {split}: {n} sequences")
    lines.append("")
    lines.append("This export intentionally strips calibration metadata to support the uncalibrated MV-SOT setting.")
    (out_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="UE5 output root, shards root, or one sequence dir")
    parser.add_argument("--output", required=True, help="Destination VISTA-MV-SOT root")
    parser.add_argument("--dataset-prefix", default="vista_iclr")
    parser.add_argument("--train", type=float, default=0.8)
    parser.add_argument("--val", type=float, default=0.1)
    parser.add_argument("--test", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--link-mode", choices=["copy", "hardlink", "symlink"], default="hardlink")
    parser.add_argument("--include-masks", action="store_true")
    parser.add_argument("--min-initial-visible-views", type=int, default=2,
                        help="Trim leading frames until at least this many views have a valid target box.")
    args = parser.parse_args()

    in_root = pathlib.Path(args.input).expanduser().resolve()
    out_root = pathlib.Path(args.output).expanduser().resolve()
    seq_dirs = find_sequences(in_root)
    if not seq_dirs:
        raise SystemExit(f"No UE5 sequences found under {in_root}")
    if abs(args.train + args.val + args.test - 1.0) > 1e-6:
        raise SystemExit("--train + --val + --test must equal 1.0")

    rng = random.Random(args.seed)
    seq_dirs = sorted(seq_dirs)
    rng.shuffle(seq_dirs)
    out_root.mkdir(parents=True, exist_ok=True)

    index = []
    for idx, seq_dir in enumerate(seq_dirs):
        split = split_name(idx, len(seq_dirs), args.train, args.val)
        public_seq_id = f"{args.dataset_prefix}_{idx:06d}"
        print(f"[export] {seq_dir} -> {split}/{public_seq_id}")
        item = export_sequence(
            seq_dir=seq_dir,
            out_split_dir=out_root / split,
            public_seq_id=public_seq_id,
            split=split,
            link_mode=args.link_mode,
            include_masks=args.include_masks,
            min_initial_visible_views=args.min_initial_visible_views,
        )
        index.append(item)

    (out_root / "sequence_index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    write_dataset_card(out_root, index)
    print(f"[export] wrote {len(index)} sequences to {out_root}")
    print("[export] calibration_available_to_methods=false")


if __name__ == "__main__":
    main()
