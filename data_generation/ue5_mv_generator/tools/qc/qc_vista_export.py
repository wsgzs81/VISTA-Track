#!/usr/bin/env python3
"""Validate exported VISTA-MV-SOT uncalibrated benchmark folders."""

import argparse
import json
import math
import pathlib
from typing import Dict, List, Tuple


FORBIDDEN_SUBSTRINGS = [
    "camera_to_world",
    "world_to_camera",
    "k_matrix_row_major",
    "intrinsic",
    "extrinsic",
    "pose_world",
    "bbox_3d_world",
    "focal_length",
    "sensor_width",
    "sensor_height",
]


def read_json(path: pathlib.Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_lines(path: pathlib.Path) -> List[str]:
    return path.read_text(encoding="utf-8").splitlines()


def parse_bbox(line: str) -> Tuple[float, float, float, float]:
    parts = [float(v) for v in line.split(",")[:4]]
    if len(parts) != 4:
        return 0.0, 0.0, 0.0, 0.0
    return tuple(parts)


def bbox_bad(bbox: Tuple[float, float, float, float], width: int, height: int) -> bool:
    x, y, w, h = bbox
    if not all(math.isfinite(v) for v in bbox):
        return True
    if w <= 1.0 or h <= 1.0:
        return True
    if w > width * 0.98 or h > height * 0.98:
        return True
    if x + w <= 0.0 or y + h <= 0.0 or x >= width or y >= height:
        return True
    return False


def find_sequences(root: pathlib.Path) -> List[pathlib.Path]:
    return sorted(p for p in root.glob("*/*") if (p / "meta.json").exists())


def check_sequence(
    seq_dir: pathlib.Path,
    width: int,
    height: int,
    min_initial_visible_views: int,
    max_bad_bbox_ratio: float,
) -> Tuple[bool, Dict]:
    meta = read_json(seq_dir / "meta.json")
    errors = []
    blob = json.dumps(meta).lower()
    leaked = [key for key in FORBIDDEN_SUBSTRINGS if key in blob]
    if leaked:
        errors.append("CALIBRATION_LEAK:" + ",".join(leaked))
    if meta.get("calibration_available_to_methods", True) is not False:
        errors.append("CALIBRATION_FLAG_NOT_FALSE")

    frame_count = int(meta.get("frame_count", 0) or 0)
    view_count = int(meta.get("view_count", 0) or 0)
    if frame_count <= 0:
        errors.append("EMPTY_FRAME_COUNT")
    if view_count < min_initial_visible_views:
        errors.append("LOW_VIEW_COUNT")

    first_visible = 0
    total_boxes = 0
    bad_boxes = 0
    total_visible = 0
    for view_idx in range(view_count):
        view_dir = seq_dir / f"view_{view_idx:03d}"
        if not view_dir.exists():
            errors.append(f"MISSING_VIEW:view_{view_idx:03d}")
            continue
        for name in ("groundtruth.txt", "visible.txt", "occlusion.txt"):
            if not (view_dir / name).exists():
                errors.append(f"MISSING_{name}:view_{view_idx:03d}")
        gt = read_lines(view_dir / "groundtruth.txt")
        vis = read_lines(view_dir / "visible.txt")
        if len(gt) != frame_count:
            errors.append(f"GT_LEN_MISMATCH:view_{view_idx:03d}:{len(gt)}/{frame_count}")
        if len(vis) != frame_count:
            errors.append(f"VIS_LEN_MISMATCH:view_{view_idx:03d}:{len(vis)}/{frame_count}")
        image_count = len(list((view_dir / "img").glob("*.png")))
        if image_count != frame_count:
            errors.append(f"IMAGE_COUNT_MISMATCH:view_{view_idx:03d}:{image_count}/{frame_count}")
        if vis and vis[0].strip() == "1":
            first_visible += 1
        for line, visible_line in zip(gt, vis):
            bbox = parse_bbox(line)
            visible = visible_line.strip() == "1"
            if visible:
                total_boxes += 1
                total_visible += 1
                if bbox_bad(bbox, width, height):
                    bad_boxes += 1

    bad_ratio = bad_boxes / total_boxes if total_boxes else 1.0
    if first_visible < min_initial_visible_views:
        errors.append(f"LOW_INITIAL_VISIBLE_VIEWS:{first_visible}")
    if bad_ratio > max_bad_bbox_ratio:
        errors.append(f"HIGH_BAD_BBOX_RATIO:{bad_ratio:.3f}")
    if total_visible == 0:
        errors.append("NO_VISIBLE_BOXES")

    return not errors, {
        "sequence": seq_dir.name,
        "frame_count": frame_count,
        "view_count": view_count,
        "source_start_frame": meta.get("source_start_frame"),
        "first_visible_views": first_visible,
        "bad_bbox_ratio": bad_ratio,
        "errors": errors,
        "ok": not errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--min-initial-visible-views", type=int, default=2)
    parser.add_argument("--max-bad-bbox-ratio", type=float, default=0.10)
    parser.add_argument("--report-json", default=None)
    args = parser.parse_args()

    root = pathlib.Path(args.input)
    seq_dirs = find_sequences(root)
    report = []
    passed = 0
    for seq_dir in seq_dirs:
        ok, item = check_sequence(
            seq_dir,
            width=args.width,
            height=args.height,
            min_initial_visible_views=args.min_initial_visible_views,
            max_bad_bbox_ratio=args.max_bad_bbox_ratio,
        )
        report.append(item)
        if ok:
            passed += 1
        else:
            print("FAIL {}: {}".format(seq_dir.name, "; ".join(item["errors"])))
    print("Result: {} passed, {} failed out of {}".format(passed, len(seq_dirs) - passed, len(seq_dirs)))
    if report:
        print("Mean bad bbox ratio: {:.4f}".format(
            sum(r["bad_bbox_ratio"] for r in report) / len(report)))
        print("Mean first visible views: {:.2f}".format(
            sum(r["first_visible_views"] for r in report) / len(report)))
    if args.report_json:
        pathlib.Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.report_json).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    raise SystemExit(0 if passed == len(seq_dirs) else 1)


if __name__ == "__main__":
    main()
