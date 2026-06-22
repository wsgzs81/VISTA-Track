#!/usr/bin/env python3
"""Offline QC: validate generated sequences for integrity and benchmark quality."""
import json
import math
import pathlib
import sys
from typing import Dict, List, Tuple


def read_json(path: pathlib.Path):
    return json.loads(path.read_text(encoding="utf-8"))


def find_sequences(root: pathlib.Path) -> List[pathlib.Path]:
    if (root / "seq_meta.json").exists():
        return [root]
    seq_dirs = []
    for meta_path in root.rglob("seq_meta.json"):
        seq_dir = meta_path.parent
        if (seq_dir / "frames").exists():
            seq_dirs.append(seq_dir)
    return sorted(set(seq_dirs))


def get_visibility_stats(ann):
    vis = ann.get("visibility", {}) or {}
    try:
        ratio = float(vis.get("visibility_ratio", vis.get("visible_ratio", 0.0)) or 0.0)
    except Exception:
        ratio = 0.0
    try:
        pixels = int(vis.get("visible_pixels", 0) or 0)
    except Exception:
        pixels = 0
    state = str(vis.get("occlusion_state", "")).lower()
    visible = pixels > 0 or ratio > 0.0
    occluded = "occlud" in state or (0.0 < ratio < 0.98)
    invisible = "invisible" in state or (pixels <= 0 and ratio <= 0.0)
    return visible, occluded, invisible, ratio, pixels


def get_bbox(ann):
    bbox = ann.get("bbox_2d_amodal_xywh") or ann.get("bbox_2d_visible_xywh") or [0, 0, 0, 0]
    if not isinstance(bbox, list) or len(bbox) != 4:
        return 0.0, 0.0, 0.0, 0.0
    try:
        return tuple(float(v) for v in bbox)
    except Exception:
        return 0.0, 0.0, 0.0, 0.0


def bbox_is_bad(bbox, width: int, height: int) -> bool:
    x, y, w, h = bbox
    if not all(math.isfinite(v) for v in bbox):
        return True
    if w <= 1.0 or h <= 1.0:
        return True
    if w > width * 0.95 or h > height * 0.95:
        return True
    if x + w <= 0.0 or y + h <= 0.0 or x >= width or y >= height:
        return True
    return False


def check_sequence(seq_dir: pathlib.Path, expected_cameras: int, expected_frames: int,
                   require_occlusion: bool = False,
                   width: int = 1280,
                   height: int = 720,
                   min_first_visible_views: int = 2,
                   min_mean_motion_px: float = 4.0,
                   max_bbox_bad_ratio: float = 0.05,
                   min_mean_visibility: float = 0.15,
                   max_invisible_fraction: float = 0.35) -> Tuple[bool, List[str], Dict]:
    errors = []
    stats = {
        "sequence": seq_dir.name,
        "total_annotations": 0,
        "valid_annotations": 0,
        "visible_annotations": 0,
        "occluded_annotations": 0,
        "invisible_annotations": 0,
        "bbox_bad_annotations": 0,
        "first_frame_visible_views": 0,
        "mean_visibility": 0.0,
        "mean_motion_px": 0.0,
    }

    meta_path = seq_dir / "seq_meta.json"
    if not meta_path.exists():
        return False, ["MISSING_SEQ_META"], stats

    try:
        meta = read_json(meta_path)
    except Exception as e:
        return False, [f"CORRUPT_SEQ_META: {e}"], stats

    if not meta.get("success", False):
        return False, ["UE_REPORTED_FAILURE"], stats

    expected_cameras = expected_cameras or int(meta.get("num_cameras", 0) or 0)
    expected_frames = expected_frames or int(meta.get("num_frames", 0) or 0)
    stats["expected_cameras"] = expected_cameras
    stats["expected_frames"] = expected_frames

    rendered_rgb = int(meta.get("rendered_rgb_frames", 0) or 0)
    if rendered_rgb < expected_frames:
        errors.append("FEW_RENDERED_RGB_FRAMES: {}/{}".format(rendered_rgb, expected_frames))

    cams_dir = seq_dir / "cameras"
    if not cams_dir.exists():
        errors.append("MISSING_CAMERAS_DIR")

    cam_files = sorted(cams_dir.glob("cam_*.json")) if cams_dir.exists() else []
    if len(cam_files) < expected_cameras:
        errors.append(f"CAMERA_CALIB_MISSING ({len(cam_files)}/{expected_cameras})")

    for cf in cam_files:
        try:
            cal = read_json(cf)
            assert "camera_id" in cal
            assert "K_matrix_row_major" in cal
            assert len(cal["K_matrix_row_major"]) == 9
        except Exception as e:
            errors.append(f"CORRUPT_CALIB {cf.name}: {e}")

    frames_dir = seq_dir / "frames"
    if not frames_dir.exists():
        errors.append("MISSING_FRAMES_DIR")
        return False, errors, stats

    for cam_idx in range(expected_cameras):
        cam_id = "cam_{:03d}".format(cam_idx)
        cam_dir = frames_dir / cam_id

        if not cam_dir.exists():
            errors.append("MISSING_CAM_DIR " + cam_id)
            continue

        rgb_dir = cam_dir / "rgb"
        rgb_files = sorted(rgb_dir.glob("*.png")) if rgb_dir.exists() else []
        if len(rgb_files) < expected_frames:
            errors.append("FEW_RGB_FILES {}: {}/{}".format(cam_id, len(rgb_files), expected_frames))

        ann_dir = cam_dir / "ann"
        if ann_dir.exists():
            ann_files = sorted(ann_dir.glob("*.json"))
            if len(ann_files) < expected_frames:
                errors.append("FEW_ANNOTATIONS {}: {}/{}".format(cam_id, len(ann_files), expected_frames))

            for sample in ([ann_files[0], ann_files[-1]] if ann_files else []):
                try:
                    ann = read_json(sample)
                    assert "frame_index" in ann
                    assert "bbox_2d_amodal_xywh" in ann
                    assert "visibility" in ann
                except Exception as e:
                    errors.append("CORRUPT_ANN {}: {}".format(sample.name, e))

    total_ann = 0
    valid_ann = 0
    visible_ann = 0
    occluded_ann = 0
    invisible_ann = 0
    bbox_bad_ann = 0
    first_frame_visible_views = 0
    ratio_sum = 0.0
    motion_values = []
    for cam_idx in range(expected_cameras):
        cam_id = "cam_{:03d}".format(cam_idx)
        ann_dir = frames_dir / cam_id / "ann"
        centers = []
        if ann_dir.exists():
            for af in sorted(ann_dir.glob("*.json")):
                total_ann += 1
                try:
                    a = read_json(af)
                    bbox = get_bbox(a)
                    bad_bbox = bbox_is_bad(bbox, width, height)
                    if bad_bbox:
                        bbox_bad_ann += 1
                    if a.get("valid", False):
                        valid_ann += 1
                    visible, occluded, invisible, ratio, _ = get_visibility_stats(a)
                    if visible and not bad_bbox:
                        visible_ann += 1
                        centers.append((bbox[0] + bbox[2] * 0.5, bbox[1] + bbox[3] * 0.5))
                    if occluded:
                        occluded_ann += 1
                    if invisible:
                        invisible_ann += 1
                    if int(a.get("frame_index", -1)) == 0 and visible and not bad_bbox:
                        first_frame_visible_views += 1
                    ratio_sum += ratio
                except Exception:
                    pass
        for c0, c1 in zip(centers[:-1], centers[1:]):
            motion_values.append(math.hypot(c1[0] - c0[0], c1[1] - c0[1]))

    if total_ann > 0:
        valid_ratio = valid_ann / total_ann
        if valid_ratio < 0.8:
            errors.append("LOW_VALID_RATIO: {:.2f} ({}/{})".format(valid_ratio, valid_ann, total_ann))
        if visible_ann == 0:
            errors.append("ALL_ZERO_VISIBILITY")
        mean_visibility = ratio_sum / total_ann
        if mean_visibility < min_mean_visibility:
            errors.append("LOW_MEAN_VISIBILITY: {:.4f}".format(mean_visibility))
        if require_occlusion and occluded_ann == 0:
            errors.append("NO_OCCLUSION_ANNOTATIONS")
        if invisible_ann == total_ann:
            errors.append("ALL_ANNOTATIONS_INVISIBLE")
        invisible_fraction = invisible_ann / total_ann
        if invisible_fraction > max_invisible_fraction:
            errors.append("HIGH_INVISIBLE_FRACTION: {:.2f}".format(invisible_fraction))
        bbox_bad_ratio = bbox_bad_ann / total_ann
        if bbox_bad_ratio > max_bbox_bad_ratio:
            errors.append("HIGH_BBOX_BAD_RATIO: {:.2f}".format(bbox_bad_ratio))
        if first_frame_visible_views < min_first_visible_views:
            errors.append("LOW_FIRST_FRAME_VISIBLE_VIEWS: {}".format(first_frame_visible_views))
        mean_motion = sum(motion_values) / len(motion_values) if motion_values else 0.0
        if mean_motion < min_mean_motion_px:
            errors.append("LOW_MEAN_MOTION_PX: {:.2f}".format(mean_motion))
        stats.update({
            "total_annotations": total_ann,
            "valid_annotations": valid_ann,
            "visible_annotations": visible_ann,
            "occluded_annotations": occluded_ann,
            "invisible_annotations": invisible_ann,
            "bbox_bad_annotations": bbox_bad_ann,
            "first_frame_visible_views": first_frame_visible_views,
            "mean_visibility": mean_visibility,
            "mean_motion_px": mean_motion,
            "bbox_bad_ratio": bbox_bad_ratio,
            "invisible_fraction": invisible_fraction,
        })
    else:
        errors.append("NO_ANNOTATIONS")

    return len(errors) == 0, errors, stats


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Sequence dir or shards root")
    parser.add_argument("--cameras", type=int, default=0,
                        help="Expected camera count. Use 0 to infer from seq_meta.json.")
    parser.add_argument("--frames", type=int, default=0,
                        help="Expected frame count. Use 0 to infer from seq_meta.json.")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--require-occlusion", action="store_true",
                        help="Fail sequences that do not contain any occluded annotation.")
    parser.add_argument("--min-first-visible-views", type=int, default=2)
    parser.add_argument("--min-mean-motion-px", type=float, default=4.0)
    parser.add_argument("--max-bbox-bad-ratio", type=float, default=0.05)
    parser.add_argument("--min-mean-visibility", type=float, default=0.15)
    parser.add_argument("--max-invisible-fraction", type=float, default=0.35)
    parser.add_argument("--report-json", default=None)
    args = parser.parse_args()

    root = pathlib.Path(args.input)
    if not root.exists():
        print("ERROR: {} does not exist".format(root))
        sys.exit(1)

    seq_dirs = find_sequences(root)

    print("Found {} sequences to validate".format(len(seq_dirs)))

    passed = 0
    failed = 0
    report = []
    for sd in seq_dirs:
        ok, errors, stats = check_sequence(
            sd, args.cameras, args.frames,
            require_occlusion=args.require_occlusion,
            width=args.width,
            height=args.height,
            min_first_visible_views=args.min_first_visible_views,
            min_mean_motion_px=args.min_mean_motion_px,
            max_bbox_bad_ratio=args.max_bbox_bad_ratio,
            min_mean_visibility=args.min_mean_visibility,
            max_invisible_fraction=args.max_invisible_fraction)
        stats["ok"] = ok
        stats["errors"] = errors
        report.append(stats)
        if ok:
            passed += 1
        else:
            failed += 1
            print("  FAIL {}: {}".format(sd.name, "; ".join(errors)))

    print("\nResult: {} passed, {} failed out of {}".format(passed, failed, len(seq_dirs)))
    if report:
        mean_motion = sum(r.get("mean_motion_px", 0.0) for r in report) / len(report)
        mean_visibility = sum(r.get("mean_visibility", 0.0) for r in report) / len(report)
        print("Mean motion px/frame: {:.2f}".format(mean_motion))
        print("Mean visibility: {:.3f}".format(mean_visibility))
    if args.report_json:
        pathlib.Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.report_json).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
