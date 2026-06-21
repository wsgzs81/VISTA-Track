#!/usr/bin/env python3
"""Offline QC: validate generated sequences for integrity and quality."""
import json
import pathlib
import sys
from typing import List, Tuple


def read_json(path: pathlib.Path):
    return json.loads(path.read_text(encoding="utf-8"))


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


def check_sequence(seq_dir: pathlib.Path, expected_cameras: int, expected_frames: int,
                   require_occlusion: bool = False) -> Tuple[bool, List[str]]:
    errors = []

    meta_path = seq_dir / "seq_meta.json"
    if not meta_path.exists():
        return False, ["MISSING_SEQ_META"]

    try:
        meta = read_json(meta_path)
    except Exception as e:
        return False, [f"CORRUPT_SEQ_META: {e}"]

    if not meta.get("success", False):
        return False, ["UE_REPORTED_FAILURE"]

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
        return False, errors

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
    ratio_sum = 0.0
    for cam_idx in range(expected_cameras):
        cam_id = "cam_{:03d}".format(cam_idx)
        ann_dir = frames_dir / cam_id / "ann"
        if ann_dir.exists():
            for af in ann_dir.glob("*.json"):
                total_ann += 1
                try:
                    a = read_json(af)
                    if a.get("valid", False):
                        valid_ann += 1
                    visible, occluded, invisible, ratio, _ = get_visibility_stats(a)
                    if visible:
                        visible_ann += 1
                    if occluded:
                        occluded_ann += 1
                    if invisible:
                        invisible_ann += 1
                    ratio_sum += ratio
                except Exception:
                    pass

    if total_ann > 0:
        valid_ratio = valid_ann / total_ann
        if valid_ratio < 0.8:
            errors.append("LOW_VALID_RATIO: {:.2f} ({}/{})".format(valid_ratio, valid_ann, total_ann))
        if visible_ann == 0:
            errors.append("ALL_ZERO_VISIBILITY")
        mean_visibility = ratio_sum / total_ann
        if mean_visibility <= 0.01:
            errors.append("LOW_MEAN_VISIBILITY: {:.4f}".format(mean_visibility))
        if require_occlusion and occluded_ann == 0:
            errors.append("NO_OCCLUSION_ANNOTATIONS")
        if invisible_ann == total_ann:
            errors.append("ALL_ANNOTATIONS_INVISIBLE")
    else:
        errors.append("NO_ANNOTATIONS")

    return len(errors) == 0, errors


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Sequence dir or shards root")
    parser.add_argument("--cameras", type=int, default=4)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--require-occlusion", action="store_true",
                        help="Fail sequences that do not contain any occluded annotation.")
    args = parser.parse_args()

    root = pathlib.Path(args.input)
    if not root.exists():
        print("ERROR: {} does not exist".format(root))
        sys.exit(1)

    seq_dirs = []
    if (root / "seq_meta.json").exists():
        seq_dirs = [root]
    else:
        for shard in sorted(root.iterdir()):
            if shard.is_dir() and shard.name.startswith("shard_"):
                for seq in sorted(shard.iterdir()):
                    if seq.is_dir() and (seq / "seq_meta.json").exists():
                        seq_dirs.append(seq)

    print("Found {} sequences to validate".format(len(seq_dirs)))

    passed = 0
    failed = 0
    for sd in seq_dirs:
        ok, errors = check_sequence(sd, args.cameras, args.frames,
                                    require_occlusion=args.require_occlusion)
        if ok:
            passed += 1
        else:
            failed += 1
            print("  FAIL {}: {}".format(sd.name, "; ".join(errors)))

    print("\nResult: {} passed, {} failed out of {}".format(passed, failed, len(seq_dirs)))
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
