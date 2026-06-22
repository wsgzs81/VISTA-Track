#!/usr/bin/env python3
"""Create contact sheets for exported VISTA-MV-SOT sequences."""

import argparse
import json
import pathlib
from typing import List

import cv2
import numpy as np


def read_gt(path: pathlib.Path) -> List[List[float]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append([float(v) for v in line.split(",")[:4]])
    return rows


def draw_cell(view_dir: pathlib.Path, frame_idx: int, label: str):
    img_path = view_dir / "img" / f"{frame_idx:06d}.png"
    img = cv2.imread(str(img_path))
    if img is None:
        img = np.zeros((360, 640, 3), dtype=np.uint8)

    gt = read_gt(view_dir / "groundtruth.txt")
    visible = (view_dir / "visible.txt").read_text(encoding="utf-8").splitlines()
    occ = (view_dir / "occlusion.txt").read_text(encoding="utf-8").splitlines()
    if frame_idx < len(gt):
        x, y, w, h = [int(round(v)) for v in gt[frame_idx]]
        if w > 0 and h > 0:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
    vis = visible[frame_idx] if frame_idx < len(visible) else "?"
    occ_state = occ[frame_idx] if frame_idx < len(occ) else "?"
    text = f"{label} f{frame_idx:03d} vis={vis} {occ_state[:18]}"
    cv2.rectangle(img, (0, 0), (min(img.shape[1], 900), 34), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return img


def make_sheet(seq_dir: pathlib.Path, out_path: pathlib.Path, max_views: int = 6) -> None:
    meta = json.loads((seq_dir / "meta.json").read_text(encoding="utf-8"))
    frame_count = int(meta["frame_count"])
    frame_ids = [0, frame_count // 2, max(0, frame_count - 1)]
    rows = []
    for view_dir in sorted(seq_dir.glob("view_*"))[:max_views]:
        cells = [draw_cell(view_dir, fid, view_dir.name) for fid in frame_ids]
        min_h = min(c.shape[0] for c in cells)
        cells = [cv2.resize(c, (int(c.shape[1] * min_h / c.shape[0]), min_h)) for c in cells]
        rows.append(np.concatenate(cells, axis=1))
    if not rows:
        raise RuntimeError(f"no views found under {seq_dir}")
    width = max(r.shape[1] for r in rows)
    padded = []
    for row in rows:
        if row.shape[1] < width:
            pad = np.zeros((row.shape[0], width - row.shape[1], 3), dtype=np.uint8)
            row = np.concatenate([row, pad], axis=1)
        padded.append(row)
    sheet = np.concatenate(padded, axis=0)
    if sheet.shape[1] > 2400:
        scale = 2400 / sheet.shape[1]
        sheet = cv2.resize(sheet, (2400, int(sheet.shape[0] * scale)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 92])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Exported VISTA-MV-SOT root")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()

    root = pathlib.Path(args.input)
    out_dir = pathlib.Path(args.output_dir)
    seq_dirs = sorted(p for p in root.glob("*/*") if (p / "meta.json").exists())
    for seq_dir in seq_dirs[: args.limit]:
        out_path = out_dir / f"{seq_dir.name}.jpg"
        make_sheet(seq_dir, out_path)
        print(out_path)


if __name__ == "__main__":
    main()
