"""Lightweight tracking metrics used for internal ablations."""

from __future__ import annotations

import numpy as np


def box_iou_xywh(a, b) -> float:
    ax, ay, aw, ah = map(float, a)
    bx, by, bw, bh = map(float, b)
    ax2, ay2 = ax + max(0.0, aw), ay + max(0.0, ah)
    bx2, by2 = bx + max(0.0, bw), by + max(0.0, bh)
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = max(0.0, aw) * max(0.0, ah) + max(0.0, bw) * max(0.0, bh) - inter
    return inter / union if union > 0 else 0.0


def average_overlap(pred_boxes, gt_boxes) -> float:
    overlaps = [box_iou_xywh(p, g) for p, g in zip(pred_boxes, gt_boxes)]
    return float(np.mean(overlaps)) if overlaps else 0.0


def success_auc(pred_boxes, gt_boxes, thresholds=None) -> float:
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)
    overlaps = np.asarray([box_iou_xywh(p, g) for p, g in zip(pred_boxes, gt_boxes)], dtype=np.float32)
    if overlaps.size == 0:
        return 0.0
    success = np.asarray([(overlaps >= t).mean() for t in thresholds], dtype=np.float32)
    return float(np.trapz(success, thresholds) / (thresholds[-1] - thresholds[0]))
