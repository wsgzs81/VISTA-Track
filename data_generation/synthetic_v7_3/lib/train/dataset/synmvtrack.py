#!/usr/bin/env python3
"""
synmvtrack.py - VISTA-Track dataset adapter for SynMVTrack.

Reads the MVTrack-style directory layout:

    SynMVTrack/
      seq_0000/
        seq_0000-1/
          img/00000001.jpg
          masks/00000001.png
          groundtruth.txt
          invisible.txt
          visibility.txt
          attributes.json
          full_projected_bbox.txt
        seq_0000-2/ ...
        seq_0000-3/ ...
        BEV/
          target_3d.txt
          target_bev.txt
          xyz_index.txt
        calibs.json
        meta.json

Interface:
    __init__(cfg)          — config dict with root, split, max_views, etc.
    __len__()              — number of (sequence, frame) pairs
    __getitem__(index)     — returns dict of tensors

Each sample returns data for one frame across all views.
"""

import json
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Optional: import base class if it exists in the project
# ---------------------------------------------------------------------------

try:
    from lib.train.dataset.base_dataset import BaseDataset  # noqa: F401
    _BASE = BaseDataset
except ImportError:
    _BASE = Dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_image(path, size=None):
    """Load image as CHW float32 tensor [0,1]."""
    try:
        from PIL import Image
        img = Image.open(path).convert("RGB")
        if size is not None:
            img = img.resize(size, Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)  # C,H,W
    except Exception:
        return torch.zeros(3, 480, 640, dtype=torch.float32)


def _load_mask(path, size=None):
    """Load mask as 1xHxW float32 tensor [0,1]."""
    try:
        from PIL import Image
        m = Image.open(path).convert("L")
        if size is not None:
            m = m.resize(size, Image.NEAREST)
        arr = np.array(m, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)  # 1,H,W
    except Exception:
        h, w = (size[1], size[0]) if size else (480, 640)
        return torch.zeros(1, h, w, dtype=torch.float32)


def _load_lines(path):
    if not os.path.isfile(path):
        return []
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


def _parse_bbox(line):
    """Parse 'x,y,w,h' → tensor [4]."""
    parts = [int(v) for v in line.strip().split(",")]
    return torch.tensor(parts, dtype=torch.float32)


def _parse_bbox_xyxy(line):
    """Parse 'x,y,w,h' → tensor [x1,y1,x2,y2]."""
    x, y, w, h = [int(v) for v in line.strip().split(",")]
    return torch.tensor([x, y, x + w, y + h], dtype=torch.float32)


def _load_calibs(calibs_path):
    """Load calibs.json → list of dicts with K, R, T as tensors."""
    with open(calibs_path) as f:
        raw = json.load(f)

    cams = []
    keys = sorted(raw.keys(), key=lambda k: int("".join(filter(str.isdigit, k)) or 0))
    for ck in keys:
        c = raw[ck]
        cams.append({
            "K": torch.tensor(c["K"], dtype=torch.float32),      # 3x3
            "R": torch.tensor(c["R"], dtype=torch.float32),      # 3x3
            "T": torch.tensor(c["T"], dtype=torch.float32).view(3, 1),  # 3x1
            "image_size": c.get("image_size", [640, 480]),
            "unit": c.get("unit", "mm"),
        })
    return cams


def _load_bev(bev_dir):
    """Load BEV data from xyz_index.txt or fallback to target_3d + target_bev.

    Returns:
        xyz_mm: (N, 3) float32 tensor — x,y,z in mm
        bev_grid: (N, 2) int64 tensor — grid_x, grid_y
    """
    xyz_path = os.path.join(bev_dir, "target_3d.txt")
    bev_path = os.path.join(bev_dir, "target_bev.txt")

    xyz_list = []
    bev_list = []

    # Try xyz_index.txt first (combined format)
    idx_path = os.path.join(bev_dir, "xyz_index.txt")
    if os.path.isfile(idx_path):
        with open(idx_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(",")
                if len(parts) >= 5:
                    xyz_list.append([float(parts[0]), float(parts[1]), float(parts[2])])
                    bev_list.append([int(parts[3]), int(parts[4])])
                elif len(parts) >= 3:
                    xyz_list.append([float(parts[0]), float(parts[1]), float(parts[2])])

    # Fallback: separate files
    if not xyz_list and os.path.isfile(xyz_path):
        with open(xyz_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    xyz_list.append([float(v) for v in line.split(",")])

    if not bev_list and os.path.isfile(bev_path):
        with open(bev_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    bev_list.append([int(v) for v in line.split(",")])

    xyz = torch.tensor(xyz_list, dtype=torch.float32) if xyz_list else torch.zeros(0, 3)
    bev = torch.tensor(bev_list, dtype=torch.long) if bev_list else torch.zeros(0, 2, dtype=torch.long)

    return xyz, bev


# ---------------------------------------------------------------------------
# Sequence index
# ---------------------------------------------------------------------------

def _build_sequence_index(root, split_file=None):
    """Build list of (seq_dir, seq_base, n_frames, n_views) tuples.

    If split_file is given, only include listed sequences.
    """
    if split_file and os.path.isfile(split_file):
        with open(split_file) as f:
            seq_ids = [l.strip() for l in f if l.strip()]
    else:
        seq_ids = sorted([
            d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d)) and d.startswith("seq_")
        ])

    sequences = []
    for sid in seq_ids:
        seq_dir = os.path.join(root, sid)
        if not os.path.isdir(seq_dir):
            continue

        attrs_path = os.path.join(seq_dir, "attributes.json")
        if not os.path.isfile(attrs_path):
            continue

        with open(attrs_path) as f:
            attrs = json.load(f)

        n_frames = attrs["num_frames"]
        n_views = attrs.get("num_views", 0)
        if n_views == 0:
            subdirs = [
                d for d in os.listdir(seq_dir)
                if os.path.isdir(os.path.join(seq_dir, d)) and d.startswith(sid + "-")
            ]
            n_views = len(subdirs)

        if n_views > 0 and n_frames > 0:
            sequences.append((seq_dir, sid, n_frames, n_views))

    return sequences


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SynMVTrackDataset(_BASE):
    """SynMVTrack dataset adapter for VISTA-Track training.

    Each sample = one frame from one sequence, with data from all views.

    Config keys:
        root (str):         path to SynMVTrack output directory
        split (str):        'train' | 'val' | 'test'
        max_views (int):    pad/truncate to this many views (default: 4)
        image_size (tuple): (W, H) resize images (default: (640, 480))
        frame_sample (str): 'all' | 'random' | 'stride_N'
        num_template (int): number of template frames (default: 1)
        num_search (int):   number of search frames (default: 1)
    """

    def __init__(self, cfg):
        super().__init__()
        self.root = cfg["root"]
        self.split = cfg.get("split", "train")
        self.max_views = cfg.get("max_views", 4)
        self.image_size = tuple(cfg.get("image_size", [640, 480]))
        self.frame_sample = cfg.get("frame_sample", "all")
        self.num_template = cfg.get("num_template", 1)
        self.num_search = cfg.get("num_search", 1)

        # Load split
        split_file = os.path.join(self.root, f"{self.split}_split.txt")
        self.sequences = _build_sequence_index(self.root, split_file)

        # Build flat frame index: [(seq_idx, frame_idx), ...]
        self.frame_index = []
        for si, (_, _, n_frames, _) in enumerate(self.sequences):
            if self.frame_sample == "all":
                for fi in range(n_frames):
                    self.frame_index.append((si, fi))
            elif self.frame_sample.startswith("stride_"):
                stride = int(self.frame_sample.split("_")[1])
                for fi in range(0, n_frames, stride):
                    self.frame_index.append((si, fi))
            else:
                # random — handled in __getitem__ via random frame
                for fi in range(n_frames):
                    self.frame_index.append((si, fi))

        # Preload calibs (small, avoid per-sample I/O)
        self._calib_cache = {}
        for si, (seq_dir, _, _, _) in enumerate(self.sequences):
            calib_path = os.path.join(seq_dir, "calibs.json")
            if os.path.isfile(calib_path):
                self._calib_cache[si] = _load_calibs(calib_path)

        # Preload BEV (small)
        self._bev_cache = {}
        for si, (seq_dir, _, _, _) in enumerate(self.sequences):
            bev_dir = os.path.join(seq_dir, "BEV")
            if os.path.isdir(bev_dir):
                self._bev_cache[si] = _load_bev(bev_dir)

    def __len__(self):
        return len(self.frame_index)

    def _get_image_path(self, seq_dir, seq_base, view_idx, frame_idx):
        """Resolve image path — try MVTrack format, then flat format."""
        # MVTrack format: seq_XXXX-N/img/00000001.jpg (1-indexed)
        p = os.path.join(seq_dir, f"{seq_base}-{view_idx + 1}", "img", f"{frame_idx + 1:08d}.jpg")
        if os.path.isfile(p):
            return p
        # Flat format: img/0004/000000.jpg (0-indexed)
        p = os.path.join(seq_dir, "img", f"{view_idx:04d}", f"{frame_idx:06d}.jpg")
        if os.path.isfile(p):
            return p
        return None

    def _get_mask_path(self, seq_dir, seq_base, view_idx, frame_idx):
        """Resolve mask path."""
        p = os.path.join(seq_dir, f"{seq_base}-{view_idx + 1}", "masks", f"{frame_idx + 1:08d}.png")
        if os.path.isfile(p):
            return p
        p = os.path.join(seq_dir, "masks", f"{view_idx:04d}", f"{frame_idx:06d}.png")
        if os.path.isfile(p):
            return p
        return None

    def _load_view_data(self, seq_dir, seq_base, view_idx, frame_idx, n_frames):
        """Load per-view data for one frame.

        Returns dict with image, mask, bbox, invisible, visibility.
        """
        # Image
        img_path = self._get_image_path(seq_dir, seq_base, view_idx, frame_idx)
        if img_path:
            image = _load_image(img_path, size=self.image_size)
        else:
            image = torch.zeros(3, self.image_size[1], self.image_size[0])

        # Mask
        mask_path = self._get_mask_path(seq_dir, seq_base, view_idx, frame_idx)
        if mask_path:
            mask = _load_mask(mask_path, size=self.image_size)
        else:
            mask = torch.zeros(1, self.image_size[1], self.image_size[0])

        # Groundtruth bbox
        view_dir = os.path.join(seq_dir, f"{seq_base}-{view_idx + 1}")
        gt_lines = _load_lines(os.path.join(view_dir, "groundtruth.txt"))
        if frame_idx < len(gt_lines):
            bbox_xywh = _parse_bbox(gt_lines[frame_idx])
            bbox_xyxy = _parse_bbox_xyxy(gt_lines[frame_idx])
        else:
            bbox_xywh = torch.zeros(4)
            bbox_xyxy = torch.zeros(4)

        # Invisible
        inv_lines = _load_lines(os.path.join(view_dir, "invisible.txt"))
        if frame_idx < len(inv_lines):
            invisible = torch.tensor(int(inv_lines[frame_idx]), dtype=torch.long)
        else:
            invisible = torch.tensor(0, dtype=torch.long)

        # Visibility ratio (from per-view file)
        vis_path = os.path.join(view_dir, "visibility.txt")
        visibility = 1.0
        if os.path.isfile(vis_path):
            vis_lines = _load_lines(vis_path)
            if frame_idx < len(vis_lines):
                try:
                    visibility = float(vis_lines[frame_idx])
                except ValueError:
                    pass

        visibility = torch.tensor(visibility, dtype=torch.float32)

        return {
            "image": image,               # 3,H,W
            "mask": mask,                 # 1,H,W
            "bbox_xywh": bbox_xywh,       # 4
            "bbox_xyxy": bbox_xyxy,       # 4
            "invisible": invisible,       # scalar long
            "visibility": visibility,     # scalar float
        }

    def __getitem__(self, index):
        """Get one (sequence, frame) sample across all views.

        Returns dict:
            images:          (V, 3, H, W)   float32
            masks:           (V, 1, H, W)   float32
            bbox_xywh:       (V, 4)         float32
            bbox_xyxy:       (V, 4)         float32
            invisible:       (V,)           long  (0=visible, 1=invisible)
            visibility:      (V,)           float32
            calib_K:         (V, 3, 3)      float32
            calib_R:         (V, 3, 3)      float32
            calib_T:         (V, 3, 1)      float32
            bev_xyz:         (N, 3)         float32  (full sequence, for reference)
            bev_grid:        (N, 2)         long
            frame_idx:       scalar         long
            seq_idx:         scalar         long
            num_views:       scalar         long  (actual views before padding)
            num_frames:      scalar         long
        """
        seq_idx, frame_idx = self.frame_index[index]
        seq_dir, seq_base, n_frames, n_views = self.sequences[seq_idx]

        actual_views = min(n_views, self.max_views)

        # Collect per-view data
        images = []
        masks = []
        bboxes_xywh = []
        bboxes_xyxy = []
        invisibles = []
        visibilities = []

        for vi in range(actual_views):
            vd = self._load_view_data(seq_dir, seq_base, vi, frame_idx, n_frames)
            images.append(vd["image"])
            masks.append(vd["mask"])
            bboxes_xywh.append(vd["bbox_xywh"])
            bboxes_xyxy.append(vd["bbox_xyxy"])
            invisibles.append(vd["invisible"])
            visibilities.append(vd["visibility"])

        # Pad to max_views if needed
        H, W = self.image_size[1], self.image_size[0]
        while len(images) < self.max_views:
            images.append(torch.zeros(3, H, W))
            masks.append(torch.zeros(1, H, W))
            bboxes_xywh.append(torch.zeros(4))
            bboxes_xyxy.append(torch.zeros(4))
            invisibles.append(torch.tensor(1, dtype=torch.long))  # pad = invisible
            visibilities.append(torch.tensor(0.0))

        # Stack
        images = torch.stack(images)                    # V,3,H,W
        masks = torch.stack(masks)                      # V,1,H,W
        bbox_xywh = torch.stack(bboxes_xywh)            # V,4
        bbox_xyxy = torch.stack(bboxes_xyxy)            # V,4
        invisible = torch.stack(invisibles)             # V,
        visibility = torch.stack(visibilities)          # V,

        # Calibration
        calibs = self._calib_cache.get(seq_idx, [])
        if calibs:
            K = torch.stack([c["K"] for c in calibs[:self.max_views]])
            R = torch.stack([c["R"] for c in calibs[:self.max_views]])
            T = torch.stack([c["T"] for c in calibs[:self.max_views]])
        else:
            K = torch.eye(3).unsqueeze(0).repeat(self.max_views, 1, 1)
            R = torch.eye(3).unsqueeze(0).repeat(self.max_views, 1, 1)
            T = torch.zeros(self.max_views, 3, 1)

        # Pad calib tensors
        if K.shape[0] < self.max_views:
            pad_n = self.max_views - K.shape[0]
            K = torch.cat([K, torch.eye(3).unsqueeze(0).repeat(pad_n, 1, 1)])
            R = torch.cat([R, torch.eye(3).unsqueeze(0).repeat(pad_n, 1, 1)])
            T = torch.cat([T, torch.zeros(pad_n, 3, 1)])

        # BEV
        bev_xyz, bev_grid = self._bev_cache.get(seq_idx, (torch.zeros(0, 3), torch.zeros(0, 2, dtype=torch.long)))

        return {
            "images": images,                # V,3,H,W
            "masks": masks,                  # V,1,H,W
            "bbox_xywh": bbox_xywh,          # V,4
            "bbox_xyxy": bbox_xyxy,          # V,4
            "invisible": invisible,          # V,
            "visibility": visibility,        # V,
            "calib_K": K,                    # V,3,3
            "calib_R": R,                    # V,3,3
            "calib_T": T,                    # V,3,1
            "bev_xyz": bev_xyz,              # N,3
            "bev_grid": bev_grid,            # N,2
            "frame_idx": torch.tensor(frame_idx, dtype=torch.long),
            "seq_idx": torch.tensor(seq_idx, dtype=torch.long),
            "num_views": torch.tensor(actual_views, dtype=torch.long),
            "num_frames": torch.tensor(n_frames, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Collate helper (for DataLoader)
# ---------------------------------------------------------------------------

def synmvtrack_collate(batch):
    """Custom collate that stacks variable-length BEV tensors.

    Standard torch.stack works for everything except bev_xyz / bev_grid
    which have sequence-dependent lengths.
    """
    keys = batch[0].keys()
    collated = {}

    for k in keys:
        tensors = [b[k] for b in batch]

        if k in ("bev_xyz", "bev_grid"):
            # Variable-length: pad to max
            max_len = max(t.shape[0] for t in tensors)
            if max_len == 0:
                if k == "bev_xyz":
                    collated[k] = torch.zeros(len(batch), 0, 3)
                else:
                    collated[k] = torch.zeros(len(batch), 0, 2, dtype=torch.long)
                continue

            if k == "bev_xyz":
                padded = torch.zeros(len(batch), max_len, 3)
                for i, t in enumerate(tensors):
                    if t.shape[0] > 0:
                        padded[i, :t.shape[0]] = t
                collated[k] = padded
            else:
                padded = torch.zeros(len(batch), max_len, 2, dtype=torch.long)
                for i, t in enumerate(tensors):
                    if t.shape[0] > 0:
                        padded[i, :t.shape[0]] = t
                collated[k] = padded
        elif k in ("frame_idx", "seq_idx", "num_views", "num_frames"):
            collated[k] = torch.stack(tensors)
        else:
            collated[k] = torch.stack(tensors)

    return collated


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def smoke_test(root, split="train", batch_size=2, max_views=4):
    """Quick smoke test: sample a batch and verify shapes.

    Usage:
        python synmvtrack.py --root output/SynMVTrack
    """
    from torch.utils.data import DataLoader

    print(f"SynMVTrack smoke test")
    print(f"  root: {root}")
    print(f"  split: {split}")

    cfg = {
        "root": root,
        "split": split,
        "max_views": max_views,
        "image_size": [640, 480],
        "frame_sample": "stride_10",  # subsample for speed
    }

    ds = SynMVTrackDataset(cfg)
    print(f"  sequences: {len(ds.sequences)}")
    print(f"  frames (stride_10): {len(ds)}")

    if len(ds) == 0:
        print("  ERROR: empty dataset")
        return False

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=synmvtrack_collate, num_workers=0)

    batch = next(iter(loader))

    print(f"\n  Batch shapes:")
    for k, v in sorted(batch.items()):
        if isinstance(v, torch.Tensor):
            print(f"    {k:20s}: {str(v.shape):20s} {v.dtype}")

    V = max_views
    H, W = 480, 640
    checks = {
        "images":     (batch["images"].shape == (batch_size, V, 3, H, W)),
        "masks":      (batch["masks"].shape == (batch_size, V, 1, H, W)),
        "bbox_xywh":  (batch["bbox_xywh"].shape == (batch_size, V, 4)),
        "bbox_xyxy":  (batch["bbox_xyxy"].shape == (batch_size, V, 4)),
        "calib_K":    (batch["calib_K"].shape == (batch_size, V, 3, 3)),
        "calib_R":    (batch["calib_R"].shape == (batch_size, V, 3, 3)),
        "calib_T":    (batch["calib_T"].shape == (batch_size, V, 3, 1)),
        "invisible":  (batch["invisible"].shape == (batch_size, V)),
        "visibility": (batch["visibility"].shape == (batch_size, V)),
        "bev_xyz":    (batch["bev_xyz"].shape[0] == batch_size),
        "bev_grid":   (batch["bev_grid"].shape[0] == batch_size),
    }

    print(f"\n  Shape checks:")
    all_ok = True
    for name, ok in checks.items():
        status = "OK" if ok else "FAIL"
        if not ok:
            all_ok = False
            expected = {
                "images": f"({batch_size},{V},3,{H},{W})",
                "masks": f"({batch_size},{V},1,{H},{W})",
                "bbox_xywh": f"({batch_size},{V},4)",
                "bbox_xyxy": f"({batch_size},{V},4)",
                "calib_K": f"({batch_size},{V},3,3)",
                "calib_R": f"({batch_size},{V},3,3)",
                "calib_T": f"({batch_size},{V},3,1)",
                "invisible": f"({batch_size},{V})",
                "visibility": f"({batch_size},{V})",
            }.get(name, "?")
            print(f"    {name:20s}: {status} (expected {expected}, got {batch[name].shape})")
        else:
            print(f"    {name:20s}: {status}")

    # Value checks
    print(f"\n  Value checks:")
    print(f"    invisible dtype: {batch['invisible'].dtype} (expected torch.long)")
    print(f"    calib_K det[0,0]: {torch.det(batch['calib_K'][0,0]).item():.4f} (expect ~1.0 for identity or valid K)")
    print(f"    bbox_xywh nonzero: {(batch['bbox_xywh'] != 0).any().item()}")
    print(f"    visibility range: [{batch['visibility'].min():.3f}, {batch['visibility'].max():.3f}]")

    print(f"\n  {'PASS' if all_ok else 'FAIL'}")
    return all_ok


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SynMVTrack dataset smoke test")
    parser.add_argument("--root", default="output/SynMVTrack")
    parser.add_argument("--split", default="train")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-views", type=int, default=4)
    args = parser.parse_args()

    ok = smoke_test(args.root, split=args.split,
                    batch_size=args.batch_size, max_views=args.max_views)
    exit(0 if ok else 1)
