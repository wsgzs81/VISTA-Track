# VISTA-MV-SOT Uncalibrated Protocol

VISTA-MV-SOT is designed as an uncalibrated multi-view single-object tracking
benchmark. UE5 can use private camera geometry internally for rendering,
annotation, and quality control, but released methods must not receive or use
camera intrinsics, extrinsics, world poses, or 3D target poses.

## Public Data

Each exported sequence contains synchronized views:

```text
VISTA-MV-SOT/
  train|val|test/
    vista_iclr_000000/
      meta.json
      view_000/
        img/000000.png
        groundtruth.txt
        visible.txt
        occlusion.txt
        mask/000000.png      # optional
      view_001/
        ...
```

`groundtruth.txt` stores one `x,y,w,h` bbox per frame in pixel coordinates.
`visible.txt` stores `1` when the target is visible in that view and `0`
otherwise. `occlusion.txt` stores coarse per-frame labels such as `full`,
`partial`, `heavy_occlusion`, `tiny_visible`, or `invisible`.

## Calibration Rule

Public training and evaluation code must treat every camera as an unknown view.
The export script enforces this rule by stripping calibration-like fields from
`meta.json` and by not copying the renderer `cameras/` directory.

Allowed:

- Synchronized RGB frames.
- Per-view 2D bounding boxes.
- Per-view visibility flags.
- Optional masks for segmentation-assisted training ablations.
- Sequence-level category and challenge labels.

Forbidden for public benchmark methods:

- Camera intrinsics or extrinsics.
- Camera-to-world or world-to-camera transforms.
- 3D target position, 3D boxes, or renderer world coordinates.
- Any private UE5 scene state not present in the exported benchmark folder.

## Quality Gate

Before export, a generated sequence should pass offline QC:

- At least two views see the target in the first frame.
- Mean target center motion is non-trivial.
- Bboxes are mask-derived and mostly inside reasonable image bounds.
- The sequence contains occlusion but is not dominated by invisibility.
- Multi-view cameras remain fixed while the target moves.

This makes the benchmark suitable for evaluating uncalibrated view association,
cross-view target correspondence, and multi-view temporal fusion.
