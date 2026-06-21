"""TrackingNet frame loader that supports extracted frames and zip shards."""

from collections import OrderedDict
from pathlib import Path
from typing import Callable
import zipfile

import cv2
import numpy as np


class TrackingNetZipReader:
    """Read TrackingNet frames from extracted folders or per-video zip files."""

    def __init__(self, root, image_loader: Callable[[str], np.ndarray] | None = None, max_open_zips: int = 16):
        self.root = Path(root)
        self.image_loader = image_loader
        self.max_open_zips = max_open_zips
        self._zip_handles: OrderedDict[Path, zipfile.ZipFile] = OrderedDict()

    def close(self) -> None:
        for zf in self._zip_handles.values():
            zf.close()
        self._zip_handles.clear()

    def read_frame(self, set_id: int, video_name: str, frame_id: int):
        frame_path = self.root / f"TRAIN_{set_id}" / "frames" / video_name / f"{frame_id}.jpg"
        if frame_path.is_file():
            if self.image_loader is not None:
                return self.image_loader(str(frame_path))
            image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if image is None:
                raise IOError(f"Could not decode TrackingNet frame: {frame_path}")
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        zip_path = self.root / f"TRAIN_{set_id}" / "zips" / f"{video_name}.zip"
        if not zip_path.is_file():
            raise FileNotFoundError(f"TrackingNet frame not found as file or zip member: {frame_path}")

        zf = self._get_zip(zip_path)
        member_name = f"{frame_id}.jpg"
        try:
            raw = zf.read(member_name)
        except KeyError as exc:
            raise FileNotFoundError(f"TrackingNet zip member missing: {zip_path}:{member_name}") from exc

        arr = np.frombuffer(raw, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            raise IOError(f"Could not decode TrackingNet zip frame: {zip_path}:{member_name}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _get_zip(self, zip_path: Path) -> zipfile.ZipFile:
        zf = self._zip_handles.get(zip_path)
        if zf is not None:
            self._zip_handles.move_to_end(zip_path)
            return zf

        zf = zipfile.ZipFile(zip_path, "r")
        self._zip_handles[zip_path] = zf
        while len(self._zip_handles) > self.max_open_zips:
            _, old_zf = self._zip_handles.popitem(last=False)
            old_zf.close()
        return zf
