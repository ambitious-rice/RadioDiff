from __future__ import annotations

import json
import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class DynamicRadioRecord:
    scene_id: str
    episode_id: str
    tx_id: str
    sample_meta_path: Path
    tx_dir: Path
    episode_dir: Path
    scene_dir: Path
    rss_npz_path: Path
    rss_png_dir: Path
    static_rss_path: Path
    building_mask_path: Path
    traffic_grid_path: Path
    frame_indices_path: Path


class DynamicRadioBase(Dataset):
    """Shared reader for the CARLA dynamic RadioMapSeer-style package."""

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        split_file: str = "split.json",
        source: str = "png",
        frame_stride: int = 1,
        cache_size: int = 8,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.split = split
        self.split_file = split_file
        self.source = source
        self.frame_stride = max(1, int(frame_stride))
        self.cache_size = max(0, int(cache_size))
        self._cache: OrderedDict[tuple[str, str], Any] = OrderedDict()

        if source not in {"png", "npz_uint8"}:
            raise ValueError(f"Unsupported dynamic radio source: {source!r}")

        self.dataset_meta = self._load_json_if_exists(self.root / "dataset_meta.json")
        self.scene_meta_by_id = {
            str(scene.get("scene_id")): scene
            for scene in self.dataset_meta.get("scenes", [])
            if scene.get("scene_id") is not None
        }
        self.split_meta = self._load_json(self.root / split_file)
        self.records = self._build_records(self._split_samples(split))
        self.frame_index = self._build_frame_index()

    def _split_samples(self, split: str) -> list[str]:
        samples = self.split_meta.get("samples", {}).get(split)
        if samples is None:
            raise KeyError(f"Split {split!r} not found in {self.root / self.split_file}")
        return [str(path) for path in samples]

    def _build_records(self, sample_paths: list[str]) -> list[DynamicRadioRecord]:
        records: list[DynamicRadioRecord] = []
        for rel_path in sample_paths:
            sample_meta_path = self.root / rel_path
            tx_dir = sample_meta_path.parent
            episode_dir = tx_dir.parent
            scene_dir = episode_dir.parents[1]
            scene_id = scene_dir.name
            episode_id = episode_dir.name
            tx_id = tx_dir.name

            records.append(
                DynamicRadioRecord(
                    scene_id=scene_id,
                    episode_id=episode_id,
                    tx_id=tx_id,
                    sample_meta_path=sample_meta_path,
                    tx_dir=tx_dir,
                    episode_dir=episode_dir,
                    scene_dir=scene_dir,
                    rss_npz_path=tx_dir / "rss_maps.npz",
                    rss_png_dir=tx_dir / "png",
                    static_rss_path=scene_dir / "static" / tx_id / "static_rss.npz",
                    building_mask_path=scene_dir / "building" / "building_mask.npz",
                    traffic_grid_path=episode_dir / "traffic" / "traffic_grid_uint8.npz",
                    frame_indices_path=episode_dir / "frame_indices.npy",
                )
            )
        return records

    def _build_frame_index(self) -> list[tuple[int, int]]:
        frame_count = int(self.split_meta.get("frame_count_per_tx_sample", 0) or 0)
        frame_index: list[tuple[int, int]] = []
        for record_idx, record in enumerate(self.records):
            if frame_count <= 0:
                indices = self.load_frame_indices(record)
                frame_ids = [int(value) for value in indices.tolist()]
            else:
                frame_ids = list(range(frame_count))
            for frame_id in frame_ids[:: self.frame_stride]:
                frame_index.append((record_idx, int(frame_id)))
        return frame_index

    def __len__(self) -> int:
        return len(self.frame_index)

    def _load_json(self, path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def _load_json_if_exists(self, path: Path) -> dict[str, Any]:
        return self._load_json(path) if path.exists() else {}

    def _cache_get(self, kind: str, path: Path) -> Any | None:
        key = (kind, str(path))
        if key not in self._cache:
            return None
        value = self._cache.pop(key)
        self._cache[key] = value
        return value

    def _cache_put(self, kind: str, path: Path, value: Any) -> Any:
        if self.cache_size <= 0:
            return value
        key = (kind, str(path))
        self._cache[key] = value
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return value

    def _load_npz_array(self, path: Path, key: str) -> np.ndarray:
        cached = self._cache_get(f"npz:{key}", path)
        if cached is not None:
            return cached
        with np.load(path) as data:
            array = np.asarray(data[key])
        return self._cache_put(f"npz:{key}", path, array)

    def load_frame_indices(self, record: DynamicRadioRecord) -> np.ndarray:
        cached = self._cache_get("frame_indices", record.frame_indices_path)
        if cached is not None:
            return cached
        return self._cache_put("frame_indices", record.frame_indices_path, np.load(record.frame_indices_path))

    def load_dynamic_frame_uint8(self, record: DynamicRadioRecord, frame_idx: int) -> np.ndarray:
        if self.source == "png":
            image_path = record.rss_png_dir / f"frame_{int(frame_idx):06d}.png"
            with Image.open(image_path) as image:
                return np.asarray(image.convert("L"), dtype=np.uint8)
        rss_uint8 = self._load_npz_array(record.rss_npz_path, "rss_uint8")
        return np.asarray(rss_uint8[int(frame_idx)], dtype=np.uint8)

    def load_dynamic_clip_uint8(
        self,
        record: DynamicRadioRecord,
        start: int,
        length: int,
        stride: int = 1,
    ) -> np.ndarray:
        rss_uint8 = self._load_npz_array(record.rss_npz_path, "rss_uint8")
        frame_ids = [int(start) + i * int(stride) for i in range(int(length))]
        return np.asarray(rss_uint8[frame_ids], dtype=np.uint8)

    def load_static_map_uint8(self, record: DynamicRadioRecord) -> np.ndarray:
        return np.asarray(self._load_npz_array(record.static_rss_path, "rss_uint8"), dtype=np.uint8)

    def load_building_mask(self, record: DynamicRadioRecord) -> np.ndarray:
        return np.asarray(self._load_npz_array(record.building_mask_path, "building_mask"), dtype=np.uint8)

    def load_traffic_grid(self, record: DynamicRadioRecord) -> np.ndarray:
        return np.asarray(self._load_npz_array(record.traffic_grid_path, "traffic_grid_uint8"), dtype=np.uint8)

    def load_traffic_frame(self, record: DynamicRadioRecord, frame_idx: int) -> np.ndarray:
        return np.asarray(self.load_traffic_grid(record)[int(frame_idx)], dtype=np.uint8)

    def load_tx_position(self, record: DynamicRadioRecord) -> Any:
        sample_meta = self._load_json(record.sample_meta_path)
        return sample_meta.get("tx_position")

    def make_tx_heatmap(self, record: DynamicRadioRecord, sigma_px: float = 1.5) -> np.ndarray:
        scene_meta = self.scene_meta_by_id.get(record.scene_id)
        if scene_meta is None:
            raise KeyError(f"Scene metadata not found for {record.scene_id!r}")

        region = scene_meta.get("valid_crop") or scene_meta.get("support_region")
        support = scene_meta["support_region"]
        center = region.get("center", support["center"])
        width_m = float(region.get("width_m", scene_meta.get("valid_size_m", {}).get("width_m")))
        height_m = float(region.get("height_m", scene_meta.get("valid_size_m", {}).get("height_m")))
        yaw = math.radians(float(region.get("yaw_deg", support.get("yaw_deg", 0.0))))
        resolution = scene_meta.get("resolution_hw") or scene_meta.get("resolution") or [128, 128]
        height, width = int(resolution[0]), int(resolution[1])

        tx_position = self.load_tx_position(record)
        if tx_position is None:
            raise KeyError(f"tx_position not found in {record.sample_meta_path}")
        dx = float(tx_position[0]) - float(center["x"])
        dy = float(tx_position[1]) - float(center["y"])

        cos_yaw = math.cos(-yaw)
        sin_yaw = math.sin(-yaw)
        local_x = cos_yaw * dx - sin_yaw * dy
        local_y = sin_yaw * dx + cos_yaw * dy

        cell_size_x = width_m / float(width)
        cell_size_y = height_m / float(height)
        col = local_x / cell_size_x + width / 2.0 - 0.5
        row = local_y / cell_size_y + height / 2.0 - 0.5

        yy, xx = np.mgrid[0:height, 0:width]
        sigma = max(float(sigma_px), 1e-6)
        heatmap = np.exp(-((xx - col) ** 2 + (yy - row) ** 2) / (2.0 * sigma ** 2))
        return np.asarray(np.clip(heatmap, 0.0, 1.0), dtype=np.float32)

    @staticmethod
    def uint8_to_image_tensor(array: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(np.asarray(array, dtype=np.float32) / 255.0)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 3:
            tensor = tensor.unsqueeze(1)
        return tensor.mul(2.0).sub(1.0).contiguous()

    @staticmethod
    def mask_to_tensor(array: np.ndarray, scale: float = 1.0) -> torch.Tensor:
        tensor = torch.from_numpy(np.asarray(array, dtype=np.float32) / float(scale))
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 3:
            tensor = tensor.unsqueeze(1)
        return tensor.contiguous()

    @staticmethod
    def condition_to_tensor(array: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(np.asarray(array, dtype=np.float32))
        if tensor.ndim != 3:
            raise ValueError(f"Condition array must be [C,H,W], got shape {array.shape}")
        return tensor.mul(2.0).sub(1.0).contiguous()


class DynamicRadioSingleFrameVaeDataset(DynamicRadioBase):
    """Single-frame VAE dataset: one dynamic RSS frame is one training sample."""

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record_idx, frame_idx = self.frame_index[int(idx)]
        record = self.records[record_idx]
        image = self.uint8_to_image_tensor(self.load_dynamic_frame_uint8(record, frame_idx))
        return {
            "image": image,
            "img_name": f"{record.scene_id}/{record.episode_id}/{record.tx_id}/frame_{frame_idx:06d}.png",
        }


class DynamicRadioSingleFrameDiffusionDataset(DynamicRadioBase):
    """Single-frame diffusion dataset: building + Tx + traffic condition to dynamic RSS."""

    def __init__(
        self,
        *args: Any,
        tx_heatmap_sigma_px: float = 1.5,
        **kwargs: Any,
    ) -> None:
        self.tx_heatmap_sigma_px = float(tx_heatmap_sigma_px)
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record_idx, frame_idx = self.frame_index[int(idx)]
        record = self.records[record_idx]

        image = self.uint8_to_image_tensor(self.load_dynamic_frame_uint8(record, frame_idx))

        building = np.asarray(self.load_building_mask(record), dtype=np.float32)
        if building.max(initial=0.0) > 1.0:
            building = building / 255.0
        tx_heatmap = self.make_tx_heatmap(record, sigma_px=self.tx_heatmap_sigma_px)
        traffic = np.asarray(self.load_traffic_frame(record, frame_idx), dtype=np.float32) / 255.0

        cond = self.condition_to_tensor(np.stack([building, tx_heatmap, traffic], axis=0))
        img_name = f"{record.scene_id}/{record.episode_id}/{record.tx_id}/frame_{frame_idx:06d}.png"

        return {
            "image": image,
            "cond": cond,
            "img_name": img_name,
        }
