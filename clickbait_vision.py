from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from clickbait_common import resolve_thumbnail_path


IMAGE_NET_MEAN = (0.485, 0.456, 0.406)
IMAGE_NET_STD = (0.229, 0.224, 0.225)


def build_image_transforms(image_size: int = 224, train: bool = False):
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                transforms.ToTensor(),
                transforms.Normalize(IMAGE_NET_MEAN, IMAGE_NET_STD),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.15)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGE_NET_MEAN, IMAGE_NET_STD),
        ]
    )


class ThumbnailDataset(Dataset):
    def __init__(self, records: Any, image_size: int = 224, train: bool = False) -> None:
        self.records = records
        self.transform = build_image_transforms(image_size=image_size, train=train)

    def __len__(self) -> int:
        return len(self.records)

    def _image_path(self, row: dict[str, Any]) -> Path:
        explicit_path = str(row.get("thumbnail_path") or row.get("image_path") or "").strip()
        if explicit_path:
            candidate = Path(explicit_path)
            if candidate.exists():
                return candidate

        video_id = str(row.get("videoID") or "").strip()
        candidate = resolve_thumbnail_path(video_id)
        if candidate is None:
            raise FileNotFoundError(f"Could not resolve thumbnail for videoID={video_id}")
        return candidate

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.records[index]
        image_path = self._image_path(row)
        image = Image.open(image_path).convert("RGB")
        return {
            "pixel_values": self.transform(image),
            "labels": int(row.get("is_clickbait", 0)),
            "video_id": str(row.get("videoID") or ""),
            "title": str(row.get("title") or ""),
            "image_path": str(image_path),
        }


def batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}