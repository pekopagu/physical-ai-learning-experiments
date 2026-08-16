from __future__ import annotations

"""metadata.csvとPNGをPyTorchの教師あり回帰データへ変換する。"""

import csv
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


DATASET_DIR = Path(__file__).with_name("dataset")
METADATA_PATH = DATASET_DIR / "metadata.csv"


def load_metadata() -> list[dict[str, str]]:
    with METADATA_PATH.open(encoding="utf-8") as file:
        return list(csv.DictReader(file))


class TargetImageDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, split: str) -> None:
        self.rows = [row for row in load_metadata() if row["split"] == split]
        if not self.rows:
            raise ValueError(f"No images found for split: {split}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        image = np.asarray(Image.open(DATASET_DIR / row["image"]).convert("RGB"))
        # [H,W,C]の0～255画像を、[C,H,W]の0～1テンソルへ変換する。
        image_tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255.0
        target_tensor = torch.tensor(
            [float(row["target_x"]), float(row["target_y"])], dtype=torch.float32
        )
        return image_tensor, target_tensor
