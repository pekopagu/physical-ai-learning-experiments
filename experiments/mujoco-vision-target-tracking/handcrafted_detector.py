from __future__ import annotations

"""人間が明示した色・面積・円形度規則による目標位置推定。"""

from dataclasses import dataclass

import cv2
import numpy as np

from camera_mapping import pixel_to_world


# テスト画像を見る前に固定する初期規則（handcrafted-v1）。
RED_LOW_1 = np.array([0, 75, 40], dtype=np.uint8)
RED_HIGH_1 = np.array([16, 255, 255], dtype=np.uint8)
RED_LOW_2 = np.array([165, 75, 40], dtype=np.uint8)
RED_HIGH_2 = np.array([179, 255, 255], dtype=np.uint8)
MIN_AREA = 3
MAX_AREA = 90


@dataclass(frozen=True)
class Detection:
    world_x: float
    world_y: float
    pixel_x: float
    pixel_y: float
    area: float
    circularity: float


def detect_target(image_rgb: np.ndarray) -> Detection | None:
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, RED_LOW_1, RED_HIGH_1) | cv2.inRange(
        hsv, RED_LOW_2, RED_HIGH_2
    )

    # 小さい目標を消さないため、初期方式ではopening/closingを適用しない。
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[tuple[float, float, float, float, float]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if not MIN_AREA <= area <= MAX_AREA:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 0:
            continue
        circularity = float(4.0 * np.pi * area / (perimeter * perimeter))
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        pixel_x = float(moments["m10"] / moments["m00"])
        pixel_y = float(moments["m01"] / moments["m00"])
        # 円形度を主にしつつ、1～2画素のノイズ領域を面積で抑える。
        score = circularity * float(np.sqrt(area))
        candidates.append((score, pixel_x, pixel_y, area, circularity))

    if not candidates:
        return None
    _, pixel_x, pixel_y, area, circularity = max(candidates, key=lambda item: item[0])
    world_x, world_y = pixel_to_world(pixel_x, pixel_y)
    return Detection(world_x, world_y, pixel_x, pixel_y, area, circularity)
