from __future__ import annotations

"""EXPERIMENT-003で固定した画像条件A～CをMuJoCoへ適用する。"""

from dataclasses import asdict, dataclass
import math

import mujoco
import numpy as np


TARGET_RADIUS_MIN = 0.30
TARGET_RADIUS_MAX = 0.95
TARGET_ANGLE_MIN = 0.15
TARGET_ANGLE_MAX = 2.85
DISTRACTOR_NAMES = (
    "distractor_box",
    "distractor_capsule",
    "distractor_ellipsoid",
)
HIDDEN_POSITION = np.array([0.0, 0.0, -1.0])


@dataclass(frozen=True)
class ImageCondition:
    group: str
    target_x: float
    target_y: float
    target_radius: float
    target_rgb: tuple[float, float, float]
    background_rgb: tuple[float, float, float]
    light_scale: float
    noise_std: float
    distractor_count: int
    ood_kind: str = "none"

    def metadata(self) -> dict[str, object]:
        return asdict(self)


def sample_target(rng: np.random.Generator) -> tuple[float, float]:
    radius = rng.uniform(TARGET_RADIUS_MIN, TARGET_RADIUS_MAX)
    angle = rng.uniform(TARGET_ANGLE_MIN, TARGET_ANGLE_MAX)
    return float(radius * math.cos(angle)), float(radius * math.sin(angle))


def sample_condition(
    group: str,
    rng: np.random.Generator,
    sample_index: int,
    target_override: tuple[float, float] | None = None,
) -> ImageCondition:
    # 学習時は各画像で乱数位置を使う。比較テストではA～Cへ同じ位置列を渡せる。
    x, y = sample_target(rng) if target_override is None else target_override
    if group == "A":
        return ImageCondition(
            group="A",
            target_x=x,
            target_y=y,
            target_radius=0.07,
            target_rgb=(1.0, 0.1, 0.1),
            background_rgb=(0.15, 0.15, 0.15),
            light_scale=1.0,
            noise_std=0.0,
            distractor_count=0,
        )

    if group == "B":
        background = tuple(float(value) for value in rng.uniform(0.08, 0.30, size=3))
        # 赤だけが極端に強い背景を避け、目標と背景が意味的に区別不能になるのを防ぐ。
        if background[0] - max(background[1], background[2]) >= 0.15:
            background = (background[0], min(0.30, background[1] + 0.10), background[2])
        return ImageCondition(
            group="B",
            target_x=x,
            target_y=y,
            target_radius=float(rng.uniform(0.045, 0.080)),
            target_rgb=(
                float(rng.uniform(0.65, 1.00)),
                float(rng.uniform(0.03, 0.30)),
                float(rng.uniform(0.03, 0.30)),
            ),
            background_rgb=background,
            light_scale=float(rng.uniform(0.65, 1.35)),
            noise_std=float(rng.uniform(0.0, 0.03)),
            distractor_count=int(rng.integers(0, 4)),
        )

    if group != "C":
        raise ValueError(f"Unknown condition group: {group}")

    # Cでは一度にすべてを変えず、4種類を順番に出して失敗原因を追えるようにする。
    ood_kind = ("lighting", "target", "background", "occlusion")[sample_index % 4]
    background = (0.15, 0.15, 0.15)
    light_scale = 1.0
    target_rgb = (0.78, 0.18, 0.18)
    target_radius = 0.06
    # Cも人間が画像だけから正解を識別できる範囲に保つ。
    # 難しくしすぎて正解自体が曖昧になると、一般化性能を測れない。
    noise_std = float(rng.uniform(0.025, 0.040))
    if ood_kind == "lighting":
        light_scale = float(rng.choice([rng.uniform(0.60, 0.75), rng.uniform(1.35, 1.50)]))
    elif ood_kind == "target":
        target_rgb = (
            float(rng.uniform(0.58, 0.72)),
            float(rng.uniform(0.25, 0.38)),
            float(rng.uniform(0.18, 0.35)),
        )
        target_radius = float(rng.choice([rng.uniform(0.040, 0.050), rng.uniform(0.085, 0.095)]))
    elif ood_kind == "background":
        background = (
            float(rng.uniform(0.30, 0.40)),
            float(rng.uniform(0.16, 0.28)),
            float(rng.uniform(0.16, 0.28)),
        )

    return ImageCondition(
        group="C",
        target_x=x,
        target_y=y,
        target_radius=target_radius,
        target_rgb=target_rgb,
        background_rgb=background,
        light_scale=light_scale,
        noise_std=noise_std,
        distractor_count=int(rng.integers(1, 4)),
        ood_kind=ood_kind,
    )


def _sample_distractor_position(
    rng: np.random.Generator, target: np.ndarray
) -> np.ndarray:
    for _ in range(100):
        candidate = np.array([rng.uniform(-0.9, 0.9), rng.uniform(-0.05, 1.0)])
        if np.linalg.norm(candidate - target) > 0.18:
            return candidate
    raise RuntimeError("Could not place a distractor away from the target")


def apply_condition(
    model: mujoco.MjModel,
    condition: ImageCondition,
    rng: np.random.Generator,
) -> None:
    target_id = model.site("target").id
    floor_id = model.geom("floor").id
    light_id = model.light("main_light").id
    target_xy = np.array([condition.target_x, condition.target_y])

    model.site_pos[target_id, :2] = target_xy
    model.site_size[target_id, 0] = condition.target_radius
    model.site_rgba[target_id] = (*condition.target_rgb, 1.0)
    model.geom_rgba[floor_id] = (*condition.background_rgb, 1.0)
    model.light_diffuse[light_id] = condition.light_scale

    for name in (*DISTRACTOR_NAMES, "ood_cylinder", "occluder"):
        geom_id = model.geom(name).id
        model.geom_pos[geom_id] = HIDDEN_POSITION
        model.geom_rgba[geom_id, 3] = 0.0

    colors = (
        (0.85, 0.22, 0.08),
        (0.72, 0.10, 0.34),
        (0.68, 0.24, 0.55),
    )
    for index, name in enumerate(DISTRACTOR_NAMES[: condition.distractor_count]):
        geom_id = model.geom(name).id
        position = _sample_distractor_position(rng, target_xy)
        model.geom_pos[geom_id] = (position[0], position[1], 0.10)
        model.geom_rgba[geom_id] = (*colors[index], 1.0)

    if condition.group == "C":
        cylinder_id = model.geom("ood_cylinder").id
        position = _sample_distractor_position(rng, target_xy)
        model.geom_pos[cylinder_id] = (position[0], position[1], 0.10)
        model.geom_rgba[cylinder_id] = (0.62, 0.25, 0.22, 1.0)

    if condition.ood_kind == "occlusion":
        occluder_id = model.geom("occluder").id
        offset = condition.target_radius * 0.55
        model.geom_pos[occluder_id] = (
            condition.target_x + offset,
            condition.target_y,
            0.16,
        )
        model.geom_size[occluder_id, :2] = (
            condition.target_radius * 0.35,
            condition.target_radius * 0.20,
        )
        model.geom_rgba[occluder_id] = (0.10, 0.60, 0.80, 1.0)


def add_image_noise(
    image: np.ndarray, noise_std: float, rng: np.random.Generator
) -> np.ndarray:
    if noise_std == 0.0:
        return image
    normalized = image.astype(np.float32) / 255.0
    noisy = np.clip(normalized + rng.normal(0.0, noise_std, normalized.shape), 0.0, 1.0)
    return np.round(noisy * 255.0).astype(np.uint8)
