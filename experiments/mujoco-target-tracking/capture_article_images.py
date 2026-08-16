from __future__ import annotations

import math
from pathlib import Path

import mujoco
from PIL import Image, ImageDraw, ImageFont

from run_experiment import (
    MODEL_PATH,
    STEPS,
    distance_to_target,
    end_effector_xy,
    inverse_kinematics,
)


OUTPUT_DIR = Path(__file__).with_name("article-images")
WIDTH = 1200
SCENE_HEIGHT = 760
CAPTION_HEIGHT = 180
EXPERIMENTS = (
    ("experiment-1", (0.65, 0.55)),
    ("experiment-2", (0.30, 0.80)),
)


def set_target(model: mujoco.MjModel, target: tuple[float, float]) -> None:
    target_id = model.site("target").id
    model.site_pos[target_id, 0] = target[0]
    model.site_pos[target_id, 1] = target[1]


def render_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    label: str,
    phase: str,
    target: tuple[float, float],
    commanded_angles: tuple[float, float],
) -> Image.Image:
    renderer = mujoco.Renderer(model, height=SCENE_HEIGHT, width=WIDTH)
    renderer.update_scene(data, camera="overview")
    pixels = renderer.render()
    renderer.close()

    scene = Image.fromarray(pixels)
    output = Image.new("RGB", (WIDTH, SCENE_HEIGHT + CAPTION_HEIGHT), "white")
    output.paste(scene, (0, 0))

    position = end_effector_xy(model, data)
    distance = distance_to_target(position, target)
    draw = ImageDraw.Draw(output)
    font = ImageFont.load_default(size=24)
    lines = [
        f"{label} / {phase}",
        f"Target: ({target[0]:.2f}, {target[1]:.2f}) m    End effector: ({position[0]:.6f}, {position[1]:.6f}) m",
        f"Distance: {distance:.6f} m    Joint angles: shoulder={data.qpos[0]:.6f}, elbow={data.qpos[1]:.6f} rad",
    ]
    if phase == "start":
        lines.append(
            f"Commanded angles: shoulder={commanded_angles[0]:.6f}, elbow={commanded_angles[1]:.6f} rad"
        )
    else:
        lines.append(f"Reached target (< 0.01 m): {distance < 0.01}")

    y = SCENE_HEIGHT + 12
    for line in lines:
        draw.text((20, y), line, fill="black", font=font)
        y += 38
    return output


def capture_experiment(label: str, target: tuple[float, float]) -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    set_target(model, target)
    mujoco.mj_forward(model, data)

    target_angles = inverse_kinematics(*target)
    start = render_frame(
        model,
        data,
        label=label,
        phase="start",
        target=target,
        commanded_angles=target_angles,
    )
    start_path = OUTPUT_DIR / f"{label}-start.png"
    start.save(start_path)

    data.ctrl[:] = target_angles
    for _ in range(STEPS):
        mujoco.mj_step(model, data)

    end = render_frame(
        model,
        data,
        label=label,
        phase="end",
        target=target,
        commanded_angles=target_angles,
    )
    end_path = OUTPUT_DIR / f"{label}-end.png"
    end.save(end_path)

    print(start_path)
    print(end_path)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    for label, target in EXPERIMENTS:
        capture_experiment(label, target)


if __name__ == "__main__":
    main()
