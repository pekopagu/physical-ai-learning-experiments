from __future__ import annotations

"""選んだ知覚方式によるアーム動作を、本人の比較用MP4へ保存する。"""

import argparse
import csv
from pathlib import Path

import cv2
import mujoco
import numpy as np

from arm_control import ArmSimulation, CONTROL_STEPS


OUTPUT_DIR = Path(__file__).with_name("outputs")
PERCEPTION_PATH = OUTPUT_DIR / "perception-evaluation.csv"
METHOD_MAP = {"ground_truth": "ground_truth", "rule_based": "handcrafted", "cnn": "cnn"}
FPS = 25
WIDTH = 960
HEIGHT = 720
PAUSE_SECONDS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record arm reaching from one perception estimate.")
    parser.add_argument("method", choices=METHOD_MAP)
    parser.add_argument("--condition", choices=("a", "b", "c"), default="c")
    parser.add_argument("--image", type=int, choices=range(1, 201), default=1)
    return parser.parse_args()


def selected_row(condition: str, image_number: int, method: str) -> dict[str, str]:
    if not PERCEPTION_PATH.exists():
        raise SystemExit("知覚評価がありません。先にevaluate_perception.pyを実行してください。")
    with PERCEPTION_PATH.open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    split = f"test_{condition}"
    # 画面表示と集計で同じ推定値を使い、動画用に再推論して結果を変えない。
    selected = sorted(
        (row for row in rows if row["split"] == split and row["method"] == METHOD_MAP[method]),
        key=lambda row: row["image"],
    )
    return selected[image_number - 1]


def annotated_frame(renderer: mujoco.Renderer, simulation: ArmSimulation, method: str) -> np.ndarray:
    renderer.update_scene(simulation.data, camera="recording")
    frame = renderer.render().copy()
    # OpenCVの動画出力はRGBではなくBGR順を使う。
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.putText(frame, f"Method: {method}", (24, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, "Red: true  Purple: estimate  Yellow: end effector", (24, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return frame


def main() -> None:
    args = parse_args()
    row = selected_row(args.condition, args.image, args.method)
    detected = row["detected"] == "True"
    estimate = (
        (float(row["predicted_x"]), float(row["predicted_y"])) if detected else None
    )
    true_target = (float(row["true_x"]), float(row["true_y"]))
    # 赤は真の目標、紫は知覚方式の推定、黄色は手先として同時に表示する。
    simulation = ArmSimulation(true_target, estimate)

    video_dir = OUTPUT_DIR / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    output_path = video_dir / f"arm-{args.method}-{args.condition}-image-{args.image:03d}.mp4"
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT)
    )
    if not writer.isOpened():
        raise SystemExit("MP4動画の出力を開始できませんでした。")

    try:
        with mujoco.Renderer(simulation.model, height=HEIGHT, width=WIDTH) as renderer:
            first = annotated_frame(renderer, simulation, args.method)
            # 動作前の位置関係を観察できるよう、初期姿勢を5秒間保持する。
            for _ in range(FPS * PAUSE_SECONDS):
                writer.write(first)

            # 100制御周期を25fpsで記録すると動作部分が4秒となり、実時間の約2倍になる。
            for _ in range(CONTROL_STEPS):
                simulation.control_step()
                writer.write(annotated_frame(renderer, simulation, args.method))

            last = annotated_frame(renderer, simulation, args.method)
            # 最終距離を画面で確認できるよう、終了姿勢も5秒間保持する。
            for _ in range(FPS * PAUSE_SECONDS):
                writer.write(last)
    finally:
        writer.release()

    print(f"Saved video: {output_path}")
    print(f"Image: {row['image']}")
    print(f"Final distance: {simulation.true_distance:.6f} m")
    print(f"Reached 3 cm: {simulation.true_distance < 0.03}")
    print("Playback: about 2x slower motion, 5s initial and final pauses")


if __name__ == "__main__":
    main()
