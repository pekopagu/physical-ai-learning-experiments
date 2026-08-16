from __future__ import annotations

"""選んだ1画像について、1方式の推定座標へ動くアームを本人が観察する。"""

import argparse
import csv
from pathlib import Path
import time

import mujoco.viewer

from arm_control import ArmSimulation, CONTROL_STEPS


OUTPUT_DIR = Path(__file__).with_name("outputs")
PERCEPTION_PATH = OUTPUT_DIR / "perception-evaluation.csv"
METHOD_MAP = {"ground_truth": "ground_truth", "rule_based": "handcrafted", "cnn": "cnn"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="View arm reaching from one perception estimate.")
    parser.add_argument("method", choices=METHOD_MAP)
    parser.add_argument("--condition", choices=("a", "b", "c"), default="c")
    parser.add_argument("--image", type=int, choices=range(1, 201), default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not PERCEPTION_PATH.exists():
        raise SystemExit("知覚評価がありません。先にevaluate_perception.pyを実行してください。")
    with PERCEPTION_PATH.open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    split = f"test_{args.condition}"
    method = METHOD_MAP[args.method]
    selected = sorted(
        (row for row in rows if row["split"] == split and row["method"] == method),
        key=lambda row: row["image"],
    )
    row = selected[args.image - 1]
    detected = row["detected"] == "True"
    estimate = (
        (float(row["predicted_x"]), float(row["predicted_y"])) if detected else None
    )
    true_target = (float(row["true_x"]), float(row["true_y"]))
    simulation = ArmSimulation(true_target, estimate)

    print(f"Method: {args.method}")
    print(f"Image: {row['image']}")
    print(f"True target: ({true_target[0]:.6f}, {true_target[1]:.6f})")
    print("Estimated target: not detected" if estimate is None else f"Estimated target: ({estimate[0]:.6f}, {estimate[1]:.6f})")
    print("Red: true target, Purple: estimated target, Yellow: end effector")
    print("The initial pose stays still for 5 seconds.")

    with mujoco.viewer.launch_passive(simulation.model, simulation.data) as viewer:
        # 自由カメラの初期値に依存せず、アームの到達領域全体を映す固定俯瞰へ切り替える。
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = simulation.model.camera("overview").id
        viewer.sync()
        time.sleep(5)
        for _ in range(CONTROL_STEPS):
            if not viewer.is_running():
                break
            simulation.control_step()
            viewer.sync()
            time.sleep(0.02)
        print(f"Final distance: {simulation.true_distance:.6f} m")
        print(f"Reached 3 cm: {simulation.true_distance < 0.03}")
        print("Inspect the final pose, then close the viewer window.")
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.05)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # 最終姿勢を確認後、Ctrl+Cで終了してもエラー表示にしない。
        print("Viewer closed by user.")
