from __future__ import annotations

"""2リンクロボットアームを目標位置まで動かす古典制御の実験。

今回の実験では、学習済みモデル、ニューラルネットワーク、強化学習は使用しない。
目標位置から逆運動学で関節の目標角度を計算し、MuJoCoの位置制御で腕を動かす。

処理と「観測・判断・行動」の対応は次のとおり。

1. 観測: 関節角度、手先位置、目標位置を取得する
2. 判断: 逆運動学により、肩と肘の目標角度を計算する
3. 行動: 計算した目標角度をアクチュエータへ設定する
4. 次状態: MuJoCoで物理シミュレーションを進め、腕を動かす
5. 再観測: 移動後の手先位置と目標までの距離を確認する
"""

import argparse
import math
import time
from pathlib import Path

import mujoco
import mujoco.viewer


MODEL_PATH = Path(__file__).with_name("arm.xml")
LINK_1 = 0.60
LINK_2 = 0.45
STEPS = 2_000
VIEW_STEP_DELAY = 0.004
START_PAUSE_SECONDS = 5


def inverse_kinematics(x: float, y: float) -> tuple[float, float]:
    """目標座標 (x, y) から、肩と肘の目標角度を数式で求める。"""
    # 2リンクアームの幾何学と余弦定理を使う古典的な逆運動学であり、
    # AIモデルが動かし方を推論または学習しているわけではない。
    cos_elbow = (x * x + y * y - LINK_1**2 - LINK_2**2) / (2 * LINK_1 * LINK_2)
    cos_elbow = max(-1.0, min(1.0, cos_elbow))
    elbow = math.acos(cos_elbow)
    shoulder = math.atan2(y, x) - math.atan2(
        LINK_2 * math.sin(elbow), LINK_1 + LINK_2 * math.cos(elbow)
    )
    return shoulder, elbow


def end_effector_xy(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[float, float]:
    """黄色い手先（end_effector）の現在位置を取得する。"""
    site_id = model.site("end_effector").id
    return float(data.site_xpos[site_id, 0]), float(data.site_xpos[site_id, 1])


def target_xy(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[float, float]:
    """赤い目標（target）の位置を取得する。"""
    site_id = model.site("target").id
    return float(data.site_xpos[site_id, 0]), float(data.site_xpos[site_id, 1])


def distance_to_target(
    position: tuple[float, float], target: tuple[float, float]
) -> float:
    """手先から目標までの直線距離を求める。"""
    return math.dist(position, target)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--view",
        action="store_true",
        help="Open MuJoCo viewer and animate the arm in approximately real time.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # MuJoCoへロボット、関節、モーター、目標などの定義を読み込む。
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # 1. 観測: 動作前の手先位置と目標位置を取得する。
    # 関節角度は data.qpos に格納されている。
    initial_position = end_effector_xy(model, data)
    target = target_xy(model, data)

    # 2. 判断: 目標へ届く肩と肘の角度を逆運動学で一度だけ計算する。
    # 今回は、動作中に観測と判断を繰り返す制御方式ではない。
    target_angles = inverse_kinematics(*target)

    print("Observation before action")
    print(f"  joint angles [rad] : {data.qpos.tolist()}")
    print(f"  end effector [m]   : {initial_position}")
    print(f"  target [m]         : {target}")
    print(f"  distance [m]       : {distance_to_target(initial_position, target):.6f}")
    print("Decision")
    print(f"  target angles [rad]: {[round(value, 6) for value in target_angles]}")

    # 3. 行動: 肩と肘の位置制御アクチュエータへ目標角度を指令する。
    data.ctrl[:] = target_angles
    if args.view:
        print("Viewer guide")
        print("  blue/green capsules: arm links")
        print("  yellow sphere       : end effector")
        print("  red transparent ball: target")
        print(f"  1. The initial pose stays still for {START_PAUSE_SECONDS} seconds.")
        print("  2. Watch both joints rotate and the yellow sphere approach the red target.")
        print("  3. The final pose stays visible until you close the window.")
        with mujoco.viewer.launch_passive(
            model,
            data,
            show_left_ui=False,
            show_right_ui=False,
        ) as viewer:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = model.camera("overview").id
            viewer.sync()
            time.sleep(START_PAUSE_SECONDS)

            # 4. 次状態: 物理シミュレーションを少しずつ進める。
            # MuJoCo内部の位置制御器が目標角度との差を縮めるように関節を動かす。
            for _ in range(STEPS):
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(VIEW_STEP_DELAY)

            print("Motion complete. Inspect the final pose, then close the viewer window.")
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.05)
    else:
        # 画面を表示しない場合も、同じ回数だけ物理シミュレーションを進める。
        for _ in range(STEPS):
            mujoco.mj_step(model, data)

    # 5. 再観測: 動作後の手先位置と目標までの距離を取得する。
    # 距離が1 cm未満なら、目標へ到達したと判定する。
    final_position = end_effector_xy(model, data)
    final_distance = distance_to_target(final_position, target)

    print("Observation after action")
    print(f"  joint angles [rad] : {[round(value, 6) for value in data.qpos.tolist()]}")
    print(f"  end effector [m]   : {tuple(round(value, 6) for value in final_position)}")
    print(f"  distance [m]       : {final_distance:.6f}")
    print(f"  reached target     : {final_distance < 0.01}")


if __name__ == "__main__":
    main()
