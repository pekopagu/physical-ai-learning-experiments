from __future__ import annotations

"""画像認識の推定座標を古典的な逆運動学へ渡す共通処理。"""

import math
from pathlib import Path

import mujoco
import numpy as np


MODEL_PATH = Path(__file__).with_name("arm.xml")
LINK_1 = 0.60
LINK_2 = 0.45
MAX_ANGLE_DELTA = 0.08
CONTROL_STEPS = 100
PHYSICS_STEPS_PER_CONTROL = 10


def inverse_kinematics(x: float, y: float) -> np.ndarray:
    """推定された手先座標から、肩と肘の目標角を幾何学的に求める。"""
    cos_elbow = (x * x + y * y - LINK_1**2 - LINK_2**2) / (2 * LINK_1 * LINK_2)
    cos_elbow = max(-1.0, min(1.0, cos_elbow))
    elbow = math.acos(cos_elbow)
    shoulder = math.atan2(y, x) - math.atan2(
        LINK_2 * math.sin(elbow), LINK_1 + LINK_2 * math.cos(elbow)
    )
    return np.array([shoulder, elbow], dtype=np.float64)


class ArmSimulation:
    """本当の目標と制御に使う推定位置を分離したMuJoCoシミュレーション。"""

    def __init__(self, true_target: tuple[float, float], estimate: tuple[float, float] | None):
        self.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.data = mujoco.MjData(self.model)
        self.true_target = np.asarray(true_target, dtype=np.float64)
        self.estimate = None if estimate is None else np.asarray(estimate, dtype=np.float64)
        self.target_id = self.model.site("target").id
        self.estimate_id = self.model.site("estimate").id
        self.end_effector_id = self.model.site("end_effector").id
        self.model.site_pos[self.target_id, :2] = self.true_target
        if self.estimate is None:
            self.model.site_rgba[self.estimate_id, 3] = 0.0
        else:
            self.model.site_pos[self.estimate_id, :2] = self.estimate
        mujoco.mj_forward(self.model, self.data)
        self.command = self.data.qpos[:2].copy()
        self.desired = self.command.copy() if self.estimate is None else inverse_kinematics(*self.estimate)

    @property
    def end_effector(self) -> np.ndarray:
        return self.data.site_xpos[self.end_effector_id, :2].copy()

    @property
    def true_distance(self) -> float:
        return float(np.linalg.norm(self.true_target - self.end_effector))

    def control_step(self) -> None:
        # 従来実験と同じく、一度に変える指令角を0.08 radまでに制限する。
        delta = np.clip(self.desired - self.command, -MAX_ANGLE_DELTA, MAX_ANGLE_DELTA)
        self.command += delta
        self.data.ctrl[:] = self.command
        for _ in range(PHYSICS_STEPS_PER_CONTROL):
            mujoco.mj_step(self.model, self.data)

    def run(self) -> dict[str, float | bool]:
        initial_distance = self.true_distance
        for _ in range(CONTROL_STEPS):
            self.control_step()
        final_distance = self.true_distance
        return {
            "initial_distance": initial_distance,
            "final_distance": final_distance,
            "success_3cm": final_distance < 0.03,
            "success_1cm": final_distance < 0.01,
            "final_x": float(self.end_effector[0]),
            "final_y": float(self.end_effector[1]),
        }
