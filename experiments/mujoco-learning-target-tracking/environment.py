from __future__ import annotations

"""未学習、PPO、古典制御で共用するGymnasium環境。"""

from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


MODEL_PATH = Path(__file__).with_name("arm.xml")
MAX_EPISODE_STEPS = 100
PHYSICS_STEPS_PER_ACTION = 10
MAX_ANGLE_DELTA = 0.08
SUCCESS_DISTANCE = 0.03
TARGET_RADIUS_MIN = 0.30
TARGET_RADIUS_MAX = 0.95
TARGET_ANGLE_MIN = 0.15
TARGET_ANGLE_MAX = 2.85


class TwoLinkReachEnv(gym.Env[np.ndarray, np.ndarray]):
    """安全な関節角増分を行動とする、CPU用2リンク目標追従環境。"""

    metadata = {"render_modes": []}

    def __init__(self, target: tuple[float, float] | None = None) -> None:
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.data = mujoco.MjData(self.model)
        self._end_effector_id = self.model.site("end_effector").id
        self._target_id = self.model.site("target").id
        self._fixed_target = target
        self._step_count = 0
        self._target = np.zeros(2, dtype=np.float64)
        self._command = np.zeros(2, dtype=np.float64)

        # 行動の2要素は肩と肘の指令角増分である。
        # PPOの出力を[-1, 1]へ制限し、実際の角度変化はstep()で縮小する。
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        # 観測は肩・肘角のcos/sin、角速度、手先から目標までの差の8要素。
        # 角度をsin/cosで表すことで、-piとpiの境界を不連続にしない。
        self.observation_space = spaces.Box(
            low=np.array(
                [-1, -1, -1, -1, -1e6, -1e6, -2.1, -2.1], dtype=np.float32
            ),
            high=np.array(
                [1, 1, 1, 1, 1e6, 1e6, 2.1, 2.1], dtype=np.float32
            ),
            dtype=np.float32,
        )

    def _sample_target(self) -> np.ndarray:
        # 到達可能領域の半径と方向を乱数で選び、学習目標を固定しない。
        radius = self.np_random.uniform(TARGET_RADIUS_MIN, TARGET_RADIUS_MAX)
        angle = self.np_random.uniform(TARGET_ANGLE_MIN, TARGET_ANGLE_MAX)
        return np.array([radius * np.cos(angle), radius * np.sin(angle)])

    def _set_target(self, target: np.ndarray) -> None:
        self._target = np.asarray(target, dtype=np.float64)
        self.model.site_pos[self._target_id, :2] = self._target

    def _end_effector(self) -> np.ndarray:
        return self.data.site_xpos[self._end_effector_id, :2].copy()

    def _distance(self) -> float:
        return float(np.linalg.norm(self._target - self._end_effector()))

    def _observation(self) -> np.ndarray:
        angles = self.data.qpos[:2]
        velocity = self.data.qvel[:2]
        target_delta = self._target - self._end_effector()
        return np.array(
            [
                np.cos(angles[0]),
                np.sin(angles[0]),
                np.cos(angles[1]),
                np.sin(angles[1]),
                velocity[0],
                velocity[1],
                target_delta[0],
                target_delta[1],
            ],
            dtype=np.float32,
        )

    @property
    def target(self) -> np.ndarray:
        return self._target.copy()

    @property
    def command(self) -> np.ndarray:
        return self._command.copy()

    @property
    def end_effector(self) -> np.ndarray:
        return self._end_effector()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, float]]:
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self._step_count = 0
        self._command[:] = self.data.qpos[:2]

        # 学習時はランダム目標、評価時はoptionsまたは生成時の固定目標を使う。
        option_target = None if options is None else options.get("target")
        selected_target = self._fixed_target if option_target is None else option_target
        target = self._sample_target() if selected_target is None else np.asarray(selected_target)
        self._set_target(target)
        self.data.ctrl[:] = self._command
        mujoco.mj_forward(self.model, self.data)
        distance = self._distance()
        return self._observation(), {"distance": distance}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, float]]:
        # 1. 方策が出した行動を安全範囲へ制限する。
        safe_action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        # 2. 正規化された行動を、肩・肘の小さな指令角変化へ変換する。
        self._command += safe_action * MAX_ANGLE_DELTA
        self._command[:] = np.clip(self._command, -np.pi, np.pi)
        self.data.ctrl[:] = self._command

        # 3. 同じ関節指令のままMuJoCoの物理状態を10回進める。
        for _ in range(PHYSICS_STEPS_PER_ACTION):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        distance = self._distance()
        # 4. 目標距離を主成分とし、大きな行動へ小さな罰を与える。
        control_cost = 0.01 * float(np.square(safe_action).sum())
        reached = distance < SUCCESS_DISTANCE
        reward = -distance - control_cost + (1.0 if reached else 0.0)
        # 3cm未満ならterminated、100制御ステップならtruncatedとして終了する。
        truncated = self._step_count >= MAX_EPISODE_STEPS
        info = {
            "distance": distance,
            "control_cost": control_cost,
            "action_magnitude": float(np.linalg.norm(safe_action)),
        }
        return self._observation(), reward, reached, truncated, info

    def close(self) -> None:
        pass


# 3方式を公平に比較するため、評価時は同じ20座標を使用する。
FIXED_EVALUATION_TARGETS: tuple[tuple[float, float], ...] = (
    (0.65, 0.55),
    (0.30, 0.80),
    (0.80, 0.30),
    (0.55, 0.65),
    (0.20, 0.70),
    (-0.20, 0.70),
    (-0.45, 0.55),
    (-0.65, 0.30),
    (0.90, 0.10),
    (0.10, 0.90),
    (0.45, 0.20),
    (0.20, 0.45),
    (-0.20, 0.45),
    (-0.45, 0.20),
    (0.70, 0.45),
    (0.45, 0.70),
    (-0.45, 0.70),
    (-0.70, 0.45),
    (0.35, 0.35),
    (-0.35, 0.35),
)
