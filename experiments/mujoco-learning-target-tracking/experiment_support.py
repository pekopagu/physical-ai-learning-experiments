from __future__ import annotations

import math
from pathlib import Path
from typing import Protocol

import numpy as np

from environment import MAX_ANGLE_DELTA, TwoLinkReachEnv


LINK_1 = 0.60
LINK_2 = 0.45
PREDICTIONS_PATH = Path(__file__).with_name("predictions.md")


class Policy(Protocol):
    def __call__(self, observation: np.ndarray, env: TwoLinkReachEnv) -> np.ndarray: ...


def require_predictions() -> None:
    # エージェントの予備結果が本人の予想へ影響しないよう、実行前記録を必須にする。
    contents = PREDICTIONS_PATH.read_text(encoding="utf-8")
    if "prediction_status: completed" not in contents:
        raise SystemExit(
            "学習前に predictions.md を本人が記入し、"
            "prediction_status を completed に変更してください。"
        )


def inverse_kinematics(x: float, y: float) -> np.ndarray:
    """目標座標から肩・肘の最終角を幾何学的に求める古典的逆運動学。"""
    cos_elbow = (x * x + y * y - LINK_1**2 - LINK_2**2) / (2 * LINK_1 * LINK_2)
    cos_elbow = max(-1.0, min(1.0, cos_elbow))
    elbow = math.acos(cos_elbow)
    shoulder = math.atan2(y, x) - math.atan2(
        LINK_2 * math.sin(elbow), LINK_1 + LINK_2 * math.cos(elbow)
    )
    return np.array([shoulder, elbow], dtype=np.float64)


def classical_policy(_observation: np.ndarray, env: TwoLinkReachEnv) -> np.ndarray:
    # PPOと同じ行動インターフェースで比較するため、正解角との差を[-1, 1]へ変換する。
    desired = inverse_kinematics(*env.target)
    return np.clip((desired - env.command) / MAX_ANGLE_DELTA, -1.0, 1.0).astype(
        np.float32
    )


def run_episode(
    env: TwoLinkReachEnv,
    policy: Policy,
    target: tuple[float, float],
    seed: int,
) -> dict[str, float | int | bool]:
    observation, _ = env.reset(seed=seed, options={"target": target})
    total_reward = 0.0
    total_action = 0.0
    step_count = 0
    terminated = truncated = False

    # 成功または100ステップ上限まで、観測→判断→行動→次状態を繰り返す。
    while not (terminated or truncated):
        action = policy(observation, env)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        total_action += info["action_magnitude"]
        step_count += 1

    distance = float(info["distance"])
    return {
        "steps": step_count,
        "total_reward": total_reward,
        "final_distance": distance,
        "success_3cm": distance < 0.03,
        "success_1cm": distance < 0.01,
        "total_action": total_action,
    }
