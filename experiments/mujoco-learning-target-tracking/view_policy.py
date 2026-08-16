from __future__ import annotations

import argparse
from pathlib import Path
import time

import mujoco.viewer
import numpy as np
from stable_baselines3 import PPO

from environment import FIXED_EVALUATION_TARGETS, TwoLinkReachEnv
from experiment_support import classical_policy, require_predictions


MODEL_PATH = Path(__file__).with_name("outputs") / "ppo_reacher.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="View one policy on one fixed target.")
    parser.add_argument("policy", choices=("random", "ppo", "classical"))
    parser.add_argument("--target", type=int, default=1, choices=range(1, 21))
    parser.add_argument("--seed", type=int, default=20260815)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_predictions()
    target = FIXED_EVALUATION_TARGETS[args.target - 1]
    env = TwoLinkReachEnv(target=target)
    observation, info = env.reset(seed=args.seed)
    rng = np.random.default_rng(args.seed)

    if args.policy == "ppo":
        if not MODEL_PATH.exists():
            raise SystemExit("学習済みモデルがありません。先に train.py を実行してください。")
        model = PPO.load(MODEL_PATH, device="cpu")

        def policy(obs: np.ndarray) -> np.ndarray:
            # 画面観察でも評価と同じ確定的なPPO行動を使う。
            action, _ = model.predict(obs, deterministic=True)
            return np.asarray(action, dtype=np.float32)

    elif args.policy == "classical":

        def policy(obs: np.ndarray) -> np.ndarray:
            return classical_policy(obs, env)

    else:

        def policy(_obs: np.ndarray) -> np.ndarray:
            # 未学習基準は観測を判断に使わず、再現可能な乱数行動を出す。
            return rng.uniform(-1.0, 1.0, size=2).astype(np.float32)

    print(f"Policy: {args.policy}")
    print(f"Target {args.target}: ({target[0]:.2f}, {target[1]:.2f})")
    print(f"Initial distance: {info['distance']:.6f} m")
    print("The initial pose stays still for 5 seconds.")

    terminated = truncated = False
    # passive viewerを使い、シミュレーションの進行はこのプログラム側で管理する。
    with mujoco.viewer.launch_passive(
        env.model, env.data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = env.model.camera("overview").id
        viewer.sync()
        time.sleep(5)

        # 画面表示だけを追加し、評価と同じenv.step()で制御する。
        while viewer.is_running() and not (terminated or truncated):
            action = policy(observation)
            observation, _, terminated, truncated, info = env.step(action)
            viewer.sync()
            time.sleep(0.04)

        print(f"Final distance: {info['distance']:.6f} m")
        print(f"Reached 3 cm: {info['distance'] < 0.03}")
        print("Inspect the final pose, then close the viewer window.")
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.05)
    env.close()


if __name__ == "__main__":
    main()
