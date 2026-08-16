from __future__ import annotations

"""2関節アームの目標追従方策をPPOでCPU学習する。"""

import argparse
from pathlib import Path
import time

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from environment import TwoLinkReachEnv
from experiment_support import require_predictions


OUTPUT_DIR = Path(__file__).with_name("outputs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the PPO reacher policy on CPU.")
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # 本人が実行前予想を記録するまで、結果を生成して予想へ影響させない。
    require_predictions()
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Monitorは各エピソードの合計報酬と長さをCSVへ保存する。
    env = Monitor(TwoLinkReachEnv(), str(OUTPUT_DIR / "training"))
    # MlpPolicyは8要素の数値観測を入力するActor-Critic型ニューラルネット。
    # device="cpu"を明示し、GPUやCUDA、Apple MPSは今回使用しない。
    model = PPO(
        "MlpPolicy",
        env,
        seed=args.seed,
        device="cpu",
        verbose=1,
        tensorboard_log=None,
    )
    started = time.perf_counter()
    # PPOは既定で2,048ステップ収集するごとに方策を更新する。
    # そのため指定値100,000に対し、実際は100,352ステップになる。
    model.learn(total_timesteps=args.steps, progress_bar=False)
    elapsed = time.perf_counter() - started
    # 学習済みのActor、Critic、正規化情報などをzipとして保存する。
    model.save(OUTPUT_DIR / "ppo_reacher")
    (OUTPUT_DIR / "training-time.txt").write_text(
        f"steps={args.steps}\nseed={args.seed}\nelapsed_seconds={elapsed:.3f}\n",
        encoding="utf-8",
    )
    env.close()
    print(f"Training complete: {args.steps} steps in {elapsed:.1f} seconds (CPU)")
    print(f"Model: {OUTPUT_DIR / 'ppo_reacher.zip'}")


if __name__ == "__main__":
    main()
