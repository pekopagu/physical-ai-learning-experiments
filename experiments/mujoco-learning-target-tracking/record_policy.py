from __future__ import annotations

"""未学習、PPO、古典制御の動作をMP4動画として保存する。"""

import argparse
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from stable_baselines3 import PPO

from environment import FIXED_EVALUATION_TARGETS, TwoLinkReachEnv
from experiment_support import classical_policy, require_predictions


OUTPUT_DIR = Path(__file__).with_name("outputs")
MODEL_PATH = OUTPUT_DIR / "ppo_reacher.zip"
FPS = 25
INITIAL_PAUSE_FRAMES = 5 * FPS
DEFAULT_FRAME_REPEAT = 2
DEFAULT_FINAL_PAUSE_SECONDS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy", choices=("random", "ppo", "classical"))
    parser.add_argument("--target", type=int, default=1, choices=range(1, 21))
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument(
        "--frame-repeat",
        type=int,
        default=DEFAULT_FRAME_REPEAT,
        help="Repeat each motion frame to slow playback without changing the episode.",
    )
    parser.add_argument(
        "--final-pause",
        type=float,
        default=DEFAULT_FINAL_PAUSE_SECONDS,
        help="Seconds to hold the final frame.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frame_repeat < 1:
        raise SystemExit("--frame-repeat は1以上を指定してください。")
    if args.final_pause < 0:
        raise SystemExit("--final-pause は0以上を指定してください。")
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
            action, _ = model.predict(obs, deterministic=True)
            return np.asarray(action, dtype=np.float32)

    elif args.policy == "classical":

        def policy(obs: np.ndarray) -> np.ndarray:
            return classical_policy(obs, env)

    else:

        def policy(_obs: np.ndarray) -> np.ndarray:
            return rng.uniform(-1.0, 1.0, size=2).astype(np.float32)

    video_dir = OUTPUT_DIR / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    output_path = video_dir / f"{args.policy}-target-{args.target:02d}.mp4"

    terminated = truncated = False
    # オフスクリーン描画により、viewer画面を録画せず同一カメラの画像を直接得る。
    with mujoco.Renderer(env.model, height=760, width=1200) as renderer:
        renderer.update_scene(env.data, camera="overview")
        initial_frame = renderer.render()
        with imageio.get_writer(
            output_path,
            fps=FPS,
            codec="libx264",
            quality=8,
            macro_block_size=None,
        ) as writer:
            # 動作開始前の位置関係を確認できるよう、初期姿勢を5秒保持する。
            for _ in range(INITIAL_PAUSE_FRAMES):
                writer.append_data(initial_frame)

            while not (terminated or truncated):
                action = policy(observation)
                observation, _, terminated, truncated, info = env.step(action)
                renderer.update_scene(env.data, camera="overview")
                frame = renderer.render()
                # 同じフレームを繰り返して再生だけを遅くし、制御結果は変えない。
                for _ in range(args.frame_repeat):
                    writer.append_data(frame)

            # 動作終了直後に動画が切れないよう、最終姿勢を指定秒数保持する。
            final_frame_count = round(args.final_pause * FPS)
            for _ in range(final_frame_count):
                writer.append_data(frame)

    env.close()
    print(f"Saved video: {output_path}")
    print(f"Initial distance: {np.linalg.norm(np.array(target) - np.array([1.05, 0.0])):.6f} m")
    print(f"Final distance: {info['distance']:.6f} m")
    print(f"Reached 3 cm: {info['distance'] < 0.03}")
    print(
        f"Playback: {args.frame_repeat}x slower motion, "
        f"{args.final_pause:g}s final pause"
    )


if __name__ == "__main__":
    main()
