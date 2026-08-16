from __future__ import annotations

"""未学習・PPO・古典制御を固定20目標で比較する。"""

import csv
from pathlib import Path
from statistics import mean

import numpy as np
from stable_baselines3 import PPO

from environment import FIXED_EVALUATION_TARGETS, TwoLinkReachEnv
from experiment_support import classical_policy, require_predictions, run_episode


OUTPUT_DIR = Path(__file__).with_name("outputs")
MODEL_PATH = OUTPUT_DIR / "ppo_reacher.zip"


def main() -> None:
    require_predictions()
    if not MODEL_PATH.exists():
        raise SystemExit("学習済みモデルがありません。先に train.py を実行してください。")

    # 評価ではニューラルネットを更新せず、保存済み方策をCPUへ読み込む。
    model = PPO.load(MODEL_PATH, device="cpu")
    rng = np.random.default_rng(20260815)

    def random_policy(_observation: np.ndarray, _env: TwoLinkReachEnv) -> np.ndarray:
        # 未学習時の基準として、観測に関係なく再現可能な乱数行動を返す。
        return rng.uniform(-1.0, 1.0, size=2).astype(np.float32)

    def learned_policy(observation: np.ndarray, _env: TwoLinkReachEnv) -> np.ndarray:
        # 学習中の探索乱数を使わず、方策分布の代表値で再現可能に評価する。
        action, _ = model.predict(observation, deterministic=True)
        return np.asarray(action, dtype=np.float32)

    policies = {
        "untrained_random": random_policy,
        "ppo": learned_policy,
        "classical": classical_policy,
    }
    rows: list[dict[str, object]] = []
    env = TwoLinkReachEnv()
    # 未学習、PPO、古典制御へ同じ20目標を与える。
    for policy_name, policy in policies.items():
        for target_index, target in enumerate(FIXED_EVALUATION_TARGETS, start=1):
            result = run_episode(env, policy, target, seed=10_000 + target_index)
            rows.append(
                {
                    "policy": policy_name,
                    "target_index": target_index,
                    "target_x": target[0],
                    "target_y": target[1],
                    **result,
                }
            )
    env.close()

    OUTPUT_DIR.mkdir(exist_ok=True)
    csv_path = OUTPUT_DIR / "evaluation.csv"
    # 目標ごとの結果を残し、平均値だけで失敗位置を隠さない。
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    lines = []
    for policy_name in policies:
        # 平均だけでなく最大誤差と成功件数を出し、大外れと精密さを分けて見る。
        selected = [row for row in rows if row["policy"] == policy_name]
        average_distance = mean(float(row["final_distance"]) for row in selected)
        maximum_distance = max(float(row["final_distance"]) for row in selected)
        success_3cm = sum(bool(row["success_3cm"]) for row in selected)
        success_1cm = sum(bool(row["success_1cm"]) for row in selected)
        line = (
            f"{policy_name}: mean={average_distance:.6f}m, "
            f"max={maximum_distance:.6f}m, 3cm={success_3cm}/20, 1cm={success_1cm}/20"
        )
        lines.append(line)
        print(line)

    (OUTPUT_DIR / "evaluation-summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"Details: {csv_path}")


if __name__ == "__main__":
    main()
