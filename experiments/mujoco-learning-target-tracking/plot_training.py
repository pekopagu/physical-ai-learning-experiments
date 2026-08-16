from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

from experiment_support import require_predictions


OUTPUT_DIR = Path(__file__).with_name("outputs")
MONITOR_PATH = OUTPUT_DIR / "training.monitor.csv"


def main() -> None:
    require_predictions()
    if not MONITOR_PATH.exists():
        raise SystemExit("学習ログがありません。先に train.py を実行してください。")

    episodes: list[int] = []
    rewards: list[float] = []
    # Monitor CSVの先頭にあるメタデータ行（#）を除き、エピソード報酬を読む。
    with MONITOR_PATH.open(encoding="utf-8") as file:
        reader = csv.DictReader(line for line in file if not line.startswith("#"))
        for index, row in enumerate(reader, start=1):
            episodes.append(index)
            rewards.append(float(row["r"]))

    # 個々の試行はばらつくため、最大50エピソードの移動平均も表示する。
    window = min(50, len(rewards))
    moving = [
        sum(rewards[max(0, index - window + 1) : index + 1])
        / len(rewards[max(0, index - window + 1) : index + 1])
        for index in range(len(rewards))
    ]
    plt.figure(figsize=(9, 5))
    plt.plot(episodes, rewards, alpha=0.25, label="episode reward")
    plt.plot(episodes, moving, label=f"moving average ({window})")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("PPO training curve (CPU)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output = OUTPUT_DIR / "training-curve.png"
    plt.savefig(output, dpi=140)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
