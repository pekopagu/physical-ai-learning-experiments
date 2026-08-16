from __future__ import annotations

"""小型CNNを正解座標で教師あり学習する。"""

import argparse
import csv
from pathlib import Path
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from model import TargetCoordinateCNN
from vision_dataset import TargetImageDataset


OUTPUT_DIR = Path(__file__).with_name("outputs")
PREDICTIONS_PATH = Path(__file__).with_name("predictions.md")


def require_predictions() -> None:
    # 本人の実行前予想を保存してから学習し、結果を見た後の予想にならないようにする。
    if "prediction_status: completed" not in PREDICTIONS_PATH.read_text(encoding="utf-8"):
        raise SystemExit("学習前にpredictions.mdを本人が記入してください。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def mean_loss(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, training: bool,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    # training=Trueのときだけ勾配計算と重み更新を行う。
    # 検証時は同じ損失計算を使うが、モデルの重みは変更しない。
    model.train(training)
    total_loss = 0.0
    total_samples = 0
    for images, targets in loader:
        if training:
            assert optimizer is not None
            optimizer.zero_grad()
        predictions = model(images)
        # CNNが出した(x, y)と正解座標の二乗誤差を学習の損失とする。
        loss = criterion(predictions, targets)
        if training:
            loss.backward()
            optimizer.step()
        total_loss += float(loss.item()) * images.shape[0]
        total_samples += images.shape[0]
    return total_loss / total_samples


def main() -> None:
    args = parse_args()
    require_predictions()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # CPU負荷がPC操作を妨げにくいよう、PyTorchの使用スレッド数を最大8へ抑える。
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))

    # trainは重み更新用、validationは未使用画像で学習途中の汎化を確認するために使う。
    train_loader = DataLoader(
        TargetImageDataset("train"), batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    validation_loader = DataLoader(
        TargetImageDataset("validation"), batch_size=args.batch_size, shuffle=False
    )
    model = TargetCoordinateCNN().to("cpu")
    # 座標回帰なので分類用の交差エントロピーではなく平均二乗誤差を使う。
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        # 1エポックは訓練画像2,000枚を一巡する単位である。
        training_loss = mean_loss(model, train_loader, criterion, True, optimizer)
        with torch.no_grad():
            # validation_lossは学習に使わず、未知画像でも誤差が下がるかを見る。
            validation_loss = mean_loss(model, validation_loader, criterion, False)
        history.append(
            {"epoch": epoch, "train_loss": training_loss, "validation_loss": validation_loss}
        )
        print(
            f"epoch={epoch:02d} train_loss={training_loss:.6f} "
            f"validation_loss={validation_loss:.6f}"
        )

    elapsed = time.perf_counter() - started
    OUTPUT_DIR.mkdir(exist_ok=True)
    torch.save(
        # 重みだけでなく学習条件も保存し、後から実験条件を確認できるようにする。
        {
            "model_state_dict": model.state_dict(),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
        },
        OUTPUT_DIR / "target_cnn.pt",
    )
    with (OUTPUT_DIR / "training-history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    epochs = [int(row["epoch"]) for row in history]
    # 数値CSVに加え、序盤から終盤までの損失変化を目視できるグラフも保存する。
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [float(row["train_loss"]) for row in history], label="train")
    plt.plot(epochs, [float(row["validation_loss"]) for row in history], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Mean squared error")
    plt.title("CNN coordinate regression (CPU)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training-curve.png", dpi=140)
    plt.close()
    (OUTPUT_DIR / "training-time.txt").write_text(
        f"elapsed_seconds={elapsed:.3f}\n", encoding="utf-8"
    )
    print(f"Training complete in {elapsed:.1f} seconds (CPU)")
    print(f"Model: {OUTPUT_DIR / 'target_cnn.pt'}")


if __name__ == "__main__":
    main()
