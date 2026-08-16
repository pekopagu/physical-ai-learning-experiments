from __future__ import annotations

"""正解座標、手書き画像処理、CNNを同じA～C画像で評価する。"""

import csv
from pathlib import Path
from statistics import mean, median

import numpy as np
from PIL import Image
import torch

from handcrafted_detector import detect_target
from model import TargetCoordinateCNN
from vision_dataset import DATASET_DIR, load_metadata


OUTPUT_DIR = Path(__file__).with_name("outputs")
MODEL_PATH = OUTPUT_DIR / "target_cnn.pt"
TEST_SPLITS = ("test_a", "test_b", "test_c")


def load_model() -> TargetCoordinateCNN:
    if not MODEL_PATH.exists():
        raise SystemExit("学習済みCNNがありません。先にtrain.pyを実行してください。")
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model = TargetCoordinateCNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    # Dropout等はない小型CNNだが、評価モードを明示して学習処理と分離する。
    model.eval()
    return model


def cnn_predict(model: TargetCoordinateCNN, image: np.ndarray) -> tuple[float, float]:
    # 64×64 RGB画像をCNN入力の[batch, channel, height, width]へ変換する。
    tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    with torch.no_grad():
        # 評価では逆伝播せず、学習済み重みから座標を1回推論する。
        prediction = model(tensor)[0].numpy()
    return float(prediction[0]), float(prediction[1])


def result_row(
    row: dict[str, str], method: str, predicted: tuple[float, float] | None
) -> dict[str, object]:
    true_x = float(row["target_x"])
    true_y = float(row["target_y"])
    if predicted is None:
        # ルールベース方式が候補を発見できなかった場合も、失敗行としてCSVへ残す。
        return {
            "image": row["image"], "split": row["split"], "condition_group": row["condition_group"],
            "method": method, "true_x": true_x, "true_y": true_y,
            "predicted_x": "", "predicted_y": "", "error_m": "",
            "within_3cm": False, "within_1cm": False, "detected": False,
        }
    error = float(np.linalg.norm(np.asarray(predicted) - np.asarray((true_x, true_y))))
    return {
        "image": row["image"], "split": row["split"], "condition_group": row["condition_group"],
        "method": method, "true_x": true_x, "true_y": true_y,
        "predicted_x": predicted[0], "predicted_y": predicted[1], "error_m": error,
        "within_3cm": error < 0.03, "within_1cm": error < 0.01, "detected": True,
    }


def main() -> None:
    model = load_model()
    metadata = [row for row in load_metadata() if row["split"] in TEST_SPLITS]
    rows: list[dict[str, object]] = []

    for row in metadata:
        image = np.asarray(Image.open(DATASET_DIR / row["image"]).convert("RGB"))
        truth = (float(row["target_x"]), float(row["target_y"]))
        # 同じ1枚に正解座標、固定ルール、学習済みCNNの3方式を適用する。
        hand = detect_target(image)
        hand_xy = None if hand is None else (hand.world_x, hand.world_y)
        cnn_xy = cnn_predict(model, image)
        rows.append(result_row(row, "ground_truth", truth))
        rows.append(result_row(row, "handcrafted", hand_xy))
        rows.append(result_row(row, "cnn", cnn_xy))

    OUTPUT_DIR.mkdir(exist_ok=True)
    csv_path = OUTPUT_DIR / "perception-evaluation.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    lines: list[str] = []
    for split in TEST_SPLITS:
        for method in ("ground_truth", "handcrafted", "cnn"):
            # 検出率、代表誤差、最大誤差、精密到達件数を条件・方式ごとに集計する。
            selected = [row for row in rows if row["split"] == split and row["method"] == method]
            detected = [row for row in selected if bool(row["detected"])]
            errors = [float(row["error_m"]) for row in detected]
            within_3cm = sum(bool(row["within_3cm"]) for row in selected)
            within_1cm = sum(bool(row["within_1cm"]) for row in selected)
            line = (
                f"{split} {method}: detected={len(detected)}/{len(selected)}, "
                f"mean={mean(errors):.6f}m, median={median(errors):.6f}m, "
                f"max={max(errors):.6f}m, 3cm={within_3cm}/{len(selected)}, "
                f"1cm={within_1cm}/{len(selected)}"
                if errors
                else f"{split} {method}: detected=0/{len(selected)}"
            )
            lines.append(line)
            print(line)

    summary_path = OUTPUT_DIR / "perception-summary.txt"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Details: {csv_path}")


if __name__ == "__main__":
    main()
