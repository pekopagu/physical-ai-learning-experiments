from __future__ import annotations

"""正解、手書き、CNNの推定位置をテスト画像へ重ねて保存する。"""

import csv
from pathlib import Path

from PIL import Image, ImageDraw

from camera_mapping import world_to_pixel
from vision_dataset import DATASET_DIR


OUTPUT_DIR = Path(__file__).with_name("outputs")
EVALUATION_PATH = OUTPUT_DIR / "perception-evaluation.csv"
COLORS = {"ground_truth": "lime", "handcrafted": "yellow", "cnn": "cyan"}


def main() -> None:
    if not EVALUATION_PATH.exists():
        raise SystemExit("知覚評価がありません。先にevaluate_perception.pyを実行してください。")
    with EVALUATION_PATH.open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    by_image: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        # 評価CSVは1画像につき3方式の行があるため、画像単位へまとめ直す。
        by_image.setdefault(row["image"], []).append(row)

    # 各条件の先頭5枚を固定選択し、都合のよい成功例だけを選ばない。
    selected_images: list[str] = []
    for split in ("test_a", "test_b", "test_c"):
        selected_images.extend(sorted(path for path in by_image if path.startswith(split))[:5])

    comparison_dir = OUTPUT_DIR / "prediction-comparisons"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    for image_path in selected_images:
        image = Image.open(DATASET_DIR / image_path).convert("RGB").resize((512, 512))
        draw = ImageDraw.Draw(image)
        scale = 512 / 64
        for row in by_image[image_path]:
            if row["detected"] != "True":
                # 未検出方式には描画できる座標がないため、マーカーを付けない。
                continue
            px, py = world_to_pixel(float(row["predicted_x"]), float(row["predicted_y"]))
            px *= scale
            py *= scale
            radius = 8 if row["method"] == "ground_truth" else 6
            # 正解を少し大きく描き、同じ位置に重なった推定マーカーも判別しやすくする。
            color = COLORS[row["method"]]
            draw.ellipse((px - radius, py - radius, px + radius, py + radius), outline=color, width=3)
            draw.text((px + 10, py - 8), row["method"], fill=color)
        output_name = image_path.replace("/", "-")
        image.save(comparison_dir / output_name)
    print(f"Saved comparison images: {len(selected_images)}")
    print(f"Directory: {comparison_dir}")


if __name__ == "__main__":
    main()
