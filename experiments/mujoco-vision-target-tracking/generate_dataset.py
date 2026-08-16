from __future__ import annotations

"""条件A～Cの画像とMuJoCo内部の正解座標ラベルを生成する。"""

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from image_conditions import (
    add_image_noise,
    apply_condition,
    sample_condition,
    sample_target,
)


MODEL_PATH = Path(__file__).with_name("scene.xml")
DATASET_DIR = Path(__file__).with_name("dataset")
IMAGE_SIZE = 64

SPLITS = (
    ("handcrafted_tuning", "B", 50, 1101, None),
    ("train", "B", 2_000, 2201, None),
    ("validation", "B", 200, 3301, None),
    # 3テストは同じtarget_seedを使い、画像条件だけを変えて比較する。
    ("test_a", "A", 200, 4401, 8801),
    ("test_b", "B", 200, 5501, 8801),
    ("test_c", "C", 200, 6601, 8801),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preview",
        action="store_true",
        help="各条件1枚だけpreview/へ保存し、全データセットは生成しない。",
    )
    return parser.parse_args()


def generate_split(
    renderer: mujoco.Renderer,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    split: str,
    group: str,
    count: int,
    seed: int,
    target_seed: int | None,
    output_root: Path,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    target_rng = None if target_seed is None else np.random.default_rng(target_seed)
    image_dir = output_root / split
    image_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for index in range(count):
        target = None if target_rng is None else sample_target(target_rng)
        condition = sample_condition(group, rng, index, target_override=target)
        # 3枚のプレビューではBとCの妨害物を必ず見せ、条件差を確認しやすくする。
        if split in {"preview_b", "preview_c"}:
            condition = replace(condition, distractor_count=max(2, condition.distractor_count))
        apply_condition(model, condition, rng)
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera="vision")
        image = add_image_noise(renderer.render(), condition.noise_std, rng)
        relative_path = Path(split) / f"{index:05d}.png"
        Image.fromarray(image).save(output_root / relative_path)
        rows.append(
            {
                "image": relative_path.as_posix(),
                "split": split,
                "condition_group": group,
                "target_x": condition.target_x,
                "target_y": condition.target_y,
                "seed": seed,
                "sample_index": index,
                "condition_json": json.dumps(
                    condition.metadata(), ensure_ascii=False, separators=(",", ":")
                ),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    output_root = DATASET_DIR / "preview" if args.preview else DATASET_DIR
    selected_splits = (
        (
            ("preview_a", "A", 1, 7701, 9901),
            ("preview_b", "B", 1, 7702, 9901),
            ("preview_c", "C", 1, 7703, 9901),
        )
        if args.preview
        else SPLITS
    )

    all_rows: list[dict[str, object]] = []
    with mujoco.Renderer(model, height=IMAGE_SIZE, width=IMAGE_SIZE) as renderer:
        for split, group, count, seed, target_seed in selected_splits:
            print(f"Generating {split}: condition {group}, {count} images")
            all_rows.extend(
                generate_split(
                    renderer,
                    model,
                    data,
                    split,
                    group,
                    count,
                    seed,
                    target_seed,
                    output_root,
                )
            )

    metadata_path = output_root / "metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Saved metadata: {metadata_path}")
    print(f"Generated images: {len(all_rows)}")


if __name__ == "__main__":
    main()
