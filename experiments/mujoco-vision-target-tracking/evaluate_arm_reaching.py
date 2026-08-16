from __future__ import annotations

"""3種類の知覚座標を同じ古典制御へ渡し、最終的な到達精度を比較する。"""

import csv
from pathlib import Path
from statistics import mean, median

from arm_control import ArmSimulation


OUTPUT_DIR = Path(__file__).with_name("outputs")
PERCEPTION_PATH = OUTPUT_DIR / "perception-evaluation.csv"
METHOD_LABELS = {
    "ground_truth": "ground_truth",
    "handcrafted": "rule_based",
    "cnn": "cnn",
}


def main() -> None:
    if not PERCEPTION_PATH.exists():
        raise SystemExit("知覚評価がありません。先にevaluate_perception.pyを実行してください。")
    with PERCEPTION_PATH.open(encoding="utf-8") as file:
        perception_rows = list(csv.DictReader(file))

    results: list[dict[str, object]] = []
    for row in perception_rows:
        # 未検出は架空の座標で補わず、指令なしとして到達失敗に含める。
        detected = row["detected"] == "True"
        estimate = (
            (float(row["predicted_x"]), float(row["predicted_y"])) if detected else None
        )
        true_target = (float(row["true_x"]), float(row["true_y"]))
        # 3方式で変えるのは推定座標だけで、アームと古典制御器は共通にする。
        simulation = ArmSimulation(true_target, estimate)
        reaching = simulation.run()
        results.append(
            {
                "image": row["image"],
                "split": row["split"],
                "condition_group": row["condition_group"],
                "method": METHOD_LABELS[row["method"]],
                "detected": detected,
                "true_x": true_target[0],
                "true_y": true_target[1],
                "command_x": "" if estimate is None else estimate[0],
                "command_y": "" if estimate is None else estimate[1],
                **reaching,
            }
        )

    output_path = OUTPUT_DIR / "arm-reaching-evaluation.csv"
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    for split in ("test_a", "test_b", "test_c"):
        for method in ("ground_truth", "rule_based", "cnn"):
            # 知覚誤差を最終的な手先距離へ変換し、実システムへの影響として比較する。
            selected = [r for r in results if r["split"] == split and r["method"] == method]
            distances = [float(r["final_distance"]) for r in selected]
            detected_count = sum(bool(r["detected"]) for r in selected)
            success_3cm = sum(bool(r["success_3cm"]) for r in selected)
            success_1cm = sum(bool(r["success_1cm"]) for r in selected)
            print(
                f"{split} {method}: detected={detected_count}/{len(selected)}, "
                f"mean={mean(distances):.6f}m, median={median(distances):.6f}m, "
                f"max={max(distances):.6f}m, 3cm={success_3cm}/{len(selected)}, "
                f"1cm={success_1cm}/{len(selected)}"
            )
    print(f"Details: {output_path}")


if __name__ == "__main__":
    main()
