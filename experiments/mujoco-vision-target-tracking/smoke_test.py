from __future__ import annotations

"""画像を保存せず、A～Cのレンダリングとラベルを技術確認する。"""

from pathlib import Path

import mujoco
import numpy as np
from PIL import Image
import torch

from arm_control import ArmSimulation
from handcrafted_detector import detect_target
from image_conditions import add_image_noise, apply_condition, sample_condition
from model import TargetCoordinateCNN


MODEL_PATH = Path(__file__).with_name("scene.xml")


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    rng = np.random.default_rng(7)

    with mujoco.Renderer(model, height=64, width=64) as renderer:
        for index, group in enumerate(("A", "B", "C")):
            condition = sample_condition(group, rng, index)
            apply_condition(model, condition, rng)
            mujoco.mj_resetData(model, data)
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera="vision")
            image = add_image_noise(renderer.render(), condition.noise_std, rng)
            assert image.shape == (64, 64, 3)
            assert image.dtype == np.uint8
            assert np.isfinite([condition.target_x, condition.target_y]).all()

    # 調整用画像1枚だけで手書き方式の戻り値形式を確認し、テスト集計は行わない。
    tuning_image = Path(__file__).with_name("dataset") / "handcrafted_tuning" / "00000.png"
    if tuning_image.exists():
        detection = detect_target(np.asarray(Image.open(tuning_image).convert("RGB")))
        assert detection is None or np.isfinite([detection.world_x, detection.world_y]).all()

    # 未学習CNNの入出力形状だけを確認し、値や精度は評価しない。
    cnn = TargetCoordinateCNN()
    with torch.no_grad():
        output = cnn(torch.zeros((1, 3, 64, 64), dtype=torch.float32))
    assert output.shape == (1, 2)

    # 全件評価は行わず、真の目標と異なる推定位置を1制御周期だけ処理できるか確認する。
    arm = ArmSimulation((0.65, 0.55), (0.60, 0.50))
    assert np.allclose(arm.true_target, (0.65, 0.55))
    assert np.allclose(arm.estimate, (0.60, 0.50))
    arm.control_step()
    assert np.isfinite(arm.end_effector).all()

    print("Technical rendering check passed (no dataset or experiment result was saved).")


if __name__ == "__main__":
    main()
