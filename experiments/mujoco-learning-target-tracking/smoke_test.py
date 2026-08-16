from __future__ import annotations

"""開発用の技術確認。本人の実験や成功率評価として扱わない。"""

import numpy as np
from gymnasium.utils.env_checker import check_env

from environment import FIXED_EVALUATION_TARGETS, TwoLinkReachEnv


def main() -> None:
    env = TwoLinkReachEnv()
    # Gymnasium APIの戻り値、空間、seed処理などの基本契約を検査する。
    check_env(env, skip_render_check=True)

    observation, info = env.reset(seed=7, options={"target": FIXED_EVALUATION_TARGETS[0]})
    assert observation.shape == (8,)
    assert np.isfinite(observation).all()
    assert np.isfinite(info["distance"])

    # 学習や成功率測定は行わず、5ステップだけ数値異常がないことを確認する。
    for _ in range(5):
        observation, reward, terminated, truncated, info = env.step(
            env.action_space.sample()
        )
        assert np.isfinite(observation).all()
        assert np.isfinite(reward)
        assert np.isfinite(info["distance"])
        if terminated or truncated:
            break

    env.close()
    print("Technical environment check passed (no experiment result was evaluated).")


if __name__ == "__main__":
    main()
