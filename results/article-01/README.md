# フィジカルAIをPCで学ぶ #1 実測結果

## 制御方式の固定20目標評価

| 方式 | 平均最終距離 | 最大最終距離 | 3cm以内 | 1cm以内 |
|---|---:|---:|---:|---:|
| 未学習 | 1.101532m | 1.751687m | 0/20 | 0/20 |
| PPO | 0.689075m | 1.054918m | 1/20 | 0/20 |
| 古典制御 | 0.024523m | 0.029701m | 20/20 | 0/20 |

詳細:

- [`evaluation.csv`](../../experiments/mujoco-learning-target-tracking/outputs/evaluation.csv)
- [`evaluation-summary.txt`](../../experiments/mujoco-learning-target-tracking/outputs/evaluation-summary.txt)
- [`training-curve.png`](../../experiments/mujoco-learning-target-tracking/outputs/training-curve.png)

## 条件Cのアーム到達評価

| 方式 | 平均最終距離 | 最大最終距離 | 3cm以内 | 1cm以内 |
|---|---:|---:|---:|---:|
| 正解座標 | 0.004745m | 0.044363m | 194/200 | 171/200 |
| ルールベース | 0.434341m | 1.819334m | 93/200 | 70/200 |
| CNN | 0.135514m | 1.030744m | 53/200 | 5/200 |

詳細:

- [`perception-evaluation.csv`](../../experiments/mujoco-vision-target-tracking/outputs/perception-evaluation.csv)
- [`arm-reaching-evaluation.csv`](../../experiments/mujoco-vision-target-tracking/outputs/arm-reaching-evaluation.csv)
- [`training-curve.png`](../../experiments/mujoco-vision-target-tracking/outputs/training-curve.png)
- [`prediction-comparisons/`](../../experiments/mujoco-vision-target-tracking/outputs/prediction-comparisons/)

## 代表動画

- [正解座標](https://youtu.be/ygZ20lCNCGo): 最終距離0.000879m
- [ルールベース](https://youtu.be/W-2Yz4nqdgA): 最終距離0.819421m
- [CNN](https://youtu.be/74Q4uafuRI0): 最終距離0.251010m

動画は条件C・画像2を同じ古典制御へ渡した比較です。
