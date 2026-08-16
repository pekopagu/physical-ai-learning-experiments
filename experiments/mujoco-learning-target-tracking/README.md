# MuJoCo学習ベース目標追従実験

`EXPERIMENT-001` の古典制御と、PPOで学習する方策を同じ2関節アームで比較するための独立環境です。

観測、行動、報酬、安全制限、固定評価目標に加え、CPUでのPPO学習、3方式の評価、学習曲線、画面表示を含みます。

## 現在実装されているもの

- 観測: 2関節角のsin/cos、角速度、目標までのx/y差
- 行動: 2関節の目標角度の制限付き増分
- 報酬: 距離、行動量ペナルティ、3cm未満の到達ボーナス
- 最大100制御ステップ
- 到達可能範囲からのランダム目標
- 比較に使用する固定20目標
- Gymnasium APIと数値の開発用チェック
- CPU固定のPPO学習
- 未学習方策、PPO、古典制御の固定20目標評価
- 学習曲線の画像保存
- 3方式を同じ固定目標で見る画面表示

## 重要な区別

`smoke_test.py` は環境が壊れていないことを開発者が確かめるだけのものです。成功率や学習結果を評価せず、本人の実験完了として扱いません。

本人による実験は、予想項目を記入してから未学習方策、学習、学習後方策、古典制御の順に実行します。

## 本人が実行する順序

### 1. 実行前の予想

`predictions.md` の8項目を記入し、先頭の状態を次のように変更します。

```text
prediction_status: completed
```

未記入のままでは、学習、評価、画面表示は開始できません。

### 2. 未学習方策を画面で確認

```text
uv run python view_policy.py random --target 1
```

最初の5秒で初期姿勢を確認し、その後の動き、目標への近づき方、振動や回り道を観察します。最後は本人がウィンドウを閉じます。

### 2.1 未学習方策を動画で保存

画面確認と同じ目標1、乱数シードを使い、初期姿勢5秒と動作をMP4へ保存します。既定では動作を半速にし、終了姿勢を5秒保持します。評価する100制御ステップと最終距離は変わりません。

```text
uv run python record_policy.py random --target 1
```

さらにゆっくり見たい場合は、各動作フレームの繰り返し数と最終保持秒数を指定できます。

```text
uv run python record_policy.py random --target 1 --frame-repeat 3 --final-pause 8
```

保存先:

```text
outputs/videos/random-target-01.mp4
```

学習後と古典制御も、同じ形式で保存できます。

```text
uv run python record_policy.py ppo --target 1
uv run python record_policy.py classical --target 1
```

### 3. CPUでPPOを学習

```text
uv run python train.py
```

既定値は100,000ステップ、乱数シード42、CPU固定です。30分を超えても完了しない場合や、PC操作へ支障が出る場合は中断します。

### 4. 学習曲線を保存

```text
uv run python plot_training.py
```

`outputs/training-curve.png` の移動平均が全体として上昇しているか確認します。

### 5. PPOと古典制御を同じ目標で確認

```text
uv run python view_policy.py ppo --target 1
uv run python view_policy.py classical --target 1
```

必要に応じて `--target 2` から `--target 20` を指定します。中央寄りと範囲端の目標を最低1件ずつ観察します。

### 6. 固定20目標を数値評価

```text
uv run python evaluate.py
```

結果は次へ保存されます。

```text
outputs/evaluation.csv
outputs/evaluation-summary.txt
```

比較する値は平均・最大最終距離、3cm未満と1cm未満の成功数です。

## 開発者による技術確認

```text
uv run python smoke_test.py
```

これは5ステップの有限値確認だけであり、成功率や学習結果を測りません。本人の実験として数えません。
