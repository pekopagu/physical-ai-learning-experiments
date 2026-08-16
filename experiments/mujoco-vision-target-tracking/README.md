# MuJoCo画像目標位置推定実験

カメラ画像から目標座標を推定し、古典的な逆運動学へ渡すハイブリッド構成を、正解座標・手書き画像処理・小型CNNで比較します。

## 現在の実装段階

実装済み:

- 64×64 RGBの固定正投影カメラ
- 条件A、B、Cの背景、照明、目標、妨害物、ノイズ
- 再現可能な乱数シード
- 画像と正解x、y座標、全生成条件の保存
- 学習前予想の記入場所
- データを保存しないレンダリング技術確認

追加実装済み:

- handcrafted-v1のHSV・面積・円形度による手書き画像処理
- 3畳み込み層の小型CNNとCPU教師あり学習
- 条件A～Cの知覚評価とCSV集計
- 各条件の先頭5枚を固定選択する比較画像生成
- 正解座標、ルールベース推定、CNN推定を同じ古典制御へ渡すアーム到達評価
- 任意の条件・画像・方式を画面で観察するアームビューア

未実装:

- 3方式を1画面へ同時表示する合成動画

## データ構成

| split | 条件 | 枚数 | 用途 |
|---|---|---:|---|
| handcrafted_tuning | B | 50 | 手書き方式の事前調整 |
| train | B | 2,000 | CNN学習 |
| validation | B | 200 | CNN検証 |
| test_a | A | 200 | 単純条件テスト |
| test_b | B | 200 | 学習範囲内テスト |
| test_c | C | 200 | 学習範囲外テスト |

テストA、B、Cは同じ200個の目標座標を使用し、見た目の条件だけを変える。プレビュー3枚も同じ目標座標を使用し、BとCには妨害物を必ず2個以上表示する。

## 重要な境界

`smoke_test.py`は画像を保存せず、レンダリングと数値形式だけを確認します。本人の画像観察、学習、評価には数えません。

本人の実験は`predictions.md`への回答後に開始します。現段階では、本人が予想する前に全データセットを生成しません。

## データセット生成

最初に条件A～Cを各1枚だけ生成し、人間が目標と妨害物を区別できることを確認する。

```text
uv run python generate_dataset.py --preview
```

確認後、学習・検証・テスト用の全2,850枚を生成する。

```text
uv run python generate_dataset.py
```

生成画像と座標ラベルは`dataset/`へ保存される。固定seedを使うため、公開リポジトリには生成済み画像を含めず、このコマンドで再生成する。

## CNN学習

```text
uv run python train.py
```

既定値はCPU、30エポック、batch size 64、学習率0.001、seed 42。学習後にモデル、学習履歴、曲線、時間を`outputs/`へ保存する。

## 知覚評価

```text
uv run python evaluate_perception.py
uv run python visualize_predictions.py
```

テストA、B、Cについて、正解座標、handcrafted-v1、CNNの検出数、平均・中央値・最大誤差、3cm・1cm以内の件数を保存する。

## アーム到達評価

```text
uv run python evaluate_arm_reaching.py
```

赤い本当の目標は固定したまま、3方式の座標を同じ逆運動学へ渡す。未検出も含む各200画像について、最終距離と3cm・1cm到達数を`outputs/arm-reaching-evaluation.csv`へ保存する。

1画像の動きを観察する例:

```text
uv run python view_arm_reaching.py rule_based --condition c --image 1
uv run python view_arm_reaching.py cnn --condition c --image 1
```

画面の赤は本当の目標、紫は画像認識の推定位置、黄色は手先を表す。`ground_truth`、`rule_based`、`cnn`から方式を選べる。

## アーム動画保存

```text
uv run python record_arm_reaching.py ground_truth --condition c --image 2
uv run python record_arm_reaching.py rule_based --condition c --image 2
uv run python record_arm_reaching.py cnn --condition c --image 2
```

開始姿勢5秒、約2倍のスロー動作、最終姿勢5秒を含むMP4を`outputs/videos/`へ保存する。
