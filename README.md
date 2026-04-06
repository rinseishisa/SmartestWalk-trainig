# SmartestWalk-training

歩行方向（LEFT / RIGHT / STRAIGHT）を分類する画像・動画分類モデルの学習・推論・エクスポートスクリプト集。  
学習済みモデルは TorchScript 形式で Android アプリ（SmartestWalk-android）に組み込む。

## 必要環境

- Python 3.8+
- PyTorch / torchvision
- OpenCV (`cv2`)
- Pillow

## ディレクトリ構成

```
src/
├── dataset/
│   └── dataset_images/      # 学習用画像（クラス別フォルダ）
│       ├── LEFT/
│       ├── RIGHT/
│       └── STRAIGHT/
├── models/                  # 保存済みチェックポイント（.pt / .ptl）
├── train_images.py          # 画像分類モデルの学習（メイン）
├── train.py                 # シンプルな動画モデルの学習
├── infer.py                 # 動画ファイル 1 本を推論
├── infer_folder.py          # 画像フォルダをバッチ推論 → CSV 出力
├── dataset.py               # 画像フォルダをバッチ推論（旧版）
├── export_tirchscript.py    # モデルを TorchScript(.ptl)にエクスポート
└── show_content.py          # チェックポイントの中身を確認するユーティリティ
```

## データセット

`src/dataset/dataset_images/` 以下をクラス名のフォルダで分けて配置する（`torchvision.datasets.ImageFolder` 形式）。

```
dataset_images/
├── LEFT/
│   ├── MOVE_LEFT_01_000001.jpg
│   └── ...
├── RIGHT/
│   └── ...
└── STRAIGHT/
    └── ...
```

## 学習

### 画像分類（メイン）

ResNet18 を ImageNet 事前学習済み重みで初期化し、最終層を 3 クラスに置き換えてファインチューニングする。

```bash
python src/train_images.py \
  --data_dir   src/dataset/dataset_images \
  --out_dir    src/models \
  --epochs     10 \
  --batch_size 32 \
  --lr         1e-4
```

バリデーション精度が更新されるたびに `src/models/best.pt` を上書き保存する。

### 動画モデル（実験用）

フレームを時間方向に平均するシンプルな CNN で動画を分類する。

```bash
python src/train.py \
  --data_dir   src/dataset \
  --epochs     5 \
  --batch_size 2
```

## 推論

### 動画 1 本を推論

```bash
python src/infer.py \
  --model_path src/models/best.pt \
  --video_path path/to/video.mp4
```

`--data_dir` と `--index` を指定してディレクトリ内の動画を選ぶこともできる。

```bash
python src/infer.py \
  --model_path src/models/best.pt \
  --data_dir   path/to/videos \
  --index      0
```

### 画像フォルダをバッチ推論

```bash
python src/infer_folder.py \
  --ckpt      src/models/best.pt \
  --image_dir src/dataset/dataset_test \
  --out_csv   predictions.csv
```

結果は CSV（`filename`, `pred`, `prob`）に出力される。

## Android 向けエクスポート

`export_tirchscript.py` 内の `CKPT`（入力 `.pt`）と `OUT`（出力 `.ptl`）のパスを編集してから実行する。

```python
# export_tirchscript.py
CKPT = "models/best_model_02.pt"
OUT  = "models/best_model_02.ptl"
```

```bash
python src/export_tirchscript.py
```

生成された `.ptl` ファイルを SmartestWalk-android プロジェクトに組み込む。

## クラス定義

| ラベル   | 内容           |
|----------|----------------|
| LEFT     | 左方向への移動 |
| RIGHT    | 右方向への移動 |
| STRAIGHT | 直進           |
