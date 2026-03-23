import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

# labels.csv方式ではなく、ファイル名から推定 or --video_path で指定する
LABEL_MAP = {"LEFT": 0, "RIGHT": 1, "STRAIGHT": 2}
ID2LABEL = {v: k for k, v in LABEL_MAP.items()}

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".MOV", ".MP4"}


def infer_label_from_name(fname: str) -> str:
    name = fname.upper()
    if "RIGHT" in name:
        return "RIGHT"
    if "LEFT" in name:
        return "LEFT"
    if "STRAIGHT" in name:
        return "STRAIGHT"
    return "UNKNOWN"


class FrameAvgModel(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        backbone = models.resnet18(weights=None)  # 推論ではweights不要（state_dictで上書きする）
        self.feature = nn.Sequential(*list(backbone.children())[:-1])  # (B,512,1,1)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        # x: (B,T,C,H,W)
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        feat = self.feature(x).flatten(1)          # (B*T,512)
        feat = feat.view(b, t, -1).mean(dim=1)     # (B,512)
        return self.classifier(feat)


def sample_frames(video_path: str, num_frames: int):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"動画が開けない: {video_path}")

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if length <= 0:
        cap.release()
        raise RuntimeError(f"フレーム数が取得できない: {video_path}")

    idxs = np.linspace(0, max(0, length - 1), num_frames).astype(int)
    targets = set(idxs.tolist())

    frames = []
    cur = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if cur in targets:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cur += 1
    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"フレーム抽出に失敗: {video_path}")

    while len(frames) < num_frames:
        frames.append(frames[-1])
    return frames[:num_frames]


def build_transform(img_size: int):
    # 学習時と同じ
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def load_video_tensor(video_path: str, num_frames: int, img_size: int):
    tf = build_transform(img_size)
    frames = sample_frames(video_path, num_frames)
    x = torch.stack([tf(f) for f in frames], dim=0)  # (T,C,H,W)
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)

    # どちらかで指定：
    parser.add_argument("--video_path", type=str, default=None, help="推論したい動画ファイル1本")
    parser.add_argument("--data_dir", type=str, default=None, help="動画が入ったディレクトリ（中の1本を選ぶ）")
    parser.add_argument("--index", type=int, default=0, help="data_dir内のソート順で何番目を推論するか")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    ckpt = torch.load(args.model_path, map_location=device)
    num_frames = int(ckpt.get("num_frames", 8))
    img_size = int(ckpt.get("img_size", 224))
    label_map = ckpt.get("label_map", LABEL_MAP)

    print("ckpt num_frames:", num_frames, "img_size:", img_size)
    print("ckpt label_map:", label_map)

    # 推論対象の動画を決める
    if args.video_path is not None:
        video_path = args.video_path
    else:
        if args.data_dir is None:
            raise SystemExit("ERROR: --video_path か --data_dir のどちらかを指定してください")
        p = Path(args.data_dir)
        files = sorted([str(x) for x in p.glob("*") if x.suffix in VIDEO_EXTS or x.suffix.lower() in VIDEO_EXTS])
        if len(files) == 0:
            raise SystemExit(f"動画が見つかりません: {p}")
        if not (0 <= args.index < len(files)):
            raise SystemExit(f"--index が範囲外です: index={args.index}, files={len(files)}")
        video_path = files[args.index]

    print("video_path:", video_path)

    # 動画→テンソル
    x = load_video_tensor(video_path, num_frames=num_frames, img_size=img_size)
    x = x.unsqueeze(0).to(device)  # (1,T,C,H,W)

    # モデル構築 & ロード
    model = FrameAvgModel(num_classes=3).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    with torch.no_grad():
        logits = model(x)
        pred = int(logits.argmax(1).item())

    pred_label = ID2LABEL[pred]
    gt_label = infer_label_from_name(Path(video_path).name)

    print("GT (from filename):", gt_label)
    print("Pred:", pred_label)
    print("Logits:", logits.detach().cpu().numpy())


if __name__ == "__main__":
    main()
