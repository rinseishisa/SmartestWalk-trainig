import argparse
from pathlib import Path
import csv

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

def build_model(arch: str, num_classes: int):
    if arch == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    raise ValueError(f"Unsupported arch: {arch}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True, help="path to best.pt")
    p.add_argument("--image_dir", type=str, required=True, help="directory containing .jpg files")
    p.add_argument("--out_csv", type=str, default="predictions.csv")
    p.add_argument("--glob", type=str, default="*.jpg", help="file pattern, e.g. *.jpg or *.jpeg")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.ckpt, map_location="cpu")
    classes = ckpt.get("classes", None)
    if classes is None:
        raise ValueError("Checkpoint missing 'classes'.")

    img_size = int(ckpt.get("img_size", 224))
    mean = ckpt.get("mean", [0.485, 0.456, 0.406])
    std  = ckpt.get("std",  [0.229, 0.224, 0.225])
    arch = ckpt.get("arch", "resnet18")

    model = build_model(arch, num_classes=len(classes))
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)

    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    image_dir = Path(args.image_dir)
    images = sorted(image_dir.glob(args.glob))
    if len(images) == 0:
        raise FileNotFoundError(f"No images matched: {image_dir}/{args.glob}")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "pred", "prob"])

        with torch.no_grad():
            for img_path in images:
                img = Image.open(img_path).convert("RGB")
                x = tf(img).unsqueeze(0).to(device)

                logits = model(x)
                probs = torch.softmax(logits, dim=1).squeeze(0)

                idx = int(torch.argmax(probs).item())
                writer.writerow([img_path.name, classes[idx], float(probs[idx])])

    print(f"saved: {out_csv}  images={len(images)}")

if __name__ == "__main__":
    main()
