import torch
import torch.nn as nn
from torchvision import models

CKPT = "models/best_model_02.pt"
OUT  = "models/best_model_02.ptl"  # Androidで使うファイル

def build_model(num_classes: int):
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

ckpt = torch.load(CKPT, map_location="cpu")
classes = ckpt["classes"]
model = build_model(len(classes))
model.load_state_dict(ckpt["model"])
model.eval()

# TorchScript化（入力shape固定のほうが安定）
example = torch.randn(1, 3, 224, 224)
scripted = torch.jit.trace(model, example)
scripted.save(OUT)

print("saved:", OUT)
print("classes:", classes)
