from ultralytics import YOLO

model = YOLO("last.pt")

print("=" * 50)
print("📋 YOUR MODEL DETAILS")
print("=" * 50)

# Model type and size
print(f"Model type:      {model.task}")
print(f"Model name:      {model.ckpt_path}")

# Architecture (nano/small/medium/large/xlarge)
num_params = sum(p.numel() for p in model.model.parameters())
print(f"Total parameters: {num_params:,}")

if num_params < 4_000_000:
    size = "YOLOv8n (NANO) — Smallest, least accurate"
elif num_params < 12_000_000:
    size = "YOLOv8s (SMALL) — Fast but limited accuracy"
elif num_params < 30_000_000:
    size = "YOLOv8m (MEDIUM) — Good balance ✅"
elif num_params < 55_000_000:
    size = "YOLOv8l (LARGE) — High accuracy"
else:
    size = "YOLOv8x (XLARGE) — Maximum accuracy"

print(f"Model size:      {size}")

# Classes it can detect
print(f"\nNumber of classes: {len(model.names)}")
print(f"\nAll pest classes this model can detect:")
for idx, name in model.names.items():
    print(f"  [{idx}] {name}")

# Training info (if available)
if hasattr(model, 'ckpt') and model.ckpt:
    meta = model.ckpt.get('train_args', {})
    if meta:
        print(f"\n📊 Training info:")
        print(f"  Epochs trained: {meta.get('epochs', 'Unknown')}")
        print(f"  Image size:     {meta.get('imgsz', 'Unknown')}")
        print(f"  Batch size:     {meta.get('batch', 'Unknown')}")
        print(f"  Dataset:        {meta.get('data', 'Unknown')}")

# File size
import os
file_size = os.path.getsize("last.pt") / (1024 * 1024)
print(f"\nFile size:       {file_size:.1f} MB")
print("=" * 50)