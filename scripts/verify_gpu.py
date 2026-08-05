import torch
from ultralytics import checks

# Verify PyTorch sees your 6GB VRAM GPU
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)} GB")

# Verify Ultralytics environment
checks()