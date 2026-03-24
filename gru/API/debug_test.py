#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Debug script: Compare differences between original test and API implementation
"""

import sys
import os
import torch

# Add project root directory to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from PIL import Image
from dataset import _to_tensor
from unet import UNet_GRU
from API.preprocess import preprocess_image

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model
model_path = "./gru/output4/gru_best.pt"
model = UNet_GRU(in_chns=4, hidden_dim=256, gru_layers=2, dropout=0.1).to(device)

# Load weights
checkpoint = torch.load(model_path, map_location=device)
if isinstance(checkpoint, dict):
    if "model" in checkpoint and isinstance(checkpoint["model"], dict):
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
else:
    state_dict = checkpoint

model.load_state_dict(state_dict, strict=False)
model.eval()

# Test image
image_path = "/home/zsh107552403865/新slam/LSG-SLAM/gru/data/room/frame000049.jpg"

print("\n" + "="*60)
print("Method 1: Original test script approach")
print("="*60)

with Image.open(image_path) as img:
    img = img.convert("RGB")
    rgb = _to_tensor(img)  # C, H, W

# Downsample to 16x8 and calculate grayscale
rgb_16x8 = torch.nn.functional.interpolate(
    rgb.unsqueeze(0), size=(16, 8), mode="bilinear", align_corners=False
).squeeze(0)

# Calculate grayscale
gray_small = (
    0.299 * rgb_16x8[0] + 0.587 * rgb_16x8[1] + 0.114 * rgb_16x8[2]
)  # (16, 8)
gray_16x1 = gray_small.mean(dim=1, keepdim=True)  # (16, 1)

rgb_16x8 = rgb_16x8.unsqueeze(0).to(device)  # (1, 3, 16, 8)
gray_16x1 = gray_16x1.unsqueeze(0).to(device)  # (1, 16, 1)

print(f"rgb_16x8 shape: {rgb_16x8.shape}")
print(f"gray_16x1 shape: {gray_16x1.shape}")

# GRU model input format
gray_expanded = gray_16x1.unsqueeze(1).unsqueeze(2).repeat(1, 1, 1, 1, 8)  # (1, 1, 1, 16, 8)
x_original = torch.cat([rgb_16x8.unsqueeze(1), gray_expanded], dim=2)  # (1, 1, 4, 16, 8)

print(f"x_original shape: {x_original.shape}")
print(f"x_original min: {x_original.min().item():.6f}, max: {x_original.max().item():.6f}")

with torch.no_grad():
    affine_pred_orig, tone_pred_orig, hidden = model(x_original, None)
    print(f"\nModel output shapes:")
    print(f"  affine_pred_orig: {affine_pred_orig.shape}")
    print(f"  tone_pred_orig: {tone_pred_orig.shape}")
    print(f"  affine_pred_orig value: {affine_pred_orig[0].cpu().numpy()}")
    print(f"  tone_pred_orig value: {tone_pred_orig[0].cpu().numpy()}")

print("\n" + "="*60)
print("Method 2: API encapsulated approach")
print("="*60)

x_api = preprocess_image(image_path)  # (1, 1, 4, 16, 8)
x_api = x_api.to(device)

print(f"x_api shape: {x_api.shape}")
print(f"x_api min: {x_api.min().item():.6f}, max: {x_api.max().item():.6f}")

# Check if they are the same
print(f"\nAre inputs the same: {torch.allclose(x_original, x_api, atol=1e-6)}")
if not torch.allclose(x_original, x_api, atol=1e-6):
    print(f"Difference: {(x_original - x_api).abs().max().item():.6f}")

with torch.no_grad():
    affine_pred_api, tone_pred_api, hidden = model(x_api, None)
    print(f"\nModel output shapes:")
    print(f"  affine_pred_api: {affine_pred_api.shape}")
    print(f"  tone_pred_api: {tone_pred_api.shape}")
    print(f"  affine_pred_api value: {affine_pred_api[0].cpu().numpy()}")
    print(f"  tone_pred_api value: {tone_pred_api[0].cpu().numpy()}")

print("\n" + "="*60)
print("Comparison results")
print("="*60)
print(f"Are affine predictions the same: {torch.allclose(affine_pred_orig, affine_pred_api, atol=1e-6)}")
print(f"Are tone predictions the same: {torch.allclose(tone_pred_orig, tone_pred_api, atol=1e-6)}")

