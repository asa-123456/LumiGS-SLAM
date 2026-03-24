#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证输入格式是否与原始测试脚本一致
"""

import sys
import os
import torch
from PIL import Image

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dataset import _to_tensor
from API.preprocess import preprocess_image

# 测试图片
image_path = "./gru/data/room/frame000049.jpg"

print("="*60)
print("原始测试脚本的预处理方式")
print("="*60)

with Image.open(image_path) as img:
    img = img.convert("RGB")
    rgb = _to_tensor(img)  # C, H, W

# 下采样到16x8并计算灰度
rgb_16x8 = torch.nn.functional.interpolate(
    rgb.unsqueeze(0), size=(16, 8), mode="bilinear", align_corners=False
).squeeze(0)

# 计算灰度
gray_small = (
    0.299 * rgb_16x8[0] + 0.587 * rgb_16x8[1] + 0.114 * rgb_16x8[2]
)  # (16, 8)
gray_16x1 = gray_small.mean(dim=1, keepdim=True)  # (16, 1)

rgb_16x8 = rgb_16x8.unsqueeze(0)  # (1, 3, 16, 8)
gray_16x1 = gray_16x1.unsqueeze(0)  # (1, 16, 1)

print(f"rgb_16x8 shape: {rgb_16x8.shape}")
print(f"gray_16x1 shape: {gray_16x1.shape}")

# GRU模型输入格式
gray_expanded = gray_16x1.unsqueeze(1).unsqueeze(2).repeat(1, 1, 1, 1, 8)  # (1, 1, 1, 16, 8)
x_original = torch.cat([rgb_16x8.unsqueeze(1), gray_expanded], dim=2)  # (1, 1, 4, 16, 8)

print(f"\nx_original shape: {x_original.shape}")
print(f"x_original[0, 0, :, :, :].shape: {x_original[0, 0, :, :, :].shape}")
print(f"x_original min: {x_original.min().item():.6f}, max: {x_original.max().item():.6f}")
print(f"x_original mean: {x_original.mean().item():.6f}")

print("\n" + "="*60)
print("API的预处理方式")
print("="*60)

x_api = preprocess_image(image_path)  # (1, 1, 4, 16, 8)

print(f"x_api shape: {x_api.shape}")
print(f"x_api[0, 0, :, :, :].shape: {x_api[0, 0, :, :, :].shape}")
print(f"x_api min: {x_api.min().item():.6f}, max: {x_api.max().item():.6f}")
print(f"x_api mean: {x_api.mean().item():.6f}")

print("\n" + "="*60)
print("对比")
print("="*60)
print(f"形状是否相同: {x_original.shape == x_api.shape}")
print(f"值是否相同: {torch.allclose(x_original, x_api, atol=1e-6)}")
if not torch.allclose(x_original, x_api, atol=1e-6):
    diff = (x_original - x_api).abs()
    print(f"最大差异: {diff.max().item():.6f}")
    print(f"平均差异: {diff.mean().item():.6f}")
    print(f"差异位置: {torch.nonzero(diff > 1e-6)}")

