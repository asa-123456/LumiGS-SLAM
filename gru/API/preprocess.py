#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Image preprocessing module: Convert input images to the format required by the model
"""

import os
import sys
from typing import Union, List
import torch
from PIL import Image

# Add project root directory to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dataset import _to_tensor


def preprocess_image(image_input: Union[str, Image.Image, torch.Tensor]) -> torch.Tensor:
    """
    Preprocess a single image, convert to model input format
    
    Args:
        image_input: Can be one of the following types:
            - str: Image file path
            - PIL.Image: PIL image object
            - torch.Tensor: Already tensor-formatted image (C, H, W) or (H, W, C)
    
    Returns:
        x: Preprocessed 4-channel image tensor, shape (1, 4, 16, 8)
           Channel order: [R, G, B, Gray]
    """
    # 1. Load image
    if isinstance(image_input, str):
        # File path
        with Image.open(image_input) as img:
            img = img.convert("RGB")
            rgb = _to_tensor(img)  # C, H, W
    elif isinstance(image_input, Image.Image):
        # PIL image object
        img = image_input.convert("RGB")
        rgb = _to_tensor(img)  # C, H, W
    elif isinstance(image_input, torch.Tensor):
        # Already tensor
        rgb = image_input
        if rgb.dim() == 3:
            # (H, W, C) -> (C, H, W)
            if rgb.shape[2] == 3 or rgb.shape[2] == 4:
                rgb = rgb.permute(2, 0, 1)
        elif rgb.dim() == 4:
            # (B, C, H, W) -> take first image
            rgb = rgb[0]
        
        # Ensure values are in [0, 1] range
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")
    
    # 2. Downsample to 16x8
    rgb_16x8 = torch.nn.functional.interpolate(
        rgb.unsqueeze(0), size=(16, 8), mode="bilinear", align_corners=False
    ).squeeze(0)  # (3, 16, 8)
    
    # 3. Calculate grayscale: Y' = 0.299 R + 0.587 G + 0.114 B
    gray_small = (
        0.299 * rgb_16x8[0] + 0.587 * rgb_16x8[1] + 0.114 * rgb_16x8[2]
    )  # (16, 8)
    
    # 4. Take mean along width to get 1 column
    gray_16x1 = gray_small.mean(dim=1, keepdim=True)  # (16, 1)
    
    # 5. Add batch dimension (consistent with original test script)
    rgb_16x8 = rgb_16x8.unsqueeze(0)  # (1, 3, 16, 8)
    gray_16x1 = gray_16x1.unsqueeze(0)  # (1, 16, 1)
    
    # 6. Expand grayscale to 16x8 to match RGB size (consistent with original test script)
    # gray_16x1: (1, 16, 1) -> unsqueeze(1) -> (1, 1, 16, 1) -> unsqueeze(2) -> (1, 1, 1, 16, 1) -> repeat -> (1, 1, 1, 16, 8)
    gray_expanded = gray_16x1.unsqueeze(1).unsqueeze(2).repeat(1, 1, 1, 1, 8)  # (1, 1, 1, 16, 8)
    
    # 7. Combine RGB and grayscale to form 4-channel image (consistent with original test script)
    # rgb_16x8: (1, 3, 16, 8) -> unsqueeze(1) -> (1, 1, 3, 16, 8)
    # cat dim=2: (1, 1, 4, 16, 8)
    x = torch.cat([rgb_16x8.unsqueeze(1), gray_expanded], dim=2)  # (1, 1, 4, 16, 8)
    
    return x


def preprocess_images(image_inputs: List[Union[str, Image.Image, torch.Tensor]]) -> torch.Tensor:
    """
    Preprocess multiple images, convert to model input format
    
    Args:
        image_inputs: List of images, each element can be a file path, PIL image, or tensor
    
    Returns:
        x: Preprocessed 4-channel image tensor, shape (N, 1, 4, 16, 8)
           where N is the number of images
    """
    processed_images = []
    for img_input in image_inputs:
        processed = preprocess_image(img_input)  # (1, 1, 4, 16, 8)
        # Remove batch dimension, keep sequence dimension
        processed_images.append(processed.squeeze(0))  # (1, 4, 16, 8)
    
    # Stack into batch
    x = torch.cat(processed_images, dim=0)  # (N, 4, 16, 8)
    # Add sequence dimension
    x = x.unsqueeze(1)  # (N, 1, 4, 16, 8)
    return x

