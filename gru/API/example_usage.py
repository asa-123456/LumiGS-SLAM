#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
API Usage Examples
"""

import sys
import os

# Add project root directory to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from API import predict, predict_sequence
import numpy as np


def example_single_image():
    """Example 1: Single image prediction"""
    print("=" * 50)
    print("Example 1: Single image prediction")
    print("=" * 50)
    
    # Note: Replace with actual image path
    image_path = "./gru/data/room/frame000049.jpg"
    
    if not os.path.exists(image_path):
        print(f"⚠ Image file does not exist: {image_path}")
        print("Please modify image_path to an actual existing image path")
        return
    
    # Prediction
    affine_matrix, tone_params = predict(image_path)
    
    # Print results
    print(f"\nInput image: {image_path}")
    print(f"\nAffine transformation matrix (3x4):")
    print(affine_matrix)
    print(f"\nTone mapping parameters (4,):")
    print(f"  gamma:    {tone_params[0]:.6f}")
    print(f"  alpha:    {tone_params[1]:.6f}")
    print(f"  beta:     {tone_params[2]:.6f}")
    print(f"  contrast: {tone_params[3]:.6f}")
    print(f"\nOutput shapes:")
    print(f"  Affine matrix: {affine_matrix.shape}")
    print(f"  Tone parameters: {tone_params.shape}")


def example_multiple_images():
    """Example 2: Multiple image prediction"""
    print("\n" + "=" * 50)
    print("Example 2: Multiple image prediction")
    print("=" * 50)
    
    # Note: Replace with actual image directory
    image_dir = "./gru/data/room"
    
    if not os.path.exists(image_dir):
        print(f"⚠ Image directory does not exist: {image_dir}")
        print("Please modify image_dir to an actual existing directory path")
        return
    
    # Get first 3 images
    image_files = sorted([
        os.path.join(image_dir, f) 
        for f in os.listdir(image_dir) 
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])[:3]
    
    if len(image_files) == 0:
        print(f"⚠ No image files found in {image_dir}")
        return
    
    print(f"\nNumber of input images: {len(image_files)}")
    for i, img_path in enumerate(image_files):
        print(f"  {i+1}. {os.path.basename(img_path)}")
    
    # Prediction
    affine_matrices, tone_params = predict(image_files)
    
    # Print results
    print(f"\nOutput shapes:")
    print(f"  Affine matrices: {affine_matrices.shape}")  # (N, 3, 4)
    print(f"  Tone parameters: {tone_params.shape}")     # (N, 4)
    
    print(f"\nResults for each image:")
    for i, (affine, tone) in enumerate(zip(affine_matrices, tone_params)):
        print(f"\nImage {i+1}:")
        print(f"  Affine matrix:\n{affine}")
        print(f"  Tone parameters: gamma={tone[0]:.6f}, alpha={tone[1]:.6f}, "
              f"beta={tone[2]:.6f}, contrast={tone[3]:.6f}")


def example_sequence():
    """Example 3: Sequence prediction (maintaining hidden state)"""
    print("\n" + "=" * 50)
    print("Example 3: Sequence prediction (maintaining hidden state)")
    print("=" * 50)
    
    # Note: Replace with actual image directory
    image_dir = "./gru/data/room"
    
    if not os.path.exists(image_dir):
        print(f"⚠ Image directory does not exist: {image_dir}")
        print("Please modify image_dir to an actual existing directory path")
        return
    
    # Get first 5 images as sequence
    image_files = sorted([
        os.path.join(image_dir, f) 
        for f in os.listdir(image_dir) 
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])[:5]
    
    if len(image_files) == 0:
        print(f"⚠ No image files found in {image_dir}")
        return
    
    print(f"\nInput sequence length: {len(image_files)}")
    for i, img_path in enumerate(image_files):
        print(f"  Frame {i+1}: {os.path.basename(img_path)}")
    
    # Sequence prediction (maintaining hidden state)
    affine_matrices, tone_params = predict_sequence(image_files)
    
    # Print results
    print(f"\nOutput shapes:")
    print(f"  Affine matrices: {affine_matrices.shape}")  # (N, 3, 4)
    print(f"  Tone parameters: {tone_params.shape}")     # (N, 4)
    
    print(f"\nSequence prediction results (GRU uses information from previous frames):")
    for i, (affine, tone) in enumerate(zip(affine_matrices, tone_params)):
        print(f"\nFrame {i+1}:")
        print(f"  Tone parameters: gamma={tone[0]:.6f}, alpha={tone[1]:.6f}, "
              f"beta={tone[2]:.6f}, contrast={tone[3]:.6f}")


if __name__ == "__main__":
    print("GRU Model API Usage Examples")
    print("=" * 50)
    
    # Run examples
    try:
        example_single_image()
        example_multiple_images()
        example_sequence()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

