#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GRU Model API Module

Provides functional interfaces that input images and return affine transformation matrices and tone mapping curve parameters.

Main interfaces:
    - predict(): Single or multiple image prediction (resets hidden state on each call)
    - predict_sequence(): Sequence prediction (maintains hidden state, suitable for continuous video frames)

Example:
    >>> from API import predict, predict_sequence
    >>> 
    >>> # Single image
    >>> affine, tone = predict("path/to/image.jpg")
    >>> print(f"Affine matrix shape: {affine.shape}")  # (3, 4)
    >>> print(f"Tone parameter shape: {tone.shape}")     # (4,)
    >>> 
    >>> # Multiple images
    >>> images = ["img1.jpg", "img2.jpg", "img3.jpg"]
    >>> affine, tone = predict(images)
    >>> print(f"Affine matrix shape: {affine.shape}")  # (3, 3, 4)
    >>> print(f"Tone parameter shape: {tone.shape}")     # (3, 4)
    >>> 
    >>> # Sequence prediction (maintains hidden state)
    >>> affine, tone = predict_sequence(images)
"""

try:
    from .gru_api import predict, predict_sequence
    from .model_loader import ModelLoader
    from .preprocess import preprocess_image, preprocess_images
except ImportError:
    import sys
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from API.gru_api import predict, predict_sequence
    from API.model_loader import ModelLoader
    from API.preprocess import preprocess_image, preprocess_images

__all__ = [
    'predict',
    'predict_sequence',
    'ModelLoader',
    'preprocess_image',
    'preprocess_images',
]

__version__ = '1.0.0'

