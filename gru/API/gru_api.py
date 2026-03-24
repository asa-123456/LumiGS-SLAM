#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GRU Model API: Provides a functional interface that takes images and returns affine transformation matrices and tone mapping curve parameters
"""

import os
import sys
from typing import Union, List, Tuple, Optional
import torch
import numpy as np
from PIL import Image

# Add project root directory to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from .model_loader import ModelLoader
    from .preprocess import preprocess_image, preprocess_images
except ImportError:
    # If relative import fails, use absolute import
    from API.model_loader import ModelLoader
    from API.preprocess import preprocess_image, preprocess_images


def predict(
    image_input: Union[str, Image.Image, torch.Tensor, List[Union[str, Image.Image, torch.Tensor]]],
    model_path: str = "./gru/output4/gru_best.pt",
    gru_hidden_dim: int = 256,
    gru_layers: int = 2,
    dropout: float = 0.1,
    device: Optional[torch.device] = None,
    reset_hidden_state: bool = True,
    return_numpy: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    GRU model prediction API
    
    Takes a single image or a set of images and returns affine transformation matrices and tone mapping curve parameters.
    
    Args:
        image_input: Input image, can be one of the following types:
            - str: Single image file path
            - PIL.Image: Single PIL image object
            - torch.Tensor: Single image tensor (C, H, W) or (H, W, C)
            - List: List of images (supports any combination of the above types)
        model_path: Model weight file path
        gru_hidden_dim: GRU hidden layer dimension (default 256)
        gru_layers: Number of GRU layers (default 2)
        dropout: Dropout rate (default 0.1)
        device: Device (CPU/GPU), automatically selected if None
        reset_hidden_state: Whether to reset GRU hidden state (default True, reset on each call)
        return_numpy: Whether to return numpy arrays (default True), returns torch.Tensor if False
    
    Returns:
        (affine_matrix, tone_params): 
            - affine_matrix: Affine transformation matrix, shape (N, 3, 4) or (3, 4) (for single image)
            - tone_params: Tone mapping curve parameters, shape (N, 4) or (4,) (for single image)
            where N is the number of input images
        
    Example:
        >>> # Single image
        >>> affine, tone = predict("path/to/image.jpg")
        >>> print(affine.shape)  # (3, 4)
        >>> print(tone.shape)    # (4,)
        
        >>> # Multiple images
        >>> images = ["img1.jpg", "img2.jpg", "img3.jpg"]
        >>> affine, tone = predict(images)
        >>> print(affine.shape)  # (3, 3, 4)
        >>> print(tone.shape)    # (3, 4)
    """
    # Load model (singleton pattern)
    loader = ModelLoader()
    model, device = loader.load_model(
        model_path=model_path,
        gru_hidden_dim=gru_hidden_dim,
        gru_layers=gru_layers,
        dropout=dropout,
        device=device
    )
    
    # Determine if it's a single image or multiple images
    is_single = not isinstance(image_input, list)
    
    if is_single:
        # Single image
        x = preprocess_image(image_input)  # (1, 1, 4, 16, 8) - already includes sequence dimension
        x = x.to(device)
        
        # Reset hidden state (always reset for single image)
        hidden = None
        
        # Inference
        with torch.no_grad():
            affine_pred, tone_pred, hidden = model(x, hidden)
        
        # Remove sequence dimension (consistent with original test script)
        # Note: Although the model automatically squeezes(1) when seq_len=1, we also do it here for compatibility
        # If the dimension doesn't exist, squeeze won't error
        affine_pred = affine_pred.squeeze(1)  # (1, 12) or (batch, 12)
        tone_pred = tone_pred.squeeze(1)  # (1, 4) or (batch, 4)
        
        # Remove batch dimension
        affine_pred = affine_pred.squeeze(0)  # (12,)
        tone_pred = tone_pred.squeeze(0)  # (4,)
        
        # Convert to 3x4 matrix
        affine_matrix = affine_pred.view(3, 4)  # (3, 4)
        
    else:
        # Multiple images
        x = preprocess_images(image_input)  # (N, 1, 4, 16, 8) - already includes sequence dimension
        x = x.to(device)
        
        # Reset hidden state (determined by parameter for multiple images)
        hidden = None
        
        # Inference (for multiple images, model processes in parallel, but hidden states are independent for each image)
        with torch.no_grad():
            affine_pred, tone_pred, hidden = model(x, hidden)
        
        # Remove sequence dimension (consistent with original test script)
        # Note: Although the model automatically squeezes(1) when seq_len=1, we also do it here for compatibility
        affine_pred = affine_pred.squeeze(1)  # (N, 12)
        tone_pred = tone_pred.squeeze(1)  # (N, 4)
        
        # Convert to 3x4 matrix
        affine_matrix = affine_pred.view(-1, 3, 4)  # (N, 3, 4)
    
    # Convert to numpy or keep as tensor
    if return_numpy:
        affine_matrix = affine_matrix.cpu().numpy()
        tone_params = tone_pred.cpu().numpy()
    else:
        affine_matrix = affine_matrix.cpu()
        tone_params = tone_pred.cpu()
    
    return affine_matrix, tone_params


def predict_sequence(
    image_inputs: List[Union[str, Image.Image, torch.Tensor]],
    model_path: str = "./gru/output4/gru_best.pt",
    gru_hidden_dim: int = 256,
    gru_layers: int = 2,
    dropout: float = 0.1,
    device: Optional[torch.device] = None,
    return_numpy: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    GRU model sequence prediction API (maintains hidden state)
    
    Takes a sequence of images, and GRU uses information from previous images to predict parameters for subsequent images.
    The difference from predict() is that this function maintains the GRU hidden state, suitable for processing continuous video frames.
    
    Args:
        image_inputs: List of image sequence
        model_path: Model weight file path
        gru_hidden_dim: GRU hidden layer dimension (default 256)
        gru_layers: Number of GRU layers (default 2)
        dropout: Dropout rate (default 0.1)
        device: Device (CPU/GPU), automatically selected if None
        return_numpy: Whether to return numpy arrays (default True)
    
    Returns:
        (affine_matrix, tone_params): 
            - affine_matrix: Affine transformation matrix, shape (N, 3, 4)
            - tone_params: Tone mapping curve parameters, shape (N, 4)
            where N is the number of input images
    
    Example:
        >>> images = ["frame1.jpg", "frame2.jpg", "frame3.jpg"]
        >>> affine, tone = predict_sequence(images)
        >>> print(affine.shape)  # (3, 3, 4)
        >>> print(tone.shape)    # (3, 4)
    """
    # Load model
    loader = ModelLoader()
    model, device = loader.load_model(
        model_path=model_path,
        gru_hidden_dim=gru_hidden_dim,
        gru_layers=gru_layers,
        dropout=dropout,
        device=device
    )
    
    # Preprocess all images
    x = preprocess_images(image_inputs)  # (N, 1, 4, 16, 8) - already includes sequence dimension
    x = x.to(device)
    
    # Initialize hidden state
    hidden = None
    
    # Inference frame by frame, maintaining hidden state
    affine_list = []
    tone_list = []
    
    with torch.no_grad():
        for i in range(len(image_inputs)):
            # Get the i-th image
            x_i = x[i:i+1]  # (1, 1, 4, 16, 8)
            
            # Inference
            affine_pred, tone_pred, hidden = model(x_i, hidden)
            
            # Remove sequence dimension (consistent with original test script)
            affine_pred = affine_pred.squeeze(1)  # (1, 12) or (batch, 12)
            tone_pred = tone_pred.squeeze(1)  # (1, 4) or (batch, 4)
            
            # Remove batch dimension
            affine_pred = affine_pred.squeeze(0)  # (12,)
            tone_pred = tone_pred.squeeze(0)  # (4,)
            
            affine_list.append(affine_pred)
            tone_list.append(tone_pred)
    
    # Stack results
    affine_pred = torch.stack(affine_list, dim=0)  # (N, 12)
    tone_pred = torch.stack(tone_list, dim=0)  # (N, 4)
    
    # Convert to 3x4 matrix
    affine_matrix = affine_pred.view(-1, 3, 4)  # (N, 3, 4)
    
    # Convert to numpy or keep as tensor
    if return_numpy:
        affine_matrix = affine_matrix.cpu().numpy()
        tone_params = tone_pred.cpu().numpy()
    else:
        affine_matrix = affine_matrix.cpu()
        tone_params = tone_pred.cpu()
    
    return affine_matrix, tone_params

