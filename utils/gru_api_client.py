"""
GRU API Client
Used to call GRU service for image normalization through functional API
"""

import os
import sys
import torch
import numpy as np
from typing import Optional, Tuple
import torch.nn.functional as F

# Add GRU module path
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRU_DIR = os.path.join(_BASE_DIR, "gru")
if GRU_DIR not in sys.path:
    sys.path.insert(0, GRU_DIR)

try:
    from API.gru_api import predict
    GRU_API_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Failed to import GRU API: {e}")
    GRU_API_AVAILABLE = False
    predict = None


class GRUAPIClient:
    """GRU API Client, used to call GRU service through functional API"""
    
    def __init__(self, api_url: str = None, timeout: int = 30, model_path: str = None, 
                 gru_hidden_dim: int = 256, gru_layers: int = 2, dropout: float = 0.1,
                 device: Optional[torch.device] = None):
        """
        Initialize GRU API Client
        
        Args:
            api_url: Deprecated, kept for backward compatibility
            timeout: Deprecated, kept for backward compatibility
            model_path: Path to GRU model weight file
            gru_hidden_dim: GRU hidden layer dimension (default 256)
            gru_layers: Number of GRU layers (default 2)
            dropout: Dropout rate (default 0.1)
            device: Device (CPU/GPU), automatically selected if None
        """
        self.model_path = model_path
        self.gru_hidden_dim = gru_hidden_dim
        self.gru_layers = gru_layers
        self.dropout = dropout
        self.device = device
        self.is_available = GRU_API_AVAILABLE
        
        if not self.is_available:
            print(f"Warning: GRU API not available. GRU normalization will be disabled.")
        elif self.model_path is None:
            # Try using default path
            default_path = os.path.join(GRU_DIR, "output4", "gru_best.pt")
            if os.path.exists(default_path):
                self.model_path = default_path
                print(f"Using default GRU model path: {default_path}")
            else:
                print(f"Warning: GRU model path not specified and default path not found: {default_path}")
                self.is_available = False
    
    def predict_params(self, rgb_image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict affine transformation parameters and tone mapping parameters for image through functional API
        
        Args:
            rgb_image: RGB image tensor with shape (C, H, W) or (1, C, H, W), range [0, 1]
        
        Returns:
            affine_params: Affine transformation parameters (12,), flattened 3x4 matrix
            tone_params: Tone mapping parameters (4,), [alpha, beta, gamma, contrast]
        """
        if not self.is_available or predict is None:
            # Return identity transformation (no transformation)
            device = rgb_image.device if isinstance(rgb_image, torch.Tensor) else torch.device("cpu")
            affine_identity = torch.tensor([
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
            ], dtype=torch.float32, device=device)
            tone_neutral = torch.tensor([1.0, 0.0, 1.0, 1.0], dtype=torch.float32, device=device)
            return affine_identity, tone_neutral
        
        # Record original device
        image_device = rgb_image.device if isinstance(rgb_image, torch.Tensor) else torch.device("cpu")
        
        # Ensure 3D tensor (C, H, W)
        if rgb_image.dim() == 4:
            rgb_image = rgb_image.squeeze(0)
        elif rgb_image.dim() == 2:
            raise ValueError(f"Invalid image dimensions: {rgb_image.shape}")
        
        try:
            # Call functional API
            # predict function accepts torch.Tensor with shape (C, H, W) or (H, W, C)
            affine_matrix, tone_params = predict(
                image_input=rgb_image,
                model_path=self.model_path,
                gru_hidden_dim=self.gru_hidden_dim,
                gru_layers=self.gru_layers,
                dropout=self.dropout,
                device=self.device,
                reset_hidden_state=True,
                return_numpy=False  # Return torch.Tensor
            )
            
            # affine_matrix has shape (3, 4), tone_params has shape (4,)
            # Flatten affine_matrix to 12 parameters
            affine_params = affine_matrix.flatten()  # (12,)
            
            # Convert to correct device and data type
            affine_params = affine_params.to(image_device).float()
            tone_params = tone_params.to(image_device).float()
            
            return affine_params, tone_params
            
        except Exception as e:
            print(f"Error calling GRU API: {e}")
            import traceback
            traceback.print_exc()
            # Return identity transformation
            device = rgb_image.device if isinstance(rgb_image, torch.Tensor) else torch.device("cpu")
            affine_identity = torch.tensor([
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
            ], dtype=torch.float32, device=device)
            tone_neutral = torch.tensor([1.0, 0.0, 1.0, 1.0], dtype=torch.float32, device=device)
            return affine_identity, tone_neutral
    
    @staticmethod
    def apply_affine_transform(image: torch.Tensor, affine_params: torch.Tensor) -> torch.Tensor:
        """
        Apply affine transformation to image (same implementation as in GRUProcessor)
        
        Args:
            image: RGB image tensor with shape (C, H, W) or (B, C, H, W), range [0, 1]
            affine_params: Affine transformation parameters (12,), flattened 3x4 matrix
        
        Returns:
            transformed_image: Transformed image with the same shape as input
        """
        # Ensure 4D tensor
        if image.dim() == 3:
            image = image.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        B, C, H, W = image.shape
        
        # Reshape affine parameters to 3x4 matrix
        affine_matrix = affine_params.view(3, 4)  # (3, 4)
        
        # Expand batch dimension
        if affine_matrix.dim() == 2:
            affine_matrix = affine_matrix.unsqueeze(0).repeat(B, 1, 1)  # (B, 3, 4)
        
        # Reshape image to (B, C, H*W)
        image_flat = image.view(B, C, H * W)  # (B, C, H*W)
        
        # Add homogeneous coordinates
        ones = torch.ones(B, 1, H * W, device=image.device, dtype=image.dtype)
        image_homo = torch.cat([image_flat, ones], dim=1)  # (B, 4, H*W)
        
        # Apply affine transformation: (B, 3, 4) @ (B, 4, H*W) -> (B, 3, H*W)
        transformed_flat = torch.einsum('bij,bjk->bik', affine_matrix, image_homo)  # (B, 3, H*W)
        
        # Reshape back to image shape
        transformed_image = transformed_flat.view(B, C, H, W)
        
        # Clamp to [0, 1] range
        transformed_image = torch.clamp(transformed_image, 0.0, 1.0)
        
        if squeeze_output:
            transformed_image = transformed_image.squeeze(0)
        
        return transformed_image
    
    @staticmethod
    def apply_tone_mapping(image: torch.Tensor, tone_params: torch.Tensor) -> torch.Tensor:
        """Apply tone mapping to image."""
        from utils.gru_utils import GRUProcessor
        return GRUProcessor.apply_tone_mapping(image, tone_params)
    
    @staticmethod
    def preprocess_image(rgb_image: torch.Tensor):
        """
        Preprocess image for GRU input (extract 16x8 RGB and 16x1 grayscale)
        
        Args:
            rgb_image: RGB image tensor with shape (C, H, W) or (1, C, H, W), range [0, 1]
        
        Returns:
            rgb_16x8: Downsampled RGB image (B, 3, 16, 8)
            gray_16x1: Grayscale image (B, 16, 1)
        """
        # Ensure 4D tensor (B, C, H, W)
        if rgb_image.dim() == 3:
            rgb_image = rgb_image.unsqueeze(0)
        
        # Downsample to 16x8
        rgb_16x8 = F.interpolate(
            rgb_image, size=(16, 8), mode="bilinear", align_corners=False
        )
        
        # Calculate grayscale: Y' = 0.299 R + 0.587 G + 0.114 B
        gray_small = (
            0.299 * rgb_16x8[:, 0] + 0.587 * rgb_16x8[:, 1] + 0.114 * rgb_16x8[:, 2]
        )  # (B, 16, 8)
        
        # Take mean along width dimension to get (B, 16, 1)
        gray_16x1 = gray_small.mean(dim=2, keepdim=True)
        
        return rgb_16x8, gray_16x1
    
    def normalize_image(self, rgb_image: torch.Tensor) -> torch.Tensor:
        """
        Normalize image using GRU API (apply affine transformation and tone mapping)
        
        Args:
            rgb_image: RGB image tensor with shape (C, H, W) or (1, C, H, W), range [0, 1]
        
        Returns:
            normalized_image: Normalized image with the same shape as input
        """
        # Predict parameters (via API)
        affine_params, tone_params = self.predict_params(rgb_image)
        
        # Apply affine transformation
        transformed = self.apply_affine_transform(rgb_image, affine_params)
        
        # Apply tone mapping
        normalized = self.apply_tone_mapping(transformed, tone_params)
        
        return normalized

