"""
GRU Image Normalization Utility Module
Used to map images to normalized/standardized appearance space, removing lighting factors

Supports two usage methods:
1. Direct model call (GRUProcessor): Faster, no network overhead
2. HTTP API call (GRUAPIClient): More flexible, can be deployed independently
"""

import os
import sys
import torch
import torch.nn.functional as F

# Add GRU module path
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRU_DIR = os.path.join(_BASE_DIR, "gru")
if GRU_DIR not in sys.path:
    sys.path.insert(0, GRU_DIR)

try:
    from unet import UNet_DualOutput
    from model import GRUNet, MobileNetGRU, build_input_vector, build_4ch_image
    from dataset import _to_tensor
except ImportError as e:
    print(f"Warning: Failed to import GRU modules: {e}")
    UNet_DualOutput = None
    GRUNet = None
    MobileNetGRU = None

# Import API client
try:
    from utils.gru_api_client import GRUAPIClient
    GRU_API_CLIENT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: GRU API client not available: {e}")
    GRU_API_CLIENT_AVAILABLE = False
    GRUAPIClient = None


class GRUProcessor:
    """GRU Image Processor for normalizing image appearance"""
    
    def __init__(self, checkpoint_path=None, model_type="unet", device=None, **kwargs):
        """
        Initialize GRU Processor
        
        Args:
            checkpoint_path: Path to GRU model weights
            model_type: Model type ("unet", "mobilenet", "mlp")
            device: Device (None for automatic selection)
            **kwargs: Model initialization parameters
        """
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self.model = None
        self.is_loaded = False
        
        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            self.load_model(checkpoint_path, model_type, **kwargs)
        elif checkpoint_path is not None:
            print(f"Warning: GRU checkpoint not found at {checkpoint_path}, GRU will be disabled")
    
    def load_model(self, checkpoint_path, model_type="unet", **kwargs):
        """Load GRU model"""
        try:
            if model_type == "mobilenet":
                variant = kwargs.get("mobilenet_variant", "v2")
                dropout = kwargs.get("dropout", 0.1)
                if MobileNetGRU is None:
                    raise ImportError("MobileNetGRU is not available")
                self.model = MobileNetGRU(variant=variant, pretrained=False, dropout=dropout)
            elif model_type == "unet":
                if UNet_DualOutput is None:
                    raise ImportError("UNet_DualOutput is not available")
                self.model = UNet_DualOutput(in_chns=4)  # 4-channel input: RGB + grayscale
            elif model_type == "mlp":
                if GRUNet is None:
                    raise ImportError("GRUNet is not available")
                hidden_dim = kwargs.get("hidden_dim", 512)
                dropout = kwargs.get("dropout", 0.1)
                self.model = GRUNet(hidden_dim=hidden_dim, dropout=dropout)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Load weights
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            except TypeError:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Compatible with multiple storage formats
            if isinstance(checkpoint, dict):
                if "model" in checkpoint and isinstance(checkpoint["model"], dict):
                    state_dict = checkpoint["model"]
                elif "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
                    state_dict = checkpoint["state_dict"]
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            self.model.load_state_dict(state_dict, strict=False)
            self.model.to(self.device)
            self.model.eval()
            # Freeze model parameters (don't update during training)
            for param in self.model.parameters():
                param.requires_grad = False
            self.model_type = model_type
            self.is_loaded = True
            print(f"GRU model loaded successfully from {checkpoint_path}")
            return True
        except Exception as e:
            print(f"Failed to load GRU model: {str(e)}")
            self.model = None
            self.is_loaded = False
            return False
    
    def preprocess_image(self, rgb_image):
        """
        Preprocess image for GRU input
        
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
    
    def predict_params(self, rgb_image):
        """
        Predict affine transformation parameters and tone mapping parameters for image
        
        Args:
            rgb_image: RGB image tensor with shape (C, H, W) or (1, C, H, W), range [0, 1]
        
        Returns:
            affine_params: Affine transformation parameters (12,), flattened 3x4 matrix
            tone_params: Tone mapping parameters (4,), [alpha, beta, gamma, contrast]
        """
        if not self.is_loaded or self.model is None:
            # Return identity transformation (no transformation)
            affine_identity = torch.tensor([
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
            ], dtype=torch.float32, device=self.device)
            tone_neutral = torch.tensor([1.0, 0.0, 1.0, 1.0], dtype=torch.float32, device=self.device)
            return affine_identity, tone_neutral
        
        rgb_16x8, gray_16x1 = self.preprocess_image(rgb_image)
        rgb_16x8 = rgb_16x8.to(self.device)
        gray_16x1 = gray_16x1.to(self.device)
        
        with torch.no_grad():
            if self.model_type == "mobilenet":
                x = build_4ch_image(rgb_16x8, gray_16x1)
            elif self.model_type == "unet":
                gray_expanded = gray_16x1.unsqueeze(1).repeat(1, 1, 1, 8)  # (B,1,16,8)
                x = torch.cat([rgb_16x8, gray_expanded], dim=1)  # (B,4,16,8)
            else:  # mlp
                x = build_input_vector(rgb_16x8, gray_16x1)
            
            affine_pred, tone_pred = self.model(x)
        
        # Remove batch dimension
        if affine_pred.dim() > 1:
            affine_pred = affine_pred[0]
        if tone_pred.dim() > 1:
            tone_pred = tone_pred[0]
        
        return affine_pred, tone_pred
    
    @staticmethod
    def apply_affine_transform(image, affine_params):
        """
        Apply affine transformation to image
        
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
        # Use einsum for batch matrix multiplication
        transformed_flat = torch.einsum('bij,bjk->bik', affine_matrix, image_homo)  # (B, 3, H*W)
        
        # Reshape back to image shape
        transformed_image = transformed_flat.view(B, C, H, W)
        
        # Clamp to [0, 1] range
        transformed_image = torch.clamp(transformed_image, 0.0, 1.0)
        
        if squeeze_output:
            transformed_image = transformed_image.squeeze(0)
        
        return transformed_image
    
    @staticmethod
    def apply_tone_mapping(image, tone_params):
        """
        Apply tone mapping to image
        
        Args:
            image: RGB image tensor with shape (C, H, W) or (B, C, H, W), range [0, 1]
            tone_params: Tone mapping parameters (4,), [alpha, beta, gamma, contrast]
        
        Returns:
            transformed_image: Transformed image with the same shape as input
        """
        # Ensure 4D tensor
        if image.dim() == 3:
            image = image.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Extract parameters (training data order is [alpha, beta, gamma, contrast])
        alpha = tone_params[0] if tone_params.dim() == 1 else tone_params[:, 0]
        beta = tone_params[1] if tone_params.dim() == 1 else tone_params[:, 1]
        gamma = tone_params[2] if tone_params.dim() == 1 else tone_params[:, 2]
        contrast = tone_params[3] if tone_params.dim() == 1 else tone_params[:, 3]
        
        # Ensure parameters have correct dimensions
        if alpha.dim() == 0:
            alpha = alpha.unsqueeze(0)
            beta = beta.unsqueeze(0)
            gamma = gamma.unsqueeze(0)
            contrast = contrast.unsqueeze(0)
        
        B = image.shape[0]
        if alpha.shape[0] == 1 and B > 1:
            alpha = alpha.expand(B)
            beta = beta.expand(B)
            gamma = gamma.expand(B)
            contrast = contrast.expand(B)
        
        # Expand dimensions to match image
        while alpha.dim() < image.dim():
            alpha = alpha.unsqueeze(-1)
            beta = beta.unsqueeze(-1)
            gamma = gamma.unsqueeze(-1)
            contrast = contrast.unsqueeze(-1)
        
        # Apply tone mapping
        # 1. Gamma correction (ensure gamma is positive using abs+small epsilon)
        epsilon = 1e-8
        gamma_safe = torch.abs(gamma) + epsilon
        transformed = image ** gamma_safe
        
        # 2. Alpha and Beta adjustment (linear transformation)
        transformed = alpha * transformed + beta
        
        # 3. Contrast adjustment
        mean_val = transformed.mean(dim=[2, 3], keepdim=True)
        transformed = contrast * (transformed - mean_val) + mean_val
        
        # Clamp to [0, 1] range
        transformed = torch.clamp(transformed, 0.0, 1.0)
        
        if squeeze_output:
            transformed = transformed.squeeze(0)
        
        return transformed
    
    def normalize_image(self, rgb_image):
        """
        Normalize image using GRU (apply affine transformation and tone mapping)
        
        Args:
            rgb_image: RGB image tensor with shape (C, H, W) or (1, C, H, W), range [0, 1]
        
        Returns:
            normalized_image: Normalized image with the same shape as input
        """
        # Predict parameters
        affine_params, tone_params = self.predict_params(rgb_image)
        
        # Apply affine transformation
        transformed = self.apply_affine_transform(rgb_image, affine_params)
        
        # Apply tone mapping
        normalized = self.apply_tone_mapping(transformed, tone_params)
        
        return normalized

