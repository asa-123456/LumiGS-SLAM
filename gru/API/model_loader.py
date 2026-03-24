#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Model Loader: Singleton pattern to ensure model is loaded only once
"""

import os
import sys
import torch
from typing import Optional

# Add project root directory to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from unet import UNet_GRU


class ModelLoader:
    """
    Singleton pattern model loader
    """
    _instance = None
    _model = None
    _device = None
    _model_path = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
        return cls._instance
    
    def load_model(
        self,
        model_path: str = "./gru/output4/gru_best.pt",
        gru_hidden_dim: int = 256,
        gru_layers: int = 2,
        dropout: float = 0.1,
        device: Optional[torch.device] = None,
        force_reload: bool = False
    ):
        """
        Load GRU model
        
        Args:
            model_path: Model weight file path
            gru_hidden_dim: GRU hidden layer dimension
            gru_layers: Number of GRU layers
            dropout: Dropout rate
            device: Device (CPU/GPU), automatically selected if None
            force_reload: Whether to force reload the model
        
        Returns:
            (model, device): Loaded model and device
        """
        # If model is already loaded with the same path and not forcing reload, return directly
        if self._model is not None and self._model_path == model_path and not force_reload:
            return self._model, self._device
        
        # Set device
        if device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = device
        
        print(f"Using device: {self._device}")
        print(f"Loading model weights: {model_path}")
        
        # Create model
        self._model = UNet_GRU(
            in_chns=4,
            hidden_dim=gru_hidden_dim,
            gru_layers=gru_layers,
            dropout=dropout
        ).to(self._device)
        
        # Load weights
        try:
            checkpoint = torch.load(model_path, map_location=self._device, weights_only=True)
        except TypeError:
            checkpoint = torch.load(model_path, map_location=self._device)
        
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
        
        # Load weights
        missing_keys, unexpected_keys = self._model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"Warning: The following parameters were not loaded: {missing_keys[:5]}..." if len(missing_keys) > 5 else f"Warning: The following parameters were not loaded: {missing_keys}")
        if unexpected_keys:
            print(f"Warning: The following parameters were not used: {unexpected_keys[:5]}..." if len(unexpected_keys) > 5 else f"Warning: The following parameters were not used: {unexpected_keys}")
        
        # Check if there are output layer related keys in checkpoint, try manual loading (consistent with original test script)
        output_layer_keys = [k for k in state_dict.keys() if ('affine' in k.lower() and 'head' in k.lower()) or ('tone' in k.lower() and 'head' in k.lower())]
        if output_layer_keys:
            print(f"  Found output layer related keys: {output_layer_keys}")
            # Try manual mapping of output layer weights
            for ckpt_key in output_layer_keys:
                try:
                    ckpt_tensor = state_dict[ckpt_key]
                    if 'affine_head' in ckpt_key and 'weight' in ckpt_key:
                        model_weight = self._model.affine_head[-1].weight  # (12, 128)
                        if ckpt_tensor.shape == model_weight.shape:
                            model_weight.data.copy_(ckpt_tensor)
                            print(f"  ✓ Manually loaded: {ckpt_key} -> affine_head[-1].weight")
                        else:
                            # Dimension mismatch, try adaptation
                            print(f"  ⚠ Dimension mismatch: {ckpt_key} (checkpoint: {ckpt_tensor.shape}, model: {model_weight.shape})")
                            if ckpt_tensor.dim() == 2 and model_weight.dim() == 2:
                                # checkpoint: (out_dim, in_dim_ckpt), model: (out_dim, in_dim_model)
                                if ckpt_tensor.shape[0] == model_weight.shape[0]:  # Same output dimension
                                    if ckpt_tensor.shape[1] < model_weight.shape[1]:
                                        # Checkpoint input dimension smaller, use zero padding
                                        padded = torch.zeros_like(model_weight)
                                        padded[:, :ckpt_tensor.shape[1]] = ckpt_tensor
                                        model_weight.data.copy_(padded)
                                        print(f"  ✓ Adapted with zero padding: {ckpt_key} -> affine_head[-1].weight")
                                    elif ckpt_tensor.shape[1] > model_weight.shape[1]:
                                        # Checkpoint input dimension larger, truncate to first N
                                        model_weight.data.copy_(ckpt_tensor[:, :model_weight.shape[1]])
                                        print(f"  ✓ Adapted with truncation: {ckpt_key} -> affine_head[-1].weight")
                                    else:
                                        print(f"  ✗ Unable to adapt dimensions: {ckpt_key}")
                                else:
                                    print(f"  ✗ Output dimension mismatch: {ckpt_key}")
                            else:
                                print(f"  ✗ Unable to adapt dimensions: {ckpt_key}")
                    elif 'affine_head' in ckpt_key and 'bias' in ckpt_key:
                        model_bias = self._model.affine_head[-1].bias
                        if ckpt_tensor.shape == model_bias.shape:
                            model_bias.data.copy_(ckpt_tensor)
                            print(f"  ✓ Manually loaded: {ckpt_key} -> affine_head[-1].bias")
                        else:
                            print(f"  ⚠ Bias dimension mismatch: {ckpt_key} (checkpoint: {ckpt_tensor.shape}, model: {model_bias.shape})")
                    elif 'tone_head' in ckpt_key and 'weight' in ckpt_key:
                        model_weight = self._model.tone_head[-1].weight  # (4, 128)
                        if ckpt_tensor.shape == model_weight.shape:
                            model_weight.data.copy_(ckpt_tensor)
                            print(f"  ✓ Manually loaded: {ckpt_key} -> tone_head[-1].weight")
                        else:
                            # Dimension mismatch, try adaptation
                            print(f"  ⚠ Dimension mismatch: {ckpt_key} (checkpoint: {ckpt_tensor.shape}, model: {model_weight.shape})")
                            if ckpt_tensor.dim() == 2 and model_weight.dim() == 2:
                                # checkpoint: (out_dim, in_dim_ckpt), model: (out_dim, in_dim_model)
                                if ckpt_tensor.shape[0] == model_weight.shape[0]:  # Same output dimension
                                    if ckpt_tensor.shape[1] < model_weight.shape[1]:
                                        # Checkpoint input dimension smaller, use zero padding
                                        padded = torch.zeros_like(model_weight)
                                        padded[:, :ckpt_tensor.shape[1]] = ckpt_tensor
                                        model_weight.data.copy_(padded)
                                        print(f"  ✓ Adapted with zero padding: {ckpt_key} -> tone_head[-1].weight")
                                    elif ckpt_tensor.shape[1] > model_weight.shape[1]:
                                        # Checkpoint input dimension larger, truncate to first N
                                        model_weight.data.copy_(ckpt_tensor[:, :model_weight.shape[1]])
                                        print(f"  ✓ Adapted with truncation: {ckpt_key} -> tone_head[-1].weight")
                                    else:
                                        print(f"  ✗ Unable to adapt dimensions: {ckpt_key}")
                                else:
                                    print(f"  ✗ Output dimension mismatch: {ckpt_key}")
                            else:
                                print(f"  ✗ Unable to adapt dimensions: {ckpt_key}")
                    elif 'tone_head' in ckpt_key and 'bias' in ckpt_key:
                        model_bias = self._model.tone_head[-1].bias
                        if ckpt_tensor.shape == model_bias.shape:
                            model_bias.data.copy_(ckpt_tensor)
                            print(f"  ✓ Manually loaded: {ckpt_key} -> tone_head[-1].bias")
                        else:
                            print(f"  ⚠ Bias dimension mismatch: {ckpt_key} (checkpoint: {ckpt_tensor.shape}, model: {model_bias.shape})")
                except Exception as e:
                    print(f"  ✗ Loading failed: {ckpt_key} -> {e}")
                    import traceback
                    traceback.print_exc()
                    pass  # Ignore loading failures
        
        # Check output layer weights (critical check)
        affine_head_weight = self._model.affine_head[-1].weight
        affine_head_bias = self._model.affine_head[-1].bias
        tone_head_weight = self._model.tone_head[-1].weight
        tone_head_bias = self._model.tone_head[-1].bias
        
        print(f"\nOutput layer weight check:")
        print(f"  affine_head weight: shape={affine_head_weight.shape}, "
              f"min={affine_head_weight.min().item():.6f}, max={affine_head_weight.max().item():.6f}, "
              f"all zero={(affine_head_weight == 0).all().item()}")
        print(f"  affine_head bias: shape={affine_head_bias.shape}, "
              f"min={affine_head_bias.min().item():.6f}, max={affine_head_bias.max().item():.6f}, "
              f"all zero={(affine_head_bias == 0).all().item()}")
        print(f"  tone_head weight: shape={tone_head_weight.shape}, "
              f"min={tone_head_weight.min().item():.6f}, max={tone_head_weight.max().item():.6f}, "
              f"all zero={(tone_head_weight == 0).all().item()}")
        print(f"  tone_head bias: shape={tone_head_bias.shape}, "
              f"min={tone_head_bias.min().item():.6f}, max={tone_head_bias.max().item():.6f}, "
              f"all zero={(tone_head_bias == 0).all().item()}")
        
        # Check if weights are zero (only check weights, not biases since biases can be zero normally)
        if (affine_head_weight == 0).all() or (tone_head_weight == 0).all():
            print("\n⚠⚠⚠ Warning: Output layer weights are all zero!")
            if (affine_head_weight == 0).all():
                print("  - affine_head weights are all zero, model output will only depend on bias")
            if (tone_head_weight == 0).all():
                print("  - tone_head weights are all zero, model output will only depend on bias")
            print("Possible reasons:")
            print("  1. Missing weight parameters in checkpoint (only biases present)")
            print("  2. Weight key name mismatch, unable to load automatically")
            print("  3. Output layer weights in weight file not properly trained")
            print("  4. Model structure mismatch (different model structure used during training)")
            print("\nRecommendations:")
            print("  1. Check if checkpoint contains affine_head.*.weight and tone_head.*.weight keys")
            print("  2. If only biases are present, model can still be used but performance may be limited")
        
        # Set to evaluation mode and disable gradient calculation to save memory
        self._model.eval()
        for param in self._model.parameters():
            param.requires_grad = False
        self._model_path = model_path
        print("Model loaded (gradient calculation disabled)")
        
        return self._model, self._device
    
    def get_model(self):
        """Get loaded model"""
        if self._model is None:
            raise RuntimeError("Model not loaded yet, please call load_model() first")
        return self._model, self._device

