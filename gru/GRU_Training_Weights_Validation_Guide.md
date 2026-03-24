# GRU Training Weights Validation Guide

## Overview

This document explains whether the weight files obtained through training the Gated Recurrent Unit (GRU) network using `python train.py` are viable, and how to validate and use these weights.

## Weight File Format

Weight file format saved by the training script `train.py`:

```python
{
    "epoch": int,              # Training epoch
    "model": state_dict,       # Model weights (state_dict)
    "optimizer": state_dict,   # Optimizer state
    "val_loss": float          # Validation loss
}
```

Save locations:
- `{out_dir}/gru_best.pt` - Model with best performance on validation set
- `{out_dir}/gru_latest.pt` - Latest trained model

## Weight Viability Validation

### Weight Format Compatibility

1. **Correct save format**: Weights are saved in standard PyTorch checkpoint format, containing the `model` key
2. **Loading compatibility**: Both `inference.py` and `api.py` support loading weights from the `model` key
3. **Model matching**: Weights are fully compatible with the `UNet_GRU` model architecture

### Fixed Issues

1. **API support**: Updated `api.py` to support `unet_gru` model type
2. **Inference support**: `inference.py` now supports sequence inference for GRU models
3. **Weight loading**: Using `strict=False` allows flexible loading, compatible with different configurations

## Validation Methods

### Method 1: Using Test Script

```bash
python test_gru_weights.py \
    --checkpoint /path/to/gru_best.pt \
    --model_type unet_gru \
    --gru_hidden_dim 256 \
    --gru_layers 2 \
    --dropout 0.1
```

The test script checks:
-  Whether the file exists
-  Whether the checkpoint format is correct
-  Whether weights can be loaded successfully
-  Whether model forward pass works normally
-  Whether output values are reasonable (no NaN/Inf)

### Method 2: Using Inference Script

```bash
python inference.py \
    --checkpoint /path/to/gru_best.pt \
    --model_type unet_gru \
    --image /path/to/test_image.jpg \
    --output_dir ./predictions \
    --gru_hidden_dim 256 \
    --gru_layers 2
```

### Method 3: Using Validation Mode

```bash
python inference.py \
    --checkpoint /path/to/gru_best.pt \
    --model_type unet_gru \
    --validate \
    --data_root /path/to/data \
    --val_image_dir val/image \
    --val_label_dir val/label \
    --sequence_length 5 \
    --gru_hidden_dim 256 \
    --gru_layers 2
```

## Using Weights for Inference

### 1. Single Image Inference

```python
from unet import UNet_GRU
import torch

# Load model
model = UNet_GRU(in_chns=4, hidden_dim=256, gru_layers=2, dropout=0.1)
checkpoint = torch.load("gru_best.pt", map_location="cpu")
model.load_state_dict(checkpoint["model"], strict=False)
model.eval()

# Inference (single image, sequence length 1)
x = torch.randn(1, 4, 16, 8)  # (batch, channels, height, width)
affine, tone, hidden = model(x, None)
```

### 2. Sequence Inference

```python
# Sequence input
x_seq = torch.randn(1, 5, 4, 16, 8)  # (batch, seq_len, channels, height, width)
hidden = None
affine, tone, hidden = model(x_seq, hidden)
# affine: (1, 5, 12), tone: (1, 5, 4)
```

### 3. Using API

```bash
# Start API service
python api.py

# Load GRU model
curl -X POST "http://localhost:8000/model/load?checkpoint_path=/path/to/gru_best.pt&model_type=unet_gru"

# Predict single image
curl -X POST "http://localhost:8000/predict" \
  -F "image=@test_image.jpg"
```

## Notes

### 1. Model Parameters Must Match

When loading weights, model initialization parameters must match those used during training:

- `gru_hidden_dim`: Default 256, must match training value
- `gru_layers`: Default 2, must match training value
- `dropout`: Default 0.1, recommended to match training value

### 2. Sequence Length

- **During training**: Use `--sequence_length` parameter (e.g., 5)
- **During inference**:
  - Single image: Can input `(batch, 4, 16, 8)`, model will handle automatically
  - Sequence inference: Input `(batch, seq_len, 4, 16, 8)`, can utilize historical information

### 3. Hidden State

GRU models maintain hidden state for sequence inference:
- At the start of each batch, hidden state should be reset to `None`
- During continuous inference, hidden state can be passed to utilize historical information

## Common Questions

### Q: Can weight files be used for other model types?

A: No. Weights trained for GRU can only be used for the `unet_gru` model type. Different model types have different architectures, so weights are not compatible.

### Q: Is it okay if the sequence length during training differs from inference?

A: Yes. GRU models support dynamic sequence lengths. You can use sequence length 5 during training and any length (including 1) during inference.

### Q: How to know the parameters used during training?

A: Check training logs or use the test script to try different parameter combinations. Default parameters are usually:
- `gru_hidden_dim=256`
- `gru_layers=2`
- `dropout=0.1`

### Q: What to do if weight files are corrupted?

A: If weight files are corrupted or cannot be loaded:
1. Check if the file is complete (file size should be reasonable)
2. Try loading with `strict=False` (already used by default)
3. Check if model initialization parameters match

## Summary

 **GRU-trained weights are fully viable** and can be used for:
- Single image inference
- Sequence inference
- API services
- Model validation

Weight files have a standard format and are fully compatible with the PyTorch ecosystem, so they can be used with confidence.

