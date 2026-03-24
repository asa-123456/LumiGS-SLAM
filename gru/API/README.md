# GRU Model API Usage Instructions

## Overview

This API module provides a functional interface for the trained GRU model. It can take a single image or a set of images as input and return:
1. **3x4 affine transformation matrix**: for spatial transformation of images
2. **1x4 nonlinear tone mapping curve parameters**: for image tone adjustment

## Installation Dependencies

Ensure the following dependencies are installed:
```bash
pip install torch torchvision pillow numpy
```

## Quick Start

### Basic Usage

```python
from API import predict

# Single image prediction
affine_matrix, tone_params = predict("path/to/image.jpg")

print(f"Affine transformation matrix (3x4):\n{affine_matrix}")
print(f"Tone mapping parameters (4,): {tone_params}")
```

### Multiple Image Prediction

```python
from API import predict

# Multiple image prediction (resets hidden state each call)
images = ["img1.jpg", "img2.jpg", "img3.jpg"]
affine_matrices, tone_params = predict(images)

print(f"Affine transformation matrix shape: {affine_matrices.shape}")  # (3, 3, 4)
print(f"Tone parameters shape: {tone_params.shape}")           # (3, 4)
```

### Sequence Prediction (Maintaining Hidden State)

```python
from API import predict_sequence

# Sequence prediction (GRU uses information from previous images)
images = ["frame1.jpg", "frame2.jpg", "frame3.jpg"]
affine_matrices, tone_params = predict_sequence(images)

print(f"Affine transformation matrix shape: {affine_matrices.shape}")  # (3, 3, 4)
print(f"Tone parameters shape: {tone_params.shape}")           # (3, 4)
```

## API Reference

### `predict()`

Function for predicting on single or multiple images.

**Parameters:**
- `image_input`: Input image, can be:
  - `str`: Image file path
  - `PIL.Image`: PIL image object
  - `torch.Tensor`: Image tensor
  - `List`: List of the above types (multiple images)
- `model_path` (str, optional): Model weight file path, defaults to using trained weights
- `gru_hidden_dim` (int, optional): GRU hidden layer dimension, default 256
- `gru_layers` (int, optional): Number of GRU layers, default 2
- `dropout` (float, optional): Dropout rate, default 0.1
- `device` (torch.device, optional): Device (CPU/GPU), default automatically selected
- `reset_hidden_state` (bool, optional): Whether to reset GRU hidden state, default True
- `return_numpy` (bool, optional): Whether to return numpy arrays, default True

**Returns:**
- `affine_matrix`: Affine transformation matrix, shape `(3, 4)` or `(N, 3, 4)`
- `tone_params`: Tone mapping parameters, shape `(4,)` or `(N, 4)`

### `predict_sequence()`

Sequence prediction function that maintains GRU hidden state, suitable for processing continuous video frames.

**Parameters:**
- `image_inputs`: List of image sequence
- Other parameters are the same as `predict()`

**Returns:**
- `affine_matrix`: Affine transformation matrix, shape `(N, 3, 4)`
- `tone_params`: Tone mapping parameters, shape `(N, 4)`

## Output Format Description

### Affine Transformation Matrix (3x4)

The affine transformation matrix is used for spatial transformation of images, formatted as:
```
[[a11, a12, a13, tx],
 [a21, a22, a23, ty],
 [a31, a32, a33, tz]]
```

### Tone Mapping Parameters (4)

Tone mapping curve parameters, in order:
- `[0]`: gamma
- `[1]`: alpha
- `[2]`: beta
- `[3]`: contrast

## Example Code

### Example 1: Processing a Single Image

```python
from API import predict
import numpy as np

# Prediction
affine, tone = predict("test_image.jpg")

# Print results
print("Affine transformation matrix:")
print(affine)
print("\nTone mapping parameters:")
print(f"gamma: {tone[0]:.6f}")
print(f"alpha: {tone[1]:.6f}")
print(f"beta: {tone[2]:.6f}")
print(f"contrast: {tone[3]:.6f}")
```

### Example 2: Batch Processing Images

```python
from API import predict
import os

# Get all image files
image_dir = "path/to/images"
image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) 
               if f.endswith(('.jpg', '.png', '.JPG', '.PNG'))]

# Batch prediction
affine_matrices, tone_params = predict(image_files)

# Process results for each image
for i, (affine, tone) in enumerate(zip(affine_matrices, tone_params)):
    print(f"Image {i+1}:")
    print(f"  Affine matrix: {affine}")
    print(f"  Tone parameters: {tone}")
```

### Example 3: Video Frame Sequence Processing

```python
from API import predict_sequence
import cv2

# Read video frames
cap = cv2.VideoCapture("video.mp4")
frames = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
cap.release()

# Sequence prediction (maintaining hidden state)
affine_matrices, tone_params = predict_sequence(frames)

# Process results
for i, (affine, tone) in enumerate(zip(affine_matrices, tone_params)):
    print(f"Frame {i+1}: Affine matrix shape={affine.shape}, Tone parameters={tone}")
```

## Notes

1. **Model Loading**: The model uses a singleton pattern. It loads the model on the first call and reuses the loaded model for subsequent calls to improve efficiency.

2. **Hidden State**:
   - `predict()`: Resets hidden state each call, suitable for independent image prediction
   - `predict_sequence()`: Maintains hidden state, suitable for continuous video frame sequences

3. **Input Format**:
   - Supports common image formats (JPG, PNG, etc.)
   - Images are automatically preprocessed: downsampled to 16x8, grayscale calculated, combined into 4 channels

4. **Device Selection**: Automatically selects GPU if available, otherwise uses CPU.

5. **Return Value**: Returns numpy arrays by default. Set `return_numpy=False` to return torch.Tensor.

## File Structure

```
API/
├── __init__.py          # Module initialization, exports main interfaces
├── model_loader.py      # Model loader (singleton pattern)
├── preprocess.py        # Image preprocessing module
├── gru_api.py           # Main API functions
└── README.md            # This file
```

## Troubleshooting

### Issue 1: Model Loading Failure
- Check if the model weight file path is correct
- Ensure the model weight file exists and is complete

### Issue 2: CUDA Out of Memory
- Set `device=torch.device('cpu')` to use CPU
- Or reduce the number of images processed in batch

### Issue 3: Input Format Error
- Ensure the input is a valid image file path, PIL image object, or tensor
- Check if the image can be opened normally

## License

Consistent with the main project.

