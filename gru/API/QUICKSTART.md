# GRU API Quick Start Guide

## Simplest Usage

```python
from API import predict

# Single image
affine_matrix, tone_params = predict("path/to/image.jpg")

print("Affine transformation matrix (3x4):")
print(affine_matrix)
print("\nTone mapping parameters (4,):")
print(tone_params)
```

## Complete Examples

### 1. Single Image Prediction

```python
from API import predict

# Input image path
image_path = "your_image.jpg"

# Call API
affine_matrix, tone_params = predict(image_path)

# Results
# affine_matrix: numpy array, shape (3, 4)
# tone_params: numpy array, shape (4,)
```

### 2. Batch Prediction for Multiple Images

```python
from API import predict

# Input image list
images = ["img1.jpg", "img2.jpg", "img3.jpg"]

# Call API
affine_matrices, tone_params = predict(images)

# Results
# affine_matrices: numpy array, shape (3, 3, 4) - 3 images, each with 3x4 matrix
# tone_params: numpy array, shape (3, 4) - 3 images, each with 4 parameters
```

### 3. Video Frame Sequence Prediction (Maintaining Hidden State)

```python
from API import predict_sequence

# Input image sequence
frames = ["frame1.jpg", "frame2.jpg", "frame3.jpg"]

# Call API (GRU uses information from previous frames)
affine_matrices, tone_params = predict_sequence(frames)

# Results
# affine_matrices: numpy array, shape (3, 3, 4)
# tone_params: numpy array, shape (3, 4)
```

## Output Format Description

### Affine Transformation Matrix (3x4)

```python
affine_matrix = [
    [a11, a12, a13, tx],  # First row
    [a21, a22, a23, ty],  # Second row
    [a31, a32, a33, tz]   # Third row
]
```

### Tone Mapping Parameters (4)

```python
tone_params = [
    gamma,     # Gamma value
    alpha,     # Alpha value
    beta,      # Beta value
    contrast   # Contrast
]
```

## Common Questions

### Q: How to specify the model path?

A: By default, trained weights are used, but you can also specify:

```python
affine, tone = predict("image.jpg", model_path="/path/to/your/model.pt")
```

### Q: How to use CPU instead of GPU?

A: 

```python
import torch
affine, tone = predict("image.jpg", device=torch.device('cpu'))
```

### Q: How to return torch.Tensor instead of numpy arrays?

A:

```python
affine, tone = predict("image.jpg", return_numpy=False)
```

## More Information

For detailed documentation, please refer to `README.md`

