# GRU Image Transformation API Usage Guide

This API encapsulates the trained GRU model for predicting affine transformation matrices (3x4) and nonlinear tone mapping curve parameters (1x4) for images.

## Features

- Supports single image prediction
- Supports batch image prediction
- Provides model loading interface
- Health check interface
- RESTful API design, easy to integrate

## Installation Dependencies

First, ensure you have Python 3.8 or higher installed, then install the necessary dependencies:

```bash
# Install PyTorch (choose the appropriate command based on your CUDA version)
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# For CPU version
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install fastapi uvicorn pydantic pillow numpy requests
```

Or use the provided requirements.txt file:

```bash
pip install -r requirements.txt
```

## API File Structure

- `api.py`: GRU model API service implementation
- `test_api.py`: API test script
- `requirements.txt`: Project dependency list
- `README_API.md`: API usage instructions

## Starting the API Service

Run the following command in the project root directory to start the API service:

```bash
# Default start (host=0.0.0.0, port=8000)
python api.py

# Or customize host and port via environment variables (Windows CMD)
set GRU_HOST=127.0.0.1 && set GRU_PORT=8080 && python api.py

# Or customize host and port via environment variables (Windows PowerShell)
$env:GRU_HOST='127.0.0.1'; $env:GRU_PORT='8080'; python api.py

# Or customize host and port via environment variables (Linux/Mac)
export GRU_HOST=127.0.0.1 && export GRU_PORT=8080 && python api.py
```

The service runs by default at `http://0.0.0.0:8000` (accessible on the local network). You can customize the server address and port by setting the environment variables `GRU_HOST` and `GRU_PORT`, or by modifying the `uvicorn.run` parameters in the `api.py` file.

## API Interface Description

### 1. Health Check

**URL**: `http://0.0.0.0:8000/health` or `http://[local IP]:8000/health`
**Method**: `GET`
**Description**: Check API service and model status

**Response Example**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda",
  "model_type": "unet"
}
```

### 2. Load Model

**URL**: `http://0.0.0.0:8000/model/load` or `http://[local IP]:8000/model/load`
**Method**: `POST`
**Parameters**:
- `checkpoint_path`: Model weight file path
- `model_type`: Model type ("unet"|"mobilenet"|"mlp")

**Description**: Load the specified model weight file

**Response Example**:
```json
{
  "status": "success",
  "message": "Successfully loaded model: d:/wendang/gru/output/gru_best.pt",
  "model_type": "unet",
  "device": "cuda"
}
```

### 3. Single Image Prediction

**URL**: `http://0.0.0.0:8000/predict/single` or `http://[local IP]:8000/predict/single`
**Method**: `POST`
**Content Type**: `multipart/form-data`
**Parameters**:
- `image`: Image file to predict

**Description**: Predict a single image and return affine transformation matrix and tone mapping parameters

**Response Example**:
```json
{
  "affine_3x4": [
    [1.012, 0.003, 0.001, -0.023],
    [0.002, 1.008, 0.000, 0.015],
    [0.000, 0.000, 1.000, 0.001]
  ],
  "tone_params": [1.05, 0.02, 0.98, 1.02],
  "order": ["gamma", "alpha", "beta", "contrast"]
}
```

### 4. Batch Image Prediction

**URL**: `http://0.0.0.0:8000/predict/batch` or `http://[local IP]:8000/predict/batch`
**Method**: `POST`
**Content Type**: `multipart/form-data`
**Parameters**:
- `images`: List of multiple image files

**Description**: Batch predict multiple images and return affine transformation matrices and tone mapping parameters for each image

**Response Example**:
```json
{
  "total_images": 2,
  "results": [
    {
      "filename": "frame000050.jpg",
      "predictions": {
        "affine_3x4": [[...], [...], [...]],
        "tone_params": [...],
        "order": [...]  
      }
    },
    {
      "filename": "frame000051.jpg",
      "predictions": {
        "affine_3x4": [[...], [...], [...]],
        "tone_params": [...],
        "order": [...]  
      }
    }
  ]
}
```

## Using the Test Script

The project provides a test script `test_api.py` that can be used to test various API functions:

```bash
python test_api.py --checkpoint d:/wendang/gru/output/gru_best.pt
```

Optional parameters:
- `--image`: Single test image path
- `--images`: Multiple test image paths
- `--data-dir`: Directory path containing test images

## Usage Examples

### Python Request Example

```python
import requests

# Single image prediction
url = "http://0.0.0.0:8000/predict/single"  # Or use the actual server IP, e.g., http://192.168.1.x:8000/predict/single
with open("test_image.jpg", "rb") as f:
    files = {"image": f}
    response = requests.post(url, files=files)
    result = response.json()
    print("Affine transformation matrix:", result["affine_3x4"])
    print("Tone mapping parameters:", result["tone_params"])

# Batch image prediction
url = "http://0.0.0.0:8000/predict/batch"  # Or use the actual server IP, e.g., http://192.168.1.x:8000/predict/batch
files = []
for img_path in ["image1.jpg", "image2.jpg"]:
    with open(img_path, "rb") as f:
        files.append(("images", (img_path, f, "image/jpeg")))
response = requests.post(url, files=files)
results = response.json()
```

## Notes

1. Ensure the model weight file exists and is accessible
2. Image formats support common formats like JPG, PNG, etc.
3. The service will attempt to automatically load the default model at startup, with the path `d:/wendang/gru/output/gru_best.pt`
4. For production environment use, it is recommended to add appropriate error handling, logging, and authentication

## Troubleshooting

- If you encounter `ModuleNotFoundError`, ensure all dependencies are installed correctly
- If model loading fails, check if the model file path is correct
- If prediction results are abnormal, check if the input image format is correct