import argparse
import json
import os
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader

from dataset import GRUDataset, GRUSequenceDataset, _to_tensor
from model import GRUNet, MobileNetGRU, build_input_vector, build_4ch_image
from unet import UNet_DualOutput, UNet_GRU


def predict_single_image(
    model: torch.nn.Module,
    image_path: str,
    device: torch.device,
    model_type: str = "mobilenet",
    hidden_state: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Predict a single image and return affine parameters and tone parameters.

    Args:
        model: Trained model
        image_path: Input image path
        device: Device (CPU/GPU)
        model_type: Model type ("mobilenet", "mlp", "unet", "unet_gru")
        hidden_state: GRU hidden state (only used for unet_gru model, for sequence inference)

    Returns:
        (affine, tone, hidden_state): Affine parameters (12,), tone parameters (4,), and updated hidden state
    """

    if model is not None:
        model.eval()
    
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        rgb = _to_tensor(img)  # C, H, W
    
    # Downsample to 16x8 and compute grayscale
    rgb_16x8 = torch.nn.functional.interpolate(
        rgb.unsqueeze(0), size=(16, 8), mode="bilinear", align_corners=False
    ).squeeze(0)
    
    # Compute grayscale: Y' = 0.299 R + 0.587 G + 0.114 B
    gray_small = (
        0.299 * rgb_16x8[0] + 0.587 * rgb_16x8[1] + 0.114 * rgb_16x8[2]
    )  # (16, 8)
    gray_16x1 = gray_small.mean(dim=1, keepdim=True)  # (16, 1)
    
    rgb_16x8 = rgb_16x8.unsqueeze(0).to(device)  # (1, 3, 16, 8)
    gray_16x1 = gray_16x1.unsqueeze(0).to(device)  # (1, 16, 1)
    
    if model is None:
        # When no model weights are provided, return "identity" affine and "neutral" tone parameters
        affine_identity = torch.tensor([
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
        ], dtype=torch.float32)
        tone_neutral = torch.tensor([1.0, 0.0, 1.0, 1.0], dtype=torch.float32)
        return affine_identity, tone_neutral, None

    with torch.no_grad():
        if model_type == "unet_gru":
            # GRU model: requires sequence input
            gray_expanded = gray_16x1.unsqueeze(1).repeat(1, 1, 1, 8)  # (1, 1, 16, 8)
            x = torch.cat([rgb_16x8.unsqueeze(1), gray_expanded], dim=2)  # (1, 1, 4, 16, 8)
            affine_pred, tone_pred, hidden_state = model(x, hidden_state)
            # Remove sequence dimension
            affine_pred = affine_pred.squeeze(1)  # (1, 12)
            tone_pred = tone_pred.squeeze(1)  # (1, 4)
        elif model_type == "mobilenet":
            x = build_4ch_image(rgb_16x8, gray_16x1)
            affine_pred, tone_pred = model(x)
            hidden_state = None
        elif model_type == "unet":
            gray_expanded = gray_16x1.unsqueeze(1).repeat(1, 1, 1, 8)  # (B,1,16,8)
            x = torch.cat([rgb_16x8, gray_expanded], dim=1)  # (B,4,16,8)
            affine_pred, tone_pred = model(x)
            hidden_state = None
        else:  # mlp
            x = build_input_vector(rgb_16x8, gray_16x1)
            affine_pred, tone_pred = model(x)
            hidden_state = None
    return affine_pred[0].cpu(), tone_pred[0].cpu(), hidden_state  # 去掉batch维度


def save_results(
    affine: torch.Tensor,
    tone: torch.Tensor,
    output_path: str,
):
    """
    Save prediction results to a JSON file.

    Args:
        affine: Affine parameters (12,), row-wise flattened 3x4 matrix
        tone: Tone parameters (4,)
        output_path: Output JSON file path
    """
    affine_np = affine.numpy().tolist()
    tone_np = tone.numpy().tolist()
    
    # Convert to 3x4 matrix format (consistent with original data format)
    affine_matrix = [
        affine_np[0:4],   # First row
        affine_np[4:8],   # Second row
        affine_np[8:12],  # Third row
    ]
    
    result = {
        "affine_3x4": affine_matrix,
        "tone_params": tone_np,
        "order": ["gamma", "alpha", "beta", "contrast"]  # Tone parameter order explanation
    }
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"Prediction results saved to: {output_path}")


def _list_images_in_dir(directory: str) -> List[str]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    files = []
    for name in sorted(os.listdir(directory)):
        p = os.path.join(directory, name)
        if os.path.isfile(p) and Path(p).suffix.lower() in exts:
            files.append(p)
    return files


def validate_model(
    model: torch.nn.Module,
    data_root: str,
    image_dir: str = "val/image",
    label_dir: str = "val/label",
    batch_size: int = 64,
    device: torch.device = torch.device("cuda"),
    model_type: str = "unet",
    sequence_length: int = 1,
    sequence_stride: int = 1,
) -> Dict[str, float]:
    """
    Evaluate model performance on validation set
    
    Args:
        model: Trained model
        data_root: Dataset root directory
        image_dir: Image directory (relative to data_root)
        label_dir: Label directory (relative to data_root)
        batch_size: Batch size
        device: Device
        model_type: Model type
        
    Returns:
        Dictionary containing evaluation metrics
    """

    # Build full paths
    image_path = os.path.join(data_root, image_dir)
    label_path = os.path.join(data_root, label_dir)
    
    # Create dataset based on model type
    if model_type == "unet_gru":
        dataset = GRUSequenceDataset(
            image_dir=image_path,
            label_dir=label_path,
            sequence_length=sequence_length,
            stride=sequence_stride
        )
    else:
        dataset = GRUDataset(image_dir=image_path, label_dir=label_path)
    
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    model.eval()
    l1_criterion = nn.SmoothL1Loss(beta=0.02)
    
    total_loss = 0.0
    total_affine_loss = 0.0
    total_tone_loss = 0.0
    num_batches = 0
    hidden = None
    
    with torch.no_grad():
        for batch in loader:
            if model_type == "unet_gru":
                # Sequence data
                rgb_16x8 = batch["rgb_16x8"].to(device)  # (B, seq_len, 3, 16, 8)
                gray_16x1 = batch["gray_16x1"].to(device)  # (B, seq_len, 16, 1)
                affine_tgt = batch["affine"].to(device)  # (B, seq_len, 12)
                tone_tgt = batch["tone"].to(device)  # (B, seq_len, 4)
                
                gray_expanded = gray_16x1.unsqueeze(2).repeat(1, 1, 1, 1, 8)  # (B, seq_len, 1, 16, 8)
                x = torch.cat([rgb_16x8, gray_expanded], dim=2)  # (B, seq_len, 4, 16, 8)
                
                hidden = None  # Reset hidden state for each batch
                affine_pred, tone_pred, hidden = model(x, hidden)
                
                # Calculate loss for all time steps
                loss_affine = l1_criterion(affine_pred.view(-1, 12), affine_tgt.view(-1, 12))
                loss_tone = l1_criterion(tone_pred.view(-1, 4), tone_tgt.view(-1, 4))
            else:
                # Non-sequence data
                rgb_16x8 = batch["rgb_16x8"].to(device)
                gray_16x1 = batch["gray_16x1"].to(device)
                affine_tgt = batch["affine"].to(device)
                tone_tgt = batch["tone"].to(device)
                
                if model_type == "mobilenet":
                    x = build_4ch_image(rgb_16x8, gray_16x1)
                elif model_type == "unet":
                    gray_expanded = gray_16x1.unsqueeze(1).repeat(1, 1, 1, 8)  # (B,1,16,8)
                    x = torch.cat([rgb_16x8, gray_expanded], dim=1)  # (B,4,16,8)
                else:  # mlp
                    x = build_input_vector(rgb_16x8, gray_16x1)
                
                affine_pred, tone_pred = model(x)
                loss_affine = l1_criterion(affine_pred, affine_tgt)
                loss_tone = l1_criterion(tone_pred, tone_tgt)
            
            loss = loss_affine + loss_tone
            
            total_loss += loss.item()
            total_affine_loss += loss_affine.item()
            total_tone_loss += loss_tone.item()
            num_batches += 1
    
    # Calculate average loss
    avg_loss = total_loss / num_batches
    avg_affine_loss = total_affine_loss / num_batches
    avg_tone_loss = total_tone_loss / num_batches
    
    metrics = {
        "total_loss": avg_loss,
        "affine_loss": avg_affine_loss,
        "tone_loss": avg_tone_loss
    }
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Use trained model for inference and validation")
    parser.add_argument("--checkpoint", type=str, default="/home/zsh/GRU训练/Luminance-GS/Luminance-GS/pinggu1/output/gru_best.pt",
                        help="Model weight file path (.pt file)")
    parser.add_argument("--image", type=str, default=None,
                        help="Single input image path (.png or .jpg)")
    parser.add_argument("--renders_dir", type=str, default=None,
                        help="Image directory (if specified, perform inference on all images in the directory)")
    parser.add_argument("--output_dir", type=str, default="/home/zsh/GRU训练/Luminance-GS/Luminance-GS/pinggu1/output/predictions",
                        help="Output JSON file save directory")
    parser.add_argument("--model_type", type=str, choices=["mobilenet", "mlp", "unet", "unet_gru"], default="unet",
                        help="Model architecture type")
    parser.add_argument("--sequence_length", type=int, default=1,
                        help="Sequence length (only for unet_gru model)")
    parser.add_argument("--sequence_stride", type=int, default=1,
                        help="Sequence sliding window step (only for unet_gru model)")
    parser.add_argument("--gru_hidden_dim", type=int, default=256,
                        help="GRU hidden layer dimension (only for unet_gru model)")
    parser.add_argument("--gru_layers", type=int, default=2,
                        help="GRU layers (only for unet_gru model)")
    parser.add_argument("--mobilenet_variant", type=str, choices=["v2", "v3"], default="v2")
    parser.add_argument("--hidden_dim", type=int, default=512,
                        help="MLP model hidden layer dimension (only used when model_type=mlp)")
    parser.add_argument("--dropout", type=float, default=0.1)
    
    # Validation related parameters
    parser.add_argument("--validate", action="store_true",
                        help="Evaluate model performance on validation set")
    parser.add_argument("--data_root", type=str, default="/home/zsh/GRU训练/Luminance-GS/Luminance-GS/data2",
                        help="Dataset root directory")
    parser.add_argument("--val_image_dir", type=str, default="val/image",
                        help="Validation image directory (relative to data_root)")
    parser.add_argument("--val_label_dir", type=str, default="val/label",
                        help="Validation label directory (relative to data_root)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for validation")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = None
    if args.checkpoint:
        # Safely load checkpoint, prefer weights only
        try:
            checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)  # PyTorch >= 2.4
        except TypeError:
            checkpoint = torch.load(args.checkpoint, map_location=device)
        
        # Instantiate model based on model type
        if args.model_type == "unet_gru":
            model = UNet_GRU(
                in_chns=4,
                hidden_dim=args.gru_hidden_dim,
                gru_layers=args.gru_layers,
                dropout=args.dropout
            )
        elif args.model_type == "mobilenet":
            model = MobileNetGRU(variant=args.mobilenet_variant, pretrained=False, dropout=args.dropout)
        elif args.model_type == "unet":
            model = UNet_DualOutput(in_chns=4)  # 4-channel input: RGB + grayscale
        else:  # mlp
            model = GRUNet(hidden_dim=args.hidden_dim, dropout=args.dropout)
        
        # Compatible with multiple storage formats: {'model': state_dict} or {'state_dict': ...} or direct state_dict
        if isinstance(checkpoint, dict):
            if "model" in checkpoint and isinstance(checkpoint["model"], dict):
                state_dict = checkpoint["model"]
            elif "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
                state_dict = checkpoint["state_dict"]
            else:
                # May be direct state_dict or contain other nesting
                # Try to use as state_dict
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        print(f"Model loaded: {args.checkpoint}")
    else:
        print("Warning: No model weight file provided, will use default parameters for prediction")
    
    # Validation mode
    if args.validate:
        if model is None:
            parser.error("Validation mode requires model weight file")
        
        print(f"\nStarting model evaluation on validation set...")
        metrics = validate_model(
            model=model,
            data_root=args.data_root,
            image_dir=args.val_image_dir,
            label_dir=args.val_label_dir,
            batch_size=args.batch_size,
            device=device,
            model_type=args.model_type,
            sequence_length=args.sequence_length,
            sequence_stride=args.sequence_stride
        )
        
        print(f"Validation results:")
        print(f"  Total loss: {metrics['total_loss']:.6f}")
        print(f"  Affine matrix loss: {metrics['affine_loss']:.6f}")
        print(f"  Tone parameter loss: {metrics['tone_loss']:.6f}")
    
    # Inference mode
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Process single image
        if args.image:
            image_path = args.image
            hidden = None
            affine, tone, hidden = predict_single_image(model, image_path, device, args.model_type, hidden)
            
            # Generate output filename
            image_name = Path(image_path).stem
            output_path = os.path.join(args.output_dir, f"{image_name}_prediction.json")
            
            save_results(affine, tone, output_path)
        
        # Process entire directory
        elif args.renders_dir:
            image_paths = _list_images_in_dir(args.renders_dir)
            print(f"Found {len(image_paths)} images, starting inference...")
            
            # For GRU model, maintain hidden state for sequence inference
            hidden = None
            for image_path in image_paths:
                stem = Path(image_path).stem
                affine, tone, hidden = predict_single_image(model, image_path, device, args.model_type, hidden)
                output_path = os.path.join(args.output_dir, f"{stem}_prediction.json")
                save_results(affine, tone, output_path)
            
            print(f"All prediction results saved to: {args.output_dir}")
        
        else:
            parser.error("Must specify one of --validate, --image, or --renders_dir")


if __name__ == "__main__":
    main()

