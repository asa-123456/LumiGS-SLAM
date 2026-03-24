import argparse
import os
from typing import Optional, Tuple

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from dataset import GRUDataset, GRUSequenceDataset
from model import GRUNet, MobileNetGRU, build_input_vector, build_4ch_image
from unet import UNet_DualOutput, UNet_GRU


def loss_fn(
    affine_pred: torch.Tensor,
    tone_pred: torch.Tensor,
    affine_tgt: torch.Tensor,
    tone_tgt: torch.Tensor,
    affine_weight: float = 1.0,
    tone_weight: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate loss function
    
    Args:
        affine_weight: Weight for affine matrix loss (can be increased to focus more on affine matrix learning)
        tone_weight: Weight for tone parameter loss
    """
    l1 = nn.SmoothL1Loss(beta=0.02)
    loss_affine = l1(affine_pred, affine_tgt) * affine_weight
    loss_tone = l1(tone_pred, tone_tgt) * tone_weight
    loss = loss_affine + loss_tone
    return loss, loss_affine / affine_weight, loss_tone / tone_weight  # Return original loss values for monitoring


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler],
    max_grad_norm: float,
    affine_weight: float = 1.0,
    tone_weight: float = 1.0,
):
    model.train()
    total_loss = 0.0
    total_affine_loss = 0.0
    total_tone_loss = 0.0
    
    # For GRU model, need to maintain hidden state
    hidden = None
    
    for batch in loader:
        # Check if it's sequence data (UNet_GRU)
        if isinstance(model, UNet_GRU):
            # Sequence data: rgb_16x8 shape is (B, seq_len, 3, 16, 8)
            rgb_16x8 = batch["rgb_16x8"].to(device)  # (B, seq_len, 3, 16, 8)
            gray_16x1 = batch["gray_16x1"].to(device)  # (B, seq_len, 16, 1)
            affine_tgt = batch["affine"].to(device)  # (B, seq_len, 12)
            tone_tgt = batch["tone"].to(device)  # (B, seq_len, 4)
            
            # Build 4-channel input: expand grayscale image and concatenate
            batch_size, seq_len = rgb_16x8.shape[0], rgb_16x8.shape[1]
            gray_expanded = gray_16x1.unsqueeze(2).repeat(1, 1, 1, 1, 8)  # (B, seq_len, 1, 16, 8)
            x = torch.cat([rgb_16x8, gray_expanded], dim=2)  # (B, seq_len, 4, 16, 8)
            
            optimizer.zero_grad(set_to_none=True)
            use_amp = scaler is not None
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                # Reset hidden state (reset at the start of each batch)
                hidden = None
                affine_pred, tone_pred, hidden = model(x, hidden)
                # affine_pred: (B, seq_len, 12), tone_pred: (B, seq_len, 4)
                # Calculate average loss for all time steps
                loss, loss_affine, loss_tone = loss_fn(
                    affine_pred.view(-1, 12), tone_pred.view(-1, 4),
                    affine_tgt.view(-1, 12), tone_tgt.view(-1, 4),
                    affine_weight, tone_weight
                )
        else:
            # Non-sequence data: original logic
            rgb_16x8 = batch["rgb_16x8"].to(device)  # (B,3,16,8)
            gray_16x1 = batch["gray_16x1"].to(device)  # (B,16,1)
            affine_tgt = batch["affine"].to(device)  # (B,12)
            tone_tgt = batch["tone"].to(device)  # (B,4)

            # Build input based on model type
            if isinstance(model, MobileNetGRU):
                x = build_4ch_image(rgb_16x8, gray_16x1)
            elif isinstance(model, UNet_DualOutput):
                gray_expanded = gray_16x1.unsqueeze(1).repeat(1, 1, 1, 8)  # (B,1,16,8)
                x = torch.cat([rgb_16x8, gray_expanded], dim=1)  # (B,4,16,8)
            else:
                x = build_input_vector(rgb_16x8, gray_16x1)

            optimizer.zero_grad(set_to_none=True)
            use_amp = scaler is not None
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                affine_pred, tone_pred = model(x)
                loss, loss_affine, loss_tone = loss_fn(affine_pred, tone_pred, affine_tgt, tone_tgt, 
                                                       affine_weight, tone_weight)

        if scaler is not None:
            scaler.scale(loss).backward()
            if max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        total_loss += loss.item()
        total_affine_loss += loss_affine.item()
        total_tone_loss += loss_tone.item()

    n = max(1, len(loader))
    return total_loss / n, total_affine_loss / n, total_tone_loss / n


def validate(model: nn.Module, loader: DataLoader, device: torch.device, 
             affine_weight: float = 1.0, tone_weight: float = 1.0):
    model.eval()
    total_loss = 0.0
    total_affine_loss = 0.0
    total_tone_loss = 0.0
    hidden = None
    
    with torch.no_grad():
        for batch in loader:
            if isinstance(model, UNet_GRU):
                # Sequence data
                rgb_16x8 = batch["rgb_16x8"].to(device)  # (B, seq_len, 3, 16, 8)
                gray_16x1 = batch["gray_16x1"].to(device)  # (B, seq_len, 16, 1)
                affine_tgt = batch["affine"].to(device)  # (B, seq_len, 12)
                tone_tgt = batch["tone"].to(device)  # (B, seq_len, 4)
                
                gray_expanded = gray_16x1.unsqueeze(2).repeat(1, 1, 1, 1, 8)  # (B, seq_len, 1, 16, 8)
                x = torch.cat([rgb_16x8, gray_expanded], dim=2)  # (B, seq_len, 4, 16, 8)
                
                hidden = None
                affine_pred, tone_pred, hidden = model(x, hidden)
                loss, loss_affine, loss_tone = loss_fn(
                    affine_pred.view(-1, 12), tone_pred.view(-1, 4),
                    affine_tgt.view(-1, 12), tone_tgt.view(-1, 4),
                    affine_weight, tone_weight
                )
            else:
                # Non-sequence data
                rgb_16x8 = batch["rgb_16x8"].to(device)
                gray_16x1 = batch["gray_16x1"].to(device)
                affine_tgt = batch["affine"].to(device)
                tone_tgt = batch["tone"].to(device)
                if isinstance(model, MobileNetGRU):
                    x = build_4ch_image(rgb_16x8, gray_16x1)
                elif isinstance(model, UNet_DualOutput):
                    gray_expanded = gray_16x1.unsqueeze(1).repeat(1, 1, 1, 8)  # (B,1,16,8)
                    x = torch.cat([rgb_16x8, gray_expanded], dim=1)  # (B,4,16,8)
                else:
                    x = build_input_vector(rgb_16x8, gray_16x1)
                affine_pred, tone_pred = model(x)
                loss, loss_affine, loss_tone = loss_fn(affine_pred, tone_pred, affine_tgt, tone_tgt,
                                                       affine_weight, tone_weight)
            total_loss += loss.item()
            total_affine_loss += loss_affine.item()
            total_tone_loss += loss_tone.item()
    n = max(1, len(loader))
    return total_loss / n, total_affine_loss / n, total_tone_loss / n


def main():
    parser = argparse.ArgumentParser(description="Train model for affine and tone estimation")
    parser.add_argument("--data_root", type=str, default="/home/zsh107552403865/gru/data4",
                        help="Root directory for data (default: /home/zsh107552403865/gru/data4)")
    parser.add_argument("--image_dir", type=str, default="train/image",
                        help="Relative path to image directory from data_root (default: train/image)")
    parser.add_argument("--label_dir", type=str, default="train/label",
                        help="Relative path to label directory from data_root (default: train/label)")
    parser.add_argument("--val_image_dir", type=str, default="val/image",
                        help="Relative path to validation image directory from data_root (default: val/image)")
    parser.add_argument("--val_label_dir", type=str, default="val/label",
                        help="Relative path to validation label directory from data_root (default: val/label)")
    parser.add_argument("--use_val", action="store_true",
                        help="Use validation dataset")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--early_stop_patience", type=int, default=0,
                        help="Early stopping patience: stop training if validation loss doesn't decrease for N consecutive epochs")
    parser.add_argument("--out_dir", type=str, default="/home/zsh107552403865/gru/output5")
    parser.add_argument("--model_type", type=str, choices=["mobilenet", "mlp", "unet", "unet_gru"], default="unet",
                        help="Choose model type: 'unet' (default), 'unet_gru' (GRU-based), 'mobilenet' or 'mlp'")
    parser.add_argument("--sequence_length", type=int, default=1,
                        help="Sequence length for GRU model (default: 1, meaning single image)")
    parser.add_argument("--sequence_stride", type=int, default=1,
                        help="Stride for sequence sliding window (default: 1)")
    parser.add_argument("--gru_hidden_dim", type=int, default=256,
                        help="GRU hidden dimension (default: 256)")
    parser.add_argument("--gru_layers", type=int, default=2,
                        help="Number of GRU layers (default: 2)")
    parser.add_argument("--mobilenet_variant", type=str, choices=["v2", "v3"], default="v2")
    parser.add_argument("--mobilenet_pretrained", action="store_true")
    parser.add_argument("--affine_weight", type=float, default=2.0,
                        help="Weight for affine matrix loss (default: 2.0, increase to focus more on affine matrix learning)")
    parser.add_argument("--tone_weight", type=float, default=1.0,
                        help="Weight for tone parameter loss (default: 1.0)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build full paths
    image_path = os.path.join(args.data_root, args.image_dir)
    label_path = os.path.join(args.data_root, args.label_dir)
    
    # Create dataset based on model type
    if args.model_type == "unet_gru":
        # Use sequence dataset
        train_set = GRUSequenceDataset(
            image_dir=image_path,
            label_dir=label_path,
            sequence_length=args.sequence_length,
            stride=args.sequence_stride
        )
        print(f"Using GRUSequenceDataset with sequence_length={args.sequence_length}, stride={args.sequence_stride}")
    else:
        # Use regular dataset
        train_set = GRUDataset(image_dir=image_path, label_dir=label_path)
    
    # Create validation dataset (if enabled)
    val_set = None
    if args.use_val:
        val_image_path = os.path.join(args.data_root, args.val_image_dir)
        val_label_path = os.path.join(args.data_root, args.val_label_dir)
        if args.model_type == "unet_gru":
            val_set = GRUSequenceDataset(
                image_dir=val_image_path,
                label_dir=val_label_path,
                sequence_length=args.sequence_length,
                stride=args.sequence_stride
            )
        else:
            val_set = GRUDataset(image_dir=val_image_path, label_dir=val_label_path)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = (
        DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        if val_set is not None else None
    )

    # Create model based on model type
    if args.model_type == "unet_gru":
        # Use UNet_GRU model, combining UNet encoder with GRU
        model = UNet_GRU(
            in_chns=4,
            hidden_dim=args.gru_hidden_dim,
            gru_layers=args.gru_layers,
            dropout=args.dropout
        ).to(device)
        print(f"Using UNet_GRU model (hidden_dim: {args.gru_hidden_dim}, layers: {args.gru_layers}, sequence_length: {args.sequence_length})")
    elif args.model_type == "unet":
        # Use UNet_DualOutput model with 4 input channels (RGB + grayscale)
        model = UNet_DualOutput(in_chns=4).to(device)
        print("Using UNet_DualOutput model with combined RGB and grayscale input")
    elif args.model_type == "mobilenet":
        model = MobileNetGRU(variant=args.mobilenet_variant, pretrained=args.mobilenet_pretrained, dropout=args.dropout).to(device)
        print(f"Using MobileNetGRU model (variant: {args.mobilenet_variant})")
    else:
        model = GRUNet(hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
        print(f"Using GRUNet model (hidden_dim: {args.hidden_dim})")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if device.type == "cuda":
        scaler = torch.amp.GradScaler('cuda')
    else:
        scaler = None  # No scaler needed for CPU

    best_val = float("inf")
    patience_counter = 0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_affine_loss, train_tone_loss = train_one_epoch(
            model, train_loader, optimizer, device, scaler, args.max_grad_norm,
            args.affine_weight, args.tone_weight
        )
        if val_loader is not None:
            val_loss, val_affine_loss, val_tone_loss = validate(
                model, val_loader, device, args.affine_weight, args.tone_weight
            )
        else:
            val_loss = train_loss
            val_affine_loss = train_affine_loss
            val_tone_loss = train_tone_loss

        print(f"Epoch {epoch:03d} | train {train_loss:.6f} (affine: {train_affine_loss:.6f}, tone: {train_tone_loss:.6f}) | "
              f"val {val_loss:.6f} (affine: {val_affine_loss:.6f}, tone: {val_tone_loss:.6f})")

        # Save latest weights
        latest_path = os.path.join(args.out_dir, "gru_latest.pt")
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_loss": val_loss,
        }, latest_path)

        # Save best weights
        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            best_path = os.path.join(args.out_dir, "gru_best.pt")
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_loss": val_loss,
            }, best_path)
            print(f"Saved best checkpoint: {best_path}")
        else:
            patience_counter += 1
        
        # Early stopping
        if args.early_stop_patience > 0 and patience_counter >= args.early_stop_patience:
            print(f"\nEarly stopping triggered: validation loss hasn't improved for {args.early_stop_patience} consecutive epochs")
            print(f"Best validation loss: {best_val:.6f} (Epoch {epoch - patience_counter})")
            break


if __name__ == "__main__":
    main()


