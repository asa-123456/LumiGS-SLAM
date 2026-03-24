#!/usr/bin/env python3
"""
Plot GRU model convergence curves based on training logs
"""
import re
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # Support Chinese display
matplotlib.rcParams['axes.unicode_minus'] = False  # Fix negative sign display issue

def parse_log_file(log_path):
    """Parse log file and extract training metrics"""
    epochs = []
    train_losses = []
    train_affine_losses = []
    train_tone_losses = []
    val_losses = []
    val_affine_losses = []
    val_tone_losses = []
    
    # Regular expression to match log format
    pattern = r'Epoch\s+(\d+)\s+\|\s+train\s+([\d.]+)\s+\(affine:\s+([\d.]+),\s+tone:\s+([\d.]+)\)\s+\|\s+val\s+([\d.]+)\s+\(affine:\s+([\d.]+),\s+tone:\s+([\d.]+)\)'
    
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                epoch = int(match.group(1))
                train_loss = float(match.group(2))
                train_affine = float(match.group(3))
                train_tone = float(match.group(4))
                val_loss = float(match.group(5))
                val_affine = float(match.group(6))
                val_tone = float(match.group(7))
                
                epochs.append(epoch)
                train_losses.append(train_loss)
                train_affine_losses.append(train_affine)
                train_tone_losses.append(train_tone)
                val_losses.append(val_loss)
                val_affine_losses.append(val_affine)
                val_tone_losses.append(val_tone)
    
    return {
        'epochs': epochs,
        'train_losses': train_losses,
        'train_affine_losses': train_affine_losses,
        'train_tone_losses': train_tone_losses,
        'val_losses': val_losses,
        'val_affine_losses': val_affine_losses,
        'val_tone_losses': val_tone_losses
    }


def plot_convergence_curves(data, output_path='gru_convergence.png'):
    """Plot convergence curves"""
    epochs = data['epochs']
    
    # Create 3 subplots: total loss, affine loss, tone loss
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    # fig.suptitle('GRU Model Training Convergence Curves', fontsize=16, fontweight='bold')
    
    # 1. Total loss
    axes[0].plot(epochs, data['train_losses'], label='Train Loss', linewidth=1.5, alpha=0.8)
    axes[0].plot(epochs, data['val_losses'], label='Validation Loss', linewidth=1.5, alpha=0.8)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Total Loss', fontsize=12)
    axes[0].set_title('Total Loss Convergence', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # 2. Affine loss
    axes[1].plot(epochs, data['train_affine_losses'], label='Train Affine Loss', linewidth=1.5, alpha=0.8, color='green')
    axes[1].plot(epochs, data['val_affine_losses'], label='Validation Affine Loss', linewidth=1.5, alpha=0.8, color='orange')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Affine Loss', fontsize=12)
    axes[1].set_title('Affine Loss Convergence', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    # 3. Tone loss
    axes[2].plot(epochs, data['train_tone_losses'], label='Train Tone Loss', linewidth=1.5, alpha=0.8, color='purple')
    axes[2].plot(epochs, data['val_tone_losses'], label='Validation Tone Loss', linewidth=1.5, alpha=0.8, color='red')
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('Tone Loss', fontsize=12)
    axes[2].set_title('Tone Loss Convergence', fontsize=13, fontweight='bold')
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Convergence curves saved to: {output_path}")
    
    # Print statistics
    print("\nTraining statistics:")
    print(f"Total epochs: {len(epochs)}")
    print(f"Final training loss: {data['train_losses'][-1]:.6f}")
    print(f"Final validation loss: {data['val_losses'][-1]:.6f}")
    print(f"Best validation loss: {min(data['val_losses']):.6f} (Epoch {epochs[data['val_losses'].index(min(data['val_losses']))]})")
    print(f"Final affine loss: {data['train_affine_losses'][-1]:.6f} (train) / {data['val_affine_losses'][-1]:.6f} (val)")
    print(f"Final tone loss: {data['train_tone_losses'][-1]:.6f} (train) / {data['val_tone_losses'][-1]:.6f} (val)")


if __name__ == '__main__':
    import sys
    
    log_path = '/home/zsh107552403865/gru/output-gru-2.log'
    output_path = '/home/zsh107552403865/gru/gru_convergence-3.png'
    
    if len(sys.argv) > 1:
        log_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    
    print(f"Parsing log file: {log_path}")
    data = parse_log_file(log_path)
    
    if not data['epochs']:
        print("Error: Failed to extract any data from the log file")
        sys.exit(1)
    
    print(f"Successfully parsed data for {len(data['epochs'])} epochs")
    plot_convergence_curves(data, output_path)

