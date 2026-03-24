#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Save prediction results for all data/room data to files
"""

import sys
import os
import json
import numpy as np

# Add project root directory to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from API import predict, predict_sequence


def save_all_results():
    """Process all images and save results"""
    data_dir = "./gru/data/room"
    
    # Get all images
    image_files = sorted([
        os.path.join(data_dir, f) 
        for f in os.listdir(data_dir) 
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    
    if len(image_files) == 0:
        print(f"No image files found in {data_dir}")
        return
    
    print("=" * 60)
    print("GRU Model API - Save All Results")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"Number of images: {len(image_files)}")
    print("=" * 60)
    
    # 1. Batch prediction (reset hidden state each call)
    print("\n[1/2] Performing batch prediction...")
    try:
        affine_matrices_batch, tone_params_batch = predict(image_files)
        print(f"Batch prediction completed: {affine_matrices_batch.shape[0]} images")
    except Exception as e:
        print(f"Batch prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 2. Sequence prediction (maintain hidden state)
    print("\n[2/2] Performing sequence prediction...")
    try:
        affine_matrices_seq, tone_params_seq = predict_sequence(image_files)
        print(f"Sequence prediction completed: {affine_matrices_seq.shape[0]} images")
    except Exception as e:
        print(f"Sequence prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 3. Save batch prediction results
    output_file_batch = "./gru/test_results_all_batch.json"
    print(f"\nSaving batch prediction results to: {output_file_batch}")
    results_batch = []
    for i, (affine, tone) in enumerate(zip(affine_matrices_batch, tone_params_batch)):
        results_batch.append({
            "image": os.path.basename(image_files[i]),
            "image_path": image_files[i],
            "affine_3x4": affine.tolist() if isinstance(affine, np.ndarray) else affine,
            "tone_params": tone.tolist() if isinstance(tone, np.ndarray) else tone,
            "tone_params_detail": {
                "gamma": float(tone[0]),
                "alpha": float(tone[1]),
                "beta": float(tone[2]),
                "contrast": float(tone[3])
            }
        })
    
    with open(output_file_batch, 'w', encoding='utf-8') as f:
        json.dump({
            "total_images": len(image_files),
            "prediction_type": "batch",
            "description": "Batch prediction results (reset hidden state each call)",
            "results": results_batch
        }, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(results_batch)} records")
    
    # 4. Save sequence prediction results
    output_file_seq = "./gru/test_results_all_sequence.json"
    print(f"\nSaving sequence prediction results to: {output_file_seq}")
    results_seq = []
    for i, (affine, tone) in enumerate(zip(affine_matrices_seq, tone_params_seq)):
        results_seq.append({
            "frame": os.path.basename(image_files[i]),
            "frame_path": image_files[i],
            "frame_index": i,
            "affine_3x4": affine.tolist() if isinstance(affine, np.ndarray) else affine,
            "tone_params": tone.tolist() if isinstance(tone, np.ndarray) else tone,
            "tone_params_detail": {
                "gamma": float(tone[0]),
                "alpha": float(tone[1]),
                "beta": float(tone[2]),
                "contrast": float(tone[3])
            }
        })
    
    with open(output_file_seq, 'w', encoding='utf-8') as f:
        json.dump({
            "total_frames": len(image_files),
            "prediction_type": "sequence",
            "description": "Sequence prediction results (GRU maintains hidden state, uses information from previous frames)",
            "results": results_seq
        }, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(results_seq)} records")
    
    # 5. Save summary information
    output_file_summary = "./gru/test_results_summary.txt"
    print(f"\nSaving summary information to: {output_file_summary}")
    with open(output_file_summary, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("GRU Model API Test Results Summary\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Data directory: {data_dir}\n")
        f.write(f"Total images: {len(image_files)}\n")
        f.write(f"Batch prediction results file: {output_file_batch}\n")
        f.write(f"Sequence prediction results file: {output_file_seq}\n\n")
        
        f.write("Prediction results for first 5 images:\n")
        f.write("-" * 60 + "\n")
        for i in range(min(5, len(image_files))):
            f.write(f"\nImage {i+1}: {os.path.basename(image_files[i])}\n")
            f.write(f"  Batch prediction - Tone parameters: gamma={tone_params_batch[i][0]:.6f}, "
                   f"alpha={tone_params_batch[i][1]:.6f}, "
                   f"beta={tone_params_batch[i][2]:.6f}, "
                   f"contrast={tone_params_batch[i][3]:.6f}\n")
            f.write(f"  Sequence prediction - Tone parameters: gamma={tone_params_seq[i][0]:.6f}, "
                   f"alpha={tone_params_seq[i][1]:.6f}, "
                   f"beta={tone_params_seq[i][2]:.6f}, "
                   f"contrast={tone_params_seq[i][3]:.6f}\n")
        
        if len(image_files) > 10:
            f.write(f"\n... (omitting {len(image_files) - 10} images) ...\n\n")
            
            f.write("Prediction results for last 5 images:\n")
            f.write("-" * 60 + "\n")
            for i in range(len(image_files) - 5, len(image_files)):
                f.write(f"\nImage {i+1}: {os.path.basename(image_files[i])}\n")
                f.write(f"  Batch prediction - Tone parameters: gamma={tone_params_batch[i][0]:.6f}, "
                       f"alpha={tone_params_batch[i][1]:.6f}, "
                       f"beta={tone_params_batch[i][2]:.6f}, "
                       f"contrast={tone_params_batch[i][3]:.6f}\n")
                f.write(f"  Sequence prediction - Tone parameters: gamma={tone_params_seq[i][0]:.6f}, "
                       f"alpha={tone_params_seq[i][1]:.6f}, "
                       f"beta={tone_params_seq[i][2]:.6f}, "
                       f"contrast={tone_params_seq[i][3]:.6f}\n")
    
    print(f"Summary information saved")
    
    print("\n" + "=" * 60)
    print("All results saved successfully!")
    print("=" * 60)
    print(f"Batch prediction results: {output_file_batch}")
    print(f"Sequence prediction results: {output_file_seq}")
    print(f"Summary information: {output_file_summary}")
    print("=" * 60)


if __name__ == "__main__":
    save_all_results()

