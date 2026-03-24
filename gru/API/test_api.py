#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test API using data/room data
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


def test_single_image():
    """Test single image"""
    print("=" * 60)
    print("Test 1: Single image prediction")
    print("=" * 60)
    
    image_path = "./gru/data/room/frame000049.jpg"
    
    if not os.path.exists(image_path):
        print(f"❌ Image file does not exist: {image_path}")
        return False
    
    print(f"Input image: {os.path.basename(image_path)}")
    
    try:
        affine_matrix, tone_params = predict(image_path)
        
        print(f"\n✅ Prediction successful!")
        print(f"\nAffine transformation matrix (3x4):")
        print(affine_matrix)
        print(f"\nTone mapping parameters (4,):")
        print(f"  gamma:    {tone_params[0]:.6f}")
        print(f"  alpha:    {tone_params[1]:.6f}")
        print(f"  beta:     {tone_params[2]:.6f}")
        print(f"  contrast: {tone_params[3]:.6f}")
        print(f"\nOutput shapes:")
        print(f"  Affine matrix: {affine_matrix.shape}")
        print(f"  Tone parameters: {tone_params.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multiple_images(limit=None, save_results=False):
    """Test multiple images"""
    print("\n" + "=" * 60)
    print("Test 2: Batch prediction for multiple images")
    print("=" * 60)
    
    data_dir = "./gru/data/room"
    
    # Get all images
    image_files = sorted([
        os.path.join(data_dir, f) 
        for f in os.listdir(data_dir) 
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    
    # If limit is specified, only take first N images
    if limit is not None:
        image_files = image_files[:limit]
    
    if len(image_files) == 0:
        print(f"❌ No image files found in {data_dir}")
        return False
    
    print(f"Number of input images: {len(image_files)}")
    if len(image_files) <= 10:
        for i, img_path in enumerate(image_files):
            print(f"  {i+1}. {os.path.basename(img_path)}")
    else:
        print(f"  First 5: {[os.path.basename(f) for f in image_files[:5]]}")
        print(f"  ... (total {len(image_files)} images)")
        print(f"  Last 5: {[os.path.basename(f) for f in image_files[-5:]]}")
    
    try:
        affine_matrices, tone_params = predict(image_files)
        
        print(f"\n✅ Batch prediction successful!")
        print(f"\nOutput shapes:")
        print(f"  Affine matrices: {affine_matrices.shape}")  # (N, 3, 4)
        print(f"  Tone parameters: {tone_params.shape}")     # (N, 4)
        
        # Save results to file
        if save_results:
            output_file = "./gru/test_results_all.json"
            results = []
            for i, (affine, tone) in enumerate(zip(affine_matrices, tone_params)):
                results.append({
                    "image": os.path.basename(image_files[i]),
                    "affine_3x4": affine.tolist() if isinstance(affine, np.ndarray) else affine,
                    "tone_params": tone.tolist() if isinstance(tone, np.ndarray) else tone,
                    "tone_params_detail": {
                        "gamma": float(tone[0]),
                        "alpha": float(tone[1]),
                        "beta": float(tone[2]),
                        "contrast": float(tone[3])
                    }
                })
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"All results saved to: {output_file}")
        
        # Display detailed results for first 3 and last 3 images
        print(f"Detailed results for first 3 images:")
        for i in range(min(3, len(image_files))):
            affine = affine_matrices[i]
            tone = tone_params[i]
            print(f"Image {i+1} ({os.path.basename(image_files[i])}):")
            print(f"Affine matrix:\n{affine}")
            print(f"Tone parameters: gamma={tone[0]:.6f}, alpha={tone[1]:.6f}, "
                  f"beta={tone[2]:.6f}, contrast={tone[3]:.6f}")
        
        if len(image_files) > 6:
            print(f"\n... (omitting results for {len(image_files) - 6} images) ...")
            print(f"\nDetailed results for last 3 images:")
            for i in range(max(3, len(image_files) - 3), len(image_files)):
                affine = affine_matrices[i]
                tone = tone_params[i]
                print(f"Image {i+1} ({os.path.basename(image_files[i])}):")
                print(f"Affine matrix:\n{affine}")
                print(f"Tone parameters: gamma={tone[0]:.6f}, alpha={tone[1]:.6f}, "
                      f"beta={tone[2]:.6f}, contrast={tone[3]:.6f}")
        elif len(image_files) > 3:
            print(f"Detailed results for remaining images:")
            for i in range(3, len(image_files)):
                affine = affine_matrices[i]
                tone = tone_params[i]
                print(f"Image {i+1} ({os.path.basename(image_files[i])}):")
                print(f"Affine matrix:\n{affine}")
                print(f"Tone parameters: gamma={tone[0]:.6f}, alpha={tone[1]:.6f}, "
                      f"beta={tone[2]:.6f}, contrast={tone[3]:.6f}")
        
        return True
    except Exception as e:
        print(f"Batch prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sequence(limit=None, save_results=False):
    """Test sequence prediction"""
    print("\n" + "=" * 60)
    print("Test 3: Sequence prediction (maintaining hidden state)")
    print("=" * 60)
    
    data_dir = "./gru/data/room"
    
    # Get all images as sequence
    image_files = sorted([
        os.path.join(data_dir, f) 
        for f in os.listdir(data_dir) 
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    
    # If limit is specified, only take first N images
    if limit is not None:
        image_files = image_files[:limit]
    
    if len(image_files) == 0:
        print(f"No image files found in {data_dir}")
        return False
    
    print(f"Input sequence length: {len(image_files)}")
    if len(image_files) <= 10:
        for i, img_path in enumerate(image_files):
            print(f"  Frame {i+1}: {os.path.basename(img_path)}")
    else:
        print(f"  First 5 frames: {[os.path.basename(f) for f in image_files[:5]]}")
        print(f"  ... (total {len(image_files)} frames)")
        print(f"  Last 5 frames: {[os.path.basename(f) for f in image_files[-5:]]}")
    
    try:
        affine_matrices, tone_params = predict_sequence(image_files)
        
        print(f"Sequence prediction successful!")
        print(f"Output shapes:")
        print(f"Affine matrices: {affine_matrices.shape}")  # (N, 3, 4)
        print(f"Tone parameters: {tone_params.shape}")     # (N, 4)
        
        # Save results to file
        if save_results:
            output_file = "./gru/test_results_sequence.json"
            results = []
            for i, (affine, tone) in enumerate(zip(affine_matrices, tone_params)):
                results.append({
                    "frame": os.path.basename(image_files[i]),
                    "affine_3x4": affine.tolist() if isinstance(affine, np.ndarray) else affine,
                    "tone_params": tone.tolist() if isinstance(tone, np.ndarray) else tone,
                    "tone_params_detail": {
                        "gamma": float(tone[0]),
                        "alpha": float(tone[1]),
                        "beta": float(tone[2]),
                        "contrast": float(tone[3])
                    }
                })
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"All results saved to: {output_file}")
        
        # Display detailed results for first 3 and last 3 frames
        print(f"Detailed results for first 3 frames (GRU uses information from previous frames):")
        for i in range(min(3, len(image_files))):
            affine = affine_matrices[i]
            tone = tone_params[i]
            print(f"Frame {i+1} ({os.path.basename(image_files[i])}):")
            print(f"  Affine matrix:\n{affine}")
            print(f"  Tone parameters: gamma={tone[0]:.6f}, alpha={tone[1]:.6f}, "
                  f"beta={tone[2]:.6f}, contrast={tone[3]:.6f}")
        
        if len(image_files) > 6:
            print(f"(omitting results for {len(image_files) - 6} frames) ...")
            print(f"Detailed results for last 3 frames:")
            for i in range(max(3, len(image_files) - 3), len(image_files)):
                affine = affine_matrices[i]
                tone = tone_params[i]
                print(f"Frame {i+1} ({os.path.basename(image_files[i])}):")
                print(f"Affine matrix:\n{affine}")
                print(f"Tone parameters: gamma={tone[0]:.6f}, alpha={tone[1]:.6f}, "
                      f"beta={tone[2]:.6f}, contrast={tone[3]:.6f}")
        elif len(image_files) > 3:
            print(f"Detailed results for remaining frames:")
            for i in range(3, len(image_files)):
                affine = affine_matrices[i]
                tone = tone_params[i]
                print(f"\nFrame {i+1} ({os.path.basename(image_files[i])}):")
                print(f"  Affine matrix:\n{affine}")
                print(f"  Tone parameters: gamma={tone[0]:.6f}, alpha={tone[1]:.6f}, "
                      f"beta={tone[2]:.6f}, contrast={tone[3]:.6f}")
        
        return True
    except Exception as e:
        print(f"Sequence prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="GRU Model API Test")
    parser.add_argument("--no-save", action="store_true", help="Do not save results to file (default: save)")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of images to process (for quick testing)")
    args = parser.parse_args()
    
    # Default to save, unless --no-save is specified
    save_results = not args.no_save
    
    print("GRU Model API Test")
    print("=" * 60)
    print(f"Test data directory: ./gru/data/room")
    if args.limit:
        print(f"Limiting to first {args.limit} images (for quick testing)")
    if save_results:
        print(f"Will save all results to JSON files")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Single image prediction", test_single_image()))
    results.append(("Batch prediction", test_multiple_images(limit=args.limit, save_results=save_results)))
    results.append(("Sequence prediction", test_sequence(limit=args.limit, save_results=save_results)))
    
    # Summarize results
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    for test_name, success in results:
        status = "Passed" if success else "Failed"
        print(f"{test_name}: {status}")
    
    total = len(results)
    passed = sum(1 for _, success in results if success)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("All tests passed!")
        return 0
    else:
        print(f"{total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())

