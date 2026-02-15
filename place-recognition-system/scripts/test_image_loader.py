#!/usr/bin/env python3
"""Test script for ImageLoader - demonstrates usage."""

import json
from pathlib import Path

from place_recognition.data import ImageLoader
from place_recognition.logging_config import setup_logging

setup_logging(level="INFO")


def main() -> None:
    """Test ImageLoader with real dataset."""
    print("\n" + "=" * 60)
    print("IMAGE LOADER TEST")
    print("=" * 60)
    
    # Setup
    dataset_root = Path("../landmarks")
    manifest_path = Path("manifests/gallery.jsonl")
    
    if not manifest_path.exists():
        print(f"ERROR: Manifest not found: {manifest_path}")
        print("Run: place-recognition build-manifest first")
        return
    
    # Load manifest
    print(f"\nLoading manifest: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    print(f"Found {len(records)} records")
    
    # Initialize loader
    print("\nInitializing ImageLoader...")
    loader = ImageLoader(
        dataset_root=dataset_root,
        image_size=224,
        normalize=True,
        augment=False,
        verbose=True,
    )
    
    # Test single image loading
    print("\n" + "-" * 60)
    print("TEST 1: Single Image Loading")
    print("-" * 60)
    record = records[0]
    print(f"Loading: {record['image_path']}")
    tensor = loader.load_image(record)
    
    if tensor is not None:
        print(f"SUCCESS!")
        print(f"  Tensor shape: {tensor.shape}")
        print(f"  Tensor dtype: {tensor.dtype}")
        print(f"  Value range: [{tensor.min():.3f}, {tensor.max():.3f}]")
    else:
        print("FAILED to load")
    
    # Test batch loading
    print("\n" + "-" * 60)
    print("TEST 2: Batch Loading")
    print("-" * 60)
    batch_size = 8
    batch_records = records[:batch_size]
    print(f"Loading batch of {batch_size} images...")
    
    batch_tensor, valid_records = loader.load_batch(batch_records)
    print(f"SUCCESS: Loaded {len(valid_records)}/{batch_size} images")
    print(f"  Batch shape: {batch_tensor.shape}")
    print(f"  Batch dtype: {batch_tensor.dtype}")
    print(f"  Memory size: {batch_tensor.numel() * batch_tensor.element_size() / 1024:.2f} KB")
    
    # Test with augmentation
    print("\n" + "-" * 60)
    print("TEST 3: With Augmentation")
    print("-" * 60)
    loader_augment = ImageLoader(
        dataset_root=dataset_root,
        image_size=224,
        normalize=True,
        augment=True,
        verbose=False,
    )
    
    tensor_aug = loader_augment.load_image(record, apply_augment=True)
    if tensor_aug is not None:
        print(f"SUCCESS: Augmented tensor shape: {tensor_aug.shape}")
    
    # Show statistics
    print("\n" + "-" * 60)
    print("TEST 4: Full Dataset Loading")
    print("-" * 60)
    print(f"Processing all {len(records)} images...")
    
    loader.reset_stats()
    all_tensors = []
    for rec in records:
        tensor = loader.load_image(rec)
        if tensor is not None:
            all_tensors.append(tensor)
    
    loader.log_stats()
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
