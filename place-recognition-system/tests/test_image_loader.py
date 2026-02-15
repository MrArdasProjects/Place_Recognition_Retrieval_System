"""Tests for ImageLoader."""

import json
from pathlib import Path

import pytest
import torch

from place_recognition.data import ImageLoader


@pytest.fixture
def sample_manifest_record(tmp_path: Path) -> dict:
    """Create a sample manifest record for testing."""
    return {
        "image_path": "test_image.jpg",
        "place_id": "test_place",
        "split": "gallery",
        "width": 800,
        "height": 600,
        "mode": "RGB",
        "is_grayscale": False,
        "md5": "abc123",
    }


def test_image_loader_initialization(tmp_path: Path) -> None:
    """Test ImageLoader initialization."""
    loader = ImageLoader(
        dataset_root=tmp_path,
        image_size=224,
        normalize=True,
        verbose=False,
    )
    
    assert loader.image_size == 224
    assert loader.normalize is True
    assert loader.dataset_root == tmp_path


def test_image_loader_stats() -> None:
    """Test statistics tracking."""
    loader = ImageLoader(dataset_root=Path("."), image_size=224)
    
    stats = loader.get_stats()
    assert stats["loaded"] == 0
    assert stats["failed"] == 0
    
    loader.reset_stats()
    assert loader.get_stats()["loaded"] == 0


def test_imagenet_constants() -> None:
    """Test ImageNet normalization constants."""
    loader = ImageLoader(dataset_root=Path("."), image_size=224)
    
    assert len(loader.IMAGENET_MEAN) == 3
    assert len(loader.IMAGENET_STD) == 3
    assert all(0 <= v <= 1 for v in loader.IMAGENET_MEAN)
    assert all(0 <= v <= 1 for v in loader.IMAGENET_STD)


def test_batch_loading_empty() -> None:
    """Test batch loading with empty list."""
    loader = ImageLoader(dataset_root=Path("."), image_size=224)
    
    batch, valid_records = loader.load_batch([])
    
    assert batch.shape[0] == 0
    assert len(valid_records) == 0


def test_transform_pipeline() -> None:
    """Test transform pipeline building."""
    # Without normalization
    loader1 = ImageLoader(dataset_root=Path("."), image_size=224, normalize=False)
    assert loader1.transform is not None
    
    # With normalization
    loader2 = ImageLoader(dataset_root=Path("."), image_size=224, normalize=True)
    assert loader2.transform is not None
    
    # With augmentation
    loader3 = ImageLoader(dataset_root=Path("."), image_size=224, augment=True)
    assert loader3.augment_transform is not None
