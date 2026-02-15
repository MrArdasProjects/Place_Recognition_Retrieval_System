"""Embedding extraction using pre-trained vision models."""

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from rich.progress import track
from torchvision import models

from place_recognition.data import ImageLoader
from place_recognition.logging_config import get_logger

logger = get_logger(__name__)


class EmbeddingExtractor:
    """Extract L2-normalized embeddings using pre-trained vision models.
    
    Args:
        model_name: Pre-trained model (resnet50, efficientnet_b0, etc.)
        device: Device ("cpu" or "cuda")
        batch_size: Batch size for inference
        cache_dir: Directory to cache embeddings
        num_workers: Parallel workers for loading (0=sequential)
        verbose: Enable verbose logging
    """
    
    SUPPORTED_MODELS = {
        "resnet50": (models.resnet50, 2048),
        "resnet101": (models.resnet101, 2048),
        "resnet152": (models.resnet152, 2048),
        "efficientnet_b0": (models.efficientnet_b0, 1280),
        "efficientnet_b1": (models.efficientnet_b1, 1280),
        "efficientnet_b2": (models.efficientnet_b2, 1408),
    }
    
    def __init__(
        self,
        model_name: str = "resnet50",
        device: str = "cpu",
        batch_size: int = 8,
        cache_dir: Path = Path("embeddings_cache"),
        num_workers: int = 0,
        verbose: bool = False,
    ) -> None:
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Model {model_name} not supported. "
                f"Choose from: {list(self.SUPPORTED_MODELS.keys())}"
            )
        
        self.model_name = model_name
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir)
        self.num_workers = num_workers
        self.verbose = verbose
        
        # Get model info
        model_fn, self.feature_dim = self.SUPPORTED_MODELS[model_name]
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        logger.info(f"Loading pre-trained model: {model_name}")
        self.model = self._load_model(model_fn)
        self.model.to(self.device)
        self.model.eval()
        
        # Statistics
        self.stats = {
            "total_images": 0,
            "batches_processed": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }
        
        logger.info(f"EmbeddingExtractor initialized:")
        logger.info(f"  Model: {model_name}")
        logger.info(f"  Feature dim: {self.feature_dim}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Batch size: {batch_size}")
    
    def _load_model(self, model_fn) -> nn.Module:
        """Load pre-trained model and remove classification head.
        
        Args:
            model_fn: Model constructor function
        
        Returns:
            Model with classification head removed
        """
        # Load pre-trained weights
        if "efficientnet" in self.model_name:
            weights = "IMAGENET1K_V1"
        else:
            weights = "IMAGENET1K_V1"
        
        model = model_fn(weights=weights)
        
        # Remove classification head based on model type
        if "resnet" in self.model_name:
            # ResNet: Remove final FC layer
            model.fc = nn.Identity()
        elif "efficientnet" in self.model_name:
            # EfficientNet: Remove classifier
            model.classifier = nn.Identity()
        
        return model
    
    @torch.no_grad()
    def extract_batch(self, batch_tensor: torch.Tensor) -> np.ndarray:
        """Extract embeddings for a batch of images.
        
        Args:
            batch_tensor: Batch of preprocessed images [B, C, H, W]
        
        Returns:
            L2-normalized embeddings [B, feature_dim]
        """
        # Move to device
        batch_tensor = batch_tensor.to(self.device)
        
        # Forward pass
        embeddings = self.model(batch_tensor)
        
        # L2 normalization
        embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
        
        # Move to CPU and convert to numpy
        embeddings = embeddings.cpu().numpy()
        
        return embeddings
    
    def _compute_cache_key(self, manifest_path: Path) -> str:
        """Compute cache key for manifest file.
        
        Args:
            manifest_path: Path to manifest file
        
        Returns:
            Cache key (MD5 hash of manifest + model name)
        """
        # Read manifest content
        with open(manifest_path, "rb") as f:
            manifest_content = f.read()
        
        # Compute hash
        hash_obj = hashlib.md5()
        hash_obj.update(manifest_content)
        hash_obj.update(self.model_name.encode())
        
        return hash_obj.hexdigest()
    
    def _get_cache_path(self, manifest_path: Path) -> Path:
        """Get cache file path for manifest.
        
        Args:
            manifest_path: Path to manifest file
        
        Returns:
            Path to cache file
        """
        cache_key = self._compute_cache_key(manifest_path)
        cache_filename = f"{manifest_path.stem}_{self.model_name}_{cache_key[:8]}.npz"
        return self.cache_dir / cache_filename
    
    def _load_from_cache(
        self, cache_path: Path
    ) -> Optional[Tuple[np.ndarray, List[Dict]]]:
        """Load embeddings from cache.
        
        Args:
            cache_path: Path to cache file
        
        Returns:
            Tuple of (embeddings, records) or None if cache not found
        """
        if not cache_path.exists():
            return None
        
        try:
            data = np.load(cache_path, allow_pickle=True)
            embeddings = data["embeddings"]
            records = data["records"].tolist()
            
            logger.info(f"Loaded {len(embeddings)} embeddings from cache")
            self.stats["cache_hits"] += 1
            
            return embeddings, records
        
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return None
    
    def _save_to_cache(
        self, cache_path: Path, embeddings: np.ndarray, records: List[Dict]
    ) -> None:
        """Save embeddings to cache.
        
        Args:
            cache_path: Path to cache file
            embeddings: Embeddings array
            records: List of manifest records
        """
        try:
            np.savez_compressed(
                cache_path,
                embeddings=embeddings,
                records=np.array(records, dtype=object),
            )
            logger.info(f"Saved {len(embeddings)} embeddings to cache")
        
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def extract_embeddings(
        self,
        manifest_path: Path,
        dataset_root: Path,
        use_cache: bool = True,
    ) -> Tuple[np.ndarray, List[Dict]]:
        """Extract embeddings for all images in manifest.
        
        Args:
            manifest_path: Path to manifest file (gallery.jsonl or query.jsonl)
            dataset_root: Root directory of the dataset
            use_cache: Whether to use cache
        
        Returns:
            Tuple of:
                - embeddings: [N, feature_dim] numpy array
                - records: List of manifest records
        """
        logger.info("=" * 60)
        logger.info(f"EMBEDDING EXTRACTION: {manifest_path.name}")
        logger.info("=" * 60)
        
        # Check cache
        cache_path = self._get_cache_path(manifest_path)
        if use_cache:
            cached = self._load_from_cache(cache_path)
            if cached is not None:
                return cached
        
        self.stats["cache_misses"] += 1
        
        # Load manifest
        logger.info(f"Loading manifest: {manifest_path}")
        with open(manifest_path, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f]
        
        logger.info(f"Found {len(records)} images")
        
        # Initialize image loader
        image_loader = ImageLoader(
            dataset_root=dataset_root,
            image_size=224,
            normalize=True,
            augment=False,
            num_workers=self.num_workers,
            verbose=False,
        )
        
        # Process in batches
        all_embeddings = []
        valid_records = []
        
        num_batches = (len(records) + self.batch_size - 1) // self.batch_size
        logger.info(f"Processing {num_batches} batches (batch_size={self.batch_size})")
        
        for i in track(
            range(0, len(records), self.batch_size),
            description=f"Extracting embeddings",
            total=num_batches,
            disable=not self.verbose,
        ):
            batch_records = records[i : i + self.batch_size]
            
            # Load batch
            batch_tensor, batch_valid_records = image_loader.load_batch(batch_records)
            
            if batch_tensor.shape[0] == 0:
                logger.warning(f"Empty batch at index {i}")
                continue
            
            # Extract embeddings
            batch_embeddings = self.extract_batch(batch_tensor)
            
            all_embeddings.append(batch_embeddings)
            valid_records.extend(batch_valid_records)
            
            self.stats["batches_processed"] += 1
        
        # Concatenate all embeddings
        if len(all_embeddings) == 0:
            logger.error("No valid embeddings extracted!")
            return np.array([]), []
        
        embeddings = np.concatenate(all_embeddings, axis=0)
        self.stats["total_images"] = len(embeddings)
        
        # Verify dimensions
        assert embeddings.shape[0] == len(valid_records)
        assert embeddings.shape[1] == self.feature_dim
        
        # Save to cache
        if use_cache:
            self._save_to_cache(cache_path, embeddings, valid_records)
        
        # Log statistics
        logger.info("=" * 60)
        logger.info(f"Extraction complete:")
        logger.info(f"  Total embeddings: {len(embeddings)}")
        logger.info(f"  Feature dimension: {self.feature_dim}")
        logger.info(f"  Shape: {embeddings.shape}")
        logger.info(f"  L2 norm check: min={np.linalg.norm(embeddings, axis=1).min():.3f}, "
                   f"max={np.linalg.norm(embeddings, axis=1).max():.3f}")
        logger.info("=" * 60)
        
        return embeddings, valid_records
    
    def get_stats(self) -> Dict:
        """Get extraction statistics."""
        return self.stats.copy()
