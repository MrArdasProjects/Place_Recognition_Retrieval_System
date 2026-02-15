"""Configuration management with YAML support and CLI overrides."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class Config:
    """Main configuration class for the place recognition system.
    
    This uses dataclass for type-safe configuration management.
    All fields have sensible defaults and can be overridden via YAML or CLI.
    """
    
    # Reproducibility
    seed: int = 42
    
    # Device configuration
    device: str = "cpu"  # Will be set to "cuda" if available
    
    # Data configuration
    batch_size: int = 32
    num_workers: int = 4
    
    # Model configuration
    model_name: str = "resnet50"
    embedding_dim: int = 512
    
    # Retrieval configuration
    top_k: int = 5
    similarity_metric: str = "cosine"  # cosine or euclidean
    
    # Index configuration
    index_type: str = "flat"  # flat, ivf, hnsw
    
    # Paths
    data_dir: Path = field(default_factory=lambda: Path("data"))
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    models_dir: Path = field(default_factory=lambda: Path("models"))
    
    # Logging
    log_level: str = "INFO"
    log_to_file: bool = False
    log_file: Optional[Path] = None
    
    def __post_init__(self) -> None:
        """Convert string paths to Path objects."""
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        if isinstance(self.models_dir, str):
            self.models_dir = Path(self.models_dir)
        if self.log_file and isinstance(self.log_file, str):
            self.log_file = Path(self.log_file)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Path):
                result[key] = str(value)
            else:
                result[key] = value
        return result
    
    def update(self, **kwargs: Any) -> None:
        """Update configuration with new values."""
        for key, value in kwargs.items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)


def load_config(
    config_path: Optional[Path] = None,
    **overrides: Any
) -> Config:
    """Load configuration from YAML file with optional CLI overrides.
    
    Args:
        config_path: Path to YAML configuration file
        **overrides: Key-value pairs to override config values
    
    Returns:
        Config object with loaded and overridden values
    
    Example:
        >>> config = load_config(Path("config/default.yaml"), seed=123, device="cuda")
    """
    # Start with default config
    config = Config()
    
    # Load from YAML if provided
    if config_path and config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f)
            if yaml_data:
                config.update(**yaml_data)
    
    # Apply CLI overrides
    if overrides:
        config.update(**overrides)
    
    return config


def save_config(config: Config, output_path: Path) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Config object to save
        output_path: Path where to save the config
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)
