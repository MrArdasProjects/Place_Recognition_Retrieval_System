"""Pytest configuration and fixtures."""

import pytest
from pathlib import Path

from place_recognition.config import Config


@pytest.fixture
def default_config() -> Config:
    """Provide a default config for tests."""
    return Config()


@pytest.fixture
def temp_config_file(tmp_path: Path) -> Path:
    """Create a temporary config file for testing."""
    config_content = """
seed: 42
device: cpu
batch_size: 16
"""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(config_content)
    return config_file
