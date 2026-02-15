"""Place Recognition & Retrieval System.

A production-ready system for image-based place recognition using deep learning.
"""

__version__ = "0.1.0"
__author__ = "Arda Uçar"

from place_recognition.config.config import Config, load_config
from place_recognition.utils.seeding import set_global_seed

__all__ = [
    "Config",
    "load_config",
    "set_global_seed",
]
