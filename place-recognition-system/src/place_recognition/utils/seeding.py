"""Global seeding utilities for reproducibility."""

import random
from typing import Optional

import numpy as np


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Set global random seed for reproducibility across all libraries.
    
    This function sets seeds for:
    - Python's random module
    - NumPy
    - PyTorch (if available)
    - CUDA (if available and PyTorch is installed)
    
    Args:
        seed: Integer seed value
        deterministic: If True, enables deterministic algorithms (may be slower)
    
    Example:
        >>> set_global_seed(42)
        >>> # All random operations will now be deterministic
    
    Note:
        Complete determinism is not always possible, especially with GPU operations.
        For more info: https://pytorch.org/docs/stable/notes/randomness.html
    """
    # Python random module
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch (if available)
    try:
        import torch
        
        torch.manual_seed(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        
        if deterministic:
            # Enable deterministic algorithms
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
            # For PyTorch >= 1.8
            if hasattr(torch, "use_deterministic_algorithms"):
                torch.use_deterministic_algorithms(True)
    
    except ImportError:
        # PyTorch not installed, skip
        pass


def get_random_state() -> dict:
    """Get current random state from all libraries.
    
    Returns:
        Dictionary containing random states
    """
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }
    
    try:
        import torch
        state["torch"] = torch.get_rng_state()
        if torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
    except ImportError:
        pass
    
    return state


def set_random_state(state: dict) -> None:
    """Restore random state from a saved state dictionary.
    
    Args:
        state: Dictionary containing random states (from get_random_state)
    """
    if "python" in state:
        random.setstate(state["python"])
    
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    
    try:
        import torch
        if "torch" in state:
            torch.set_rng_state(state["torch"])
        if "torch_cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["torch_cuda"])
    except ImportError:
        pass
