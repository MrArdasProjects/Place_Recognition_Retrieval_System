#!/usr/bin/env python3
"""Entrypoint script for extracting embeddings.

This script is a thin wrapper around the CLI module.
"""

import sys
from place_recognition.cli import app

if __name__ == "__main__":
    # Forward to CLI with extract-embeddings command
    sys.argv.insert(1, "extract-embeddings")
    app()
