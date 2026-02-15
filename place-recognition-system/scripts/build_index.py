#!/usr/bin/env python3
"""Entrypoint script for building search index.

This script is a thin wrapper around the CLI module.
"""

import sys
from place_recognition.cli import app

if __name__ == "__main__":
    sys.argv.insert(1, "build-index")
    app()
