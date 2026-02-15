#!/usr/bin/env python3
"""Entrypoint script for evaluating retrieval performance.

This script is a convenience wrapper. Users can also run:
    place-recognition evaluate [OPTIONS]
"""

from place_recognition.cli import app

if __name__ == "__main__":
    # Let user invoke with: python scripts/evaluate.py [OPTIONS]
    # No argv manipulation needed - cleaner and safer
    app()
