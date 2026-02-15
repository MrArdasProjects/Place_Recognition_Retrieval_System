#!/usr/bin/env python3
"""Entrypoint script for checking duplicates.

This script is a thin wrapper around the CLI module.
"""

from place_recognition.cli import app

if __name__ == "__main__":
    app(["check-duplicates"])
