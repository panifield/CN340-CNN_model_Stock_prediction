#!/usr/bin/env python3
"""Backward-compatible entry point; use ``python C/main.py`` for the modular project."""

from __future__ import annotations

from main import main


if __name__ == "__main__":
    main()
