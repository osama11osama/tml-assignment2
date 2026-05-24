#!/usr/bin/env python3
"""Thin wrapper around submission.py (kept for backwards compatibility)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from submission import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
