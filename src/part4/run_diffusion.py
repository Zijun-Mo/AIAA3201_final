#!/usr/bin/env python3
"""Compatibility wrapper for migrated Phase 5 Route G entrypoint."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.part3.run_diffusion import main


if __name__ == "__main__":
    main()
