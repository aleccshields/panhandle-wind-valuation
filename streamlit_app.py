"""
Root-level Streamlit entry point.

Streamlit Cloud requires the app entry point to be at the repository root.
This file simply delegates to app/main.py, keeping all app logic in one place.
"""

import sys
from pathlib import Path

# Ensure the repo root is on sys.path so `src` imports resolve correctly
# whether the app is launched from the root or from inside app/.
sys.path.insert(0, str(Path(__file__).parent))

# Import and run the main app — Streamlit re-executes this module on every
# interaction, so the import is intentionally at module level.
import app.main  # noqa: F401, E402  (side-effectful import)
