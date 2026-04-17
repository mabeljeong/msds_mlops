"""
Test configuration.

Both `scripts/fetch_walkscore.py` and `scripts/fetch_census.py` read their API
keys from environment variables at module import time. We set placeholder
values here so the modules can be imported during test collection without a
real `.env` file. We also ensure the repo root is on `sys.path` so
`scripts.*` is importable when pytest is invoked from anywhere.
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("WALKSCORE_API_KEY", "test-walkscore-key")
os.environ.setdefault("CENSUS_API_KEY", "test-census-key")
