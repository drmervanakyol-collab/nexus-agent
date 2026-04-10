"""
nexus/release/version.py
Single source of truth for Nexus Agent release metadata.

VERSION    — semantic version string (MAJOR.MINOR.PATCH)
BUILD_DATE — ISO-8601 date string; injected by build_release.bat via
             the NEXUS_BUILD_DATE environment variable at package time.
GIT_HASH   — first 8 hex chars of the HEAD commit; injected by
             build_release.bat via NEXUS_GIT_HASH at package time.

At development time (no env vars set) these default to "dev" / "unknown"
so that imports always succeed without a build step.

Usage
-----
    from nexus.release.version import VERSION, BUILD_DATE, GIT_HASH
    print(f"Nexus Agent {VERSION} ({BUILD_DATE}, {GIT_HASH})")
"""
from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Core constants — override via environment at package time
# ---------------------------------------------------------------------------

VERSION: str = os.environ.get("NEXUS_VERSION", "1.0.0")
BUILD_DATE: str = os.environ.get("NEXUS_BUILD_DATE", "dev")
GIT_HASH: str = os.environ.get("NEXUS_GIT_HASH", "unknown")

# ---------------------------------------------------------------------------
# Composite strings
# ---------------------------------------------------------------------------

VERSION_STRING: str = f"{VERSION}+{GIT_HASH}"
FULL_VERSION_STRING: str = f"Nexus Agent {VERSION} (built {BUILD_DATE}, {GIT_HASH})"


def version_tuple() -> tuple[int, int, int]:
    """
    Return VERSION as a 3-tuple of ints for programmatic comparisons.

    Falls back to (0, 0, 0) for non-release builds where VERSION may
    contain pre-release suffixes.
    """
    try:
        parts = VERSION.split(".")
        return int(parts[0]), int(parts[1]), int(parts[2])
    except (ValueError, IndexError):
        return 0, 0, 0
