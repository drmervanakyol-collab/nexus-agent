"""
nexus/memory
Persistent UI context memory — fingerprints and action corrections.

Public API
----------
UIFingerprint        — learned UI state with transport preferences
FingerprintStore     — save / find_similar / record_outcome / evict_stale
new_fingerprint      — convenience factory (generates UUID)

CorrectionRecord     — wrong→correct action mapping
CorrectionStore      — save / find_applicable / increment_apply_count
new_correction       — convenience factory (generates UUID)
"""
from nexus.memory.correction_store import (
    CorrectionRecord,
    CorrectionStore,
    new_correction,
)
from nexus.memory.fingerprint_store import (
    FingerprintStore,
    UIFingerprint,
    new_fingerprint,
)

__all__ = [
    "CorrectionRecord",
    "CorrectionStore",
    "FingerprintStore",
    "UIFingerprint",
    "new_correction",
    "new_fingerprint",
]
