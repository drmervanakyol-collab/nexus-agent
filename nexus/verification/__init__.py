"""
nexus/verification
Post-action verification layer.

Public API
----------
VerificationMode      — SKIP / VISUAL / SOURCE / AUTO
VerificationPolicy    — immutable policy dataclass
VerificationResult    — outcome dataclass returned by verifiers
VisualVerifier        — frame-diff + OCR based verifier
SourceVerifier        — structured source read-back verifier
"""
from nexus.verification.policy import VerificationMode, VerificationPolicy
from nexus.verification.result import VerificationResult
from nexus.verification.source_verification import SourceVerifier, SourceProbe
from nexus.verification.visual_verification import OcrFn, VisualVerifier

__all__ = [
    "OcrFn",
    "SourceProbe",
    "SourceVerifier",
    "VerificationMode",
    "VerificationPolicy",
    "VerificationResult",
    "VisualVerifier",
]
