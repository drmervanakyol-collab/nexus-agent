"""
tests/property/conftest.py
Shared Hypothesis settings and strategies for property tests.
"""
from __future__ import annotations

from hypothesis import HealthCheck, settings

# Suppress too_slow for CI runs; property tests are allowed to be thorough.
settings.register_profile(
    "property",
    max_examples=200,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile("property")
