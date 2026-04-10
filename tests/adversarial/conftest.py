"""
tests/adversarial/conftest.py
Shared helpers and fixtures for adversarial tests.

Adversarial tests verify that the system handles failure modes gracefully:
it must not crash, it must raise the expected error (or recover), and it
must leave state consistent.  All tests are fully injectable — no real OS,
network, or hardware calls.
"""
from __future__ import annotations

