"""Test configuration."""

import os
import pytest


@pytest.fixture(autouse=True)
def test_config():
    """Ensure ops-only backend for tests (no model needed)."""
    os.environ["ACC_CAPTURER_BACKENDS"] = "ops"
    yield
    os.environ.pop("ACC_CAPTURER_BACKENDS", None)
