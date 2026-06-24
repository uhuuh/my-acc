"""Test configuration."""

import pytest
from acc.config import config


@pytest.fixture(autouse=True)
def test_config():
    """Reset config to test-safe defaults before each test."""
    config.update(
        capturer_backends="ops",
    )
    yield
