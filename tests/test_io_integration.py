# tests/test_io_integration.py
import pytest
import tempfile
import os
from acc.serialization import SerializationSession


def test_serialization_session_async_io():
    """Test SerializationSession uses async IO for seq writes"""
    with tempfile.TemporaryDirectory() as tmpdir:
        session = SerializationSession(tmpdir, enable_async_io=True)
        session.start()

        session.end()

        # Session should have IOWriter instance
        assert hasattr(session, '_io_writer')
        assert session._io_writer is not None


def test_serialization_session_async_io_disabled():
    """Test SerializationSession without async IO"""
    with tempfile.TemporaryDirectory() as tmpdir:
        session = SerializationSession(tmpdir, enable_async_io=False)
        session.start()

        session.end()

        # Session should have IOWriter instance in sync mode
        assert hasattr(session, '_io_writer')
        assert session._io_writer is not None
        assert session._io_writer.enable_async == False


def test_serialization_session_default_async_io_enabled():
    """Test SerializationSession defaults to async IO enabled"""
    with tempfile.TemporaryDirectory() as tmpdir:
        session = SerializationSession(tmpdir)
        session.start()

        session.end()

        # By default, async IO should be enabled
        assert hasattr(session, '_io_writer')
        assert session._io_writer is not None
        assert session._io_writer.enable_async == True