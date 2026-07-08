"""Public dump API and capture-session lifecycle."""

from contextlib import nullcontext

from loguru import logger

from .cache import DumpPipeline
from .capturer import Capturer
from .config import Config
from .record import Record


class DumpSession:
    """Owns one capturer and its dump pipeline."""

    def __init__(self, config, model=None):
        self._next_seq_id = 1
        self._pipeline = DumpPipeline(config)
        self._stopped = False
        try:
            self._capturer = Capturer(config, model, self._capture)
        except BaseException:
            self._pipeline.close()
            raise
        logger.info(f"[acc] capture started: {config.dump_dir}")

    def _capture(self, capturer_type, capturer_key, args, kwargs, outputs):
        record = Record(
            seq_id=self._next_seq_id,
            capturer_type=capturer_type,
            capturer_key=capturer_key,
            args=args,
            kwargs=kwargs,
            outputs=outputs,
        )
        self._next_seq_id += 1
        self._pipeline.save(record)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False

    def stop(self):
        if self._stopped:
            return
        self._stopped = True
        try:
            self._capturer.stop()
        finally:
            self._pipeline.close()
        logger.info("[acc] capture stopped")


def acc_dump(dump_path=None, model=None, **kwargs):
    """Context manager that captures PyTorch operators during the wrapped block.

    Args:
        dump_path: Base directory for dump output. If None, uses "acc" in cwd.
        model: Optional nn.Module for module-level capture (forward hooks).
        **kwargs: Override config values such as hash_workers and
            cache_write_workers.

    Returns:
        A context manager. When ``dump_enabled=False``, returns
        ``contextlib.nullcontext()`` (a no-op).

    Usage::

        with acc_dump("my_dump", model=model, hash_workers=4):
            output = model(input)
    """
    config = Config(dump_path=dump_path, **kwargs)
    if not config.dump_enabled:
        return nullcontext()
    return DumpSession(config, model)
