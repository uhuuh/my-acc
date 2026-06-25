"""Public API: acc_dump context manager."""

from .config import config
from .capturer import Capturer
from .manager import Manager


class _AccDumpContext:
    def __init__(self, manager):
        self._manager = manager
        self.enabled = config.dump_enabled

    def __enter__(self):
        if self.enabled:
            self._manager.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled:
            self._manager.stop()
        return False


def acc_dump(dump_path, model=None, **kwargs):
    merged = {k: v for k, v in kwargs.items() if v is not None}
    if dump_path is not None:
        merged['dump_path'] = dump_path
    config.update(**merged)
    capturer = Capturer(model=model)
    manager = Manager(capturer)
    return _AccDumpContext(manager)
