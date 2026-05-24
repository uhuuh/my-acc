from .config import config
from .manager import Manager


class _OpsDumpContext:
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

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper


def ops_dump(dump_path=None, **kwargs):
    merged = {k: v for k, v in kwargs.items() if v is not None}
    if dump_path is not None:
        merged['dump_path'] = dump_path
    config.update(**merged)
    mgr = Manager()
    return _OpsDumpContext(mgr)
