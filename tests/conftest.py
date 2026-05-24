import os
os.environ["ACC_IO_FLUSH_MODE"] = "stop"

from acc.config import config
config.update(io_flush_mode="stop")
