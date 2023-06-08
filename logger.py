import logging
import os

from rich.logging import RichHandler

logger = logging.getLogger("app")
logger.setLevel(os.environ.get("LOG_LEVEL", logging.INFO))
logger.addHandler(RichHandler())