"""One-shot logging configuration.

Called once from app.py at startup. Every module then does
    log = logging.getLogger(__name__)
at the top and uses log.debug/info/warning/error/exception — no other
setup required.

Default level is INFO. Override with the TEXIQ_LOG env var (e.g. DEBUG).
"""
import logging
import os


def setup_logging() -> None:
    level_name = os.environ.get("TEXIQ_LOG", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
