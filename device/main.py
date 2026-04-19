"""Entry point — wires up all modules and runs the doorbell event loop."""

import signal
import sys

from loguru import logger

from config import cfg
from doorbell import Doorbell


def _configure_logging() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        level="INFO",
        colorize=True,
    )
    logger.add(
        "doorbell.log",
        rotation="10 MB",
        retention="7 days",
        level="DEBUG",
    )


def main() -> None:
    _configure_logging()
    logger.info("SWFT Doorbell starting up …")

    doorbell = Doorbell()

    def _shutdown(sig, frame):
        logger.info(f"Received signal {sig} — shutting down")
        doorbell.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    doorbell.start()  # blocks until stopped


if __name__ == "__main__":
    main()
