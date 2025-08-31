"""Lightweight logging setup for the benchmarking package.

Provides a single function to configure a dedicated logger that writes to a
file with timestamps and callsite information. Other modules can retrieve the
same logger via ``logging.getLogger("benchmark")`` and log messages without
reconfiguring handlers.
"""

from __future__ import annotations

from pathlib import Path
import logging


def setup_logging(log_file: str | Path, level: int = logging.INFO, to_stdout: bool = False) -> logging.Logger:
    """Configure and return the shared "benchmark" logger.

    Parameters
    ----------
    log_file:
        Destination log file path.
    level:
        Logging level (default INFO).
    to_stdout:
        If True, also echo logs to stderr/console.
    """

    logger = logging.getLogger("benchmark")
    logger.setLevel(level)

    # Avoid adding duplicate handlers across repeated setup calls
    log_path = str(Path(log_file))
    fmt = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(module)s:%(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    def _has_file_handler() -> bool:
        for h in logger.handlers:
            if isinstance(h, logging.FileHandler):
                # type: ignore[attr-defined]
                if getattr(h, "baseFilename", None) == log_path:
                    return True
        return False

    if not _has_file_handler():
        fh = logging.FileHandler(log_path)
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    if to_stdout and not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        sh = logging.StreamHandler()
        sh.setLevel(level)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    # Don't propagate to root to avoid duplicate messages if root is configured elsewhere
    logger.propagate = False
    return logger
