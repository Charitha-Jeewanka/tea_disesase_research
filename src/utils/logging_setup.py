"""
src/utils/logging_setup.py -- Centralized logging configuration.

Sets up a consistent logging format with timestamps, log levels,
and both console and file handlers.
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path


def setup_logging(
    log_dir: str = "logs",
    log_filename: str = "",
    level: int = logging.INFO,
    project_root: str = ".",
) -> logging.Logger:
    """
    Configure the root logger with console and file handlers.

    Parameters
    ----------
    log_dir : str
        Directory for log files, relative to project_root.
    log_filename : str
        Name of the log file. If empty, auto-generates based on timestamp.
    level : int
        Logging level (default: logging.INFO).
    project_root : str
        Base directory for resolving relative paths.

    Returns
    -------
    logging.Logger
        The configured root logger instance.
    """
    log_path = Path(project_root) / log_dir
    log_path.mkdir(parents=True, exist_ok=True)

    if not log_filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"run_{timestamp}.log"

    log_file = log_path / log_filename

    # -- Log format with timestamps --
    log_format = (
        "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s"
    )
    date_format = "%Y-%m-%d %H:%M:%S"

    # -- Clear any existing handlers on the root logger --
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()

    # -- Console handler (stdout) --
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(
        logging.Formatter(log_format, datefmt=date_format)
    )
    root_logger.addHandler(console_handler)

    # -- File handler --
    file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(
        logging.Formatter(log_format, datefmt=date_format)
    )
    root_logger.addHandler(file_handler)

    root_logger.info("Logging initialised. Log file: %s", log_file)

    return root_logger
