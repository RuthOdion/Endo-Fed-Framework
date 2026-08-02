"""
Logging configuration for the endometriosis FL framework.

Provides structured logging with both file and console handlers,
supporting experiment-specific log files.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.utils.config import get_results_path


def setup_logger(
    name: str = "endometriosis_fl",
    log_level: int = logging.INFO,
    log_file: Optional[str] = None,
    console_output: bool = True,
) -> logging.Logger:
    """
    Set up a logger with file and console handlers.

    Args:
        name: Logger name.
        log_level: Logging level (default: INFO).
        log_file: Path to log file. If None, creates one in results/logs/.
        console_output: Whether to add console handler.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger

    # Format with timestamp, level, and module info
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler
    if log_file is None:
        log_dir = get_results_path("logs")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = str(log_dir / f"{name}_{timestamp}.log")

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def get_experiment_logger(
    experiment_name: str,
    log_level: int = logging.INFO,
) -> logging.Logger:
    """
    Get a logger configured for a specific experiment.

    Creates a dedicated log file for the experiment in results/logs/.

    Args:
        experiment_name: Name of the experiment (used in filename).
        log_level: Logging level.

    Returns:
        Configured logger for the experiment.
    """
    log_dir = get_results_path("logs")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = str(log_dir / f"{experiment_name}_{timestamp}.log")

    logger = setup_logger(
        name=f"experiment.{experiment_name}",
        log_level=log_level,
        log_file=log_file,
        console_output=True,
    )

    logger.info(f"Experiment '{experiment_name}' logger initialised")
    logger.info(f"Log file: {log_file}")

    return logger
