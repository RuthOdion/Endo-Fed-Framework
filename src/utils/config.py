"""
Configuration loading and path management utilities.

Provides functions to load YAML configuration files and resolve
project paths relative to the project root directory.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def get_project_root() -> Path:
    """
    Get the project root directory.

    Returns:
        Path to the project root (parent of src/ directory).
    """
    current_file = Path(__file__).resolve()
    # Navigate up: utils/ -> src/ -> project_root/
    project_root = current_file.parent.parent.parent
    return project_root


def get_data_path(subpath: str = "") -> Path:
    """
    Get path to the data directory.

    Args:
        subpath: Optional subdirectory within data/.

    Returns:
        Path to the data directory or subdirectory.
    """
    data_dir = get_project_root() / "data"
    if subpath:
        return data_dir / subpath
    return data_dir


def get_results_path(subpath: str = "") -> Path:
    """
    Get path to the results directory.

    Args:
        subpath: Optional subdirectory within results/.

    Returns:
        Path to the results directory or subdirectory.
    """
    results_dir = get_project_root() / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    if subpath:
        result_subdir = results_dir / subpath
        result_subdir.mkdir(parents=True, exist_ok=True)
        return result_subdir
    return results_dir


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load YAML configuration file.

    Args:
        path: Path to the config file. If None, loads the default
              config/config.yaml from the project root.

    Returns:
        Dictionary containing all configuration parameters.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the config file is malformed.
    """
    if path is None:
        path = str(get_project_root() / "config" / "config.yaml")

    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if config is None:
        return {}

    return config


def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Get a nested configuration value using dot notation.

    Args:
        config: Configuration dictionary.
        key_path: Dot-separated path (e.g., 'model.dropout').
        default: Default value if key not found.

    Returns:
        Configuration value or default.
    """
    keys = key_path.split(".")
    value = config

    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default

    return value
