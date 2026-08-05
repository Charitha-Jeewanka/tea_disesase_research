"""
src/utils/config_loader.py -- YAML configuration loader.

Loads the single source-of-truth config.yaml and provides
typed access to settings via a plain dictionary.
"""

import logging
from pathlib import Path
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """
    Load and return the project YAML configuration.

    Parameters
    ----------
    config_path : str
        Path to the YAML config file, relative to the current working directory
        or an absolute path.

    Returns
    -------
    dict
        Parsed configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    yaml.YAMLError
        If the YAML file is malformed.
    """
    config_file = Path(config_path)
    if not config_file.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {config_file.resolve()}"
        )

    with open(config_file, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    logger.info("Configuration loaded from: %s", config_file.resolve())
    return config


def get_nested(config: Dict[str, Any], *keys, default: Any = None) -> Any:
    """
    Safely retrieve a nested value from the config dictionary.

    Parameters
    ----------
    config : dict
        The configuration dictionary.
    *keys : str
        Sequence of keys to traverse.
    default : any
        Value to return if the key path does not exist.

    Returns
    -------
    any
        The value at the nested key path, or default.

    Example
    -------
    >>> get_nested(cfg, "phase1", "augmentation", "mosaic", default=1.0)
    1.0
    """
    current = config
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key, default)
        else:
            return default
    return current
