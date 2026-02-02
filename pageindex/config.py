"""
Configuration management for PageIndex.

Handles loading configuration from YAML files and merging with user options.
"""
import yaml
import logging
from pathlib import Path
from types import SimpleNamespace as config

logger = logging.getLogger(__name__)


class ConfigLoader:
    """
    Load and manage PageIndex configuration.

    Loads default configuration from config.yaml and merges with user options.
    """

    def __init__(self, default_path: str = None):
        if default_path is None:
            default_path = Path(__file__).parent / "config.yaml"
        self._default_dict = self._load_yaml(default_path)

    @staticmethod
    def _load_yaml(path):
        """Load YAML file into dictionary."""
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _validate_keys(self, user_dict):
        """Validate that user keys exist in default config."""
        unknown_keys = set(user_dict) - set(self._default_dict)
        if unknown_keys:
            raise ValueError(f"Unknown config keys: {unknown_keys}")

    def load(self, user_opt=None) -> config:
        """
        Load the configuration, merging user options with default values.

        Args:
            user_opt: User options as dict, config(SimpleNamespace), or None

        Returns:
            SimpleNamespace with merged configuration
        """
        if user_opt is None:
            user_dict = {}
        elif isinstance(user_opt, config):
            user_dict = vars(user_opt)
        elif isinstance(user_opt, dict):
            user_dict = user_opt
        else:
            raise TypeError("user_opt must be dict, config(SimpleNamespace) or None")

        # Validate keys
        self._validate_keys(user_dict)

        # Merge with defaults
        merged = {**self._default_dict, **user_dict}
        return config(**merged)


__all__ = ['ConfigLoader']
