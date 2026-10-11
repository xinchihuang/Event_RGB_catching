#!/usr/bin/env python3
"""
Simple Configuration Loader for Event RGB Catching System
"""

import os
import sys
import yaml
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path


# Preserve direct execution of this module's configuration check.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.config_loader import ConfigLoader as BaseConfigLoader, _select_config


class ConfigLoader(BaseConfigLoader):
    """Same strict loading rules, with live bases for external overlays."""
    default_stack = "config_live"


# Global configuration instance
_config = None

def get_config() -> ConfigLoader:
    global _config
    if _config is None:
        _config = ConfigLoader(_select_config("config_live"))
    return _config

def load_config(key_path: str, default: Any = None) -> Any:
    """Quick function to get configuration value"""
    return get_config().get(key_path, default)


def get_available_configs() -> List[str]:
    """Get list of available configuration files"""
    return ConfigLoader.get_available_configs()

def get_current_config_name() -> str:
    """Get the name of the currently loaded configuration file"""
    return get_config().current_config_file


# Keep this public import for callers using the live compatibility loader.
from config.config_loader import Config as SharedConfig


class Config(SharedConfig):
    """Use the shared schema and defaults with the live loader's own state."""

    @staticmethod
    def _get_config():
        return get_config()


if __name__ == "__main__":
    # Test configuration loader
    logging.basicConfig(level=logging.INFO)
    
    config = get_config()
    
    # Test getting values
    rgb_width = config.get('image.rgb_width', 1080)
    print(f"RGB width: {rgb_width}")
    
    # Test Config class
    dims = Config.get_image_dimensions()
    print(f"Image dimensions: {dims}")
    
    timing = Config.get_timing()
    print(f"Timing: {timing}")
