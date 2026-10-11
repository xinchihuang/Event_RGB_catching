#!/usr/bin/env python3
"""
Simple Configuration Loader for Event RGB Catching System
"""

import os
import sys
import yaml
import logging
from typing import Dict, Any, List
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge overlay into base."""
    if not isinstance(base, dict) or not isinstance(overlay, dict):
        return overlay
    merged = dict(base)
    for key, value in overlay.items():
        merged[key] = _deep_merge(merged.get(key), value)
    return merged


class ConfigLoader:
    """Load required camera, tracking and overlay mappings as one stack.

    Explicit paths are never redirected by basename. Relative paths use the
    repository root. External overlays use this loader's default stack.
    """
    default_stack = "config"

    def __init__(self, config_file=None):
        self.logger = logging.getLogger(__name__)
        if config_file is None:
            config_file = f"{self.default_stack}/overlays/config_l.yaml"
        self._set_config(config_file)

    def _set_config(self, config_file):
        if not str(config_file).strip():
            raise ValueError("Configuration path must not be empty")
        config_path = (PROJECT_ROOT / Path(config_file)).resolve()
        stack_dir = PROJECT_ROOT / self.default_stack
        for name in ("config", "config_live"):
            candidate = PROJECT_ROOT / name
            if candidate in config_path.parents:
                stack_dir = candidate
                break
        camera_path = stack_dir / "camera_config.yaml"
        tracking_path = stack_dir / "tracking_config.yaml"
        # Commit a new configuration only after all required files validate.
        merged = self._read_stack(camera_path, tracking_path, config_path)
        self.camera_config_path = camera_path
        self.tracking_config_path = tracking_path
        self.config_path = config_path
        self.current_config_file = str(config_file)
        self.config = merged
        self.logger.info("Configuration loaded: camera=%s, tracking=%s, overlay=%s",
                         camera_path, tracking_path, config_path)

    def switch_config(self, config_file: str) -> None:
        """Switch the complete stack; preserve the old state on failure."""
        self._set_config(config_file)

    @classmethod
    def get_available_configs(cls) -> List[str]:
        return sorted(
            str(path.relative_to(PROJECT_ROOT))
            for path in (PROJECT_ROOT / cls.default_stack / "overlays").glob("*.yaml")
        )

    @staticmethod
    def _read_yaml(path):
        try:
            with path.open("r", encoding="utf-8") as handle:
                value = yaml.safe_load(handle)
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML configuration {path}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(
                f"Configuration must contain a YAML mapping: {path}; "
                "use {} for an intentionally empty overlay"
            )
        return value

    def _read_stack(self, camera_path, tracking_path, overlay_path):
        merged = {}
        for path in (camera_path, tracking_path, overlay_path):
            merged = _deep_merge(merged, self._read_yaml(path))
        return merged

    def _load_config(self) -> None:
        self.config = self._read_stack(
            self.camera_config_path, self.tracking_config_path, self.config_path
        )

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation

        Args:
            key_path: Configuration key path (e.g., "image.rgb_width")
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self.config

        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def reload(self) -> None:
        """Reload configuration from file"""
        self._load_config()


# Global configuration instance
_config = None

def config_file_for(task, test_mode="", default_stack="config"):
    """Select a known overlay directly, without probing for fallback files."""
    tasks = {"normal": "l", "dark": "d", "clutter": "c", "dark_clutter": "dc"}
    key = (task or "").lower()
    if key not in tasks:
        raise ValueError(f"Unknown configuration task: {task}")
    mode = (test_mode or "").strip().lower()
    suffix = f"_{mode}" if default_stack == "config" and mode in {"model", "event", "rgb"} else ""
    return f"{default_stack}/overlays/config_{tasks[key]}{suffix}.yaml"


def _select_config(default_stack="config"):
    if "ACTIVE_CONFIG_FILE" in os.environ:
        return os.environ["ACTIVE_CONFIG_FILE"]
    import argparse
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--task", default=os.environ.get("ACTIVE_TASK", "normal"))
    parser.add_argument("--test_mode", default=os.environ.get("TEST_MODE", ""))
    args, _ = parser.parse_known_args()
    return config_file_for(args.task, args.test_mode, default_stack)


def get_config() -> ConfigLoader:
    """Get the lazily loaded configuration for this process."""
    global _config
    if _config is None:
        _config = ConfigLoader(_select_config())
    return _config


def initialize_config(task, test_mode):
    """Validate before allocating resources; share the exact path with workers."""
    global _config
    requested = (os.environ["ACTIVE_CONFIG_FILE"] if "ACTIVE_CONFIG_FILE" in os.environ
                 else config_file_for(task, test_mode))
    loaded = ConfigLoader(requested)
    _config = loaded
    os.environ["ACTIVE_CONFIG_FILE"] = str(loaded.config_path)
    os.environ["ACTIVE_TASK"] = task
    os.environ["TEST_MODE"] = test_mode
    return loaded


def load_config(key_path: str, default: Any = None) -> Any:
    """Quick function to get configuration value"""
    return get_config().get(key_path, default)


def get_available_configs() -> List[str]:
    """Get list of available configuration files"""
    return ConfigLoader.get_available_configs()

def get_current_config_name() -> str:
    """Get the name of the currently loaded configuration file"""
    return get_config().current_config_file


# Shared typed accessors for both configuration stacks.
class Config:
    """Shared accessors; subclasses only choose their process-local loader."""

    @staticmethod
    def _get_config():
        return get_config()

    @classmethod
    def get_image_dimensions(cls):
        """Get image dimensions"""
        config = cls._get_config()
        return {
            'rgb_width': config.get('image.rgb_width', 1080),
            'rgb_height': config.get('image.rgb_height', 1440),
            'event_width': config.get('image.event_width', 720),
            'event_height': config.get('image.event_height', 1280)
        }

    @classmethod
    def get_timing(cls):
        """Get timing parameters"""
        config = cls._get_config()
        return {
            'startup_delay': config.get('timing.startup_delay', 3),
            'reset_delay': config.get('timing.reset_delay', 2),
            'shutdown_timeout': config.get('timing.shutdown_timeout', 5)
        }

    @classmethod
    def get_image_processing(cls):
        """Get image processing parameters"""
        config = cls._get_config()
        return {
            'accumulation_time_us': config.get('image_processing.accumulation_time_us', 1000),
            'fps': config.get('image_processing.fps', 0.0),
            'binary_threshold': config.get('image_processing.binary_threshold', 40),
            'binary_max_value': config.get('image_processing.binary_max_value', 255),
            'contrast_alpha': config.get('image_processing.contrast_alpha', 0.5),
            'contrast_beta': config.get('image_processing.contrast_beta', 30),
            'brightness_multiplier': config.get('image_processing.brightness_multiplier', 1.0),
            'display_scale': config.get('image_processing.display_scale', 0.8)
        }

    @classmethod
    def get_clustering(cls):
        """Get clustering parameters"""
        config = cls._get_config()
        return {
            'eps': config.get('clustering.eps', 9),
            'min_samples': config.get('clustering.min_samples', 70)
        }

    @classmethod
    def get_paths(cls):
        """Resolve machine-local data and project-relative resource paths."""
        from project_paths import config_paths
        return config_paths(cls._get_config())

    @classmethod
    def get_alignment(cls):
        """Get camera alignment parameters"""
        config = cls._get_config()
        return {
            'event_pixel_size': config.get('alignment.event_pixel_size', 3.45),
            'rgb_pixel_size': config.get('alignment.rgb_pixel_size', 4.86),
            'offset_x': config.get('alignment.offset_x', 30),
            'offset_y': config.get('alignment.offset_y', 29),
            'y_coordinate_offset': config.get('alignment.y_coordinate_offset', 1279),
            # Deprecated compatibility key; live mounting rotation is fixed.
            'apply_rotation': config.get('alignment.apply_rotation', False),
            'replay_apply_rotation': config.get(
                'alignment.replay_apply_rotation', False
            ),
        }

    @classmethod
    def get_camera_extrinsics(cls):
        """Get camera extrinsic rotation parameters"""
        config = cls._get_config()
        return {
            'camera_rpy_to': config.get('camera_extrinsics.camera_rpy_to', [0, 45, -90]),
            'camera_rpy_back': config.get('camera_extrinsics.camera_rpy_back', [45, 0, 90])
        }

    @classmethod
    def get_camera_intrinsics(cls):
        """Get camera intrinsic parameters"""
        config = cls._get_config()
        return {
            'fx': config.get('camera_intrinsics.fx', 987.1678420424932),
            'fy': config.get('camera_intrinsics.fy', 986.0683237433913),
            'cx': config.get('camera_intrinsics.cx', 513.7733221750243),
            'cy': config.get('camera_intrinsics.cy', 739.7418995490075),
            'dist_coeffs': config.get(
                'camera_intrinsics.distortion_coeffs',
                [
                    -0.11123899870851763,
                    -0.020663002330730863,
                    -0.0002997445722290338,
                    0.0007964973013426446,
                    0.014818755754515196,
                ],
            ),
        }

    @classmethod
    def get_data_reading(cls):
        """Get data reading parameters"""
        config = cls._get_config()
        return {
            'delta_t': config.get('data_reading.delta_t', 2000),
            'n_events': config.get('data_reading.n_events', 10000),
            'send_interval': config.get('data_reading.send_interval', 0.002),
            'video_period': config.get('data_reading.video_period', 0.02)
        }

    @classmethod
    def get_roi(cls):
        """Get ROI detection parameters"""
        config = cls._get_config()
        return {
            'hsv_lower': config.get('rgb_param.hsv_lower', [30, 90, 0]),
            'hsv_upper': config.get('rgb_param.hsv_upper', [50, 255, 255]),
            'kernel_size': config.get('rgb_param.kernel_size', [5, 5]),
            'morphology_iterations': config.get('rgb_param.morphology_iterations', 2),
            'roi_range': config.get('roi_range', 200),
            'default_bbox': config.get('default_bbox', [430, 1180, 40, 40]),
            'backup_bbox': config.get('backup_bbox', [180, 1030, 40, 40])
        }

    @classmethod
    def get_tracking(cls):
        """Get tracking parameters"""
        config = cls._get_config()
        return {
            'dbscan_eps': config.get('tracking.dbscan_eps', 15),
            'dbscan_min_samples': config.get('tracking.dbscan_min_samples', 100),
            'max_ball_buffer': config.get('tracking.max_ball_buffer', 1000),
            'hsv_display_scale': config.get('tracking.hsv_display_scale', 0.75)
        }

    @classmethod
    def get_motor(cls):
        """Get motor control parameters"""
        config = cls._get_config()
        return {
            'bitrate': config.get('motor.bitrate', 1000000),
            'motor_id': config.get('motor.motor_id', 101),
            'main_can_id': config.get('motor.main_can_id', 254),
            'kp': config.get('motor.kp', 100),
            'kd': config.get('motor.kd', 1),
            'shaft_length': config.get('motor.shaft_length', 0.5),
            'joint_height': config.get('motor.joint_height', 0.7),
            'movement_delay': config.get('motor.movement_delay', 2.5)
        }

    @classmethod
    def get_trajectory(cls):
        """World units: meters, seconds; legacy bounds_x/y mean world Y/Z."""
        config = cls._get_config()
        return {
            'inlier_threshold': config.get('trajectory.inlier_threshold', 0.35),
            'gravity': config.get('trajectory.gravity', -9.81),
            'impact_bounds_x': config.get('trajectory.impact_bounds_x', [-1.5, 1.5]),
            'impact_bounds_y': config.get('trajectory.impact_bounds_y', [0, 2]),
            'save_delay_sec': config.get('trajectory.save_delay_sec', 0.15)
        }


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
