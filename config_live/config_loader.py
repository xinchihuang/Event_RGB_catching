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


def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge overlay into base."""
    if not isinstance(base, dict) or not isinstance(overlay, dict):
        return overlay
    merged = dict(base)
    for key, value in overlay.items():
        merged[key] = _deep_merge(merged.get(key), value)
    return merged


class ConfigLoader:
    """Simple configuration loader for hardcoded parameters"""
    
    def __init__(self, config_file: str = "config_live/config_l.yaml"):
        """
        Initialize configuration loader
        
        Args:
            config_file: Configuration file name (default: config.yaml)
        """
        # Look for config file in project root
        project_root = Path(__file__).parent.parent
        self.camera_config_path = project_root / "config_live/camera_config.yaml"
        self.tracking_config_path = project_root / "config_live/tracking_config.yaml"
        self.config_path = project_root / config_file
        self.current_config_file = config_file
        
        self.config = {}
        self.logger = logging.getLogger(__name__)
        self._load_config()
    
    def switch_config(self, config_file: str) -> None:
        """
        Switch to a different configuration file
        
        Args:
            config_file: New configuration file name
        """
        project_root = Path(__file__).parent.parent
        new_config_path = project_root / config_file
        
        if not new_config_path.exists():
            self.logger.error(f"Configuration file not found: {new_config_path}")
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        self.config_path = new_config_path
        self.current_config_file = config_file
        self._load_config()
        self.logger.info(f"Switched to configuration: {config_file}")
    
    @classmethod
    def get_available_configs(cls) -> List[str]:
        """
        Get list of available configuration files
        
        Returns:
            List of available configuration file names
        """
        project_root = Path(__file__).parent.parent
        overlay_dir = project_root / "config_live/overlays"
        config_files = []

        if overlay_dir.exists():
            for config_file in overlay_dir.glob("*.yaml"):
                config_files.append(str(Path("config_live/overlays") / config_file.name))
        else:
            config_dir = project_root / "config_live"
            for config_file in config_dir.glob("*.yaml"):
                if config_file.name in {"camera_config.yaml", "tracking_config.yaml"}:
                    continue
                config_files.append(str(Path("config_live") / config_file.name))
        
        return sorted(config_files)
    
    def _load_config(self) -> None:
        """Load configuration from camera + tracking + overlay YAML files"""
        camera_config = {}
        if self.camera_config_path.exists():
            try:
                with open(self.camera_config_path, 'r', encoding='utf-8') as f:
                    camera_config = yaml.safe_load(f) or {}
            except yaml.YAMLError as e:
                self.logger.error(f"YAML parsing error in camera config: {e}")
                raise
            except Exception as e:
                self.logger.error(f"Failed to load camera config: {e}")
                raise
        else:
            self.logger.warning(f"Camera config file not found: {self.camera_config_path}")

        tracking_config = {}
        if self.tracking_config_path.exists():
            try:
                with open(self.tracking_config_path, 'r', encoding='utf-8') as f:
                    tracking_config = yaml.safe_load(f) or {}
            except yaml.YAMLError as e:
                self.logger.error(f"YAML parsing error in tracking config: {e}")
                raise
            except Exception as e:
                self.logger.error(f"Failed to load tracking config: {e}")
                raise
        else:
            self.logger.warning(f"Tracking config file not found: {self.tracking_config_path}")

        overlay_config = {}
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    overlay_config = yaml.safe_load(f) or {}
            except yaml.YAMLError as e:
                self.logger.error(f"YAML parsing error in overlay config: {e}")
                raise
            except Exception as e:
                self.logger.error(f"Failed to load overlay config: {e}")
                raise
        else:
            self.logger.warning(f"Overlay config file not found: {self.config_path}")

        self.config = _deep_merge(camera_config, tracking_config)
        self.config = _deep_merge(self.config, overlay_config)
        if overlay_config:
            self.logger.info(
                "Configuration loaded: camera=%s, tracking=%s, overlay=%s",
                self.camera_config_path,
                self.tracking_config_path,
                self.config_path,
            )
        else:
            self.logger.info(
                "Configuration loaded: camera=%s, tracking=%s",
                self.camera_config_path,
                self.tracking_config_path,
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

def get_config() -> ConfigLoader:
    """Get global configuration instance with smart config file selection"""
    global _config
    if _config is None:
        # 优先级：环境变量 > 命令行参数 > 默认值
        config_file = None
        
        # 1. 检查环境变量
        if 'ACTIVE_CONFIG_FILE' in os.environ:
            config_file = os.environ['ACTIVE_CONFIG_FILE']
        
        # 2. 尝试解析命令行参数（如果在主进程中）
        elif hasattr(sys, 'argv') and '--task' in sys.argv:
            try:
                task_idx = sys.argv.index('--task')
                if task_idx + 1 < len(sys.argv):
                    task = sys.argv[task_idx + 1]
                    test_mode = None
                    if '--test_mode' in sys.argv:
                        test_mode_idx = sys.argv.index('--test_mode')
                        if test_mode_idx + 1 < len(sys.argv):
                            test_mode = sys.argv[test_mode_idx + 1]
                    if not test_mode:
                        test_mode = os.environ.get("TEST_MODE")

                    task_key = (task or "").lower()
                    task_base = {
                        "normal": "config_l",
                        "dark": "config_d",
                        "clutter": "config_c",
                        "dark_clutter": "config_dc"
                    }
                    base = task_base.get(task_key)
                    if base:
                        config_file = f"config_live/{base}.yaml"
            except:
                pass
        
        # 3. 使用默认配置
        if config_file is None:
            config_file = "config_live/config_l.yaml"

        project_root = Path(__file__).parent.parent
        candidate = Path(config_file)
        if not candidate.is_absolute():
            overlay_candidate = Path("config_live/overlays") / candidate.name
            if (project_root / overlay_candidate).exists():
                config_file = str(overlay_candidate)
            
        _config = ConfigLoader(config_file)
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


# Hardcoded parameter mappings for easy migration
class Config:
    """Configuration constants for easy access"""
    
    @staticmethod
    def get_image_dimensions():
        """Get image dimensions"""
        config = get_config()
        return {
            'rgb_width': config.get('image.rgb_width', 1080),
            'rgb_height': config.get('image.rgb_height', 1440),
            'event_width': config.get('image.event_width', 720),
            'event_height': config.get('image.event_height', 1280)
        }
    
    @staticmethod 
    def get_timing():
        """Get timing parameters"""
        config = get_config()
        return {
            'startup_delay': config.get('timing.startup_delay', 3),
            'reset_delay': config.get('timing.reset_delay', 2),
            'shutdown_timeout': config.get('timing.shutdown_timeout', 5)
        }
    
    @staticmethod
    def get_image_processing():
        """Get image processing parameters"""
        config = get_config()
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
    
    @staticmethod
    def get_clustering():
        """Get clustering parameters"""
        config = get_config()
        return {
            'eps': config.get('clustering.eps', 9),
            'min_samples': config.get('clustering.min_samples', 70)
        }
    
    @staticmethod
    def  get_paths():
        """Get file paths"""
        config = get_config()
        return {
            'record_dir': config.get('paths.record_dir', '/media/xinchi/data/Event_RGB_catching_data/saved_data_dc'),
            'evaluation_root': config.get('paths.evaluation_root', '/media/xinchi/data/test_data/evaluation_results'),
            'template_npy': config.get('paths.template_npy', '94.npy'),
            'template_jpg': config.get('paths.template_jpg', '94.jpg')
        }

    @staticmethod
    def get_alignment():
        """Get camera alignment parameters"""
        config = get_config()
        return {
            'event_pixel_size': config.get('alignment.event_pixel_size', 3.45),
            'rgb_pixel_size': config.get('alignment.rgb_pixel_size', 4.86),
            'offset_x': config.get('alignment.offset_x', 30),
            'offset_y': config.get('alignment.offset_y', 29),
            'y_coordinate_offset': config.get('alignment.y_coordinate_offset', 1279)
        }

    @staticmethod
    def get_camera_extrinsics():
        """Get camera extrinsic rotation parameters"""
        config = get_config()
        return {
            'camera_rpy_to': config.get('camera_extrinsics.camera_rpy_to', [0, 45, -90]),
            'camera_rpy_back': config.get('camera_extrinsics.camera_rpy_back', [45, 0, 90])
        }

    @staticmethod
    def get_camera_intrinsics():
        """Get camera intrinsic parameters"""
        config = get_config()
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
    
    @staticmethod
    def get_data_reading():
        """Get data reading parameters"""
        config = get_config()
        return {
            'delta_t': config.get('data_reading.delta_t', 2000),
            'n_events': config.get('data_reading.n_events', 10000),
            'send_interval': config.get('data_reading.send_interval', 0.002),
            'video_period': config.get('data_reading.video_period', 0.02)
        }
    
    @staticmethod
    def get_roi():
        """Get ROI detection parameters"""
        config = get_config()
        return {
            'hsv_lower': config.get('rgb_param.hsv_lower', [30, 90, 0]),
            'hsv_upper': config.get('rgb_param.hsv_upper', [50, 255, 255]),
            'kernel_size': config.get('rgb_param.kernel_size', [5, 5]),
            'morphology_iterations': config.get('rgb_param.morphology_iterations', 2),
            'roi_range': config.get('roi_range', 200),
            'default_bbox': config.get('default_bbox', [430, 1180, 40, 40]),
            'backup_bbox': config.get('backup_bbox', [180, 1030, 40, 40])
        }
    
    @staticmethod
    def get_tracking():
        """Get tracking parameters"""
        config = get_config()
        return {
            'dbscan_eps': config.get('tracking.dbscan_eps', 15),
            'dbscan_min_samples': config.get('tracking.dbscan_min_samples', 100),
            'max_ball_buffer': config.get('tracking.max_ball_buffer', 1000),
            'hsv_display_scale': config.get('tracking.hsv_display_scale', 0.75)
        }
    
    @staticmethod
    def get_motor():
        """Get motor control parameters"""
        config = get_config()
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
    
    @staticmethod
    def get_trajectory():
        """Get trajectory fitting parameters"""
        config = get_config()
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
