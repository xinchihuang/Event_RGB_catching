"""Portable training paths; machine-specific overrides belong in the environment."""
import os
from pathlib import Path

AMTTRACK_ROOT = Path(__file__).resolve().parents[3]


def _path(variable, default):
    value = Path(os.environ.get(variable, str(default))).expanduser()
    return (AMTTRACK_ROOT / value).resolve()


class EnvironmentSettings:
    def __init__(self):
        workspace = _path('AMTTRACK_WORKSPACE_DIR', AMTTRACK_ROOT)
        data = _path('AMTTRACK_DATA_ROOT', AMTTRACK_ROOT / 'data')
        self.workspace_dir = str(workspace)
        self.tensorboard_dir = str(_path('AMTTRACK_TENSORBOARD_DIR', workspace / 'tensorboard'))
        self.pretrained_networks = str(_path('AMTTRACK_PRETRAINED_DIR', AMTTRACK_ROOT / 'pretrained_networks'))
        self.coesot_dir = str(data / 'COESOT/train')
        self.coesot_val_dir = str(data / 'COESOT/test')
        self.fe108_dir = str(data / 'FE108/train')
        self.fe108_val_dir = str(data / 'FE108/test')
        self.visevent_dir = str(data / 'VisEvent/train')
        self.visevent_val_dir = str(data / 'VisEvent/test')
        # Keep the existing home-relative FELT location without embedding a username.
        self.felt_dir = str(_path('AMTTRACK_FELT_DIR', Path.home() / 'FELT_dataset'))
