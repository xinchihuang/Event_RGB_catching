"""Machine-local data paths; relative paths are based on the project directory.

Precedence: explicit argument > named environment variable > local setting >
stage default. Stage defaults use data_root, never probe alternate directories.
"""
import argparse
import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
SCENES = {'normal': 'l', 'clutter': 'c', 'dark': 'd', 'dark_clutter': 'dc'}


def local_settings():
    path = PROJECT_ROOT / 'config/paths.local.yaml'
    if not path.exists():
        return {}
    with path.open(encoding='utf-8') as handle:
        values = yaml.safe_load(handle)
    if not isinstance(values, dict):
        raise ValueError(f'{path} must contain a YAML mapping')
    return values


def absolute_path(value):
    if value is None or not str(value).strip():
        raise ValueError('Path must not be empty')
    path = Path(value).expanduser()
    return (PROJECT_ROOT / path).resolve()


def data_root():
    value = (os.environ['EVENT_RGB_DATA_ROOT'] if 'EVENT_RGB_DATA_ROOT' in os.environ
             else local_settings().get('data_root'))
    if value is None:
        raise ValueError('Set EVENT_RGB_DATA_ROOT or data_root in config/paths.local.yaml '
                         '(see config/paths.local.example.yaml), or pass an explicit path')
    return absolute_path(value)


def resolve_path(value):
    text = str(value) if value is not None else ''
    if '${data_root}' in text:
        text = text.replace('${data_root}', str(data_root()))
    return absolute_path(text)


def setting_path(key, explicit=None, env=None, default=None):
    if explicit is not None:
        return resolve_path(explicit)
    if env and env in os.environ:
        return resolve_path(os.environ[env])
    value = local_settings().get(key, default)
    if value is None:
        raise ValueError(f'Specify {key} explicitly or in config/paths.local.yaml')
    return resolve_path(value)


def stage_path(stage, task='normal', explicit=None):
    case = SCENES[task]
    if stage == 'processed':
        # DATA_ROOT has historically meant processed_data, not the whole disk.
        return setting_path('processed_root', explicit, 'DATA_ROOT', '${data_root}/processed_data')
    if stage == 'replay':
        if explicit is not None:
            return resolve_path(explicit)
        for env in ('RECORD_DIR', 'DATA_DIR'):
            if env in os.environ:
                return resolve_path(os.environ[env])
        return stage_path('processed') / f'saved_data_{case}'
    if stage == 'raw':
        return setting_path('capture_dir', explicit, 'CAPTURE_DIR',
                            '${data_root}/saved_data_raw/saved_data_' + case)
    if stage == 'dataset':
        return setting_path('dataset_dir', explicit, 'DATASET_DIR',
                            '${data_root}/saved_data/dataset_' + case)
    if stage == 'selected':
        return setting_path('selected_dataset', explicit, 'SELECTED_DATASET_DIR',
                            '${data_root}/saved_data/selected_dataset')
    if stage == 'evaluation':
        return setting_path('evaluation_root', explicit, 'EVALUATION_RESULTS_DIR',
                            '${data_root}/test_data/evaluation_results')
    if stage == 'demo':
        return setting_path('comparison_root', explicit, 'COMPARISON_RESULTS_DIR', '${data_root}/demo')
    raise ValueError(f'Unknown data stage: {stage}')


def config_paths(config):
    """Resolve paths exposed by both config stacks, preserving env overrides."""
    record = config.get('paths.record_dir', '${data_root}/processed_data/saved_data_dc')
    if 'RECORD_DIR' in os.environ or 'DATA_DIR' in os.environ:
        record = os.environ.get('RECORD_DIR', os.environ.get('DATA_DIR'))
    elif 'DATA_ROOT' in os.environ:
        record = stage_path('processed') / Path(record).name
    elif str(record).startswith('${data_root}/processed_data/'):
        record = stage_path('processed') / Path(record).name
    evaluation = os.environ.get('EVALUATION_RESULTS_DIR')
    if evaluation is None:
        evaluation = local_settings().get('evaluation_root', config.get(
            'paths.evaluation_root', '${data_root}/test_data/evaluation_results'))
    return dict(
        record_dir=str(resolve_path(record)), evaluation_root=str(resolve_path(evaluation)),
        # Existing template names are relative to tracking/, not the launch cwd.
        template_npy=str((PROJECT_ROOT / 'tracking' / config.get('paths.template_npy', '94.npy')).resolve()),
        template_jpg=str((PROJECT_ROOT / 'tracking' / config.get('paths.template_jpg', '94.jpg')).resolve()),
    )


def require_recordings(path):
    path = resolve_path(path)
    for child in ('event', 'rgb'):
        if not (path / child).is_dir():
            raise FileNotFoundError(f'Missing input directory: {path / child}')
    return path


def distinct_output(source, output):
    source, output = resolve_path(source), resolve_path(output)
    if source == output or source in output.parents or output in source.parents:
        raise ValueError(f'Input and output must be separate directories: {source}, {output}')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['raw', 'processed', 'replay', 'dataset', 'evaluation', 'demo'], required=True)
    parser.add_argument('--task', choices=SCENES, default='normal')
    args = parser.parse_args()
    try:
        print(stage_path(args.stage, args.task))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
