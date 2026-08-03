import os

from lib.test.evaluation.environment import EnvSettings


def local_env_settings():
    settings = EnvSettings()

    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
    data_dir = os.path.join(project_dir, 'data')
    output_dir = os.path.join(project_dir, 'output')

    settings.coesot_path = os.path.join(data_dir, 'COESOT')
    settings.fe108_path = os.path.join(data_dir, 'FE108')
    settings.network_path = os.path.join(output_dir, 'test', 'networks')
    settings.prj_dir = project_dir
    settings.result_plot_path = os.path.join(output_dir, 'test', 'result_plots')
    settings.results_path = os.path.join(output_dir, 'test', 'tracking_results')
    settings.save_dir = output_dir
    settings.segmentation_path = os.path.join(output_dir, 'test', 'segmentation_results')
    settings.visevent_path = os.path.join(data_dir, 'VisEvent')

    return settings
