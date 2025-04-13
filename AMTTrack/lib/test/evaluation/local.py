from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.coesot_path = '/home/xinchi/Event_RGB_catching/AMTTrack/data/COESOT'
    settings.fe108_path = '/home/xinchi/Event_RGB_catching/AMTTrack/data/FE108'
    settings.network_path = '/home/xinchi/Event_RGB_catching/AMTTrack/output/test/networks'    # Where tracking networks are stored.
    settings.prj_dir = '/home/xinchi/Event_RGB_catching/AMTTrack'
    settings.result_plot_path = '/home/xinchi/Event_RGB_catching/AMTTrack/output/test/result_plots'
    settings.results_path = '/home/xinchi/Event_RGB_catching/AMTTrack/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/home/xinchi/Event_RGB_catching/AMTTrack/output'
    settings.segmentation_path = '/home/xinchi/Event_RGB_catching/AMTTrack/output/test/segmentation_results'
    settings.visevent_path = '/home/xinchi/Event_RGB_catching/AMTTrack/data/VisEvent'

    return settings

