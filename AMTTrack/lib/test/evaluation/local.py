from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.
    settings.felt_path = '/home/xinchi/Event_RGB_catching_data/AMTTrack/data/FELT'
    settings.network_path = '/home/xinchi/Event_RGB_catching_data/AMTTrack/output/test/networks'    # Where tracking networks are stored.
    settings.prj_dir = '/home/xinchi/Event_RGB_catching_data/AMTTrack'
    settings.result_plot_path = '/home/xinchi/Event_RGB_catching_data/AMTTrack/output/test/result_plots'
    settings.results_path = '/home/xinchi/Event_RGB_catching_data/AMTTrack/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/home/xinchi/Event_RGB_catching_data/AMTTrack/output'
    settings.segmentation_path = '/home/xinchi/Event_RGB_catching_data/AMTTrack/output/test/segmentation_results'
    settings.visevent_path = '/home/xinchi/Event_RGB_catching_data/AMTTrack/data/VisEvent'

    return settings

