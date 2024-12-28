from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.
    settings.felt_path = '/home/xinchi/FELT_SOT_Benchmark/AMTTrack/data/FELT'
    settings.coesot_path = '/home/xinchi/FELT_SOT_Benchmark/AMTTrack/data/COESOT'
    settings.fe108_path = '/home/xinchi/FELT_SOT_Benchmark/AMTTrack/data/FE108'
    settings.network_path = '/home/xinchi/FELT_SOT_Benchmark/AMTTrack/output/test/networks'    # Where tracking networks are stored.
    settings.prj_dir = '/home/xinchi/FELT_SOT_Benchmark/AMTTrack'
    settings.result_plot_path = '/home/xinchi/FELT_SOT_Benchmark/AMTTrack/output/test/result_plots'
    settings.results_path = '/home/xinchi/FELT_SOT_Benchmark/AMTTrack/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/home/xinchi/FELT_SOT_Benchmark/AMTTrack/output'
    settings.segmentation_path = '/home/xinchi/FELT_SOT_Benchmark/AMTTrack/output/test/segmentation_results'
    settings.visevent_path = '/home/xinchi/FELT_SOT_Benchmark/AMTTrack/data/VisEvent'

    return settings

