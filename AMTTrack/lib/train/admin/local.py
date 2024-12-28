class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/home/yifan/FELT_SOT_Benchmark/AMTTrack'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/home/yifan/FELT_SOT_Benchmark/AMTTrack/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/home/yifan/FELT_SOT_Benchmark/AMTTrack/pretrained_networks'
        self.coesot_dir = '/home/yifan/FELT_SOT_Benchmark/AMTTrack/data/COESOT/train'
        self.coesot_val_dir = '/home/yifan/FELT_SOT_Benchmark/AMTTrack/data/COESOT/test'
        self.fe108_dir = '/home/yifan/FELT_SOT_Benchmark/AMTTrack/data/FE108/train'
        self.fe108_val_dir = '/home/yifan/FELT_SOT_Benchmark/AMTTrack/data/FE108/test'
        self.visevent_dir = '/home/yifan/FELT_SOT_Benchmark/AMTTrack/data/VisEvent/train'
        self.visevent_val_dir = '/home/yifan/FELT_SOT_Benchmark/AMTTrack/data/VisEvent/test'
