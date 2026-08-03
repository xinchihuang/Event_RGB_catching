class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/home/xinchi/Event_RGB_catching/AMTTrack'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/home/xinchi/Event_RGB_catching/AMTTrack/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/home/xinchi/Event_RGB_catching/AMTTrack/pretrained_networks'
        self.coesot_dir = '/home/xinchi/Event_RGB_catching/AMTTrack/data/COESOT/train'
        self.coesot_val_dir = '/home/xinchi/Event_RGB_catching/AMTTrack/data/COESOT/test'
        self.fe108_dir = '/home/xinchi/Event_RGB_catching/AMTTrack/data/FE108/train'
        self.fe108_val_dir = '/home/xinchi/Event_RGB_catching/AMTTrack/data/FE108/test'
        self.visevent_dir = '/home/xinchi/Event_RGB_catching/AMTTrack/data/VisEvent/train'
        self.visevent_val_dir = '/home/xinchi/Event_RGB_catching/AMTTrack/data/VisEvent/test'
        self.felt_dir = '/home/xinchi/FELT_dataset'
        # self.ball_dir = '/media/xinchi/data/saved_data/selected_dataset'