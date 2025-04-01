import torch
print(torch.version.__version__)
# Path to your .tar file
checkpoint_path = '/home/xinchi/Event_RGB_catching_data/AMTTrack/output/checkpoints/train/amttrack/amttrack_felt/AMTTrack_ep0050.pth.tar'

with open(checkpoint_path, 'rb') as f:
    checkpoint = torch.load(f, map_location='cpu')

print("Checkpoint keys:", checkpoint.keys())

# Load model state dict (assuming 'net' contains the model weights)
model_state_dict = checkpoint['net']
