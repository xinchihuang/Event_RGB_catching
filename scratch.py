import torch
print(torch.version.__version__)
# Path to your .tar file
checkpoint_path = '/home/xinchi/Event_RGB_catching_data/AMTTrack/output/checkpoints/train/amttrack/amttrack_felt/AMTTrack_ep0050.pth.tar'

# Load the .tar file
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# Print keys of the checkpoint to verify
print(checkpoint.keys())

# Load model state dict (assuming 'net' contains the model weights)
model_state_dict = checkpoint['net']

# Initialize your model



# Optionally, load optimizer state or other information
# optimizer.load_state_dict(checkpoint['optimizer'])
# epoch = checkpoint['epoch']
