
# Project: Event-RGB Catching

This project is designed to track and catch a thrown object (likely a ball) using a combination of an event-based camera and a standard RGB camera. The system uses a deep learning model for object tracking and a classical physics model for trajectory prediction.

## High-Level Design

The project follows a pipeline architecture:

1.  **Data Acquisition**: Synchronized data from an event camera and an RGB camera is captured.
2.  **Data Processing**: The raw data is processed to align the event and RGB streams, detect the object of interest, and generate training data.
3.  **Object Tracking**: A deep learning model, AMTTrack, processes the synchronized sensor data to track the object in 2D.
4.  **3D Trajectory Fitting**: The 2D tracking data is used to fit a 3D trajectory of the object.
5.  **Catching**: The predicted 3D trajectory is used to calculate the impact point, which can then be used to control a robotic arm to catch the object.

## File Structure

The project is organized into the following key directories:

*   `/`: The root directory contains scripts for data collection, trajectory fitting, and other utilities.
*   `AMTTrack/`: This directory contains the core deep learning-based object tracking model.
    *   `lib/`: This subdirectory contains the main source code for the AMTTrack model.
        *   `models/`: Contains the model definitions, including the Vision Transformer (ViT) backbone and the main AMTTrack model.
        *   `train/`: Contains scripts for training the model.
        *   `test/`: Contains scripts for evaluating the model.
*   `scripts/`: This directory contains various Python scripts for data collection, dataset preparation, and 3D trajectory fitting experiments.
*   `plot/`: This directory contains scripts for visualizing data, such as plotting 3D trajectories.
*   `saved_data/`: This directory is likely used to store the raw and processed data.

## Key Files

*   `collect_data.py`: Captures synchronized data from the event and RGB cameras.
*   `prepare_dataset.py`: Processes the raw data, aligns the streams, and generates training data.
*   `scratch_fitting_3D.py`: Fits a 3D trajectory to the tracked object.
*   `AMTTrack/lib/models/amttrack/amttrack.py`: Defines the main AMTTrack model, which combines a Vision Transformer backbone with a tracking head.
*   `AMTTrack/lib/models/amttrack/vit_ce.py`: Implements a Vision Transformer with a Candidate Elimination module for efficient processing of event data.

## Core Technologies

*   **Python**: The primary programming language.
*   **PyTorch**: The deep learning framework used to build the AMTTrack model.
*   **OpenCV**: Used for image and video processing.
*   **Metavision SDK**: Used for interfacing with the event-based camera.
*   **NumPy, SciPy**: Used for numerical computation and data analysis.
