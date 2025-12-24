# Event-Based Ball Catching System

An event-based real-time ball tracking and catching system using event cameras and RGB cameras. The system performs real-time trajectory prediction and robot control for catching flying balls.

## Overview

This system combines event-based vision with traditional RGB imaging to achieve high-speed, low-latency ball tracking and trajectory prediction. The system uses DBSCAN clustering for ball detection from event camera data and implements parabolic motion models for trajectory fitting and landing prediction.

## System Architecture

The system uses a multi-process architecture with shared memory for efficient inter-process communication:

```
┌─────────────────────┐
│   Main Process      │
│   (live_real.py)    │
└──────────┬──────────┘
           │
           ├───> Event Reading Process
           │     (read_from_event.py)
           │
           ├───> Event Tracking Process
           │     (Event_tracker.py)
           │
           ├───> Trajectory Prediction
           │     (fitting_process.py)
           │
           ├───> GUI Display Process
           │     (gui_process.py)
           │
           └───> Keyboard Input Process
                 (keyboard control)
```

## Key Features

- **Event-Based Ball Detection**: Uses DBSCAN clustering on event camera data for fast ball detection
- **Direct Event Clustering**: Clusters events directly without converting to frames first
- **Trajectory Prediction**: Implements RANSAC-based parabolic trajectory fitting
- **Real-Time Display**: Provides real-time visualization of event and RGB data
- **Multi-Process Architecture**: Efficient parallel processing with shared memory
- **Configurable Parameters**: YAML-based configuration system for easy parameter tuning

## File Descriptions

### Main Entry Point
- **`live_real.py`**: Main program entry point. Orchestrates all processes and manages system lifecycle.

### Core Modules

#### Event Processing
- **`read_from_event.py`**: Reads event data from the event camera and publishes to shared memory
  - Supports both ASAP and fixed-rate modes
  - Implements time-budgeted event draining to minimize latency
  - Handles coordinate transformation and alignment with RGB camera

#### Ball Tracking
- **`Event_tracker.py`**: Event-based ball tracking module
  - `event_tracking_3D()`: Frame-based tracking using event frame generation
  - `event_tracking_3D_direct()`: Direct event clustering (faster, lower latency)
  - `ROI_filter()`: Region of Interest filtering for focused tracking

#### Trajectory Fitting
- **`fitting_process.py`**: Ball trajectory prediction and landing position calculation
  - `landing_prediction()`: Main prediction function
  - Uses MonoRansac fitter for robust trajectory estimation
  - Validates predictions against configurable bounds

- **`ransac_fit.py`**: RANSAC-based parabolic trajectory fitting
  - `adaptive_ransac_parabola_3d()`: Adaptive RANSAC fitting
  - `least_squares_solution_parabola()`: Least squares refinement
  - Handles gravity compensation in trajectory model

#### Visualization
- **`gui_process.py`**: Real-time GUI display process
  - Displays event and RGB frames side-by-side
  - Shows ball detection bounding boxes
  - Visualizes ROI regions
  - Version-based data synchronization to avoid stale frames

#### Configuration
- **`config_loader.py`**: Configuration management system
  - YAML-based configuration loading
  - Provides static accessor methods for different parameter groups
  - Supports multiple configuration files

### Configuration Files
- **`config.yaml`**: Main configuration file (see Configuration section below)

### Utility Files
- **`utils.py`**: Utility functions (currently empty placeholder)
- **`parabola_fitter.py`**: Parabola fitting utilities (currently empty placeholder)

## Installation

### Requirements
```bash
# Python dependencies
pip install numpy opencv-python pyyaml scikit-learn pynput pyttsx3 torch

# Event camera SDK (Prophesee/Metavision)
# Follow installation instructions from:
# https://docs.prophesee.ai/stable/installation/index.html
```

### Hardware Requirements
- Event Camera (compatible with Metavision SDK)
- RGB Camera (optional, for hybrid tracking)
- CUDA-capable GPU (recommended for real-time performance)

## Configuration

The system is configured via `config.yaml`. Key configuration sections:

### Image Dimensions
```yaml
image:
  rgb_width: 1080
  rgb_height: 1440
  event_width: 720
  event_height: 1280
```

### Camera Alignment
```yaml
alignment:
  event_pixel_size: 3.45
  rgb_pixel_size: 4.86
  offset_x: 30
  offset_y: 29
  y_coordinate_offset: 1279
```

### Event Data Reading
```yaml
data_reading:
  delta_t: 2000              # Event packet time window (microseconds)
  n_events: 10000            # Events per packet
  send_interval: 0.002       # Publishing interval (seconds)
```

### Ball Tracking (DBSCAN)
```yaml
tracking:
  dbscan_eps: 12             # DBSCAN epsilon (pixel distance)
  dbscan_min_samples: 110    # Minimum samples for cluster
  max_ball_buffer: 100       # Maximum ball observations to store
```

### Trajectory Prediction
```yaml
trajectory:
  required_observations: 15   # Observations needed for fitting
  inlier_threshold: 0.35      # RANSAC inlier threshold (meters)
  gravity: -9.81              # Gravity constant (m/s^2)
  impact_bounds_x: [-1.5, 1.5]  # Valid landing zone X bounds
  impact_bounds_y: [0, 2]       # Valid landing zone Y bounds
```

## Usage

### Basic Usage

```bash
# Run with default configuration
python live_real.py

# Run with specific task configuration
python live_real.py --task normal

# Run with specific test mode
python live_real.py --test_mode Event

# Disable GUI display (headless mode)
ENABLE_GUI=0 python live_real.py
```

### Command Line Arguments

```bash
python live_real.py [OPTIONS]

Options:
  --test_mode STR    Test mode selection (default: Event)
                     Options: Event
  --task STR         Task configuration to use (default: normal)
                     Options: normal
```

### Keyboard Controls

During runtime, the following keyboard controls are available:
- **`t`**: Start tracking mode (triggers voice notification "System started tracking. Throw the ball now!")
- **`w`**: Enter wait mode (pause tracking)
- **`e`**: Exit the system

### System States

The system operates in three states:
- **`'w'` (Wait)**: System is waiting, no tracking active
- **`'t'` (Tracking)**: Active tracking and prediction
- **`'e'` (Exit)**: System shutdown

## Process Communication

### Shared Memory Buffers

The system uses shared memory for efficient data sharing:

1. **Event Data**: Raw event stream data
   - Structure: (x, y, t, p) - coordinates, timestamp, polarity

2. **FLIR/RGB Data**: RGB camera image data (if used)
   - Structure: RGB image array

3. **Ball Pixel 3D Data**: Ball observation data
   - Structure: (x, y, size, timestamp)

4. **ROI Data**: Region of Interest parameters
   - Structure: (x, y, width, height)

5. **Impact Point Data**: Predicted landing position
   - Structure: (x, y, z)

### Data Versioning

Event and FLIR data use version counters to prevent processing stale data:
- `event_data_version`: Incremented each time new event data is published
- `flir_data_version`: Incremented each time new FLIR data is available

## Algorithm Details

### Event Clustering (Direct Mode)

1. Read events from shared memory
2. Filter events by Region of Interest (ROI)
3. Extract unique (x, y) coordinates from events
4. Apply DBSCAN clustering on spatial coordinates
5. Extract largest cluster as ball
6. Store ball center and size with timestamp

### Trajectory Prediction

1. Collect ball observations (x, y, size, t)
2. Convert 2D pixel coordinates to 3D world coordinates using ball size
3. Apply RANSAC-based parabolic fitting with gravity compensation
4. Predict landing position using fitted trajectory
5. Validate prediction against configured bounds
6. Publish impact point to shared memory

### Time-Budgeted Event Draining

To minimize latency, the event reading process implements time-budgeted draining:
1. Set time budget based on mode (ASAP: 100ms, Fixed-rate: 75% of period)
2. Continuously read event batches within budget
3. Keep only the latest batch (discard intermediate data)
4. Publish latest batch to avoid backlog accumulation

## Performance Considerations

### Optimization Tips

1. **DBSCAN Parameters**: Tune `eps` and `min_samples` for your ball size and camera resolution
2. **Event Send Interval**: Lower values reduce latency but increase CPU load
3. **ROI Usage**: Define tight ROI to reduce processing overhead
4. **Direct Clustering**: Use `event_tracking_3D_direct` for lower latency vs `event_tracking_3D`

### Expected Performance

- Event reading rate: 500 Hz (configurable)
- Tracking latency: < 10ms (direct mode)
- Frame generation latency: ~20ms (frame-based mode)

## Troubleshooting

### Common Issues

1. **No events detected**
   - Check event camera connection
   - Verify camera permissions
   - Check `delta_t` and `n_events` parameters

2. **Poor tracking accuracy**
   - Adjust DBSCAN parameters (`eps`, `min_samples`)
   - Calibrate camera alignment parameters
   - Ensure adequate lighting conditions

3. **High latency**
   - Use direct clustering mode
   - Reduce `send_interval`
   - Check CPU/GPU utilization

4. **Process hangs on exit**
   - The system implements graceful shutdown with timeout
   - If processes don't exit, they will be force-terminated after 5 seconds

5. **Voice notifications fail**
   - Install pyttsx3: `pip install pyttsx3`
   - System will fallback to text notifications if unavailable

## Development

### Adding New Processes

To add a new process to the system:

1. Define the process function
2. Add to `all_processes` dict in `live_real.py`
3. Add process name to appropriate `process_groups`
4. Handle shared memory and stop_event properly

Example:
```python
def my_new_process(shared_data, system_state, stop_event):
    while not stop_event.is_set():
        if system_state.value == 't':
            # Process logic here
            pass
```

### Configuration Guidelines

- Keep all tunable parameters in `config.yaml`
- Use `Config` class static methods for parameter access
- Document parameter units and valid ranges

### Process Safety

The system implements a three-phase shutdown process:
1. **Phase 1**: Wait for graceful exit (timeout: 5 seconds)
2. **Phase 2**: Force terminate remaining processes
3. **Phase 3**: Force kill stubborn processes

## Integration

### Parent System Integration

This event-only system is part of the larger Event_RGB_catching project and can be used standalone or integrated with RGB tracking for hybrid event+RGB tracking capabilities.

### Robot Integration

The system outputs predicted impact points that can be used by robot control systems for automated ball catching.

## References

### Event Camera Resources
- [Prophesee Documentation](https://docs.prophesee.ai/)
- [Event-Based Vision Resources](https://github.com/uzh-rpg/event-based_vision_resources)

### Algorithms
- DBSCAN Clustering: [scikit-learn](https://scikit-learn.org/stable/modules/clustering.html#dbscan)
- RANSAC Fitting: Fischler & Bolles (1981)

## License

Part of the Event_RGB_catching project. See parent directory for licensing information.

## Contact

For issues and questions:
1. Check troubleshooting section
2. Review system logs
3. Verify hardware connections
4. Contact system maintainers

## Acknowledgments

This project uses:
- Prophesee Metavision SDK for event camera interface
- OpenCV for image processing
- scikit-learn for clustering algorithms
- PyTorch for tensor operations
- pyttsx3 for voice notifications
