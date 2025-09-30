# Event-Only Ball Tracking System

This directory contains a standalone event-based ball tracking and catching system designed for high-speed ball tracking using event cameras.

## Overview

The event-only system provides real-time ball tracking capabilities using only event camera data, without dependence on RGB cameras or external modules. It includes trajectory prediction and robot control for automated ball catching.

## System Architecture

### Core Components

1. **Event Data Input** (`read_from_event.py`)
   - Reads event stream from event camera
   - Handles event data buffering and synchronization
   - Manages system state transitions

2. **Event Tracking** (`Event_tracker.py`)
   - 3D ball tracking using event data
   - Real-time position estimation
   - Noise filtering and trajectory smoothing

3. **Trajectory Prediction** (`fitting_process.py`)
   - Parabolic trajectory fitting
   - Landing point prediction
   - Robot control coordination

4. **Configuration Management** (`config_loader.py`)
   - System configuration loading
   - Parameter management
   - Path configuration

5. **Utilities** (`utils.py`, `parabola_fitter.py`)
   - Mathematical utilities
   - Trajectory fitting algorithms
   - Helper functions

## Quick Start

### Prerequisites
- Event camera hardware
- Robot control system (optional)
- Python dependencies: numpy, torch, multiprocessing, pynput, pyttsx3

### Running the System

```bash
cd event_only
python live_real_time.py --test_mode Event
```

### Command Line Options

```bash
python live_real_time.py [OPTIONS]

Options:
  --record_index INT    Initial record index (default: 0)
  --test_mode STR      Test mode: Event (default: Robot)
  --task STR           Task configuration: normal (default: normal)
```

## System States

The system operates with 5 main states:

- **'w' (waiting)**: System ready, waiting for ball
- **'t' (tracking)**: Actively tracking ball trajectory  
- **'c' (catching)**: Robot executing catch motion
- **'r' (reset)**: Resetting for next iteration
- **'e' (exit)**: Shutting down system

## Keyboard Controls

| Key | Action | Description |
|-----|--------|-------------|
| `t` | Start Tracking | Begin ball tracking with voice notification |
| `w` | Set Waiting | Manual waiting state control |
| Other | Exit | Shutdown system gracefully |

## Voice Notifications

The system provides audio feedback when transitioning to tracking state:
- **Message**: "System started tracking"
- **Requirements**: pyttsx3 library (optional)
- **Fallback**: Text notifications if voice unavailable

## Process Architecture

### Event Process Group

The system runs multiple concurrent processes:

1. **event_process**: Event data acquisition
2. **event_tracking_process**: Real-time ball tracking
3. **landing_prediction_process**: Trajectory prediction
4. **robot_control_process**: Robot motion control
5. **keyboard_process**: User input handling

### Shared Memory

Inter-process communication via shared memory:

- `shared_event_data`: Event camera data buffer
- `shared_flir_data`: Compatibility buffer for RGB integration
- `shared_ROI_data`: Region of Interest parameters
- `shared_ball_pixel_3D_data`: 3D ball position tracking
- `shared_impact_point_data`: Predicted landing coordinates

## Configuration

### Config File (`config.yaml`)
```yaml
# Event camera settings
event_camera:
  resolution: [1080, 1440]
  
# Tracking parameters
tracking:
  roi_size: 100
  tracking_threshold: 0.5
  
# Robot control
robot:
  catching_height: 0.3
  reaction_time: 0.1
```

## API Reference

### Main Entry Point

```python
# Start event-only tracking system
python live_real_time.py --test_mode Event
```

### Key Functions

```python
# Event data reading
read_from_event(shared_event_data, system_state, num_event, stop_event)

# 3D ball tracking  
event_tracking_3D(shared_event_data, shared_flir_data, shared_ROI_data,
                  shared_ball_pixel_3D_data, ball_observations, num_event, 
                  system_state, stop_event)

# Trajectory prediction
landing_prediction(shared_ball_pixel_3D_data, ball_observations,
                   shared_impact_point_data, record_index, system_state, stop_event)

# Robot control
robot_control(shared_impact_point_data, system_state, stop_event)
```

## Safety Features

### Graceful Shutdown
- Signal handling (Ctrl+C, SIGTERM)
- Process cleanup with timeout
- Shared memory cleanup
- Multi-phase shutdown process

### Error Handling
- Process failure detection
- Automatic restart capabilities
- Configuration fallbacks
- Voice notification fallbacks

## Performance

### Real-time Capabilities
- Event-driven processing
- Low-latency tracking
- Multiprocess architecture
- Optimized shared memory usage

### System Requirements
- CPU: Multi-core recommended
- Memory: 4GB+ RAM
- GPU: CUDA-compatible (optional)
- Storage: Minimal requirements

## Troubleshooting

### Common Issues

1. **No event data**: Check event camera connection
2. **Voice notifications fail**: Install pyttsx3 or ignore warnings
3. **Process startup errors**: Verify Python multiprocessing support
4. **Memory errors**: Check available RAM and shared memory limits

### Debug Mode

Enable debug output:
```bash
python live_real_time.py --test_mode Event --verbose
```

### Log Files

System logs are output to console. For persistent logging:
```bash
python live_real_time.py --test_mode Event 2>&1 | tee system.log
```

## Development

### Adding New Features

1. Implement new process function
2. Add to `all_processes` dictionary
3. Update process groups as needed
4. Add shared memory if required
5. Update configuration files

### Testing

```bash
# Basic functionality test
python live_real_time.py --test_mode Event --record_index 0

# Keyboard control test
# Press 't' to start tracking
# Press 'w' to set waiting state
```

## Integration

### Parent System Integration

This event-only system can be used standalone or integrated with the parent Event_RGB_catching system for hybrid event+RGB tracking capabilities.

### Robot Integration

Compatible with robot control systems that accept 3D coordinate inputs for automated ball catching.

## License

Part of the Event_RGB_catching project. See parent directory for licensing information.

## Support

For issues and questions:
1. Check troubleshooting section
2. Review system logs
3. Verify hardware connections
4. Contact system maintainers