# Configuration System

Simple configuration management for Event RGB Catching System hardcoded parameters.

## Files

- `config.yaml` - Main configuration file with all hardcoded parameters
- `config_loader.py` - Simple configuration loader utility

## Usage

### Basic Usage

```python
from config.config_loader import load_config

# Get individual values
rgb_width = load_config('image.rgb_width', 1080)
eps = load_config('clustering.eps', 9)
record_dir = load_config('paths.record_dir')
```

### Using Config Class

```python
from config.config_loader import Config

# Get grouped parameters
dims = Config.get_image_dimensions()
# Returns: {'rgb_width': 1080, 'rgb_height': 1440, ...}

timing = Config.get_timing()
# Returns: {'startup_delay': 3, 'reset_delay': 2, ...}

processing = Config.get_image_processing()
# Returns: {'accumulation_time_us': 1000, 'fps': 0.0, ...}
```

## Configuration Categories

### Image Dimensions
- `image.rgb_width` (1080)
- `image.rgb_height` (1440) 
- `image.event_width` (720)
- `image.event_height` (1280)

### Timing Parameters
- `timing.startup_delay` (3 seconds)
- `timing.reset_delay` (2 seconds)
- `timing.shutdown_timeout` (5 seconds)

### Image Processing
- `image_processing.accumulation_time_us` (1000)
- `image_processing.binary_threshold` (40)
- `image_processing.contrast_alpha` (0.5)
- `image_processing.display_scale` (0.8)

### Clustering (DBSCAN)
- `clustering.eps` (9)
- `clustering.min_samples` (70)

### File Paths
- `paths.record_dir`
- `paths.template_npy` (94.npy)
- `paths.template_jpg` (94.jpg)

## Migration Example

**Before (hardcoded):**
```python
rgb_width, rgb_height = 1080, 1440
dbscan = DBSCAN(eps=9, min_samples=70)
```

**After (configurable):**
```python
from config.config_loader import Config

dims = Config.get_image_dimensions()
rgb_width, rgb_height = dims['rgb_width'], dims['rgb_height']

clustering = Config.get_clustering()
dbscan = DBSCAN(eps=clustering['eps'], min_samples=clustering['min_samples'])
```

## Modifying Configuration

Edit `config.yaml` to change parameter values:

```yaml
clustering:
  eps: 12        # Changed from 9
  min_samples: 80 # Changed from 70
```

Configuration is loaded once at startup. Restart the application to apply changes.