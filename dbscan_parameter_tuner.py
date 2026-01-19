#!/usr/bin/env python3
"""
DBSCAN Parameter Tuner - Interactive tool for optimizing clustering parameters
Features:
- White event visualization
- Dynamic parameter adjustment via command line or interactive mode
- Customizable image processing parameters
- Real-time visual feedback
- Support for both .npy files and live event camera
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import os
import sys
import argparse
import threading
import time

# Add project path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette
from config.config_loader import Config, get_config
from data_input.read_from_raw import align_to_rgb


# Default parameters (used when config is not loaded)
DEFAULT_PARAMS = {
    'eps': 12,
    'min_samples': 70,
    'accumulation_time': 1000,
    'threshold': 40,
    'event_color': [255, 255, 255],
    'display_scale': 0.6,
    'rotation': 0,
    'delta_t_read': 2000,
    'delta_t': 10000,
    'n_events': 100000,
    'fps': 10,
    'batch_eps_range': [8, 18, 2],
    'batch_min_samples_range': [40, 100, 10]
}

# Default data file path (used in PyCharm default run)
DEFAULT_DATA_FILE = None  # Will be set from config at runtime


def load_config_for_task(task_name):
    """
    Load configuration for specific task

    Args:
        task_name: Task name (normal, dark, clutter, dark_clutter)
    """
    test_mode_key = os.environ.get("TEST_MODE", "").strip().lower()
    mode_suffix = {"model": "model", "rgb": "rgb", "event": "event"}.get(test_mode_key)
    task_key = (task_name or "").lower()
    task_base = {
        "normal": "config_l",
        "dark": "config_d",
        "clutter": "config_c",
        "dark_clutter": "config_dc"
    }
    legacy_mapping = {
        "normal": "config/config_l.yaml",
        "dark": "config/config_d.yaml",
        "clutter": "config/config_c.yaml",
        "dark_clutter": "config/config_dc.yaml"
    }

    config_file = None
    if mode_suffix:
        base = task_base.get(task_key)
        if base:
            config_file = f"config/{base}_{mode_suffix}.yaml"
    if config_file is None:
        config_file = legacy_mapping.get(task_key)
    if config_file is None:
        print(f"Warning: Unknown task '{task_name}', using default config")
        return

    # Set environment variable for config selection
    os.environ['ACTIVE_CONFIG_FILE'] = config_file

    # Reload config to pick up the new file
    config = get_config()
    config.switch_config(config_file)

    print(f"Loaded configuration: {config_file} (task: {task_name})")


def get_params_from_config():
    """
    Get parameters from existing config sections

    Returns:
        dict: Parameters from config, with defaults for missing values
    """
    try:
        # Get existing config sections
        image_proc = Config.get_image_processing()
        tracking = Config.get_tracking()
        data_reading = Config.get_data_reading()
        image_dims = Config.get_image_dimensions()
        paths = Config.get_paths()

        # Build default data file path: record_dir/event/0.raw
        record_dir = paths.get('record_dir', '')
        default_data_file = os.path.join(record_dir, 'event', '0.raw') if record_dir else None

        # Map to tuner parameters
        params = {
            # From tracking section
            'eps': tracking.get('dbscan_eps', DEFAULT_PARAMS['eps']),
            'min_samples': tracking.get('dbscan_min_samples', DEFAULT_PARAMS['min_samples']),

            # From image_processing section
            'accumulation_time': image_proc.get('accumulation_time_us', DEFAULT_PARAMS['accumulation_time']),
            'threshold': image_proc.get('binary_threshold', DEFAULT_PARAMS['threshold']),
            'display_scale': image_proc.get('display_scale', DEFAULT_PARAMS['display_scale']),

            # From data_reading section
            'delta_t_read': data_reading.get('delta_t', DEFAULT_PARAMS['delta_t_read']),
            'n_events': data_reading.get('n_events', DEFAULT_PARAMS['n_events']),

            # From image dimensions
            'width': image_dims.get('rgb_width', 1080),
            'height': image_dims.get('rgb_height', 1440),

            # From paths section
            'default_data_file': default_data_file,

            # Not in config, use defaults
            'event_color': DEFAULT_PARAMS['event_color'],
            'rotation': DEFAULT_PARAMS['rotation'],
            'delta_t': DEFAULT_PARAMS['delta_t'],
            'fps': DEFAULT_PARAMS['fps'],
            'batch_eps_range': DEFAULT_PARAMS['batch_eps_range'],
            'batch_min_samples_range': DEFAULT_PARAMS['batch_min_samples_range']
        }

        return params

    except Exception as e:
        print(f"Warning: Failed to load config ({e}), using defaults")
        params = DEFAULT_PARAMS.copy()
        params['default_data_file'] = None
        return params


class LiveEventCameraReader:
    """Real-time event camera reader using threading"""

    def __init__(self, width=1080, height=1440, delta_t=10000, n_events=100000):
        """
        Initialize live event camera reader

        Args:
            width: Image width
            height: Image height
            delta_t: Time window for events (microseconds)
            n_events: Number of events per batch
        """
        self.width = width
        self.height = height
        self.delta_t = delta_t
        self.n_events = n_events
        self.latest_events = None
        self.running = False
        self.lock = threading.Lock()
        self.event_iterator = None
        self.thread = None

        # Import align_to_rgb at init time to avoid thread issues
        from data_input.read_from_event import align_to_rgb
        from config.config_loader import Config
        self.align_to_rgb = align_to_rgb
        self.Config = Config

    def start(self):
        """Start the event camera reader thread"""
        from metavision_core.event_io import EventsIterator

        print(f"Starting live event camera (delta_t={self.delta_t}us, n_events={self.n_events})...")

        # Initialize event iterator for live camera (empty input_path)
        self.event_iterator = iter(EventsIterator(
            input_path="",  # Empty path means live camera
            mode="mixed",
            delta_t=self.delta_t,
            n_events=self.n_events
        ))

        self.running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

        print("Event camera started successfully!")

    def _read_loop(self):
        """Background thread to continuously read events"""
        while self.running:
            try:
                events = next(self.event_iterator)

                # Apply coordinate transformation (same as read_from_event.py)
                x = events['x'].copy()
                y = events['y'].copy()

                # User transform: swap and negate
                x_new = y.copy()
                y_new = -x.copy()

                # Apply y-coordinate offset
                alignment = self.Config.get_alignment()
                y_new = y_new + alignment['y_coordinate_offset']

                # Align to RGB dimensions
                x_aligned, y_aligned = self.align_to_rgb({'x': x_new, 'y': y_new})

                # Create new structured array
                ev_transformed = np.zeros(events.shape, dtype=events.dtype)
                ev_transformed['x'] = x_aligned
                ev_transformed['y'] = y_aligned
                ev_transformed['p'] = events['p'].copy()
                ev_transformed['t'] = events['t'].copy()

                with self.lock:
                    self.latest_events = ev_transformed

            except StopIteration:
                print("Event stream ended")
                break
            except Exception as e:
                print(f"Event camera error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.01)

    def get_latest_events(self):
        """Get the most recent event batch"""
        with self.lock:
            return self.latest_events.copy() if self.latest_events is not None else None

    def stop(self):
        """Stop the event camera reader"""
        print("Stopping event camera...")
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        print("Event camera stopped")


class RawFileReader:
    """
    Simple .raw file reader for DBSCAN parameter tuning
    Loads entire .raw file into memory without multiprocessing
    """

    def __init__(self, raw_file, width=1080, height=1440, delta_t=2000):
        """
        Args:
            raw_file: Path to .raw event file
            width: Image width
            height: Image height
            delta_t: Time window for event chunking (microseconds)
        """
        from metavision_core.event_io import EventsIterator

        self.width = width
        self.height = height

        print(f"Loading .raw file: {raw_file}")
        print(f"  Image size: {width}x{height}")
        print(f"  Delta_t: {delta_t}us")

        event_chunks = []
        total_events = 0

        # Read events in chunks
        iterator = EventsIterator(
            input_path=raw_file,
            mode="delta_t",
            delta_t=delta_t,
            n_events=100000
        )

        for chunk_idx, ev in enumerate(iterator):
            if len(ev) == 0:
                continue

            # Apply coordinate transformation (same as read_from_raw.py)
            # Step 1: Swap x,y and negate x
            x = ev['x'].copy()
            y = ev['y'].copy()
            x_new = y.copy()
            y_new = -x.copy()

            # Step 2: Apply y-coordinate offset
            alignment = Config.get_alignment()
            y_new = y_new + alignment['y_coordinate_offset']

            # Step 3: Align to RGB dimensions
            x_aligned, y_aligned = align_to_rgb({'x': x_new, 'y': y_new})

            # Create transformed event array
            ev_transformed = np.zeros(ev.shape, dtype=ev.dtype)
            ev_transformed['x'] = x_aligned
            ev_transformed['y'] = y_aligned
            ev_transformed['p'] = ev['p'].copy()
            ev_transformed['t'] = ev['t'].copy()

            event_chunks.append(ev_transformed)
            total_events += len(ev_transformed)

            if (chunk_idx + 1) % 100 == 0:
                print(f"  Loaded {chunk_idx + 1} chunks, {total_events} events...")

        # Concatenate all chunks
        if event_chunks:
            self.all_events = np.concatenate(event_chunks)
        else:
            self.all_events = np.array([], dtype=event_chunks[0].dtype if event_chunks else None)

        print(f"✓ Loaded {len(self.all_events)} total events from {len(event_chunks)} chunks")

        if len(self.all_events) > 0:
            print(f"  Time range: {self.all_events['t'][0]} to {self.all_events['t'][-1]} us")
            print(f"  X range: {self.all_events['x'].min()} to {self.all_events['x'].max()}")
            print(f"  Y range: {self.all_events['y'].min()} to {self.all_events['y'].max()}")

    def get_events(self, time_slice=None):
        """
        Get events (all or time slice)

        Args:
            time_slice: (start_us, end_us) or None for all events

        Returns:
            Event array
        """
        if time_slice is None:
            return self.all_events

        start_us, end_us = time_slice
        mask = (self.all_events['t'] >= start_us) & (self.all_events['t'] <= end_us)
        return self.all_events[mask]

    def get_chunk(self, chunk_index, chunk_size_us=2000):
        """
        Get events in a specific time chunk

        Args:
            chunk_index: Chunk index
            chunk_size_us: Chunk size in microseconds

        Returns:
            Event array for that chunk
        """
        if len(self.all_events) == 0:
            return self.all_events

        start_time = self.all_events['t'][0]
        chunk_start = start_time + chunk_index * chunk_size_us
        chunk_end = chunk_start + chunk_size_us

        return self.get_events((chunk_start, chunk_end))


class NPYFileReader:
    """
    NPY file reader compatible with RawFileReader interface for playback mode
    """
    
    def __init__(self, event_data):
        """
        Initialize with event data from .npy file
        
        Args:
            event_data: Event array loaded from .npy file
        """
        self.all_events = event_data
        print(f"✓ NPY Reader initialized with {len(event_data)} events")
        if len(event_data) > 0:
            print(f"  Time range: {event_data['t'][0]} to {event_data['t'][-1]} us")
    
    def get_events(self, time_slice=None):
        """
        Get events (all or time slice)
        
        Args:
            time_slice: (start_us, end_us) or None for all events
        
        Returns:
            Event array
        """
        if time_slice is None:
            return self.all_events
        
        start_us, end_us = time_slice
        mask = (self.all_events['t'] >= start_us) & (self.all_events['t'] <= end_us)
        return self.all_events[mask]
    
    def get_chunk(self, chunk_index, chunk_size_us=2000):
        """
        Get events in a specific time chunk
        
        Args:
            chunk_index: Chunk index
            chunk_size_us: Chunk size in microseconds
        
        Returns:
            Event array for that chunk
        """
        if len(self.all_events) == 0:
            return self.all_events
        
        start_time = self.all_events['t'][0]
        end_time = self.all_events['t'][-1]
        
        # Handle case where time is decreasing
        if end_time < start_time:
            chunk_start = start_time - chunk_index * chunk_size_us
            chunk_end = chunk_start - chunk_size_us
            # Ensure chunk_end >= chunk_start for proper range
            if chunk_end < chunk_start:
                chunk_start, chunk_end = chunk_end, chunk_start
        else:
            chunk_start = start_time + chunk_index * chunk_size_us
            chunk_end = chunk_start + chunk_size_us
        
        return self.get_events((chunk_start, chunk_end))


class DBSCANParameterTuner:
    """Interactive DBSCAN parameter tuning tool"""

    def __init__(self,
                 width=1080,
                 height=1440,
                 accumulation_time_us=1000,
                 binary_threshold=40,
                 event_color=(255, 255, 255),  # White by default
                 display_scale=0.6,
                 rotation=0):
        """
        Initialize the tuner

        Args:
            width: Image width
            height: Image height
            accumulation_time_us: Event accumulation time in microseconds
            binary_threshold: Threshold for binary conversion (0-255)
            event_color: RGB color for event visualization (R, G, B)
            display_scale: Scale factor for display (0.0-1.0)
            rotation: Image rotation in degrees (0, 90, 180, 270)
        """
        self.width = width
        self.height = height
        self.accumulation_time_us = accumulation_time_us
        self.binary_threshold = binary_threshold
        self.event_color = event_color
        self.display_scale = display_scale
        self.rotation = rotation

        # Initialize event frame generator
        self.event_frame_gen = PeriodicFrameGenerationAlgorithm(
            sensor_width=width,
            sensor_height=height,
            fps=0.0,
            palette=ColorPalette.Dark
        )

    def load_events(self, npy_file):
        """Load event data from .npy file"""
        print(f"Loading events from: {npy_file}")
        event_data = np.load(npy_file)
        print(f"  Shape: {event_data.shape}")
        print(f"  Dtype: {event_data.dtype}")
        if len(event_data) > 0:
            print(f"  Time range: {event_data['t'][0]} to {event_data['t'][-1]} us")
        return event_data

    def events_to_image(self, event_data):
        """
        Convert event data to images (event frame, binary, grayscale)

        Returns:
            tuple: (event_frame_colored, binary_image, gray_image)
        """
        # Generate event frame
        event_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        event_frame = np.ascontiguousarray(event_frame)
        self.event_frame_gen.generate_frame(
            event_data,
            event_frame,
            accumulation_time_us=self.accumulation_time_us
        )

        # Convert to grayscale
        gray_image = cv2.cvtColor(event_frame, cv2.COLOR_BGR2GRAY)

        # Apply binary threshold
        _, binary_image = cv2.threshold(
            gray_image,
            self.binary_threshold,
            255,
            cv2.THRESH_BINARY
        )

        # Create colored event frame with custom color
        event_frame_colored = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        mask = binary_image > 0
        event_frame_colored[mask] = self.event_color

        return event_frame_colored, binary_image, gray_image

    def cluster_with_dbscan(self, binary_image, eps, min_samples):
        """
        Apply DBSCAN clustering to binary image

        Args:
            binary_image: Binary image (0 or 255)
            eps: DBSCAN epsilon parameter
            min_samples: DBSCAN min_samples parameter

        Returns:
            tuple: (points, labels, n_clusters, n_noise)
        """
        # Extract points from binary image
        points = np.column_stack(np.where(binary_image > 0))

        if len(points) == 0:
            print("  ⚠ No points found in binary image")
            return points, np.array([]), 0, 0

        # Apply DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(points)

        # Count clusters and noise
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        return points, labels, n_clusters, n_noise

    def cluster_events_direct(self, event_data, eps, min_samples):
        """
        Apply DBSCAN clustering directly to event coordinates (like event_tracking_3D_direct)

        Args:
            event_data: Structured array with 'x', 'y', 'p', 't' fields
            eps: DBSCAN epsilon parameter
            min_samples: DBSCAN min_samples parameter

        Returns:
            tuple: (points, labels, n_clusters, n_noise)
        """
        if len(event_data) == 0:
            print("  ⚠ No events to cluster")
            return np.array([]), np.array([]), 0, 0

        # Extract (x, y) coordinates directly from events
        points = np.column_stack([event_data['x'], event_data['y']])

        print(f"  Direct clustering on {len(points)} events")

        # Apply DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(points)

        # Count clusters and noise
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        return points, labels, n_clusters, n_noise

    def visualize_clusters(self, event_frame_colored, points, labels,
                          show_bbox=True, show_labels=True):
        """
        Create visualization with colored clusters

        Args:
            event_frame_colored: Base image with white events
            points: Array of point coordinates
            labels: Cluster labels for each point
            show_bbox: Whether to draw bounding boxes
            show_labels: Whether to show cluster labels

        Returns:
            result_image: Visualization with colored clusters
        """
        result_image = event_frame_colored.copy()

        if len(points) == 0 or len(labels) == 0:
            return result_image

        # Separate noise and cluster points
        cluster_mask = labels != -1
        cluster_points = points[cluster_mask]
        cluster_labels = labels[cluster_mask]
        noise_points = points[~cluster_mask]

        # Generate colors for each cluster
        unique_labels = np.unique(cluster_labels)
        if len(unique_labels) > 0:
            colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

            # Color each cluster
            for label, color in zip(unique_labels, colors):
                label_mask = cluster_labels == label
                label_points = cluster_points[label_mask]

                # Convert color to BGR
                color_bgr = (int(color[2]*255), int(color[1]*255), int(color[0]*255))

                # Color the points
                for pt in label_points:
                    result_image[pt[0], pt[1]] = color_bgr

                # Draw bounding box
                if show_bbox and len(label_points) > 0:
                    y_min, x_min = label_points.min(axis=0)
                    y_max, x_max = label_points.max(axis=0)

                    # Draw rectangle
                    cv2.rectangle(result_image, (x_min, y_min), (x_max, y_max),
                                (0, 255, 0), 2)

                    # Draw label
                    if show_labels:
                        label_text = f"C{int(label)}"
                        cv2.putText(result_image, label_text, (x_min, y_min-5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                        # Show cluster size
                        size_text = f"{len(label_points)}pts"
                        cv2.putText(result_image, size_text, (x_min, y_max+15),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Draw noise points in red
        for pt in noise_points:
            result_image[pt[0], pt[1]] = (0, 0, 255)

        return result_image

    def run_clustering_direct(self, event_data, eps, min_samples,
                             save_path=None, show_visualization=True):
        """
        Run direct event clustering (like event_tracking_3D_direct)

        Args:
            event_data: Event data array with 'x', 'y', 'p', 't' fields
            eps: DBSCAN epsilon
            min_samples: DBSCAN min_samples
            save_path: Path to save result (optional)
            show_visualization: Whether to display result

        Returns:
            dict: Results including images and statistics
        """
        print(f"\n{'='*60}")
        print(f"Running DBSCAN DIRECT: eps={eps}, min_samples={min_samples}")
        print(f"{'='*60}")

        # Direct clustering on event coordinates
        print("Applying DBSCAN clustering directly to events...")
        points, labels, n_clusters, n_noise = self.cluster_events_direct(
            event_data, eps, min_samples
        )

        # Statistics
        total_points = len(points)
        signal_points = total_points - n_noise
        signal_ratio = (signal_points / total_points * 100) if total_points > 0 else 0

        print(f"Results:")
        print(f"  Total events: {total_points}")
        print(f"  Clusters: {n_clusters}")
        print(f"  Signal events: {signal_points} ({signal_ratio:.1f}%)")
        print(f"  Noise events: {n_noise}")

        if n_clusters > 0:
            avg_cluster_size = signal_points / n_clusters
            print(f"  Avg cluster size: {avg_cluster_size:.1f} events")

        # Create background image for visualization
        event_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Plot all events as white dots first
        for pt in points:
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < self.width and 0 <= y < self.height:
                event_frame[y, x] = (255, 255, 255)

        # Visualize clusters
        print("Creating visualization...")
        result_image = self.visualize_clusters_direct(event_frame, points, labels)

        # Add info text only if showing visualization
        if show_visualization:
            info_text = [
                f"DIRECT MODE: eps={eps}, min_samples={min_samples}",
                f"Clusters: {n_clusters}, Noise: {n_noise}",
                f"Signal: {signal_points}/{total_points} ({signal_ratio:.1f}%)"
            ]

            y_offset = 30
            for text in info_text:
                cv2.putText(result_image, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                y_offset += 30

        # Display
        if show_visualization:
            display_img = result_image.copy()
            if self.rotation == 90:
                display_img = cv2.rotate(display_img, cv2.ROTATE_90_CLOCKWISE)
            elif self.rotation == 180:
                display_img = cv2.rotate(display_img, cv2.ROTATE_180)
            elif self.rotation == 270:
                display_img = cv2.rotate(display_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

            if self.rotation in [90, 270]:
                display_width = int(self.height * self.display_scale)
                display_height = int(self.width * self.display_scale)
            else:
                display_width = int(self.width * self.display_scale)
                display_height = int(self.height * self.display_scale)

            display_img = cv2.resize(display_img, (display_width, display_height))
            cv2.imshow('DBSCAN Direct Clustering Results', display_img)

        # Save
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.',
                       exist_ok=True)
            cv2.imwrite(save_path, result_image)
            print(f"  Saved to: {save_path}")

        return {
            'result_image': result_image,
            'event_frame': event_frame,
            'points': points,
            'labels': labels,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'signal_points': signal_points,
            'signal_ratio': signal_ratio
        }

    def visualize_clusters_direct(self, event_frame, points, labels):
        """
        Visualize clusters from direct event clustering

        Args:
            event_frame: Background image
            points: Event coordinates (x, y)
            labels: Cluster labels

        Returns:
            result_image: Visualization with colored clusters
        """
        result_image = event_frame.copy()

        if len(points) == 0 or len(labels) == 0:
            return result_image

        # Separate noise and cluster points
        cluster_mask = labels != -1
        cluster_points = points[cluster_mask]
        cluster_labels = labels[cluster_mask]
        noise_points = points[~cluster_mask]

        # Generate colors for each cluster
        unique_labels = np.unique(cluster_labels)
        if len(unique_labels) > 0:
            colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

            # Color each cluster
            for label, color in zip(unique_labels, colors):
                label_mask = cluster_labels == label
                label_points = cluster_points[label_mask]

                # Convert color to BGR
                color_bgr = (int(color[2]*255), int(color[1]*255), int(color[0]*255))

                # Color the event points
                for pt in label_points:
                    x, y = int(pt[0]), int(pt[1])
                    if 0 <= x < self.width and 0 <= y < self.height:
                        result_image[y, x] = color_bgr

                # Draw bounding box
                if len(label_points) > 0:
                    x_coords = label_points[:, 0]
                    y_coords = label_points[:, 1]
                    x_min, x_max = int(x_coords.min()), int(x_coords.max())
                    y_min, y_max = int(y_coords.min()), int(y_coords.max())

                    cv2.rectangle(result_image, (x_min, y_min), (x_max, y_max),
                                (0, 255, 0), 2)

                    # Draw label
                    label_text = f"C{int(label)}"
                    cv2.putText(result_image, label_text, (x_min, y_min-5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # Show cluster size
                    size_text = f"{len(label_points)}pts"
                    cv2.putText(result_image, size_text, (x_min, y_max+15),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Draw noise points in red
        for pt in noise_points:
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < self.width and 0 <= y < self.height:
                result_image[y, x] = (0, 0, 255)

        return result_image

    def run_clustering(self, event_data, eps, min_samples,
                      save_path=None, show_visualization=True, use_direct=False):
        """
        Run complete clustering pipeline

        Args:
            event_data: Event data array
            eps: DBSCAN epsilon
            min_samples: DBSCAN min_samples
            save_path: Path to save result (optional)
            show_visualization: Whether to display result
            use_direct: Use direct event clustering (True) or image-based (False)

        Returns:
            dict: Results including images and statistics
        """
        # Use direct mode if requested
        if use_direct:
            return self.run_clustering_direct(event_data, eps, min_samples,
                                            save_path, show_visualization)

        print(f"\n{'='*60}")
        print(f"Running DBSCAN: eps={eps}, min_samples={min_samples}")
        print(f"{'='*60}")

        # Convert events to images
        print("Converting events to images...")
        event_frame_colored, binary_image, gray_image = self.events_to_image(event_data)

        # Run DBSCAN
        print("Applying DBSCAN clustering...")
        points, labels, n_clusters, n_noise = self.cluster_with_dbscan(
            binary_image, eps, min_samples
        )

        # Statistics
        total_points = len(points)
        signal_points = total_points - n_noise
        signal_ratio = (signal_points / total_points * 100) if total_points > 0 else 0

        print(f"Results:")
        print(f"  Total points: {total_points}")
        print(f"  Clusters: {n_clusters}")
        print(f"  Signal points: {signal_points} ({signal_ratio:.1f}%)")
        print(f"  Noise points: {n_noise}")

        if n_clusters > 0:
            avg_cluster_size = signal_points / n_clusters
            print(f"  Avg cluster size: {avg_cluster_size:.1f} points")

        # Create visualization
        print("Creating visualization...")
        result_image = self.visualize_clusters(event_frame_colored, points, labels)

        # Add info text on image only if showing visualization
        if show_visualization:
            info_text = [
                f"eps={eps}, min_samples={min_samples}",
                f"Clusters: {n_clusters}, Noise: {n_noise}",
                f"Signal: {signal_points}/{total_points} ({signal_ratio:.1f}%)"
            ]

            y_offset = 30
            for text in info_text:
                cv2.putText(result_image, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                y_offset += 30

        # Display
        if show_visualization:
            # Apply rotation if needed
            display_img = result_image.copy()
            if self.rotation == 90:
                display_img = cv2.rotate(display_img, cv2.ROTATE_90_CLOCKWISE)
            elif self.rotation == 180:
                display_img = cv2.rotate(display_img, cv2.ROTATE_180)
            elif self.rotation == 270:
                display_img = cv2.rotate(display_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # Calculate display dimensions after rotation
            if self.rotation in [90, 270]:
                display_width = int(self.height * self.display_scale)
                display_height = int(self.width * self.display_scale)
            else:
                display_width = int(self.width * self.display_scale)
                display_height = int(self.height * self.display_scale)

            display_img = cv2.resize(display_img, (display_width, display_height))
            cv2.imshow('DBSCAN Clustering Results', display_img)

        # Save
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.',
                       exist_ok=True)
            cv2.imwrite(save_path, result_image)
            print(f"  Saved to: {save_path}")

        return {
            'result_image': result_image,
            'event_frame': event_frame_colored,
            'binary_image': binary_image,
            'gray_image': gray_image,
            'points': points,
            'labels': labels,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'signal_points': signal_points,
            'signal_ratio': signal_ratio
        }

    def interactive_mode(self, event_data, initial_eps=9, initial_min_samples=70):
        """
        Interactive parameter tuning with keyboard controls

        Args:
            event_data: Event data array or None for live camera
            initial_eps: Starting epsilon value
            initial_min_samples: Starting min_samples value
        """
        eps = float(initial_eps)
        min_samples = int(initial_min_samples)

        print("\n" + "="*60)
        print("Interactive DBSCAN Parameter Tuning")
        print("="*60)
        print("\nKeyboard Controls:")
        print("  w/s     - Increase/decrease eps by 1")
        print("  e/d     - Increase/decrease eps by 0.5")
        print("  a/z     - Increase/decrease min_samples by 10")
        print("  q/x     - Increase/decrease min_samples by 5")
        print("  r       - Reset to initial values")
        print("  t       - Re-test with current parameters")
        print("  p       - Print current parameters")
        print("  SPACE   - Save current result")
        print("  ESC     - Exit")
        print("="*60)

        save_counter = 1

        while True:
            # Run clustering with current parameters
            results = self.run_clustering(
                event_data, eps, min_samples,
                show_visualization=True
            )

            # Wait for key press
            key = cv2.waitKey(0) & 0xFF

            if key == 27:  # ESC
                print("\nExiting interactive mode...")
                break
            elif key == ord('w'):
                eps += 1
                print(f"→ eps = {eps}")
            elif key == ord('s'):
                eps = max(0.5, eps - 1)
                print(f"→ eps = {eps}")
            elif key == ord('e'):
                eps += 0.5
                print(f"→ eps = {eps}")
            elif key == ord('d'):
                eps = max(0.5, eps - 0.5)
                print(f"→ eps = {eps}")
            elif key == ord('a'):
                min_samples += 10
                print(f"→ min_samples = {min_samples}")
            elif key == ord('z'):
                min_samples = max(1, min_samples - 10)
                print(f"→ min_samples = {min_samples}")
            elif key == ord('q'):
                min_samples += 5
                print(f"→ min_samples = {min_samples}")
            elif key == ord('x'):
                min_samples = max(1, min_samples - 5)
                print(f"→ min_samples = {min_samples}")
            elif key == ord('r'):
                eps = float(initial_eps)
                min_samples = int(initial_min_samples)
                print(f"→ Reset to eps={eps}, min_samples={min_samples}")
            elif key == ord('p'):
                print(f"\n{'='*40}")
                print(f"Current Parameters:")
                print(f"  eps = {eps}")
                print(f"  min_samples = {min_samples}")
                print(f"{'='*40}")
            elif key == ord('t'):
                print("→ Re-testing with current parameters...")
            elif key == ord(' '):  # SPACE
                save_path = f"interactive_results/result_{save_counter:03d}_eps{eps}_min{min_samples}.jpg"
                os.makedirs("interactive_results", exist_ok=True)
                cv2.imwrite(save_path, results['result_image'])
                print(f"→ Saved: {save_path}")
                save_counter += 1

        cv2.destroyAllWindows()

    def live_camera_mode(self, initial_eps=9, initial_min_samples=70,
                         delta_t=10000, n_events=100000, fps=10):
        """
        Real-time interactive mode with live event camera

        Args:
            initial_eps: Starting epsilon value
            initial_min_samples: Starting min_samples value
            delta_t: Event accumulation window (microseconds)
            n_events: Number of events per batch
            fps: Target frames per second for processing
        """
        eps = float(initial_eps)
        min_samples = int(initial_min_samples)

        # Start live camera reader
        camera = LiveEventCameraReader(
            width=self.width,
            height=self.height,
            delta_t=delta_t,
            n_events=n_events
        )
        camera.start()

        # Wait for camera to initialize
        time.sleep(1.0)

        print("\n" + "="*60)
        print("Live Event Camera - Interactive DBSCAN Tuning")
        print("="*60)
        print("\nKeyboard Controls:")
        print("  w/s     - Increase/decrease eps by 1")
        print("  e/d     - Increase/decrease eps by 0.5")
        print("  a/z     - Increase/decrease min_samples by 10")
        print("  q/x     - Increase/decrease min_samples by 5")
        print("  r       - Reset to initial values")
        print("  p       - Print current parameters")
        print("  SPACE   - Save current result")
        print("  ESC     - Exit")
        print("="*60)
        print(f"Processing at {fps} FPS")
        print("")

        save_counter = 1
        frame_interval = 1.0 / fps
        last_process_time = 0
        results = None

        try:
            while True:
                current_time = time.time()

                # Process at target FPS
                if current_time - last_process_time >= frame_interval:
                    event_data = camera.get_latest_events()

                    if event_data is not None and len(event_data) > 0:
                        results = self.run_clustering(
                            event_data, eps, min_samples,
                            show_visualization=True
                        )
                        last_process_time = current_time

                # Check for key press (non-blocking with 1ms timeout)
                key = cv2.waitKey(1) & 0xFF

                if key == 27:  # ESC
                    print("\nExiting live camera mode...")
                    break
                elif key == ord('w'):
                    eps += 1
                    print(f"→ eps = {eps}")
                elif key == ord('s'):
                    eps = max(0.5, eps - 1)
                    print(f"→ eps = {eps}")
                elif key == ord('e'):
                    eps += 0.5
                    print(f"→ eps = {eps}")
                elif key == ord('d'):
                    eps = max(0.5, eps - 0.5)
                    print(f"→ eps = {eps}")
                elif key == ord('a'):
                    min_samples += 10
                    print(f"→ min_samples = {min_samples}")
                elif key == ord('z'):
                    min_samples = max(1, min_samples - 10)
                    print(f"→ min_samples = {min_samples}")
                elif key == ord('q'):
                    min_samples += 5
                    print(f"→ min_samples = {min_samples}")
                elif key == ord('x'):
                    min_samples = max(1, min_samples - 5)
                    print(f"→ min_samples = {min_samples}")
                elif key == ord('r'):
                    eps = float(initial_eps)
                    min_samples = int(initial_min_samples)
                    print(f"→ Reset to eps={eps}, min_samples={min_samples}")
                elif key == ord('p'):
                    print(f"\n{'='*40}")
                    print(f"Current Parameters:")
                    print(f"  eps = {eps}")
                    print(f"  min_samples = {min_samples}")
                    print(f"{'='*40}")
                elif key == ord(' ') and results:  # SPACE
                    save_path = f"live_results/result_{save_counter:03d}_eps{eps}_min{min_samples}.jpg"
                    os.makedirs("live_results", exist_ok=True)
                    cv2.imwrite(save_path, results['result_image'])
                    print(f"→ Saved: {save_path}")
                    save_counter += 1

        finally:
            camera.stop()
            cv2.destroyAllWindows()

    def batch_mode(self, event_data, eps_values, min_samples_values, output_dir):
        """
        Batch test multiple parameter combinations

        Args:
            event_data: Event data array
            eps_values: List of eps values to test
            min_samples_values: List of min_samples values to test
            output_dir: Directory to save results
        """
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Batch Mode: Testing {len(eps_values) * len(min_samples_values)} combinations")
        print(f"{'='*60}")

        all_results = []

        for eps in eps_values:
            for min_samples in min_samples_values:
                save_path = f"{output_dir}/eps{eps}_min{min_samples}.jpg"

                results = self.run_clustering(
                    event_data, eps, min_samples,
                    save_path=save_path,
                    show_visualization=False
                )

                all_results.append({
                    'eps': eps,
                    'min_samples': min_samples,
                    'n_clusters': results['n_clusters'],
                    'signal_points': results['signal_points'],
                    'n_noise': results['n_noise'],
                    'signal_ratio': results['signal_ratio']
                })

        # Print summary
        print(f"\n{'='*60}")
        print("Batch Results Summary")
        print(f"{'='*60}")
        print(f"{'eps':<8} {'min_samples':<12} {'clusters':<10} {'signal%':<10}")
        print("-"*60)

        for r in all_results:
            print(f"{r['eps']:<8} {r['min_samples']:<12} {r['n_clusters']:<10} {r['signal_ratio']:>6.1f}%")

        print(f"\nAll results saved to: {output_dir}")

    def _update_playback_display(self, events, eps, min_samples, current_frame,
                                total_frames, chunk_size_us, playing):
        """
        Update the playback display with current frame clustering results
        
        Args:
            events: Event data for current frame
            eps: Current DBSCAN epsilon
            min_samples: Current DBSCAN min_samples
            current_frame: Current frame number
            total_frames: Total number of frames
            chunk_size_us: Time window per frame
            playing: Whether auto-play is active
        """
        # Run clustering on current events (without showing separate window)
        results = self.run_clustering(
            events, eps, min_samples,
            show_visualization=False,  # Don't show the separate window
            use_direct=True  # Use direct clustering
        )
        
        # Add frame info overlay
        frame_info = [
            f"Frame: {current_frame + 1}/{total_frames} ({current_frame * chunk_size_us}us)",
            f"Events: {len(events)} | eps={eps} | min_samples={min_samples}",
            f"Clusters: {results['n_clusters']} | Signal: {results['signal_ratio']:.1f}%"
        ]
        if playing:
            frame_info[0] += " [PLAYING]"
        
        # Overlay frame info on image
        overlay_img = results['result_image'].copy()
        y_pos = 30
        for info_line in frame_info:
            cv2.putText(overlay_img, info_line, (10, y_pos),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            y_pos += 30
        
        # Apply rotation and scaling for display
        display_img = overlay_img.copy()
        if self.rotation == 90:
            display_img = cv2.rotate(display_img, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotation == 180:
            display_img = cv2.rotate(display_img, cv2.ROTATE_180)
        elif self.rotation == 270:
            display_img = cv2.rotate(display_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # Calculate display size after rotation
        if self.rotation in [90, 270]:
            display_width = int(self.height * self.display_scale)
            display_height = int(self.width * self.display_scale)
        else:
            display_width = int(self.width * self.display_scale)
            display_height = int(self.height * self.display_scale)
        
        display_img = cv2.resize(display_img, (display_width, display_height))
        cv2.imshow('Frame Playback - DBSCAN Results', display_img)

    def frame_playback_mode(self, reader, initial_eps=12, initial_min_samples=70, 
                           chunk_size_us=2000, auto_fps=10):
        """
        Frame-by-frame playback mode for .raw files
        
        Args:
            reader: RawFileReader instance
            initial_eps: Initial DBSCAN epsilon
            initial_min_samples: Initial DBSCAN min_samples  
            chunk_size_us: Time window per frame (microseconds)
            auto_fps: FPS for auto-play mode
        """
        if len(reader.all_events) == 0:
            print("No events in file!")
            return
            
        # Calculate total frames
        time_span = abs(reader.all_events['t'][-1] - reader.all_events['t'][0])
        total_frames = int(time_span // chunk_size_us) + 1
        
        current_frame = 0
        eps = float(initial_eps)
        min_samples = int(initial_min_samples)
        playing = False
        frame_interval = 1.0 / auto_fps
        last_frame_time = 0
        
        print(f"\n{'='*60}")
        print("Frame-by-Frame Playback Mode")
        print(f"{'='*60}")
        print(f"Total frames: {total_frames}")
        print(f"Frame duration: {chunk_size_us}us")
        print(f"Auto-play FPS: {auto_fps}")
        print("\nKeyboard Controls:")
        print("  SPACE      - Play/Pause auto-playback")
        print("  →(Right)   - Next frame")
        print("  ←(Left)    - Previous frame") 
        print("  ↑(Up)      - Fast forward (+10 frames)")
        print("  ↓(Down)    - Fast backward (-10 frames)")
        print("  HOME       - Go to first frame")
        print("  END        - Go to last frame")
        print("  w/s        - Increase/decrease eps by 1")
        print("  e/d        - Increase/decrease eps by 0.5")
        print("  a/z        - Increase/decrease min_samples by 10")
        print("  r          - Reset parameters")
        print("  p          - Print current parameters")
        print("  ESC        - Exit")
        print(f"{'='*60}")
        print()
        
        try:
            while True:
                current_time = time.time()
                
                # Auto-play mode
                if playing and (current_time - last_frame_time >= frame_interval):
                    current_frame += 1
                    if current_frame >= total_frames:
                        current_frame = total_frames - 1
                        playing = False
                        print("→ Reached end, auto-play stopped")
                    last_frame_time = current_time
                
                # Get current frame data
                events = reader.get_chunk(current_frame, chunk_size_us)
                
                if len(events) > 0:
                    # Update display with current frame
                    self._update_playback_display(events, eps, min_samples, current_frame, 
                                                total_frames, chunk_size_us, playing)
                else:
                    print(f"No events in frame {current_frame}")
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27:  # ESC
                    print("\nExiting playback mode...")
                    break
                elif key == ord(' '):  # SPACE - Play/Pause
                    playing = not playing
                    status = "Playing" if playing else "Paused"
                    print(f"→ {status} (Frame: {current_frame + 1}/{total_frames})")
                elif key == 83:  # Right arrow - Next frame
                    current_frame = min(current_frame + 1, total_frames - 1)
                    playing = False
                    print(f"→ Next frame: {current_frame + 1}/{total_frames}")
                elif key == 81:  # Left arrow - Previous frame
                    current_frame = max(current_frame - 1, 0)
                    playing = False
                    print(f"→ Previous frame: {current_frame + 1}/{total_frames}")
                elif key == 82:  # Up arrow - Fast forward
                    current_frame = min(current_frame + 10, total_frames - 1)
                    playing = False
                    print(f"→ Fast forward: {current_frame + 1}/{total_frames}")
                elif key == 84:  # Down arrow - Fast backward
                    current_frame = max(current_frame - 10, 0)
                    playing = False
                    print(f"→ Fast backward: {current_frame + 1}/{total_frames}")
                elif key == 80:  # HOME key - First frame
                    current_frame = 0
                    playing = False
                    print(f"→ First frame: {current_frame + 1}/{total_frames}")
                elif key == 87:  # END key - Last frame
                    current_frame = total_frames - 1
                    playing = False
                    print(f"→ Last frame: {current_frame + 1}/{total_frames}")
                elif key == ord('w'):  # Increase eps
                    eps += 1
                    print(f"→ eps = {eps}")
                elif key == ord('s'):  # Decrease eps
                    eps = max(0.5, eps - 1)
                    print(f"→ eps = {eps}")
                elif key == ord('e'):  # Increase eps by 0.5
                    eps += 0.5
                    print(f"→ eps = {eps}")
                elif key == ord('d'):  # Decrease eps by 0.5
                    eps = max(0.5, eps - 0.5)
                    print(f"→ eps = {eps}")
                elif key == ord('a'):  # Increase min_samples
                    min_samples += 10
                    print(f"→ min_samples = {min_samples}")
                elif key == ord('z'):  # Decrease min_samples
                    min_samples = max(1, min_samples - 10)
                    print(f"→ min_samples = {min_samples}")
                elif key == ord('r'):  # Reset parameters
                    eps = float(initial_eps)
                    min_samples = int(initial_min_samples)
                    print(f"→ Reset: eps={eps}, min_samples={min_samples}")
                elif key == ord('p'):  # Print parameters
                    print(f"\nCurrent State:")
                    print(f"  Frame: {current_frame + 1}/{total_frames}")
                    print(f"  Parameters: eps={eps}, min_samples={min_samples}")
                    print(f"  Status: {'Playing' if playing else 'Paused'}")
                
        finally:
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description='DBSCAN Parameter Tuner - Optimize clustering parameters for event data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using configuration files:

  # Load config for specific task (normal/dark/clutter/dark_clutter)
  python dbscan_parameter_tuner.py --task normal
  python dbscan_parameter_tuner.py --task dark --no-live data.npy

  # Load default config (config/config_l.yaml)
  python dbscan_parameter_tuner.py --config

  # Config + override specific parameters
  python dbscan_parameter_tuner.py --task clutter --eps 20 --threshold 50

  # Without configuration files (manual parameters):

  # Default: Live camera + interactive mode (uses hardcoded defaults)
  python dbscan_parameter_tuner.py

  # Live camera with custom parameters
  python dbscan_parameter_tuner.py --eps 12 --min_samples 80 --fps 5

  # Use .npy file instead of live camera
  python dbscan_parameter_tuner.py data.npy --no-live

  # Use .raw file (auto-detected)
  python dbscan_parameter_tuner.py event_data.raw --no-live --mode interactive

  # File mode with task config
  python dbscan_parameter_tuner.py data.npy --no-live --task normal --mode single

  # Batch mode with config defaults
  python dbscan_parameter_tuner.py event.raw --no-live --task dark --mode batch

  # Custom parameters override config
  python dbscan_parameter_tuner.py event.raw --no-live --task clutter --eps 25 --threshold 45
        """
    )

    # Configuration
    parser.add_argument('--task', type=str, default='clutter',
                       choices=['normal', 'dark', 'clutter', 'dark_clutter'],
                       help='Task/environment type to load config for (normal, dark, clutter, dark_clutter)')
    parser.add_argument('--config', action='store_true',
                       help='Load parameters from config file (can be combined with explicit parameters)')

    # Data source
    parser.add_argument('npy_file', type=str, nargs='?', default=None,
                       help='Path to .npy or .raw event data file (optional, defaults to live camera)')
    parser.add_argument('--live', action='store_true', default=False,
                       help='Use live event camera (default: True, use --no-live for file mode)')
    parser.add_argument('--no-live', action='store_false', dest='live',
                       help='Disable live camera and use .npy or .raw file instead')
    parser.add_argument('--use-raw', action='store_true', default=True,
                       help='Input is .raw file instead of .npy (auto-detected if file ends with .raw)')
    parser.add_argument('--delta_t_read', type=int, default=None,
                       help='Delta_t for reading .raw files (us, default from config or 2000)')

    # Mode selection
    parser.add_argument('--mode', type=str, default='interactive',
                       choices=['single', 'interactive', 'batch', 'playback'],
                       help='Operation mode (default: interactive)')
    parser.add_argument('--direct', action='store_true', default=True,
                       help='Use direct event clustering (default: True, like event_tracking_3D_direct)')
    parser.add_argument('--no-direct', action='store_false', dest='direct',
                       help='Use image-based clustering (like event_tracking_3D)')

    # DBSCAN parameters
    parser.add_argument('--eps', type=float, default=None,
                       help='DBSCAN epsilon parameter (default from config)')
    parser.add_argument('--min_samples', type=int, default=None,
                       help='DBSCAN min_samples parameter (default from config)')

    # Image processing parameters
    parser.add_argument('--width', type=int, default=1080,
                       help='Image width (default: 1080)')
    parser.add_argument('--height', type=int, default=1440,
                       help='Image height (default: 1440)')
    parser.add_argument('--accumulation_time', type=int, default=None,
                       help='Event accumulation time in microseconds (default from config)')
    parser.add_argument('--threshold', type=int, default=None,
                       help='Binary threshold value 0-255 (default from config)')
    parser.add_argument('--event_color', type=int, nargs=3, default=None,
                       metavar=('R', 'G', 'B'),
                       help='Event color in RGB (default from config)')
    parser.add_argument('--display_scale', type=float, default=None,
                       help='Display scale factor 0.0-1.0 (default from config)')
    parser.add_argument('--rotation', type=int, default=None, choices=[0, 90, 180, 270],
                       help='Image rotation in degrees (default from config)')

    # Batch mode parameters
    parser.add_argument('--eps_range', type=float, nargs=3,
                       metavar=('START', 'STOP', 'STEP'),
                       help='Eps range for batch mode: start stop step')
    parser.add_argument('--min_samples_range', type=int, nargs=3,
                       metavar=('START', 'STOP', 'STEP'),
                       help='Min_samples range for batch mode: start stop step')

    # Live camera parameters
    parser.add_argument('--delta_t', type=int, default=None,
                       help='Event accumulation window for live camera (us, default from config)')
    parser.add_argument('--n_events', type=int, default=None,
                       help='Number of events per batch for live camera (default from config)')
    parser.add_argument('--fps', type=int, default=None,
                       help='Target FPS for live camera processing (default from config)')

    # Output parameters
    parser.add_argument('--output_dir', type=str, default='tuning_results',
                       help='Output directory for results (default: tuning_results)')
    parser.add_argument('--save', action='store_true',
                       help='Save result in single mode')
    parser.add_argument('--no_display', action='store_true',
                       help='Do not display results (auto-save and exit)')

    args = parser.parse_args()

    # Load configuration (always load based on --task, default is 'clutter')
    load_config_for_task(args.task)
    config_params = get_params_from_config()

    # Helper function to get value with priority: cmd_args > config > default
    def get_param(arg_value, config_key, default=None):
        if arg_value is not None:
            return arg_value
        return config_params.get(config_key, default)

    # Apply config defaults, allow command-line overrides
    eps = get_param(args.eps, 'eps')
    min_samples = get_param(args.min_samples, 'min_samples')
    accumulation_time = get_param(args.accumulation_time, 'accumulation_time')
    threshold = get_param(args.threshold, 'threshold')
    event_color = get_param(args.event_color, 'event_color')
    display_scale = get_param(args.display_scale, 'display_scale')
    rotation = get_param(args.rotation, 'rotation')
    delta_t_read = get_param(args.delta_t_read, 'delta_t_read')
    delta_t = get_param(args.delta_t, 'delta_t')
    n_events = get_param(args.n_events, 'n_events')
    fps = get_param(args.fps, 'fps')
    width = args.width if args.width != 1080 else config_params.get('width', 1080)
    height = args.height if args.height != 1440 else config_params.get('height', 1440)

    # Get default data file from config
    data_file = args.npy_file if args.npy_file else config_params.get('default_data_file')

    # Print configuration
    print(f"\nConfiguration:")
    print(f"  Task: {args.task}")
    print(f"  DBSCAN: eps={eps}, min_samples={min_samples}")
    print(f"  Image: accumulation_time={accumulation_time}us, threshold={threshold}")
    print(f"  Display: scale={display_scale}, rotation={rotation}")
    print(f"  Dimensions: {width}x{height}")
    if data_file:
        print(f"  Data file: {data_file}")
    print()

    # Initialize tuner
    tuner = DBSCANParameterTuner(
        width=width,
        height=height,
        accumulation_time_us=accumulation_time,
        binary_threshold=threshold,
        event_color=tuple(event_color),
        display_scale=display_scale,
        rotation=rotation
    )

    # Default to file mode when data_file is available
    if not args.live and data_file and os.path.exists(data_file) and data_file.endswith('.raw'):
        args.mode = 'playback'  # Set playback mode as default for raw files

    # Live camera mode
    if args.live:
        print("Using live event camera")
        tuner.live_camera_mode(
            initial_eps=eps
            initial_min_samples=min_samples,
            delta_t=delta_
            n_events=n_events,
            fps=fps
        )
        return 0

    # File-based modes
    if not data_file:
        print("Error: No data file specified and no default file in config")
        return 1

    if not os.path.exists(data_file):
        print(f"Error: File not found: {data_file}")
        return 1

    # Load event data - auto-detect .raw or .npy
    reader = None  # Initialize reader variable
    if data_file.endswith('.raw'):
        print(f"Loading .raw file: {data_file}")
        reader = RawFileReader(data_file, width=width, height=height, delta_t=delta_t_read)
        event_data = reader.get_events()
    else:
        print(f"Loading .npy file: {data_file}")
        event_data = tuner.load_events(data_file)
        # For playback mode, create a compatible reader from npy data
        if args.mode == 'playback':
            reader = NPYFileReader(event_data)

    if len(event_data) == 0:
        print("Error: No events in file")
        return 1

    # Print clustering mode
    if args.direct:
        print("Clustering mode: DIRECT (event coordinates)")
    else:
        print("Clustering mode: IMAGE-BASED (converted to image first)")
    print()

    # Run selected mode
    if args.mode == 'interactive':
        tuner.interactive_mode(event_data, eps, min_samples)

    elif args.mode == 'batch':
        # Set default ranges if not specified
        if args.eps_range:
            eps_values = np.arange(args.eps_range[0], args.eps_range[1], args.eps_range[2])
        else:
            # Use config defaults
            batch_eps_range = config_params['batch_eps_range']
            eps_values = np.arange(batch_eps_range[0], batch_eps_range[1], batch_eps_range[2])

        if args.min_samples_range:
            min_samples_values = np.arange(
                args.min_samples_range[0],
                args.min_samples_range[1],
                args.min_samples_range[2]
            )
        else:
            # Use config defaults
            batch_min_samples_range = config_params['batch_min_samples_range']
            min_samples_values = np.arange(
                batch_min_samples_range[0],
                batch_min_samples_range[1],
                batch_min_samples_range[2]
            )

        tuner.batch_mode(event_data, eps_values, min_samples_values, args.output_dir)

    elif args.mode == 'playback':
        # Frame-by-frame playback mode (supports both .raw and .npy files)
        if not data_file.endswith(('.raw', '.npy')):
            print("Error: Playback mode supports .raw and .npy files only")
            return 1
        
        print("Starting frame-by-frame playback mode...")
        tuner.frame_playback_mode(
            reader=reader,  # Use the already created reader
            initial_eps=eps,
            initial_min_samples=min_samples,
            chunk_size_us=delta_t_read,
            auto_fps=fps
        )

    else:  # single mode
        save_path = None
        if args.save or args.no_display:
            save_path = f"{args.output_dir}/result_eps{eps}_min{min_samples}.jpg"

        show_viz = not args.no_display

        results = tuner.run_clustering(
            event_data, eps, min_samples,
            save_path=save_path,
            show_visualization=show_viz,
            use_direct=args.direct
        )

        if show_viz:
            print("\nPress any key to exit...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("\nDone! (no display mode)")

    return 0


if __name__ == '__main__':
    sys.exit(main())
