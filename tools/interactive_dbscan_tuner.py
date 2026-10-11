#!/usr/bin/env python3
"""
Interactive DBSCAN parameter tuner.
Features:
- Read already rotated events from a .raw or .dat file
- Cluster with DBSCAN
- Visualize clusters on an image
- Adjust parameters via keyboard
- Step through frames
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import os
import sys
import argparse
import time

# Add project path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from config.config_loader import Config, get_config
from data_input.read_from_aligned_buffers import resolve_event_path
from project_paths import stage_path, resolve_path
from data_input.read_from_aligned_buffers import load_processed_event_file


class InteractiveDBSCANTuner:
    """Interactive DBSCAN parameter tuner."""

    def __init__(self, raw_file, display_scale=0.6):
        """
        Args:
            raw_file: Path to an already rotated .dat or legacy .raw event file.
            display_scale: Display scale for the window.
        """
        self.raw_file = raw_file
        self.display_scale = display_scale

        # Configs
        config = get_config()
        self.data_reading = Config.get_data_reading()
        self.tracking = Config.get_tracking()
        self.image_dims = Config.get_image_dimensions()

        # Image size
        self.width = self.image_dims['rgb_width']
        self.height = self.image_dims['rgb_height']

        # DBSCAN params
        self.eps = self.tracking.get('dbscan_eps', 12)
        self.min_samples = self.tracking.get('dbscan_min_samples', 70)

        # Read params
        self.delta_t = self.data_reading.get('delta_t', 2000)
        self.n_events = self.data_reading.get('n_events', 100000)

        # Event buffer
        self.event_chunks = []
        self.current_frame = 0
        self.total_frames = 0

        # Autoplay settings
        self.auto_play = False
        self.fps = 10
        self.last_frame_time = 0

        # Cache for cluster results
        self.cache = {}
        self.last_eps = self.eps
        self.last_min_samples = self.min_samples
        self.last_frame = -1

        print(f"Initialization parameters:")
        print(f"  Image size: {self.width}x{self.height}")
        print(f"  DBSCAN params: eps={self.eps}, min_samples={self.min_samples}")
        print(f"  Read params: delta_t={self.delta_t}us, n_events={self.n_events}")

        self._load_event_data()

    def _load_event_data(self):
        """Load processed event chunks without rotating them."""
        print(f"Loading processed event file: {self.raw_file}")

        # Share the replay reader, including DAT files historically named .raw.
        chunks = load_processed_event_file(
            self.raw_file, dict(delta_t=self.delta_t, n_events=self.n_events)
        )
        self.event_chunks.extend(chunk for chunk in chunks if len(chunk))
        total_events = sum(len(chunk) for chunk in self.event_chunks)

        self.total_frames = len(self.event_chunks)
        print(f"Loading complete: {self.total_frames} frames, total {total_events} events")

        if self.total_frames > 0:
            first_chunk = self.event_chunks[0]
            last_chunk = self.event_chunks[-1]
            if len(first_chunk) > 0 and len(last_chunk) > 0:
                print(f"  Time range: {first_chunk['t'][0]} to {last_chunk['t'][-1]} us")

    def _merge_events_by_position(self, events):
        """
        Merge events by (x, y) position and keep one event per pixel.
        """
        if len(events) == 0:
            return events

        # N x 2 coordinate array
        coords = np.column_stack([events['x'], events['y']])

        # Unique coordinates and representative indices
        unique_coords, unique_indices = np.unique(coords, axis=0, return_index=True)

        # Representative events
        merged_events = events[unique_indices].copy()
        # Ensure coordinates match unique_coords
        merged_events['x'] = unique_coords[:, 0]
        merged_events['y'] = unique_coords[:, 1]

        return merged_events

    def _cluster_events(self, events):
        """
        Run DBSCAN on events (after merging per pixel).

        Args:
            events: Event array.

        Returns:
            tuple: (points, labels, n_clusters, n_noise)
        """
        if len(events) == 0:
            return np.array([]), np.array([]), 0, 0

        # Merge events by pixel first
        merged_events = self._merge_events_by_position(events)

        # Extract coordinates
        points = np.column_stack([merged_events['x'], merged_events['y']])

        # Run DBSCAN
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        labels = dbscan.fit_predict(points)

        # Summarize clustering
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        return points, labels, n_clusters, n_noise

    def _visualize_clusters(self, points, labels):
        """
        Visualize clusters with a single bounding box covering all clusters.

        Args:
            points: Coordinate array (merged).
            labels: Cluster labels.

        Returns:
            result_image: Visualization image.
        """
        # Black background
        result_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        if len(points) == 0 or len(labels) == 0:
            return result_image

        # Split noise vs clustered points
        cluster_mask = labels != -1
        cluster_points = points[cluster_mask]
        cluster_labels = labels[cluster_mask]
        noise_points = points[~cluster_mask]

        # 1) Noise points in red (batched)
        if len(noise_points) > 0:
            noise_coords = noise_points.astype(int)
            # Keep points inside image bounds
            valid_mask = (
                    (noise_coords[:, 0] >= 0) & (noise_coords[:, 0] < self.width) &
                    (noise_coords[:, 1] >= 0) & (noise_coords[:, 1] < self.height)
            )
            valid_noise = noise_coords[valid_mask]
            if len(valid_noise) > 0:
                result_image[valid_noise[:, 1], valid_noise[:, 0]] = [0, 0, 255]

        # 2) Per-cluster colors (no per-cluster bbox)
        unique_labels = np.unique(cluster_labels)
        if len(unique_labels) > 0:
            colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

            for label, color in zip(unique_labels, colors):
                label_mask = cluster_labels == label
                label_points = cluster_points[label_mask]
                if len(label_points) == 0:
                    continue

                # Convert to BGR
                color_bgr = [int(color[2] * 255), int(color[1] * 255), int(color[0] * 255)]

                # Batch set pixel colors
                label_coords = label_points.astype(int)
                valid_mask = (
                        (label_coords[:, 0] >= 0) & (label_coords[:, 0] < self.width) &
                        (label_coords[:, 1] >= 0) & (label_coords[:, 1] < self.height)
                )
                valid_coords = label_coords[valid_mask]
                if len(valid_coords) > 0:
                    result_image[valid_coords[:, 1], valid_coords[:, 0]] = color_bgr

        # 3) One global bounding box for all cluster_points
        if len(cluster_points) > 0:
            x_coords = cluster_points[:, 0]
            y_coords = cluster_points[:, 1]
            x_min, x_max = int(x_coords.min()), int(x_coords.max())
            y_min, y_max = int(y_coords.min()), int(y_coords.max())

            # Green box
            cv2.rectangle(result_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Label in the top-left of the box
            cv2.putText(
                result_image, "ALL", (x_min, max(y_min - 5, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )

        return result_image

    def _update_display(self):
        """Update display with a simple cache."""
        if self.current_frame >= self.total_frames:
            self.current_frame = self.total_frames - 1

        if self.current_frame < 0:
            self.current_frame = 0

        # Current frame events
        events = self.event_chunks[self.current_frame]

        # Recompute if parameters or frame changed
        cache_key = (self.current_frame, self.eps, self.min_samples)
        params_changed = (self.eps != self.last_eps or
                          self.min_samples != self.last_min_samples)

        if cache_key not in self.cache or params_changed:
            # Clear cache if parameters changed
            if params_changed:
                self.cache.clear()
                self.last_eps = self.eps
                self.last_min_samples = self.min_samples

            # Compute and cache clustering
            points, labels, n_clusters, n_noise = self._cluster_events(events)
            self.cache[cache_key] = (points, labels, n_clusters, n_noise)
        else:
            # Fetch from cache
            points, labels, n_clusters, n_noise = self.cache[cache_key]

        self.last_frame = self.current_frame

        # Visualization
        result_image = self._visualize_clusters(points, labels)

        # Summary stats
        total_points = len(points)  # merged points used for clustering
        signal_points = total_points - n_noise
        signal_ratio = (signal_points / total_points * 100) if total_points > 0 else 0

        # Overlay info text
        cache_status = "CACHED" if cache_key in self.cache and not params_changed else "COMPUTED"
        info_text = [
            f"Frame: {self.current_frame + 1}/{self.total_frames} [{cache_status}]",
            f"Events: {len(events)} | unique_pixels: {total_points}",
            f"eps={self.eps} | min_samples={self.min_samples}",
            f"Clusters: {n_clusters} | Signal: {signal_ratio:.1f}% | Noise: {n_noise}",
            f"{'[AUTO]' if self.auto_play else '[MANUAL]'}"
        ]

        y_pos = 30
        for text in info_text:
            cv2.putText(result_image, text, (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            y_pos += 30

        # Scale display
        display_width = int(self.width * self.display_scale)
        display_height = int(self.height * self.display_scale)
        display_image = cv2.resize(result_image, (display_width, display_height))

        cv2.imshow('Interactive DBSCAN Tuner', display_image)

    def run(self):
        """Run the interactive tuner."""
        if self.total_frames == 0:
            print("Error: No event data loaded")
            return

        print("\n" + "=" * 60)
        print("Interactive DBSCAN Parameter Tuner")
        print("=" * 60)
        print("Keyboard Controls:")
        print("  SPACE       - Play/Pause auto-play")
        print("  j / l       - Previous / Next frame")
        print("  u / o       - Fast backward / forward (-/+10 frames)")
        print("  h / E       - First / Last frame")
        print("  e / q       - eps +1 / -1")
        print("  d / a       - min_samples +5 / -5")
        print("  r           - Reset parameters")
        print("  p           - Print current parameters")
        print("  +/-         - Adjust playback speed")
        print("  ESC         - Exit")
        print("=" * 60)
        print()

        try:
            while True:
                current_time = time.time()

                # Auto-play mode
                if self.auto_play and (current_time - self.last_frame_time >= 1.0 / self.fps):
                    self.current_frame += 1
                    if self.current_frame >= self.total_frames:
                        self.current_frame = self.total_frames - 1
                        self.auto_play = False
                        print("-> Reached last frame, auto-play stopped")
                    self.last_frame_time = current_time

                # Update display
                self._update_display()

                # Handle keyboard input
                key = cv2.waitKey(1)
                key_char = key & 0xFF  # low 8 bits for standard keys

                # --------- Exit / playback control ---------
                if key_char == 27:  # ESC
                    print("\nExiting interactive tuner...")
                    break
                elif key_char == ord(' '):  # SPACE - play/pause
                    self.auto_play = not self.auto_play
                    status = "Playing" if self.auto_play else "Paused"
                    print(f"-> {status} (Frame: {self.current_frame + 1}/{self.total_frames})")

                # --------- Frame stepping ---------
                # Next frame
                elif key_char == ord('l'):
                    self.current_frame = min(self.current_frame + 1, self.total_frames - 1)
                    self.auto_play = False
                    print(f"-> Next frame: {self.current_frame + 1}/{self.total_frames}")

                # Previous frame
                elif key_char == ord('j'):
                    self.current_frame = max(self.current_frame - 1, 0)
                    self.auto_play = False
                    print(f"-> Previous frame: {self.current_frame + 1}/{self.total_frames}")

                # Fast forward +10
                elif key_char == ord('o'):
                    self.current_frame = min(self.current_frame + 10, self.total_frames - 1)
                    self.auto_play = False
                    print(f"-> Fast forward (+10): {self.current_frame + 1}/{self.total_frames}")

                # Fast backward -10
                elif key_char == ord('u'):
                    self.current_frame = max(self.current_frame - 10, 0)
                    self.auto_play = False
                    print(f"-> Fast backward (-10): {self.current_frame + 1}/{self.total_frames}")

                # --------- Jump to first/last ---------
                elif key_char == ord('h'):  # first frame
                    self.current_frame = 0
                    self.auto_play = False
                    print(f"-> First frame: {self.current_frame + 1}/{self.total_frames}")
                elif key_char == ord('E'):  # last frame
                    self.current_frame = self.total_frames - 1
                    self.auto_play = False
                    print(f"-> Last frame: {self.current_frame + 1}/{self.total_frames}")

                # --------- DBSCAN parameter tuning ---------
                elif key_char == ord('e'):  # eps +1
                    self.eps += 1
                    print(f"-> eps = {self.eps}")
                elif key_char == ord('q'):  # eps -1
                    self.eps = max(0.5, self.eps - 1)
                    print(f"-> eps = {self.eps}")
                elif key_char == ord('d'):  # min_samples +5
                    self.min_samples += 5
                    print(f"-> min_samples = {self.min_samples}")
                elif key_char == ord('a'):  # min_samples -5
                    self.min_samples = max(1, self.min_samples - 5)
                    print(f"-> min_samples = {self.min_samples}")

                # --------- Other controls ---------
                elif key_char == ord('r'):  # reset parameters
                    self.eps = self.tracking.get('dbscan_eps', 12)
                    self.min_samples = self.tracking.get('dbscan_min_samples', 70)
                    print(f"-> Reset parameters: eps={self.eps}, min_samples={self.min_samples}")
                elif key_char == ord('p'):  # print parameters
                    print(f"\nCurrent State:")
                    print(f"  Frame: {self.current_frame + 1}/{self.total_frames}")
                    print(f"  Parameters: eps={self.eps}, min_samples={self.min_samples}")
                    print(f"  Playback: {'Auto' if self.auto_play else 'Manual'}, FPS={self.fps}")
                elif key_char == ord('+') or key_char == ord('='):  # increase FPS
                    self.fps = min(self.fps + 1, 30)
                    print(f"-> FPS = {self.fps}")
                elif key_char == ord('-'):  # decrease FPS
                    self.fps = max(self.fps - 1, 1)
                    print(f"-> FPS = {self.fps}")

        except KeyboardInterrupt:
            print("\nUser interrupted")
        finally:
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Interactive DBSCAN tuner: read processed events from a .dat or legacy .raw file and adjust parameters live."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default config
  python interactive_dbscan_tuner.py path/to/event.raw

  # Use a specific task overlay
  python interactive_dbscan_tuner.py path/to/event.raw --task clutter

  # Custom display scale
  python interactive_dbscan_tuner.py path/to/event.raw --scale 0.8
        """,
    )

    parser.add_argument('raw_file', type=str, nargs='?',
                        default=None,
                        )
    parser.add_argument('--task', type=str, default='clutter',
                        choices=['normal', 'dark', 'clutter', 'dark_clutter'],
                        help='Task name used to select the overlay (default: clutter)')
    parser.add_argument('--scale', type=float, default=0.6,
                        help='Display scale (default: 0.6)')

    args = parser.parse_args()

    try:
        args.raw_file = str(resolve_path(args.raw_file) if args.raw_file is not None
                            else resolve_event_path(stage_path('replay', args.task), 0))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))

    # Validate input file
    if not os.path.exists(args.raw_file):
        print(f"Error: File not found: {args.raw_file}")
        return 1

    from config.config_loader import initialize_config
    try:
        config = initialize_config(args.task, os.environ.get("TEST_MODE", ""))
    except (OSError, ValueError) as exc:
        parser.error(f"Configuration error: {exc}")
    print(f"Loading config file: {config.config_path} (task: {args.task})")

    # Create and run the tuner
    try:
        tuner = InteractiveDBSCANTuner(args.raw_file, args.scale)
        tuner.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
