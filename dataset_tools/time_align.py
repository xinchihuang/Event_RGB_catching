import os
import cv2
import numpy as np
import tempfile
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette
from metavision_core.event_io import EventsIterator
from metavision_core.event_io import DatWriter
import struct

def _is_dat_event_file(file_path):
    """Detect DAT-style event file by header signature."""
    try:
        with open(file_path, "rb") as f:
            head = f.read(256)
        return b"Data file containing" in head
    except OSError:
        return False

def _build_events_iterator(input_path, **kwargs):
    """Return EventsIterator for DAT files even if extension is not .dat."""
    if not _is_dat_event_file(input_path):
        return EventsIterator(input_path=input_path, **kwargs), None

    if input_path.endswith(".dat"):
        return EventsIterator(input_path=input_path, **kwargs), None

    tmp_dir = tempfile.mkdtemp(prefix="event_dat_")
    link_path = os.path.join(tmp_dir, os.path.basename(input_path) + ".dat")
    try:
        os.symlink(input_path, link_path)
        iterator = EventsIterator(input_path=link_path, **kwargs)
    except Exception as e:
        print(f"DAT symlink failed: {e}")
        try:
            if os.path.exists(link_path):
                os.unlink(link_path)
            os.rmdir(tmp_dir)
        except OSError:
            pass
        return None, None

    def _cleanup():
        try:
            os.unlink(link_path)
            os.rmdir(tmp_dir)
        except OSError:
            pass

    return iterator, _cleanup

def load_dat_events(file_path, delta_t=10000, chunk_size=200000):
    """Load events from DAT-style files regardless of extension (delta_t slicing)."""
    try:
        from metavision_core.event_io import dat_tools

        with open(file_path, "rb") as f:
            start, ev_type, ev_size, size = dat_tools.parse_header(f)
            if ev_type not in dat_tools.EV_TYPES:
                print(f"Unsupported DAT event type: {ev_type}")
                return None, None

            raw_dtype = np.dtype(dat_tools.EV_TYPES[ev_type])
            decode_dtype = np.dtype(dat_tools.DECODE_DTYPES[ev_type])

            f.seek(0, os.SEEK_END)
            end = f.tell()
            total_events = (end - start) // ev_size
            f.seek(start, os.SEEK_SET)

            if total_events == 0:
                print("No events in DAT file")
                return None, None

            event_chunks = []
            remaining = total_events
            pending = np.empty((0,), dtype=decode_dtype)
            current_time = 0

            while remaining > 0:
                count = min(chunk_size, remaining)
                buf = np.empty((count,), dtype=decode_dtype)
                dat_tools.stream_events(f, buf, raw_dtype, count)
                remaining -= count

                if pending.size > 0:
                    events = np.concatenate((pending, buf))
                else:
                    events = buf

                empty_chunk = np.empty((0,), dtype=decode_dtype)
                while events.size > 0:
                    expected_time = current_time + delta_t
                    if events["t"][0] >= expected_time:
                        event_chunks.append(empty_chunk)
                        current_time = expected_time
                        continue

                    if events["t"][-1] < expected_time:
                        break

                    idx = np.searchsorted(events["t"], expected_time, side="left")
                    event_chunks.append(events[:idx].copy())
                    events = events[idx:]
                    current_time = expected_time

                pending = events.copy() if events.size > 0 else empty_chunk

            if pending.size > 0:
                event_chunks.append(pending.copy())

        size_tuple = None
        if size and size[0] is not None and size[1] is not None:
            size_tuple = (int(size[0]), int(size[1]))
        print(f"Loaded {total_events} events in {len(event_chunks)} chunks from DAT file")
        return event_chunks, size_tuple
    except Exception as e:
        print(f"Error loading DAT events: {e}")
        return None, None

def save_event_buffer_to_raw(event_buffer, output_path, event_size=None, output_is_dat=True):
    """Save event buffer using DatWriter when output format is DAT-like."""
    try:
        from metavision_core.event_io import DatWriter

        if not output_is_dat:
            print("RAW output requested but no RAW writer is available; skipping save")
            return False
        
        # Concatenate all event packets
        all_events = np.concatenate(event_buffer) if len(event_buffer) > 0 else np.array([])
        
        if len(all_events) == 0:
            print("No events to save")
            return False
        
        # Use DatWriter to create compatible DAT file (even if extension is .raw)
        if event_size and event_size[0] and event_size[1]:
            height, width = int(event_size[0]), int(event_size[1])
        else:
            height, width = 1280, 720
        with DatWriter(output_path, height=height, width=width) as writer:
            writer.write(all_events)
        
        print(f"Saved {len(all_events)} events to {output_path}")
        return True
    except Exception as e:
        print(f"Error saving event buffer: {e}")
        return False

def save_rgb_buffer_to_mp4(rgb_buffer, output_path, fps=30):
    """Save RGB buffer to .mp4 file"""
    try:
        if len(rgb_buffer) == 0:
            print("No RGB frames to save")
            return False
            
        h, w = rgb_buffer[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        
        for frame in rgb_buffer:
            writer.write(frame)
        
        writer.release()
        print(f"Saved {len(rgb_buffer)} frames to {output_path}")
        return True
    except Exception as e:
        print(f"Error saving RGB buffer: {e}")
        return False

def load_raw_binary_events(file_path):
    """Load events from raw binary data when header is corrupted"""
    try:
        with open(file_path, 'rb') as f:
            # Skip the header by finding the end marker
            content = f.read()
            
            # Find the end of header
            header_end = content.find(b'% end\n')
            if header_end == -1:
                # Try alternative endings
                header_end = content.find(b'% end\r\n')
                if header_end == -1:
                    print("Could not find header end marker")
                    return None
            
            # Skip past the header
            header_end += len(b'% end\n')
            binary_data = content[header_end:]
            
            if len(binary_data) == 0:
                print("No binary event data found")
                return None
            
            # Try to interpret as event data (x, y, p, t format)
            # Each event should be 16 bytes based on collect_data.py EVENT_DTYPE
            event_size = 16
            num_events = len(binary_data) // event_size
            
            if num_events == 0:
                print("Insufficient data for events")
                return None
            
            # Create structured array
            dtype = np.dtype([('x', '<u2'), ('y', '<u2'), ('p', '<i2'), ('t', '<i8')])
            events = np.frombuffer(binary_data[:num_events * event_size], dtype=dtype)
            
            # Split into chunks similar to EventsIterator behavior
            chunk_size = 100000  # 100k events per chunk
            event_chunks = []
            for i in range(0, len(events), chunk_size):
                chunk = events[i:i + chunk_size].copy()
                if len(chunk) > 0:
                    event_chunks.append(chunk)
            
            print(f"Loaded {len(events)} events in {len(event_chunks)} chunks from binary data")
            return event_chunks
            
    except Exception as e:
        print(f"Error loading binary event data: {e}")
        return None

def process_loaded_data(event_buffer, raw_data_dir, index, event_size=None, output_is_dat=True):
    """Process already loaded event data with RGB data"""
    # Load RGB video (same as before)
    vid_path = f"{raw_data_dir}/rgb/{index}.mp4"
    print(f"Loading: {vid_path}")
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        print(f"Cannot open video file: {vid_path}")
        return None

    rgb_buffer = []
    try:
        ok, frame = cap.read()
        while ok:
            if frame is not None:
                rgb_buffer.append(frame.copy())
            ok, frame = cap.read()
    except Exception as e:
        print(f"Error reading RGB frames: {e}")
    finally:
        cap.release()

    # Continue with the rest of the processing (same as main function)
    return continue_time_align_processing(event_buffer, rgb_buffer, index, event_size, output_is_dat)

def continue_time_align_processing(event_buffer, rgb_buffer, index, event_size=None, output_is_dat=True):
    """Continue processing with loaded data"""
    rgb_w, rgb_h = 1080, 1440
    
    event_count = len(event_buffer)
    rgb_count = len(rgb_buffer)
    
    if event_count == 0 or rgb_count == 0:
        print(f"No data found. RGB frames: {rgb_count}, Event packets: {event_count}")
        return None

    print(f"Total RGB frames: {rgb_count}")
    print(f"Total event packets: {event_count}")

    # Time alignment
    ratio = 10
    print(f"Before alignment: RGB={len(rgb_buffer)}, Events={len(event_buffer)}")
    if rgb_count * 10 > event_count:
        rgb_bias = int(rgb_count - event_count / 10)
        if rgb_bias > 0:
            rgb_buffer = rgb_buffer[:-rgb_bias]
        print(f"Removed {rgb_bias} RGB frames from end")
    else:
        event_bias = int(event_count - rgb_count * 10)
        if event_bias > 0:
            event_buffer = event_buffer[event_bias:]
        print(f"Removed {event_bias} event packets from start")
    print(f"After alignment: RGB={len(rgb_buffer)}, Events={len(event_buffer)}")

    # Auto-save processed data
    base_output_dir = "/media/xinchi/data/processed_data/saved_data_c"
    event_dir = f"{base_output_dir}/event"
    rgb_dir = f"{base_output_dir}/rgb"
    os.makedirs(event_dir, exist_ok=True)
    os.makedirs(rgb_dir, exist_ok=True)
    event_output = f"{event_dir}/{index}.raw"
    rgb_output = f"{rgb_dir}/{index}.mp4"
    
    print("Auto-saving processed data...")
    save_event_buffer_to_raw(event_buffer, event_output, event_size=event_size, output_is_dat=output_is_dat)
    save_rgb_buffer_to_mp4(rgb_buffer, rgb_output, fps=30)
    print("Auto-save completed!")

    # Start the viewer
    start_viewer(event_buffer, rgb_buffer, index, event_output, rgb_output, event_size, output_is_dat)
    return True

def start_viewer(event_buffer, rgb_buffer, index, event_output, rgb_output, event_size=None, output_is_dat=True):
    """Start the interactive viewer"""
    rgb_w, rgb_h = 1080, 1440

    if event_size and event_size[0] and event_size[1]:
        event_h, event_w = event_size
    else:
        event_w, event_h = 1280, 720

    # Event frame generator - use actual event camera dimensions (1280x720 after rotation in collect_data.py)
    frame_gen = PeriodicFrameGenerationAlgorithm(
        sensor_width=event_w,
        sensor_height=event_h,
        fps=100, 
        palette=ColorPalette.Dark
    )

    print("Press 'q' to quit, 'p' to pause/resume, 's' to step frame by frame")
    print("Press 'e' to save event buffer, 'r' to save RGB buffer, 'b' to save both")
    print(f"Starting playback with {len(rgb_buffer)} RGB frames and {len(event_buffer)} event packets")
    
    frame_idx = 0
    paused = False
    step_mode = False
    ratio = 10
    
    while frame_idx < len(rgb_buffer):
        try:
            # Get current frames
            rgb_img = rgb_buffer[frame_idx]
            if rgb_img is None:
                frame_idx += 1
                continue

            # Get corresponding event packet
            ev_idx = int(frame_idx * ratio)
            ev_idx = min(max(ev_idx, 0), len(event_buffer) - 1)
            ev_local = event_buffer[ev_idx]

            # Generate event frame with correct dimensions
            event_frame_native = np.zeros((event_h, event_w, 3), dtype=np.uint8)
            
            if ev_local is not None and len(ev_local) > 0:
                # Clip coordinates to prevent buffer overflow (based on actual event sensor dimensions)
                ev_local_clipped = ev_local.copy()
                ev_local_clipped['x'] = np.clip(ev_local_clipped['x'], 0, event_w - 1)
                ev_local_clipped['y'] = np.clip(ev_local_clipped['y'], 0, event_h - 1)
                
                try:
                    frame_gen.generate_frame(ev_local_clipped, event_frame_native)
                except Exception as e:
                    print(f"Frame generation failed: {e}")
            
            # Resize event frame to match RGB frame dimensions for display
            event_frame = cv2.resize(event_frame_native, (rgb_w, rgb_h))

            # Create side-by-side display
            display_img = np.hstack((rgb_img, event_frame))

            # Scale for display
            scale = 0.6
            display_width = int(display_img.shape[1] * scale)
            display_height = int(display_img.shape[0] * scale)
            display_resized = cv2.resize(display_img, (display_width, display_height))

            cv2.imshow(f'Event-RGB Viewer - Record {index}', display_resized)
            
            # Handle key input and frame timing
            if paused or step_mode:
                key = cv2.waitKey(0) & 0xFF
            else:
                key = cv2.waitKey(20) & 0xFF  # 20 FPS playback
            
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
                step_mode = False
            elif key == ord('s'):
                step_mode = not step_mode
                paused = False
            elif key == ord('e'):
                # Save event buffer
                print("Saving event buffer...")
                save_event_buffer_to_raw(event_buffer, event_output, event_size=event_size, output_is_dat=output_is_dat)
            elif key == ord('r'):
                # Save RGB buffer
                print("Saving RGB buffer...")
                save_rgb_buffer_to_mp4(rgb_buffer, rgb_output, fps=30)
            elif key == ord('b'):
                # Save both buffers
                print("Saving both event and RGB buffers...")
                save_event_buffer_to_raw(event_buffer, event_output, event_size=event_size, output_is_dat=output_is_dat)
                save_rgb_buffer_to_mp4(rgb_buffer, rgb_output, fps=30)
            elif key == 27:  # ESC
                break
            
            if not paused:
                frame_idx += 1

        except Exception as e:
            print(f"Error processing frame {frame_idx}: {e}")
            frame_idx += 1
            continue

    cv2.destroyAllWindows()
    print("Viewer closed")

def align_to_rgb(event_data, rgb_width=1080, rgb_height=1440, event_width=720, event_height=1280):
    x_e, y_e = event_data['x'].copy(), event_data['y'].copy()
    rgb_center_x = rgb_width / 2
    rgb_center_y = rgb_height / 2
    event_center_x = event_width / 2
    event_center_y = event_height / 2
    event_pixel = 3.45
    rgb_pixel = 4.86

    x = (x_e - event_center_x) * rgb_pixel / event_pixel + rgb_center_x + 30
    y = (y_e - event_center_y) * rgb_pixel / event_pixel + rgb_center_y + 29
    y = np.clip(y, 0, rgb_height)
    return x.astype(int), y.astype(int)

def time_align(raw_data_dir, index, bias=0):
    rgb_w, rgb_h = 1080, 1440
    
    # Load event data
    ev_path = f"{raw_data_dir}/event/{index}.raw"
    event_buffer = []
    event_size = None
    cleanup = None
    output_is_dat = _is_dat_event_file(ev_path)
    
    try:
        # Try to load with EventsIterator - use parameters compatible with collect_data.py format
        try:
            # First try with the correct sensor dimensions from collect_data.py
            # The raw file was saved with rotated coordinates (720x1280 -> 1280x720)
            event_iterator, cleanup = _build_events_iterator(
                input_path=ev_path,
                mode="delta_t",
                delta_t=10000,  # Increased delta_t for better chunk size
                n_events=1_000_000
            )
            if event_iterator is None:
                raise RuntimeError("EventsIterator setup failed for DAT file")
        except Exception as e1:
            print(f"First attempt failed: {e1}")
            try:
                # Try with just the file path - let EventsIterator auto-detect format
                event_iterator, cleanup = _build_events_iterator(input_path=ev_path)
                if event_iterator is None:
                    raise RuntimeError("EventsIterator setup failed for DAT file")
            except Exception as e2:
                print(f"Second attempt failed: {e2}")
                # Try loading as DAT if header indicates a DAT-style file
                if _is_dat_event_file(ev_path):
                    print("Attempting DAT fallback reader...")
                    event_buffer, event_size = load_dat_events(ev_path, delta_t=10000)
                    if event_buffer is None:
                        print(f"All loading methods failed for {ev_path}")
                        return None
                    print(f"Successfully loaded {len(event_buffer)} event packets from DAT file")
                    return process_loaded_data(event_buffer, raw_data_dir, index, event_size, output_is_dat)

                # Try loading as simple binary data if header is corrupted
                print("Attempting to load raw binary data...")
                event_buffer = load_raw_binary_events(ev_path)
                if event_buffer is None:
                    print(f"All loading methods failed for {ev_path}")
                    return None
                print(f"Successfully loaded {len(event_buffer)} event packets from binary data")
                return process_loaded_data(event_buffer, raw_data_dir, index, output_is_dat=output_is_dat)

        event_size = event_iterator.get_size()
        for ev in iter(event_iterator):
            if ev is None:
                continue
            if ev.size == 0:
                event_buffer.append(ev.copy())
                continue

            ev_local = np.empty(ev.shape, dtype=ev.dtype)
            np.copyto(ev_local, ev)

            event_buffer.append(ev_local)
    except Exception as e:
        print(f"Error loading event file {ev_path}: {e}")
        print(f"Skipping index {index} due to corrupt RAW file")
        return None
    finally:
        if cleanup:
            cleanup()

    # If we got here, EventsIterator worked, so continue with normal processing
    return process_loaded_data(event_buffer, raw_data_dir, index, event_size, output_is_dat)


def main():
    # raw_data_dir = "/media/xinchi/data/saved_data/saved_data_c"
    raw_data_dir = '/media/xinchi/data/processed_data/saved_data_c'
    
    # Try multiple indices to find a working file
    for index in range(0, 10):
        print(f"Trying index {index}...")
        result = time_align(raw_data_dir, index)
        if result is not None:
            break

if __name__ == "__main__":
    main()
