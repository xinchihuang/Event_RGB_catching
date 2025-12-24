import numpy as np
import time
import os
import sys
from multiprocessing import shared_memory

# Import configuration
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)
from config.config_loader import Config
def align_to_rgb(event_data, rgb_width=None, rgb_height=None, event_width=None, event_height=None):
    # Get dimensions from config if not provided
    if rgb_width is None or rgb_height is None or event_width is None or event_height is None:
        dims = Config.get_image_dimensions()
        rgb_width = dims['rgb_width'] if rgb_width is None else rgb_width
        rgb_height = dims['rgb_height'] if rgb_height is None else rgb_height
        event_width = dims['event_width'] if event_width is None else event_width
        event_height = dims['event_height'] if event_height is None else event_height
    
    # Get alignment parameters from config
    alignment = Config.get_alignment()
    event_pixel = alignment['event_pixel_size']
    rgb_pixel = alignment['rgb_pixel_size']
    offset_x = alignment['offset_x']
    offset_y = alignment['offset_y']
    
    x_e, y_e = event_data['x'].copy(), event_data['y'].copy()
    rgb_center_x = rgb_width / 2
    rgb_center_y = rgb_height / 2
    event_center_x = event_width / 2
    event_center_y = event_height / 2

    x = (x_e - event_center_x) * rgb_pixel / event_pixel + rgb_center_x + offset_x
    y = (y_e - event_center_y) * rgb_pixel / event_pixel + rgb_center_y + offset_y
    y = np.clip(y, 0, rgb_height)
    return x.astype(int), y.astype(int)
def read_from_event(shared_event_data, system_state, num_event, event_data_version, stop_event):
    import time
    import numpy as np
    from multiprocessing import shared_memory
    from metavision_core.event_io import EventsIterator

    # ---- Configs
    data_reading = Config.get_data_reading()
    delta_t       = data_reading['delta_t']
    n_events      = data_reading['n_events']
    send_interval = float(data_reading['send_interval'])
    # ASAP mode (send_interval<=0): publish immediately every loop
    asap_mode = not (send_interval > 0.0)
    hz = (1.0 / send_interval) if not asap_mode else 0.0

    # Pre-fetch alignment/size config to avoid repeated access in loop
    dims = Config.get_image_dimensions()
    alignment = Config.get_alignment()
    y_offset = float(alignment['y_coordinate_offset'])

    # ---- Iterator (live if input_path="")
    event_iterator = iter(EventsIterator(
        input_path="",
        mode="mixed",
        delta_t=delta_t,
        n_events=n_events
    ))

    # ---- Shared memory view
    event_memory, event_dtype, event_shape = shared_event_data
    event_shm = shared_memory.SharedMemory(name=event_memory.name)
    shm_np = np.ndarray(event_shape, dtype=event_dtype, buffer=event_shm.buf)

    latest_events = None
    have_latest   = False
    stream_ended  = False
    last_log      = 0.0

    t0 = time.perf_counter()
    next_deadline = (t0 + send_interval) if not asap_mode else None

    # -------- Method A: Time-budgeted drain until catching up to real-time or budget exhausted --------
    def drain_events_time_budget():
        """
        Repeatedly call next(iterator) within a limited time budget to consume to the 'latest batch'.
        Only return the latest batch (for publishing) to clear backlog and avoid latency accumulation.
        """
        nonlocal latest_events, have_latest, stream_ended

        # Time budget: ASAP mode gets small budget (100ms), fixed-rate mode uses 75% of period
        if asap_mode:
            budget_ms = 100.0
        else:
            budget_ms = max(0.5, 0.75 * send_interval * 1000.0)

        start = time.perf_counter()
        got_any = False

        while True:
            # Stop when time is up (leave time for publishing/writing shared memory)
            now = time.perf_counter()
            if (now - start) * 1000.0 >= budget_ms:
                break

            try:
                ev = next(event_iterator)   # Get a batch of events
            except StopIteration:
                stream_ended = True
                break
            except Exception as e:
                # Acquisition error: exit this round, wait for next loop to retry
                print(f"Event acquisition error: {e}")
                break

            # Coordinate transformation (process in float domain, finally clip in align_to_rgb)
            try:
                x = ev['x'].astype(np.float32, copy=False)
                y = ev['y'].astype(np.float32, copy=False)

                # User-defined rotation/translation
                x_new = y
                y_new = -x + y_offset

                xa, ya = align_to_rgb({'x': x_new, 'y': y_new},
                                      rgb_width=dims['rgb_width'],
                                      rgb_height=dims['rgb_height'],
                                      event_width=dims['event_width'],
                                      event_height=dims['event_height'])

                ev2 = ev.copy()
                ev2['x'] = xa
                ev2['y'] = ya

                latest_events = ev2   # Only keep 'last batch' = latest
                have_latest   = True
                got_any       = True
            except Exception as e:
                print(f"Event processing error: {e}")
                have_latest = False
                # Don't exit immediately on error, continue trying to drain next batch
                continue

        return got_any

    try:
        while not stop_event.is_set() and not stream_ended:
            now = time.perf_counter()

            # ---------- Key: First drain internal buffer to 'latest' ----------
            got_any = drain_events_time_budget()
            if stream_ended:
                break

            # Read running state (Value('c') may be bytes)
            is_running = getattr(system_state, "value", None)
            if isinstance(is_running, (bytes, bytearray)):
                is_running = is_running.decode('ascii', errors='ignore') if is_running else ''
            elif not isinstance(is_running, str):
                is_running = str(is_running) if is_running is not None else ''

            # Publish decision (ASAP mode: publish every round; fixed-rate: only when time is reached)
            should_publish = asap_mode or (now >= next_deadline)

            if should_publish:
                if is_running == 't' and have_latest and (latest_events is not None):
                    process_start = time.perf_counter()
                    try:
                        n_avail = latest_events.shape[0] if hasattr(latest_events, "shape") else 0
                        n_cap   = int(event_shape[0])
                        n_copy  = min(n_avail, n_cap)

                        if n_copy > 0:
                            shm_np[:n_copy] = latest_events[:n_copy]
                            num_event.value = n_copy
                            # Increment version to signal new data is available
                            event_data_version.value += 1
                        else:
                            num_event.value = 0

                        proc_ms = (time.perf_counter() - process_start) * 1000.0

                        # Fixed-rate mode advances next deadline without drift
                        if not asap_mode:
                            missed = max(1, int((now - next_deadline) // send_interval) + 1)
                            next_deadline += missed * send_interval

                        # Throttled logging
                        if now - last_log >= 10.0:
                            print(f"Event: {hz:.0f} Hz target | sent {n_copy} evts | proc {proc_ms:.2f} ms")
                            last_log = now

                        if (not asap_mode) and proc_ms > send_interval * 1000.0:
                            print(f"Warning: processing {proc_ms:.2f} ms > interval {send_interval*1000:.2f} ms")
                    except Exception as e:
                        print(f"Event shared-memory write error: {e}")
                else:
                    # Paused: quickly schedule next publish time (fixed-rate)
                    if not asap_mode:
                        next_deadline = now + send_interval

            # Light sleep to avoid busy-wait
            if asap_mode:
                time.sleep(0.0002)  # ASAP mode gives a very small yield
            else:
                # Sleep to a fraction of next publish point, avoid long busy-wait
                slack = (next_deadline - time.perf_counter()) if next_deadline else 0.0
                if slack > 0:
                    time.sleep(min(slack * 0.25, 0.005))

    finally:
        print("Event Camera: Cleaning up...")
        # Shared memory is cleaned up by main process, don't close here

# def read_from_event(shared_event_data, system_state,num_event, stop_event):
#     from metavision_core.event_io import EventsIterator
#
#     # Get data reading parameters from config
#     data_reading = Config.get_data_reading()
#
#     # Initialize event iterator
#     event_iterator = iter(EventsIterator(input_path="", mode="mixed",
#                                         delta_t=data_reading['delta_t'],
#                                         n_events=data_reading['n_events']))
#
#     # Time control parameters
#     send_interval = data_reading['send_interval']  # Send interval
#     last_send_time = 0
#     latest_events = None
#     events_ready = False
#
#     # Get shared memory information
#     event_memory, event_dtype, event_shape = shared_event_data
#     event_shm = shared_memory.SharedMemory(name=event_memory.name)
#
#     print(f"Event Camera: Starting with {send_interval*1000}ms send interval (500Hz)")
#
#     def get_latest_events():
#         """
#         Acquire and process latest event data
#         """
#         nonlocal latest_events, events_ready
#
#         try:
#             # Get next batch of event data
#             event_data = next(event_iterator)
#
#             # Coordinate transformation processing
#             event_data['x'], event_data['y'] = event_data['y'].copy(), -event_data['x'].copy()
#             alignment = Config.get_alignment()
#             event_data['y'] += alignment['y_coordinate_offset']
#             event_data['x'], event_data['y'] = align_to_rgb(event_data)
#
#             # Update latest events
#             latest_events = event_data.copy()
#             events_ready = True
#
#             return True
#
#         except StopIteration:
#             print("Event stream ended")
#             return False
#         except Exception as e:
#             print(f"Event acquisition error: {e}")
#             return False
#
#     while not stop_event.is_set():
#         current_time = time.perf_counter()
#
#         # Continuously acquire latest event data, keep buffer up to date
#         try:
#             if not get_latest_events():
#                 # If no more data, exit loop
#                 break
#         except Exception as e:
#             print(f"Event processing error: {e}")
#             continue
#
#         # Check if it's time to send
#         if current_time - last_send_time >= send_interval:
#             if system_state.value == 't' and events_ready and latest_events is not None and hasattr(latest_events, 'shape'):
#                 # Record processing start time
#                 process_start = time.perf_counter()
#
#                 try:
#                     # Send latest event data to shared memory
#                     shm = np.ndarray(event_shape, dtype=event_dtype, buffer=event_shm.buf)
#
#                     # Clear shared memory
#                     shm[:] = np.zeros(event_shape, dtype=event_dtype)
#
#                     # Copy latest event data
#                     num_events = 0
#                     if latest_events is not None:
#                         try:
#                             num_events = min(latest_events.shape[0], event_shape[0])
#                             if num_events > 0:
#                                 shm[:num_events] = latest_events[:num_events]
#                         except (AttributeError, IndexError) as e:
#                             print(f"Event data format error: {e}")
#                             num_events = 0
#                         num_event.value = num_events
#
#                     # Calculate processing time
#                     process_time = time.perf_counter() - process_start
#
#                     # Update send time
#                     last_send_time = current_time
#
#                     # If processing time exceeds interval, skip next cycle
#                     if process_time > send_interval:
#                         print(f"Warning: Event processing time {process_time*1000:.2f}ms > interval {send_interval*1000}ms, skipping next cycle")
#                         last_send_time += send_interval  # Skip a cycle
#
#                     # Print statistics (optional, for debugging)
#                     if int(current_time) % 10 == 0:  # Print every 10 seconds
#                         print(f"Event: Send rate: {1/send_interval:.0f}Hz, Events: {num_events}, Process time: {process_time*1000:.2f}ms")
#
#                 except Exception as e:
#                     print(f"Event shared memory write error: {e}")
#             else:
#                 # Even if not sending data, update time to maintain sync
#                 last_send_time = current_time
#         else:
#             shm = np.ndarray(event_shape, dtype=event_dtype, buffer=event_shm.buf)
#             shm[:] = np.zeros(event_shape, dtype=event_dtype)
#         # Brief sleep to avoid excessive CPU usage
#         time.sleep(0.0001)  # 0.1ms sleep
#
#
#     # Clean up resources
#     print("Event Camera: Cleaning up...")
#     try:
#         # Event camera cleanup code can be added here
#         pass
#     except Exception as e:
#         print(f"Event cleanup error: {e}")