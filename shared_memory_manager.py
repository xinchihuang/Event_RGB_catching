"""
Shared Memory Manager Module

Centralized management of all shared memory resources used across processes.
Provides unified creation, access, and cleanup interfaces.
"""

import numpy as np
from multiprocessing import shared_memory
import gc


class SharedMemoryManager:
    """
    Unified manager for all shared memory resources.

    Responsibilities:
    1. Define specifications for all shared memory blocks
    2. Create all shared memory in main process
    3. Provide access interface for subprocesses
    4. Unified cleanup on shutdown
    """

    def __init__(self):
        """Initialize the manager with empty storage."""
        # Storage for shared memory objects
        self.memory_blocks = {}

        # Storage for numpy array references (main process only)
        self.numpy_arrays = {}

        # Storage for dtype and shape information
        self.dtypes = {}
        self.shapes = {}

    def create_all(self):
        """
        Create all shared memory blocks required by the system.

        This should only be called in the main process.
        """
        # Event data: structured array for event camera data
        event_dtype = {
            'names': ['x', 'y', 'p', 't'],
            'formats': ['<u2', '<u2', '<i2', '<i8'],
            'offsets': [0, 2, 4, 8],
            'itemsize': 16
        }
        event_shape = (100000,)
        self._create_single_buffer('event_data', event_dtype, event_shape)

        # FLIR camera data: RGB image
        flir_dtype = np.uint8
        flir_shape = (1440, 1080, 3)
        self._create_single_buffer('flir_data', flir_dtype, flir_shape)

        # ROI data: region of interest coordinates
        ROI_dtype = np.uint16
        ROI_shape = (4,)
        self._create_single_buffer('ROI_data', ROI_dtype, ROI_shape)

        # Ball pixel data (2D): ball observations in pixel space
        ball_pixel_dtype = np.float64
        ball_pixel_shape = (100, 3)
        self._create_single_buffer('ball_pixel', ball_pixel_dtype, ball_pixel_shape)

        # Historical name: rows are (u_px, v_px, size_px, event_time_s), not XYZ.
        ball_pixel_3D_dtype = np.float64
        ball_pixel_3D_shape = (1000, 4)
        self._create_single_buffer('ball_pixel_3D', ball_pixel_3D_dtype, ball_pixel_3D_shape)

        # Impact point: predicted landing point
        impact_point_dtype = np.float32
        impact_point_shape = (3,)
        self._create_single_buffer('impact_point', impact_point_dtype, impact_point_shape)

    def _create_single_buffer(self, name, dtype, shape):
        """
        Create a single shared memory buffer.

        Args:
            name: Identifier for this memory block
            dtype: Numpy dtype for the array
            shape: Shape tuple for the array
        """
        # Calculate required size
        dt = np.dtype(dtype)
        size = dt.itemsize * np.prod(shape)

        # Create shared memory
        shm = shared_memory.SharedMemory(create=True, size=size)

        # Create numpy view and initialize to zero
        np_array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        np_array[:] = 0

        # Store references
        self.memory_blocks[name] = shm
        self.numpy_arrays[name] = np_array
        self.dtypes[name] = dtype
        self.shapes[name] = shape

        print(f"Created shared memory: {name} (size={size} bytes, shape={shape})")

    def get_shared_data_tuple(self, name):
        """
        Get the (memory, dtype, shape) tuple for passing to subprocesses.

        Args:
            name: Identifier for the memory block

        Returns:
            Tuple of (SharedMemory, dtype, shape) for subprocess access
        """
        if name not in self.memory_blocks:
            raise KeyError(f"Shared memory '{name}' not found")

        return (
            self.memory_blocks[name],
            self.dtypes[name],
            self.shapes[name]
        )

    def get_numpy_array(self, name):
        """
        Get the numpy array view (main process only).

        Args:
            name: Identifier for the memory block

        Returns:
            Numpy array view of the shared memory
        """
        if name not in self.numpy_arrays:
            raise KeyError(f"Numpy array '{name}' not found")

        return self.numpy_arrays[name]

    def cleanup_all(self):
        """
        Clean up all shared memory resources.

        This should only be called in the main process during shutdown.
        Follows safe cleanup procedure:
        1. Delete numpy array references
        2. Force garbage collection
        3. Close all shared memory blocks
        4. Unlink all shared memory blocks
        """
        print("Starting shared memory cleanup...")

        # Step 1: Delete numpy array references
        print("  Deleting numpy array references...")
        self.numpy_arrays.clear()

        # Step 2: Force garbage collection
        print("  Running garbage collection...")
        gc.collect()

        # Step 3: Close all shared memory blocks
        print("  Closing shared memory blocks...")
        for name, shm in self.memory_blocks.items():
            try:
                shm.close()
                print(f"    Closed: {name}")
            except Exception as e:
                print(f"    Warning: Error closing {name}: {e}")

        # Step 4: Unlink all shared memory blocks
        print("  Unlinking shared memory blocks...")
        for name, shm in self.memory_blocks.items():
            try:
                shm.unlink()
                print(f"    Unlinked: {name}")
            except FileNotFoundError:
                print(f"    Already unlinked: {name}")
            except Exception as e:
                print(f"    Warning: Error unlinking {name}: {e}")

        # Clear the storage
        self.memory_blocks.clear()
        self.dtypes.clear()
        self.shapes.clear()

        print("Shared memory cleanup complete!")


def open_shared_memory(shared_data_tuple):
    """
    Helper function for subprocesses to open shared memory.

    Args:
        shared_data_tuple: Tuple of (SharedMemory, dtype, shape)

    Returns:
        Tuple of (SharedMemory handle, numpy array view)

    Note:
        Subprocesses should use this to ensure they don't create new memory.
        The returned numpy array will have OWNDATA=False.
    """
    memory, dtype, shape = shared_data_tuple

    # Open existing shared memory (don't create)
    shm = shared_memory.SharedMemory(name=memory.name)

    # Create numpy view
    np_array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

    # Verify this is a view, not owned data
    assert not np_array.flags['OWNDATA'], "Subprocess created owned array instead of view!"

    return shm, np_array
