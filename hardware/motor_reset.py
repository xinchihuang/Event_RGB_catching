#!/usr/bin/env python3
"""
独立测试脚本：测试 motor 归位功能
测试步骤：
1. 启动 motor 进程
2. 设置状态为 'c' (catch)
3. 提供预定位置 [0, 1, 1]
4. 观察 motor 是否执行动作并归位到 0
5. 检查 motor 是否正确 disable 和 bus shutdown
"""

import multiprocessing
from multiprocessing import shared_memory, Value
import numpy as np
import time
import sys
import os

# Add project path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from hardware.motor import robot_control

def main():
    print("=" * 60)
    print("Motor Reset Test - Starting")
    print("=" * 60)

    # Create shared memory for impact point
    impact_point_dtype = np.float32
    impact_point_shape = (3,)
    impact_point_memory = shared_memory.SharedMemory(
        create=True,
        size=np.prod(impact_point_shape) * np.dtype(impact_point_dtype).itemsize
    )

    # Create numpy array view
    impact_point_array = np.ndarray(
        impact_point_shape,
        dtype=impact_point_dtype,
        buffer=impact_point_memory.buf
    )

    # Set test position [0, 1, 1]
    impact_point_array[0] = 0.0  # x
    impact_point_array[1] = 1.0  # y
    impact_point_array[2] = 1.0  # z

    print(f"Test position set to: {impact_point_array}")

    shared_impact_point_data = (impact_point_memory, impact_point_dtype, impact_point_shape)

    # Create system state and stop event
    system_state = Value('u', 'w')  # Start in 'w' (waiting) state
    stop_event = multiprocessing.Event()

    # Start motor control process
    print("\n[1] Starting motor control process...")
    motor_process = multiprocessing.Process(
        target=robot_control,
        args=(shared_impact_point_data, system_state, stop_event)
    )
    motor_process.start()
    print(f"Motor process started (PID: {motor_process.pid})")

    # Wait for motor to initialize
    print("\n[2] Waiting for motor initialization (3 seconds)...")
    time.sleep(3)

    # Change state to 'c' to trigger catch action
    print("\n[3] Setting state to 'c' (catch) to trigger motor action...")
    print(f"Impact point: x={impact_point_array[0]}, y={impact_point_array[1]}, z={impact_point_array[2]}")
    system_state.value = 'c'

    # Wait for motor to complete action
    print("\n[4] Waiting for motor action to complete (15 seconds)...")
    print("    Expected sequence:")
    print("    - Motor moves to calculated angle")
    print("    - Motor resets to initial position (angle=0)")
    print("    - Motor disables")
    print("    - Bus shuts down")
    print("    - State changes to 'e'")

    time.sleep(15)

    # Check final state
    print(f"\n[5] Final state: '{system_state.value}'")
    if system_state.value == 'e':
        print("    ✅ State correctly changed to 'e'")
    else:
        print(f"    ❌ State should be 'e' but is '{system_state.value}'")

    # Wait for process to exit
    print("\n[6] Waiting for motor process to exit...")
    motor_process.join(timeout=5)

    if motor_process.is_alive():
        print("    ❌ Motor process still alive, terminating...")
        motor_process.terminate()
        motor_process.join(timeout=2)
        if motor_process.is_alive():
            print("    ❌ Force killing motor process...")
            motor_process.kill()
            motor_process.join()
    else:
        print("    ✅ Motor process exited normally")

    # Cleanup
    print("\n[7] Cleaning up shared memory...")
    try:
        del impact_point_array
        impact_point_memory.close()
        impact_point_memory.unlink()
        print("    ✅ Shared memory cleaned up")
    except Exception as e:
        print(f"    ❌ Error cleaning up: {e}")

    print("\n" + "=" * 60)
    print("Motor Reset Test - Complete")
    print("=" * 60)
    print("\nPlease check the output above for:")
    print("1. Motor moved to target angle")
    print("2. Motor reset to initial position (angle=0)")
    print("3. Motor disabled successfully")
    print("4. Bus shut down successfully")
    print("5. State changed from 'c' to 'e'")
    print("6. Process exited without errors")

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)