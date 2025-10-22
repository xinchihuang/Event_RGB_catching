import math
from multiprocessing import shared_memory
import numpy as np
import time
import os
def robot_control(shared_impact_point_data,system_state,stop_event):
    from .pcan_cybergear import CANMotorController
    import can
    impact_point_memory, impact_point_dtype, impact_point_shape = shared_impact_point_data
    impact_point_shm = shared_memory.SharedMemory(name=impact_point_memory.name)
    bus = can.interface.Bus(interface="pcan", channel="PCAN_USBBUS1", bitrate=1000000)
    motor1 = CANMotorController(bus, motor_id=101, main_can_id=254)
    motor1.set_0_pos()
    motor1.enable()
    motor1.send_motor_control_command(torque=0, target_angle=0, target_velocity=0, Kp=100, Kd=1)
    print("motor start")
    while not stop_event.is_set():

        # print(len(ball_pixels))
        # print(system_state.value)

        if system_state.value=='c':

            impact_point_array = np.ndarray(impact_point_shape, dtype=impact_point_dtype, buffer=impact_point_shm.buf)
            print(impact_point_array)
            x, y, z = impact_point_array
            shaft_length = 0.40
            joint_height = 0.43
            delta_height = z - joint_height

            angle = np.arctan2(y,delta_height)
            # if angle<0:
            #     angle = max(angle, np.pi/4)
            print(math.degrees(angle))
            if angle>0:
                angle=min(angle, np.pi/2)
            else:
                angle=max(angle, -np.pi/2)
            if x==0 and y==0 and z==0:
                angle=0
            print('angle',angle)
            motor1.send_motor_control_command(torque=0, target_angle=angle, target_velocity=0, Kp=100, Kd=1)
            time.sleep(2.5)
            motor1.send_motor_control_command(torque=0, target_angle=0, target_velocity=0, Kp=10, Kd=1)
            time.sleep(2.5)
            # Disable motor immediately after reset
            try:
                motor1.disable()
                print('Motor disabled successfully')
            except Exception as e:
                print(f'Motor disable error: {e}')
            try:
                bus.shutdown()
                print('Bus shutdown successfully')
            except Exception as e:
                print(f'Bus shutdown error: {e}')

            system_state.value='e'
            break
