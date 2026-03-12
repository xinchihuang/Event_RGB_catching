from multiprocessing import shared_memory
import numpy as np
import time
import os
from pcan_cybergear import CANMotorController
import can


bus = can.interface.Bus(interface="pcan", channel="PCAN_USBBUS1", bitrate=1000000)
motor1 = CANMotorController(bus, motor_id=102, main_can_id=254)
motor1.set_0_pos()
motor1.enable()
motor1.send_motor_control_command(torque=0, target_angle=0, target_velocity=0, Kp=20, Kd=1)
motor1.enable()
bus.shutdown()
