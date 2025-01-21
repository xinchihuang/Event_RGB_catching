import time
import can
from pcan_cybergear import CANMotorController


bus = can.interface.Bus(interface="pcan", channel="PCAN_USBBUS1", bitrate=1000000)
motor1 = CANMotorController(bus, motor_id=101, main_can_id=254)

motor1.set_0_pos()

motor1.enable()

motor1.send_motor_control_command(torque=0, target_angle=0.07, target_velocity=0, Kp=100, Kd=1)
time.sleep(2.5)
motor1.disable()

bus.shutdown()

