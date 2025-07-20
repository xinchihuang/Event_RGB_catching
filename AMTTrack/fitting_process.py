from multiprocessing import Process,Array,Queue,Manager,shared_memory,Value
import numpy as np
import time
import csv
def landing_prediction(shared_ball_pixel_3D_data,ball_observations,shared_impact_point_data,record_index,system_state,stop_event):

    from parabola_fitter import MonoRansac
    from utils import rotate_matrix_coordinate

    # camera_config = {
    #     'fx': 739.9,
    #     'fy': 739.9,
    #     'cx': 310,
    #     'cy': 710,
    #     'dist_'
    #     'dist_coeffs': [-0.1060764147484086, -0.001306849716448579, -0.0005920424312549498, -5.0607348313996e-05, 0]
    #
    # }
    camera_config = {
    "fx": 987.1678420424932,
    "fy": 986.0683237433913,
    "cy": 739.7418995490075,
    "cx": 513.7733221750243,
    "dist_coeffs": [
        -0.11123899870851763,
        -0.020663002330730863,
        -0.0002997445722290338,
        0.0007964973013426446,
        0.014818755754515196
    ]
}
    camera_rpy_to = [0, 45, -90]
    camera_rpy_back = [45, 0, 90]
    rotation_matrix_to = rotate_matrix_coordinate(camera_rpy_to[0], camera_rpy_to[1], camera_rpy_to[2])
    rotation_matrix_back = rotate_matrix_coordinate(camera_rpy_back[0], camera_rpy_back[1], camera_rpy_back[2])

    ball_pixel_memory, ball_pixel_dtype, ball_pixel_shape = shared_ball_pixel_3D_data
    ball_pixel_shm = shared_memory.SharedMemory(name=ball_pixel_memory.name)
    impact_point_memory, impact_point_dtype, impact_point_shape = shared_impact_point_data
    impact_point_shm = shared_memory.SharedMemory(name=impact_point_memory.name)
    impact_point_array = np.ndarray(impact_point_shape, dtype=impact_point_dtype, buffer=impact_point_shm.buf)
    fitter = MonoRansac(camera_params=camera_config,rotation_matrix_to=rotation_matrix_to,rotation_matrix_back=rotation_matrix_back)
    buffered_impact_point = [0,0,0]
    last_index=0
    saved_impact_point_dir = "impact_point_mode_l.csv"
    while not stop_event.is_set():
        ball_pixels = np.ndarray(ball_pixel_shape, dtype=ball_pixel_dtype, buffer=ball_pixel_shm.buf)
        ball_pixels = ball_pixels[:ball_observations.value]
        # print(ball_observations.value)
        try:
            if system_state.value == 't':
                if ball_observations.value == last_index:
                    continue
                else:
                    last_index = ball_observations.value
                if len(ball_pixels) >= 10:
                    # print(ball_pixels)

                    ball_memory = ball_pixels

                    observations = ball_memory
                    # print(observations)
                    observations[:, 3] = observations[:, 3] - observations[0, 3]
                    fitter.set_data(observations)
                    result = fitter.get_params()
                    impact_point=fitter.predict()
                    if -1.5<impact_point[1]<1.5 and 0<impact_point[2]<2:
                        buffered_impact_point = impact_point
                    else:
                        impact_point = buffered_impact_point

                    # print(impact_point_arrays)
                    impact_point_array[0] = impact_point[0]
                    impact_point_array[1] = impact_point[1]
                    impact_point_array[2] = impact_point[2]
                    # print(impact_point)
                    # system_state.value='c'
                    with open(saved_impact_point_dir, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            record_index.value,
                            ball_observations.value,
                            impact_point[0],
                            impact_point[1],
                            impact_point[2],
                            time.time()
                        ])

        except:
            continue
def robot_control(shared_impact_point_data,system_state,stop_event):
    from pcan_cybergear import CANMotorController
    import can
    impact_point_memory, impact_point_dtype, impact_point_shape = shared_impact_point_data
    impact_point_shm = shared_memory.SharedMemory(name=impact_point_memory.name)
    bus = can.interface.Bus(interface="pcan", channel="PCAN_USBBUS1", bitrate=1000000)
    motor1 = CANMotorController(bus, motor_id=101, main_can_id=254)
    motor1.set_0_pos()
    motor1.enable()
    motor1.send_motor_control_command(torque=0, target_angle=0, target_velocity=0, Kp=100, Kd=1)
    print("motor start")
    while system_state.value == 'e':
        continue
    while not stop_event.is_set():

        # print(len(ball_pixels))

        if system_state.value=='c':

            impact_point_array = np.ndarray(impact_point_shape, dtype=impact_point_dtype, buffer=impact_point_shm.buf)
            # print(impact_point_array)
            x, y, z = impact_point_array
            shaft_length = 0.5
            joint_height = 0.7
            delta_height = z + joint_height
            if delta_height>0:
                angle = np.pi/2 - np.arctan(delta_height/shaft_length)
                # if angle<0:
                #     angle = max(angle, np.pi/4)
                if angle>0:
                    angle=min(angle, np.pi/4)
                if y < 0:
                    angle = -angle
                print('angle',angle)
                motor1.send_motor_control_command(torque=0, target_angle=angle, target_velocity=0, Kp=100, Kd=1)
                time.sleep(2.5)
                motor1.disable()
                bus.shutdown()
            system_state.value='e'