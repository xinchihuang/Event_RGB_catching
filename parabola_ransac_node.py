#!/usr/bin/env python3

'''
ROS Node that runs a RANSAC to determine the parabola along which the projectile will move, and calculates the impact point.
Subscribes to:
- /sees/cluster_bbox: BoundingBox2DStamped: the center of the largest cluster on an image

Publishes:
- /sees/impact_marker_array: MarkerArray: the impact points
- /sees/inlier_marker_array: MarkerArray: the inlier points of the RANSAC
- /sees/outlier_marker_array: MarkerArray: the outlier points of the RANSAC
- /sees/parabola_marker: Marker: a visualization of the trajectory
- /sees/impact_marker: Marker: a visualization of the morst recent impact point
- /sim_capture/ball_states: Float32MultiArray: net center position command to our robot @ 100 Hz (can be disabled for new applications)
'''
# standard

import time
import numpy as np

# custom
from utils import load_bboxes, filter_to_max_bboxes, filter_bboxes_to_interval
from ransac_fit import fit_parabola_to_bboxes, compute_impact_point, sample_parabola

DEBUG_NODE = True

class RansacTracker:
    '''
    Takes in the center of the largest cluster on an image and tracks it using RANSAC + KF. Publishes the ball state @ 100 Hz.
    '''
    def __init__(self):
        self.points_4d = []
        self.intersection_markers = []
        self.impact_points = []
        # rolling buffer of bboxes to consider for parabola fitting
        self.bboxes_buffer_len = 50
        # discard bboxes older than 0.5 secs from buffer
        self.time_window_oldest = 2.5
        # RANSAC discards bboxes for parabola fitting, which are 0.2m away from the parabola
        self.inlier_threshold = 0.25
        self.t_step = 0.01
        self.t_since_throw = 0.0
        self.max_skipped_detections = 5
        self.skipped_detection_num = 0
        self.marker_idx = 0
        self.ball_pub_rate = 100
        self.start_time_set = False
        self.use_z_only = True
        self.point_distance_threshold = 0.05

    def init_filter(self):
        '''
        Reinitializes the Kalman Filter that is used to track the projectiles.
        '''
        self.t_since_throw = 0.0
        self.skipped_detection_num = 0
        self.points_4d = []
        self.impact_points = []
        self.delete_marker_array(self.inlier_array_pub)
        self.delete_marker_array(self.outlier_array_pub)
        self.delete_marker_array(self.impact_array_pub)
        self.marker_array.markers = []
        
    def process_measurement(self, point_msg):
        '''
        Processes an incoming cluster center 3D point
        '''
        print("Got measurement, x: ", point_msg.point.x, " y: ", point_msg.point.y, " z: ", point_msg.point.z)
        POINT_CLOUD_MAX_Z = 8        
        if point_msg.point.z > POINT_CLOUD_MAX_Z or point_msg.point.z < 0:
            #print("Skipping 3D point, too far away")
            self.skipped_detection_num += 1

            if self.skipped_detection_num >= self.max_skipped_detections:
                # rospy.loginfo(f"More than {self.max_skipped_detections} skipped detections, reinitializing...")
                self.init_filter()
                self.skipped_detection_num = 0
        else:
            time = self._get_time_from_stamp(point_msg.header.stamp)

            point_4d = np.array([point_msg.point.x, point_msg.point.y, point_msg.point.z, time]).astype("float32")
            if len(self.points_4d):
                points_4d_to_process = np.stack(self.points_4d)
                distance_vectors = points_4d_to_process[:,:3] - point_4d[:3].reshape(1,3)
                # calculate min
                if self.use_z_only:
                    distance_vectors = distance_vectors[:,2].reshape(-1,1)
                distances = np.linalg.norm(distance_vectors, axis=1)
                min_dist = np.min(distances)
                if min_dist > self.point_distance_threshold:
                    self.points_4d.append(point_4d)
                    self.skipped_detection_num = 0
                    rospy.loginfo(f"Detected object at {point_4d}")
                    rospy.loginfo(f"Num 3D points in buffer = {len(self.points_4d)}")
                else:
                    self.skipped_detection_num += 1
                    rospy.logwarn('Non-sufficient distance from previous points: Skipping detection')
                if len(self.points_4d) > self.bboxes_buffer_len:
                    self.points_4d.pop(0)
            else:
                self.points_4d.append(point_4d)
                self.skipped_detection_num = 0
                rospy.loginfo(f"Detected object at {point_4d}")
                rospy.loginfo(f"Num 3D points in buffer = {len(self.points_4d)}")

    def publish_ball_state(self, _ = None):
        '''
        Publishes the ball state & the intersection point as a Float32MultiArray message
        '''
        if len(self.points_4d) < 2:
            self._publish_ball_state_msg(0, -0.92, 0)
            return 

        start = time.time()
        points_4d_to_process = np.stack(self.points_4d)
        initial_position, initial_velocity, inliers, points_3d = fit_parabola_to_bboxes(points_4d_to_process, inlier_threshold=self.inlier_threshold)

        # print("initial position")
        if np.isnan(initial_velocity[0]):
            self._publish_ball_state_msg(0, -0.92, 0)
            return 

        # print("impact point")
        impact_point = compute_impact_point(initial_position, initial_velocity)
        rospy.loginfo(f"Impact Point at Px={impact_point[0]} Py={impact_point[1]}. Publishing msg. Running @ {1/(time.time()-start)} Hz")

        self.impact_points.append(impact_point)
        median_impact_point = np.median(np.stack(self.impact_points), axis=0)
        self._publish_ball_state_msg(median_impact_point[0], median_impact_point[1])

        if DEBUG_NODE:
            # print("Publishing markers")
            self._get_impact_marker_array(impact_point)
            self._get_inlier_outlier_marker_arrays(inliers, points_3d)
            self._get_median_impact_point(median_impact_point)
            self._get_parabola_marker_array(initial_velocity, initial_position)



    def _get_time_from_stamp(self, timestamp):
        '''
        Gives the elapsed seconds as a float from a ros timestamp
        '''
        if self.start_time_set:
            time = float(timestamp.nsecs)/10e9 + float(timestamp.secs) - self.start_time
        else:
            self.start_time = float(timestamp.nsecs)/10e9 + float(timestamp.secs)
            time = 0.0
            self.start_time_set = True
        return time

if __name__ == '__main__':

    pass