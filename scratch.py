import numpy as np
import cv2
def transform_pixel_to_coordinate(ball_pixels):
    cx = 619.674
    cy = 333.682
    fx = 673.797
    fy = 673.797
    intrinsic_matrix = np.array([[fx, 0, cx],
                                 [0, fy, cy],
                                 [0, 0, 1]])
    distort_matrix = np.array([-0.269418, 0.048704, -0.000727, 0.002795, 0])
    distorted_pixel = ball_pixels[:,:2].astype(float)

    # distorted_pixel = distorted_pixel.astype(float)
    start_t=ball_pixels[0,2]
    time_array=ball_pixels[:,2]-start_t
    time_array=time_array.reshape(-1,1)
    print(time_array)
    # return intrinsic_matrix, distorted_pixel
    undistorted_pixel = cv2.undistortPoints(distorted_pixel, intrinsic_matrix, distort_matrix, P=intrinsic_matrix).reshape(-1,2)


    # return undistorted_pixel
    undistorted_pixel[:,0] -= cx
    undistorted_pixel[:,1] -= cy
    undistorted_pixel[:,0] /= fx
    undistorted_pixel[:,1] /= fy
    coordinate=np.concatenate((undistorted_pixel,time_array),axis=1)
    return coordinate
ball_pixels=[[200,200,0],
             [200,300,1]]
ball_pixels=np.array(ball_pixels)
transform_pixel_to_coordinate(ball_pixels)