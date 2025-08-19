import numpy as np
import cv2
rgb_height= 1080
rgb_width = 1440

def draw_initial_bbox(image):
    x, y, w, h = 180, 1030, 40, 40
    rgb_image = image
    cv2.rectangle(rgb_image, (x, y), (x + h, y + w), (0, 255, 0), 2)
    cv2.imshow('Image with Bounding Box', rgb_image)
    cv2.waitKey(0)
    return x, y, w, h
image= cv2.imread('/home/xinchi/Event_RGB_catching/121.jpg')
draw_initial_bbox(image)