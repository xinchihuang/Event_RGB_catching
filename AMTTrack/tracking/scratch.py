import cv2
import numpy as np

# Load the image
image = cv2.imread('18_56_24_043.png')

# Convert from BGR to HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the range for green color in HSV
lower_green = np.array([30, 30, 0])
upper_green = np.array([60, 200, 255])

# Create a mask for green color
mask = cv2.inRange(hsv, lower_green, upper_green)

# Optional: Clean up the mask by removing noise
mask = cv2.erode(mask, None, iterations=2)
mask = cv2.dilate(mask, None, iterations=2)

# Find contours in the mask
contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Proceed only if at least one contour was found
if len(contours) > 0:
    # Find the largest contour by area
    c = max(contours, key=cv2.contourArea)
    # Get bounding box coordinates from the largest contour
    x, y, w, h = cv2.boundingRect(c)

    # Draw the bounding box on the original image
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Also, if you still want to draw the center:
    M = cv2.moments(c)
    # if M["m00"] != 0:
    #     center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    #     cv2.circle(image, center, 5, (0, 0, 255), -1)

    # Display the result
    cv2.imshow("Detected Green Ball with Bounding Box", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No green objects found.")
