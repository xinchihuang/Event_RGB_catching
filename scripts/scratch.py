import cv2
import numpy as np
from matplotlib import pyplot as plt

# ---------- 1. 读图并转换到 HSV ----------
image_path = "/home/xinchi/Event_RGB_catching/AMTTrack/83.jpg"
bgr = cv2.imread(image_path)
hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

# ---------- 2. 设定黄色阈值 ----------
# OpenCV 的 Hue∈[0,179]，黄色约 20–35° ⇒ 10–25
lower_hsv = np.array([15,  80,  80])   # H,S,V
upper_hsv = np.array([35, 255, 255])

# ---------- 3. 生成掩码并形态学去噪 ----------
mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=2)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

# ---------- 4. 轮廓检测并绘制最小外接圆 ----------
output = bgr.copy()
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    if cv2.contourArea(cnt) > 50:          # 过滤小噪声
        (x, y), r = cv2.minEnclosingCircle(cnt)
        cv2.circle(output, (int(x), int(y)), int(r), (0, 0, 255), 2)

# ---------- 5. 可视化 ----------
plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('Detected Yellow Ball (HSV threshold)')
plt.show()
