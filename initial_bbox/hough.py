import cv2
import numpy as np


def overlay_canny_edges(image: np.ndarray, edges: np.ndarray, color=(0, 0, 255)) -> np.ndarray:
    """
    将 Canny 边缘叠加到原图上

    Args:
        image: 原始 BGR 图像
        edges: Canny 边缘检测得到的二值图
        color: 边缘颜色 (BGR格式)，默认红色

    Returns:
        overlay: 边缘叠加后的图像
    """
    overlay = image.copy()
    overlay[edges > 0] = color  # 边缘像素设为红色
    return overlay


# ==== 示例 ====
img = cv2.imread("../94.jpg")

# 灰度 + 模糊 + Canny
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, 60, 180)
4
# 叠加边缘
overlay = overlay_canny_edges(img, edges, color=(0, 0, 255))

# 显示
cv2.imshow("Original", img)
cv2.imshow("Canny Edges", edges)
cv2.imshow("Overlay", overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()
