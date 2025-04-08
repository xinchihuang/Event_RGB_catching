import cv2
import numpy as np

def detect_object_by_roi_hsv(image_path,
                             delta_h=10,
                             delta_s=50,
                             delta_v=50,
                             min_area=100):
    """
    1) 用户手动从图像中选定一个ROI
    2) 获取ROI的平均HSV
    3) 给HSV一定范围后，阈值分割整张图像
    4) 找到满足颜色范围的区域并绘制外接矩形
    """
    # 1. 读取图像
    image = cv2.imread(image_path)
    image = image[800:, :, :]
    image= cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_AREA)
    if image is None:
        raise FileNotFoundError(f"无法读取图片，请检查路径: {image_path}")

    # 2. 让用户在窗口中选择ROI, 返回 (x, y, w, h)
    roi_rect = cv2.selectROI("Select ROI", image, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select ROI")  # 选完后关掉ROI窗口
    x, y, w, h = roi_rect

    if w == 0 or h == 0:
        print("未选择有效ROI，程序结束。")
        return

    # 3. 提取 ROI，并转换到 HSV
    roi = image[y:y+h, x:x+w]
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # 4. 计算ROI区域的平均HSV
    #    mean() 返回 (H_mean, S_mean, V_mean, alpha) 四个值，最后一个是透明度（没用）
    mean_hsv = cv2.mean(roi_hsv)[:3]
    mean_h, mean_s, mean_v = mean_hsv

    # 5. 为了扩大匹配范围，给平均H/S/V加减一定容忍度（delta），并做边界检查
    #    HSV空间：H ∈ [0,179], S ∈ [0,255], V ∈ [0,255]
    lower_h = max(mean_h - delta_h, 0)
    upper_h = min(mean_h + delta_h, 179)
    lower_s = max(mean_s - delta_s, 0)
    upper_s = min(mean_s + delta_s, 255)
    lower_v = max(mean_v - delta_v, 0)
    upper_v = min(mean_v + delta_v, 255)

    lower_bound = np.array([lower_h, lower_s, lower_v], dtype=np.uint8)
    upper_bound = np.array([upper_h, upper_s, upper_v], dtype=np.uint8)

    print("====== ROI 平均 HSV ======")
    print(f"Mean HSV: ({mean_h:.2f}, {mean_s:.2f}, {mean_v:.2f})")
    print("====== 颜色检测阈值范围 ======")
    print(f"Lower: {lower_bound}, Upper: {upper_bound}")

    # 6. 将整张图像转换到HSV并进行阈值分割
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # 7.（可选）形态学去噪：先腐蚀再膨胀
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # 8. 查找轮廓并在原图上绘制外接矩形
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # 根据面积过滤小噪点
        if area > min_area:
            x_cnt, y_cnt, w_cnt, h_cnt = cv2.boundingRect(cnt)
            cv2.rectangle(image,
                          (x_cnt, y_cnt),
                          (x_cnt + w_cnt, y_cnt + h_cnt),
                          (0, 0, 255), 2)

    # 9. 显示结果
    cv2.imshow("Mask", mask)
    cv2.imshow("Detection Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":

    detect_object_by_roi_hsv("/home/xinchi/Event_RGB_catching_data/dataset/rgb/23.jpg",
                             delta_h=10,
                             delta_s=50,
                             delta_v=50,
                             min_area=100)
