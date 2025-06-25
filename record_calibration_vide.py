# import PySpin
# import cv2
# import numpy as np
# import time

# def record_with_preview(
#     output_path="flir_calib_preview.mp4",
#     duration=20,
#     frame_rate=15,
#     checkerboard_size=(9, 6),
# ):
#     system = PySpin.System.GetInstance()
#     cam_list = system.GetCameras()
#     cam = cam_list.GetByIndex(0)
#
#     cam.Init()
#     cam.AcquisitionFrameRateEnable.SetValue(True)
#     cam.AcquisitionFrameRate.SetValue(frame_rate)
#     cam.BeginAcquisition()
#
#     image = cam.GetNextImage()
#     height, width = image.GetHeight(), image.GetWidth()
#     image.Release()
#
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     # 初始化 VideoWriter
#     out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (width, height), isColor=True)
#
#
#     print(f"[INFO] Recording started. Duration: {duration}s | Resolution: {width}x{height}")
#     start_time = time.time()
#
#     while time.time() - start_time < duration:
#         image = cam.GetNextImage()
#         if image.IsIncomplete():
#             print("[WARN] Incomplete frame")
#             image.Release()
#             continue
#
#         img_gray = image.GetNDArray().copy()
#         # image.Release()
#         #
#         # # # 角点检测
#         # found, corners = cv2.findChessboardCorners(img_gray, checkerboard_size)
#         preview = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
#         # if found:
#         #     cv2.drawChessboardCorners(preview, checkerboard_size, corners, found)
#         #     out.write(preview)  # 只保存角点成功帧
#         #     print(f"[INFO] Frame saved (detected corners)")
#         out.write(preview)
#         # 显示预览
#         cv2.imshow("Preview", preview)
#         cv2.waitKey(1)
#
#     out.release()
#     cv2.destroyAllWindows()
#     cam.EndAcquisition()
#     cam.DeInit()
#     del cam
#     cam_list.Clear()
#     system.ReleaseInstance()
#     print(f"[INFO] Recording finished and saved to {output_path}")

# if __name__ == "__main__":
#     record_with_preview()

import cv2
import numpy as np
import json

# === 参数设置 ===
video_path = "flir_calib_preview.mp4"       # 视频文件路径
checkerboard_size = (10, 7)           # 内角点数
square_size = 0.117                # 单位长度
max_frames = 50                      # 最多处理多少帧（选够用的）
output_json = "camera_config.json"

# === 准备数据结构 ===
objp = np.zeros((checkerboard_size[0]*checkerboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
objp *= square_size
objpoints, imgpoints = [], []

# === 打开视频并逐帧读取 ===
cap = cv2.VideoCapture(video_path)
frame_idx = 0
used_frames = 0

while cap.isOpened() and used_frames < max_frames:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret_cb, corners = cv2.findChessboardCorners(gray, checkerboard_size)

    vis_frame = frame.copy()
    if ret_cb:
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        objpoints.append(objp)
        imgpoints.append(corners2)
        used_frames += 1

        # === 绘制角点 ===
        cv2.drawChessboardCorners(vis_frame, checkerboard_size, corners2, ret_cb)
        cv2.putText(vis_frame, f"Frame {frame_idx} - Used", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 2)
    else:
        cv2.putText(vis_frame, f"Frame {frame_idx} - Skipped", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Chessboard Detection", vis_frame)
    cv2.waitKey(1)

    frame_idx += 1

# === 标定 ===
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# === 构建相机参数字典 ===
camera_config = {
    'fx': float(K[0, 0]),
    'fy': float(K[1, 1]),
    'cx': float(K[0, 2]),
    'cy': float(K[1, 2]),
    'dist_coeffs': dist.flatten().tolist()
}

# === 保存配置 ===
with open(output_json, 'w') as f:
    json.dump(camera_config, f, indent=4)

print("\n=== Camera Intrinsic Parameters ===")
print(json.dumps(camera_config, indent=4))
print(f"[INFO] Saved to {output_json}")

