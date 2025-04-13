import cv2
video_path = "/home/xinchi/Event_RGB_catching_data/saved_data/rgb/0.mp4"
cap = cv2.VideoCapture(video_path)

# 检查视频文件是否成功打开
if not cap.isOpened():
    print(f"无法打开视频文件: {video_path}")
else:
    print(f"视频文件打开成功: {video_path}")

# 循环读取每一帧
count=0
while True:
    ret, frame = cap.read()

    # 检查是否成功读取帧
    if not ret:
        print("读取视频帧失败或视频结束")
        break

    # 处理帧（例如显示帧）
    cv2.imshow('Video Frame', frame)

    # 按键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    print(count)
    count+=1