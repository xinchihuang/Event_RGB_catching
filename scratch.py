import cv2

video_path="/home/xinchi/Event_RGB_catching_data/saved_data/rgb/0.mp4"
cap = cv2.VideoCapture(video_path)
count=0
while True:
    ret, frame = cap.read()

    if not ret:
        # If no frame is returned, the video has ended
        print("End of video.")
        break
    cv2.imshow("frame", frame)
    cv2.waitKey(1)
    cv2.imwrite("/home/xinchi/Event_RGB_catching_data/frames/"+str(count)+".jpg",frame)
    count+=1


