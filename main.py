import cv2

cam = cv2.VideoCapture(0)

while cam.isOpened():
    result, frame = cam.read()
    if not result:
        print("empty frame")
        continue

    cv2.imshow("", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
