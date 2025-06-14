import cv2
import mediapipe as mp
import math


def calc_dist(p1, p2):
    return round(math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2), 2)


cam = cv2.VideoCapture(0)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
color = (255, 255, 255)

with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cam.isOpened():
        result, frame = cam.read()
        frame = cv2.flip(frame, 1)
        if not result:
            print("empty frame")
            continue

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        if results.multi_hand_landmarks:
            h, w = frame.shape[:2]
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                hand = results.multi_handedness[i].classification[0]

                thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                param_change = calc_dist(thumb, index) - 0.2
                x1, y1 = int(index.x * w), int(index.y * h)
                x2, y2 = int(thumb.x * w), int(thumb.y * h)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                text = hand.label
                if hand.index == 0:
                    text = 'Brightness: ' + str(round(param_change, 2))
                    frame = cv2.convertScaleAbs(frame, alpha=1.0, beta=param_change * 300)
                else:
                    text = 'Blur: ' + str(round(param_change, 2))
                    blurred = cv2.GaussianBlur(frame, (0, 0), 3)
                    frame = cv2.addWeighted(frame, 1 + param_change * 10, blurred, -param_change * 10, 0)

                cv2.line(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, text, (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color, 2, 2),
                    mp_drawing.DrawingSpec(color, 2))

        cv2.imshow("hands tracking", frame)
        if cv2.waitKey(1) == ord('q'):
            break

cam.release()
cv2.destroyAllWindows()
