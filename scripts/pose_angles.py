import math

import cv2
import mediapipe as mp
from picamera2 import Picamera2

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def calculate_angle(a, b, c):
    a = (a.x, a.y)
    b = (b.x, b.y)
    c = (c.x, c.y)

    radians = (
        math.atan2(c[1] - b[1], c[0] - b[0])
        - math.atan2(a[1] - b[1], a[0] - b[0])
    )

    angle = abs(math.degrees(radians))

    if angle > 180:
        angle = 360 - angle

    return angle

picam2 = Picamera2()
picam2.configure(
    picam2.create_preview_configuration(
        main={"format": "XRGB8888", "size": (640, 480)}
    )
)
picam2.start()

rep_count = 0
stage = "Unknown"

try:
    while True:
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

            landmarks = results.pose_landmarks.landmark

            shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

            angle = calculate_angle(shoulder, elbow, wrist)

            frame_height, frame_width = frame.shape[:2]
            elbow_position = (
                int(elbow.x * frame_width),
                int(elbow.y * frame_height)
            )

            cv2.putText(
                frame,
                f"{int(angle)} deg",
                elbow_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            if angle > 160:
                stage = "Down"

            if angle < 60 and stage == "Down":
                stage = "Up"
                rep_count += 1

        cv2.rectangle(frame, (0, 0), (230, 100), (245, 117, 16), -1)

        cv2.putText(
            frame,
            "REPS",
            (15, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            1
        )

        cv2.putText(
            frame,
            str(rep_count),
            (15, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.8,
            (255, 255, 255),
            3
        )

        cv2.putText(
            frame,
            "STAGE",
            (115, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            1
        )

        cv2.putText(
            frame,
            stage,
            (115, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Press q to quit",
            (10, 465),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.imshow("Day 10 - Bicep Curl Rep Counter", frame)

        if cv2.waitKey(20) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("Stopped by user")

finally:
    pose.close()
    picam2.stop()
    cv2.destroyAllWindows()
