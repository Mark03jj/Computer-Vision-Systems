import cv2
import mediapipe as mp
from picamera2 import Picamera2

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

picam2 = Picamera2()
picam2.configure(
    picam2.create_preview_configuration(
        main={"format": "XRGB8888", "size": (640, 480)}
    )
)
picam2.start()

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

            right_shoulder = landmarks[
                mp_pose.PoseLandmark.RIGHT_SHOULDER.value
            ]
            right_wrist = landmarks[
                mp_pose.PoseLandmark.RIGHT_WRIST.value
            ]

            cv2.putText(
                frame,
                f"Right shoulder: ({right_shoulder.x:.2f}, {right_shoulder.y:.2f})",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Right wrist: ({right_wrist.x:.2f}, {right_wrist.y:.2f})",
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
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

        cv2.imshow("Day 9 - Pose Estimation", frame)

        if cv2.waitKey(20) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("Stopped by user")

finally:
    pose.close()
    picam2.stop()
    cv2.destroyAllWindows()
