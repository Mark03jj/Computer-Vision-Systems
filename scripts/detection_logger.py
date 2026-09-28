import csv
import os
import time
from datetime import datetime

import cv2
import numpy as np
import tensorflow as tf
from picamera2 import Picamera2

MODEL_PATH = "models/model.tflite"
LABELS_PATH = "models/labels.txt"
LOG_PATH = "data/detections.csv"

CONFIDENCE_THRESHOLD = 0.50
TARGET_LABELS = {"person"}
COOLDOWN_SECONDS = 5

with open(LABELS_PATH, "r") as file:
    labels = [line.strip() for line in file.readlines()]

os.makedirs("data", exist_ok=True)

if not os.path.exists(LOG_PATH):
    with open(LOG_PATH, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp", "label", "confidence"])

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_height = input_details[0]["shape"][1]
input_width = input_details[0]["shape"][2]

last_logged = {}

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
        frame_height, frame_width = frame.shape[:2]

        resized = cv2.resize(frame, (input_width, input_height))
        input_data = np.expand_dims(resized, axis=0)

        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()

        boxes = interpreter.get_tensor(output_details[0]["index"])[0]
        classes = interpreter.get_tensor(output_details[1]["index"])[0]
        scores = interpreter.get_tensor(output_details[2]["index"])[0]

        people_detected = 0

        for i, score in enumerate(scores):
            if score < CONFIDENCE_THRESHOLD:
                continue

            class_id = int(classes[i])
            label = labels[class_id] if 0 <= class_id < len(labels) else "Unknown"

            if label not in TARGET_LABELS:
                continue

            people_detected += 1

            ymin, xmin, ymax, xmax = boxes[i]
            x1 = int(xmin * frame_width)
            y1 = int(ymin * frame_height)
            x2 = int(xmax * frame_width)
            y2 = int(ymax * frame_height)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{label}: {score:.0%}",
                (x1, max(y1 - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            now = time.time()

            if label not in last_logged or now - last_logged[label] >= COOLDOWN_SECONDS:
                timestamp = datetime.now().isoformat(timespec="seconds")

                with open(LOG_PATH, "a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([timestamp, label, f"{score:.2f}"])

                last_logged[label] = now
                print(f"Logged: {timestamp}, {label}, {score:.2f}")

        cv2.putText(
            frame,
            f"People detected: {people_detected}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
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

        cv2.imshow("Day 8 - Detection Logger", frame)

        if cv2.waitKey(20) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("Stopped by user")

finally:
    picam2.stop()
    cv2.destroyAllWindows()
