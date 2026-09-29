import cv2
import numpy as np
import tensorflow as tf
from picamera2 import Picamera2

MODEL_PATH = "models/model.tflite"
LABELS_PATH = "models/labels.txt"
CONFIDENCE_THRESHOLD = 0.50
TARGET_LABEL = "person"

with open(LABELS_PATH, "r") as file:
    labels = [line.strip() for line in file.readlines()]

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_height = input_details[0]["shape"][1]
input_width = input_details[0]["shape"][2]

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

        people_count = 0

        for i, score in enumerate(scores):
            if score < CONFIDENCE_THRESHOLD:
                continue

            class_id = int(classes[i])
            label = labels[class_id] if 0 <= class_id < len(labels) else "Unknown"

            if label != TARGET_LABEL:
                continue

            people_count += 1

            ymin, xmin, ymax, xmax = boxes[i]
            x1 = int(xmin * frame_width)
            y1 = int(ymin * frame_height)
            x2 = int(xmax * frame_width)
            y2 = int(ymax * frame_height)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"Person: {score:.0%}",
                (x1, max(y1 - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        cv2.putText(
            frame,
            f"People detected: {people_count}",
            (10, 35),
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

        cv2.imshow("Smart Classroom Occupancy Monitor", frame)

        if cv2.waitKey(20) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("Stopped by user")

finally:
    picam2.stop()
    cv2.destroyAllWindows()
