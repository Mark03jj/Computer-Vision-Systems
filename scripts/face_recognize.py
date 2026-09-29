import os

import cv2
import face_recognition
from picamera2 import Picamera2

KNOWN_FACES_DIR = os.path.expanduser(
    "~/cv_project/images/known_faces"
)

known_encodings = []
known_names = []

if not os.path.exists(KNOWN_FACES_DIR):
    os.makedirs(KNOWN_FACES_DIR)

for filename in os.listdir(KNOWN_FACES_DIR):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    path = os.path.join(KNOWN_FACES_DIR, filename)
    image = face_recognition.load_image_file(path)
    encodings = face_recognition.face_encodings(image)

    if encodings:
        known_encodings.append(encodings[0])
        name = os.path.splitext(filename)[0]
        name = name.replace("_", " ").title()
        known_names.append(name)

print(f"Loaded {len(known_names)} known face(s): {known_names}")

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

        small_frame = cv2.resize(
            rgb_frame,
            (0, 0),
            fx=0.5,
            fy=0.5
        )

        face_locations = face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(
            small_frame,
            face_locations
        )

        for face_location, face_encoding in zip(
            face_locations,
            face_encodings
        ):
            top, right, bottom, left = face_location
            name = "Unknown"

            if known_encodings:
                distances = face_recognition.face_distance(
                    known_encodings,
                    face_encoding
                )

                best_match_index = distances.argmin()

                if distances[best_match_index] < 0.6:
                    name = known_names[best_match_index]

            top *= 2
            right *= 2
            bottom *= 2
            left *= 2

            cv2.rectangle(
                frame,
                (left, top),
                (right, bottom),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                name,
                (left, max(top - 10, 25)),
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

        cv2.imshow("Day 12 - Face Recognition", frame)

        if cv2.waitKey(20) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("Stopped by user")

finally:
    picam2.stop()
    cv2.destroyAllWindows()
