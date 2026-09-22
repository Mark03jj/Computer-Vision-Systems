import cv2
import numpy as np
from picamera2 import Picamera2

# HSV range for a blue object.
# You can adjust these later if your blue block is not detected well.
LOWER_BLUE = np.array([95, 100, 60])
UPPER_BLUE = np.array([130, 255, 255])

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

        # Convert the camera frame to HSV for color detection.
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # White pixels in the mask are pixels within the blue HSV range.
        mask = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)

        # Remove small noise dots.
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        status = "No blue object detected"

        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)

            if area > 300:
                (x, y), radius = cv2.minEnclosingCircle(largest)
                moments = cv2.moments(largest)

                if moments["m00"] != 0:
                    center_x = int(moments["m10"] / moments["m00"])
                    center_y = int(moments["m01"] / moments["m00"])

                    cv2.circle(
                        frame,
                        (int(x), int(y)),
                        int(radius),
                        (0, 255, 0),
                        2
                    )

                    cv2.circle(
                        frame,
                        (center_x, center_y),
                        5,
                        (0, 0, 255),
                        -1
                    )

                    status = f"BLUE BLOCK: ({center_x}, {center_y})"

        cv2.putText(
            frame,
            "Color Block Tracker",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            status,
            (10, 460),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2
        )

        cv2.imshow("Original Feed", frame)
        cv2.imshow("Blue Mask", mask)

        if cv2.waitKey(20) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("Stopped by user")

finally:
    picam2.stop()
    cv2.destroyAllWindows()
