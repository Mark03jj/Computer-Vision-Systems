import cv2
from picamera2 import Picamera2

def classify_shape(contour):
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
    sides = len(approx)

    if sides == 3:
        return "Triangle"
    elif sides == 4:
        x, y, w, h = cv2.boundingRect(approx)
        ratio = w / float(h)
        if 0.9 <= ratio <= 1.1:
            return "Square"
        return "Rectangle"
    elif sides > 6:
        return "Circle"
    else:
        return f"{sides}-gon"

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

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            if cv2.contourArea(contour) < 300:
                continue

            shape = classify_shape(contour)
            approx = cv2.approxPolyDP(
                contour,
                0.04 * cv2.arcLength(contour, True),
                True
            )

            x, y, w, h = cv2.boundingRect(approx)
            cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
            cv2.putText(
                frame,
                shape,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        cv2.imshow("Shape Detection", frame)
        cv2.imshow("Edges", edges)

        if cv2.waitKey(20) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("Stopped")

finally:
    picam2.stop()
    cv2.destroyAllWindows()
