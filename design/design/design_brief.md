# Design Brief — Smart Classroom Occupancy Monitor

## Problem Statement

Teachers and classroom staff may need a simple way to know whether people are present in a classroom or lab area. Manually checking the room or keeping attendance counts can take time and may not show when the space is actively occupied.

The intended user is a teacher, classroom monitor, or school staff member who needs a basic visual record of activity in a defined area.

## Proposed Solution

This individual project uses a Raspberry Pi Camera and a TensorFlow Lite object-detection model to detect people in a selected classroom area. The program displays bounding boxes around detected people, shows the number of people currently detected, and logs time, label, and confidence score to a CSV file. A cooldown timer prevents the program from adding duplicate logs every camera frame while a person stays in view.

## CV Technique(s) Used

The main technique is TensorFlow Lite object detection using the SSD MobileNet model. This technique fits the project because it can recognize a person as an object class, unlike color tracking or shape detection, which cannot reliably identify a person.

OpenCV is used to display the camera feed, draw bounding boxes, add text, and handle the window. CSV logging is used to save timestamped detection events for later review.

## Success Criteria

The system will be considered successful if it:

- Detects one person in the camera's selected test area in at least 8 out of 10 trials under normal classroom lighting
- Displays a bounding box, `person` label, confidence score, and current person count
- Logs a timestamp, `person` label, and confidence score to `data/detections.csv`
- Avoids repeated CSV rows for the same continuous detection by using a 5-second cooldown
- Runs live from the Raspberry Pi camera at a usable speed for a classroom demonstration
