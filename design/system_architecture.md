# System Architecture and Camera Placement

## How My System Works

```text
Camera
  |
  v
Raspberry Pi takes a picture/frame
  |
  v
Python and OpenCV prepare the frame
  |
  v
TensorFlow Lite checks the frame for objects
  |
  v
Is there a person?
  |
  +-- No --> Keep checking new frames
  |
  +-- Yes --> Draw a box around the person
                |
                v
            Show the number of people on screen
                |
                v
            Save the time, label, and confidence score in a CSV file
```

## Parts of My Project

| Part | What it does |
|---|---|
| Raspberry Pi Camera | Looks at the test area and sends video frames to the Pi |
| Picamera2 | Lets Python get frames from the camera |
| OpenCV | Changes the frame format, draws boxes, and shows the live camera window |
| TensorFlow Lite model | Tries to recognize objects in the camera view |
| Person filter | Only keeps detections labeled `person` |
| Confidence threshold | Ignores detections below 50% confidence |
| Cooldown timer | Stops the program from logging the same person every single frame |
| CSV file | Saves the time, label, and confidence score of detections |

## Camera Placement Plan

My system needs to detect a person standing in a marked area. I want the person to be large enough in the camera view for the model to recognize them, but I also need enough space around them so they do not get cut off.

The Raspberry Pi Camera Module V2 has a horizontal field of view of about 62.2 degrees. That means the camera can see a fairly wide area in front of it.

I will begin by placing the camera about **2 meters away** from the person. At that distance, the camera should see about **2.4 meters across**, which is wide enough for one person and some extra space around them.

My plan is to test three distances:

| Test distance | Why I am testing it |
|---|---|
| 1.5 meters | The person will look bigger, but they might not fully fit in the frame |
| 2.0 meters | This is my starting distance because it should show the person clearly with extra space |
| 2.5 meters | The camera will see more area, but the person may look too small for accurate detection |

I will place the camera around chest or head height and point it mostly straight at the test area. If the camera is above the person, I will angle it slightly downward.

## Field of View Math

The camera's horizontal field of view is about 62.2 degrees.

At a distance of 2 meters:

\[
\text{Visible width} = 2(2)\tan(31.1^\circ)
\]

\[
\text{Visible width} \approx 2.4\text{ meters}
\]

This means the camera should see about 2.4 meters from left to right when it is 2 meters away. That should give enough room for one person to stand in the test area.

## Final Plan

For my first test, I will place the camera about 2 meters from the area where a person will stand. I will run the detection program and test whether it detects a person correctly. Then I will compare the results at 1.5, 2.0, and 2.5 meters. I will use the distance that gives the clearest and most reliable person detections for my final presentation.
