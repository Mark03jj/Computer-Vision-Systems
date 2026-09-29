# Decision Matrix — Final Project Selection

## Scoring Scale

- 1 = weak fit
- 3 = acceptable fit
- 5 = strong fit

| Concept | Feasibility on Pi Hardware | Time Available | Usefulness to a Real User | Technical Interest | Total |
|---|---:|---:|---:|---:|---:|
| Smart Classroom Occupancy Monitor | 5 | 5 | 4 | 4 | 18 |
| Bicep Curl Rep Counter | 3 | 3 | 4 | 5 | 15 |
| Color-Based Object Finder | 5 | 5 | 2 | 2 | 14 |

## Concept Summaries

### Smart Classroom Occupancy Monitor

The system detects people using TensorFlow Lite object detection, shows the current number of people detected, and logs detection events to a CSV file. Its user could be a teacher or classroom monitor.

### Bicep Curl Rep Counter

The system uses pose estimation to calculate an elbow angle and count bicep curls. Its user could be a student or fitness user.

### Color-Based Object Finder

The system uses HSV color tracking to locate a selected colored object. Its user could be someone looking for a specific colored item or testing a sorting process.

## Selection Justification

I selected the Smart Classroom Occupancy Monitor because it earned the highest score of 18 out of 20. It is realistic to complete as an individual because it builds directly on the Day 7 object detection and Day 8 CSV logging work. It has a clear user, a measurable test plan, and a simple live demonstration: a person enters the camera view, the program labels the person, updates the count, and records a timestamped detection. The bicep-curl counter was technically interesting, but it depends on MediaPipe, which may not be compatible with the Raspberry Pi's Python 3.13 environment. The color finder is easier to build, but it solves a less useful problem.
