# Shape Maker

![Python](https://img.shields.io/badge/Python-3.11-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Tasks%20API-orange)
![Status](https://img.shields.io/badge/Part%201-Complete-brightgreen)

Webcam hand-tracking experiments with [MediaPipe](https://github.com/google-ai-edge/mediapipe) and OpenCV.

## Parts

1. **Pinch Filter** (`part1_pinch_filter/`) — complete. Tracks the thumb tip and index fingertip
   on both hands. Once all 4 points are visible, connects them into a quad shape and applies a
   selectable visual effect inside it (blur, tints, grayscale, edge detection, pixelate, sepia,
   invert) — switch effects live with number keys `1`–`9`.
2. **Shape Tracker** (coming later) — track where your hands have traveled over time and turn the
   path into shapes.

## Setup

1. **Create and activate a virtual environment:**

   ```
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. **Install dependencies:**

   ```
   pip install -r requirements.txt
   ```

3. **Download the hand landmark model** (used by `part1_pinch_filter`, not tracked in git since
   it's a large binary file):

   ```
   curl -L -o part1_pinch_filter/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
   ```

## Run Part 1 — Pinch Filter

```
python part1_pinch_filter/main.py
```

- Hold up both hands with your thumb and index finger visible on each.
- Once all 4 fingertips are detected, a quad shape connects them with a live effect applied inside.
- Press `1`–`9` to switch effects, `q` to quit.
