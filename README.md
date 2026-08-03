# Shape Maker

Webcam hand-tracking experiments with [MediaPipe](https://github.com/google-ai-edge/mediapipe) and OpenCV.

## Parts

1. **Pinch Filter** (`part1_pinch_filter/`) — Track the thumb and index finger on both hands,
   draw a rectangle between each pinch pair, and apply an image filter inside that rectangle.
2. **Shape Tracker** (coming later) — Track where your hands have traveled over time and turn the
   path into shapes.

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run Part 1

```
python part1_pinch_filter/main.py
```
