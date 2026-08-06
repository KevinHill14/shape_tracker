"""
Part 1: Pinch Filter

For each visible hand, track the thumb tip and index fingertip. Once both
hands are visible (4 points total), connect those 4 points into a quad and
apply a selectable visual effect inside it.

Uses the MediaPipe Tasks API (HandLandmarker) with the hand_landmarker.task
model file (see README.md for how to download it).
"""

import math
import time
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = "part1_pinch_filter/hand_landmarker.task"

# Pairs of landmark indices that form the hand "skeleton" (finger bones).
# The Tasks API doesn't expose this as a constant the way the legacy
# mp.solutions.hands module does, so it's listed out directly here -
# each pair is one bone segment between two of the 21 landmarks.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index finger
    (5, 9), (9, 10), (10, 11), (11, 12),   # middle finger
    (9, 13), (13, 14), (14, 15), (15, 16),  # ring finger
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                # wrist to pinky base
]

EFFECT_NAMES = {
    0: "Invisibility",
    1: "Blur",
    2: "Blue Tint",
    3: "Green Tint",
    4: "Brighten (White)",
    5: "Grayscale",
    6: "Neon Edges",
    7: "Pixelate",
    8: "Sepia",
    9: "Invert",
}


def apply_effect(frame, effect, background=None):
    """Return a full-frame version of `frame` with `effect` applied."""
    if effect == 0:  # invisibility - reveal the captured empty background
        if background is not None and background.shape == frame.shape:
            return background
        return frame

    if effect == 1:
        return cv2.GaussianBlur(frame, (25, 25), 0)

    if effect == 2:  # blue tint
        tint = frame.astype(np.int16)
        tint[:, :, 0] = np.clip(tint[:, :, 0] + 90, 0, 255)  # BGR: blue channel
        return tint.astype(np.uint8)

    if effect == 3:  # green tint
        tint = frame.astype(np.int16)
        tint[:, :, 1] = np.clip(tint[:, :, 1] + 90, 0, 255)  # BGR: green channel
        return tint.astype(np.uint8)

    if effect == 4:  # brighten toward white
        return cv2.convertScaleAbs(frame, alpha=1.0, beta=90)

    if effect == 5:  # grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if effect == 6:  # neon edge outline
        edges = cv2.Canny(frame, 100, 200)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    if effect == 7:  # pixelate
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (max(1, w // 20), max(1, h // 20)), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    if effect == 8:  # sepia
        kernel = np.array(
            [[0.272, 0.534, 0.131],
             [0.349, 0.686, 0.168],
             [0.393, 0.769, 0.189]]
        )
        sepia = cv2.transform(frame, kernel)
        return np.clip(sepia, 0, 255).astype(np.uint8)

    if effect == 9:  # invert
        return cv2.bitwise_not(frame)

    return frame


def main():
    # Point MediaPipe at the downloaded model file, and configure how it
    # should track hands: up to 2 hands, confidence thresholds for
    # detecting/tracking, and VIDEO mode since we're feeding it a live
    # frame-by-frame stream (not a single still image).
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    detector = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    current_effect = 1
    # Snapshot of the empty scene, used by the invisibility effect. Starts
    # unset - captured from the first frame, and recapturable anytime with 'b'.
    background = None
    recapture_background = True

    # Create the window up front and force it fullscreen, so it doesn't
    # require the user to manually maximize it each run.
    cv2.namedWindow("Pinch Filter", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Pinch Filter", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        # Selfie-view: mirror the frame so it feels natural to interact with.
        frame = cv2.flip(frame, 1)

        # Capture a clean (undrawn-on) copy of this frame as the "empty"
        # background for the invisibility effect. Happens once at startup,
        # and again anytime 'b' is pressed (e.g. after stepping out of frame).
        if recapture_background:
            background = frame.copy()
            recapture_background = False

        # MediaPipe expects RGB; OpenCV gives BGR. Wrap the converted frame
        # in MediaPipe's own image container before feeding it to the model.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Run hand detection on this frame. VIDEO mode needs a timestamp
        # that increases every call; wall-clock time works fine here since
        # cap.get(CAP_PROP_POS_MSEC) isn't reliable on live webcams.
        timestamp_ms = int(time.time() * 1000)
        result = detector.detect_for_video(mp_image, timestamp_ms)

        # For each detected hand, grab the thumb tip (landmark 4) and index
        # fingertip (landmark 8), convert their normalized (0-1) coordinates
        # to actual pixel coordinates, and collect all of them together.
        h, w, _ = frame.shape
        points = []
        hands_pixel_landmarks = []
        for hand_landmarks in result.hand_landmarks:
            thumb_tip = hand_landmarks[4]
            pointer_finger_tip = hand_landmarks[8]
            thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
            pointer_x, pointer_y = int(pointer_finger_tip.x * w), int(pointer_finger_tip.y * h)
            points.append((thumb_x, thumb_y))
            points.append((pointer_x, pointer_y))

            # All 21 landmarks for this hand, converted to pixel coords, so
            # we can draw the full finger skeleton further down.
            hands_pixel_landmarks.append(
                [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
            )

        # Only draw once both hands are visible (4 points: thumb+index x2).
        if len(points) == 4:
            # Find the centroid (average point) of the 4 fingertips, then
            # sort the points by their angle around that centroid. This
            # walks the points around the shape in order, instead of the
            # order they happened to be collected in, which avoids drawing
            # a self-intersecting "bowtie" outline.
            cx = sum(p[0] for p in points) / len(points)
            cy = sum(p[1] for p in points) / len(points)
            ordered = sorted(points, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
            poly_points = np.array(ordered, dtype=np.int32)

            # Build a mask of the quad shape so the effect only applies
            # inside it, then draw the outline and blend the effect in.
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [poly_points], 255)
            cv2.polylines(frame, [poly_points], isClosed=True, color=(0, 255, 0), thickness=2)
            effect_frame = apply_effect(frame, current_effect, background)
            frame = np.where(mask[:, :, None] == 255, effect_frame, frame)

        # Draw each hand's finger skeleton (joints as dots, bones as lines)
        # on top of everything else so it stays visible even inside the
        # effect region.
        for landmarks in hands_pixel_landmarks:
            for start_idx, end_idx in HAND_CONNECTIONS:
                cv2.line(frame, landmarks[start_idx], landmarks[end_idx], (255, 255, 255), 1)
            for point in landmarks:
                cv2.circle(frame, point, 4, (0, 255, 255), -1)

        # Show which effect is active and how to change it.
        cv2.putText(
            frame,
            f"[{current_effect}] {EFFECT_NAMES[current_effect]}  (0-9 effects, b = recapture background)",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Pinch Filter", frame)
        key = cv2.waitKey(5) & 0xFF
        if key == ord("q"):
            break
        elif ord("0") <= key <= ord("9"):
            current_effect = key - ord("0")
        elif key == ord("b"):
            recapture_background = True

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
