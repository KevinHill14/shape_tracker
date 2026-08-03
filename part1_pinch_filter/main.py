"""
Part 1: Pinch Filter

Goal: for each visible hand, find the thumb tip and index fingertip,
draw a rectangle between them, and apply a filter inside that rectangle.

This file has the OpenCV webcam boilerplate already wired up.
The MediaPipe-specific pieces are left as TODOs for us to fill in together.
"""

import cv2
import mediapipe as mp

def main():
    cap = cv2.VideoCapture(0)

    # TODO 1: create a MediaPipe Hands object here.
    # Look at mp.solutions.hands.Hands(...) — what parameters does it take,
    # and what do max_num_hands / min_detection_confidence control?

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        # Selfie-view: mirror the frame so it feels natural to interact with.
        frame = cv2.flip(frame, 1)

        # TODO 2: MediaPipe expects RGB images, OpenCV gives you BGR.
        # Convert the frame before running it through Hands.

        # TODO 3: run the Hands model on the frame and get the result.

        # TODO 4: for each detected hand, pull out the landmark for the
        # thumb tip and the landmark for the index fingertip.
        # (What are their landmark indices? Check the MediaPipe hand
        # landmark diagram.)

        # TODO 5: convert those normalized landmark coordinates (0-1) into
        # pixel coordinates using the frame's width/height.

        # TODO 6: draw a rectangle between the two points, and apply a
        # filter (e.g. blur, color shift) to the region inside it.

        cv2.imshow("Pinch Filter", frame)
        if cv2.waitKey(5) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
