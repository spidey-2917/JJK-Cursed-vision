import cv2
import time

from hand_tracker import HandTracker
from gesture_utils import (
    detect_gesture,
    draw_gesture_effects,
    draw_gesture_label,
    draw_fps,
    draw_idle_hint,
    EffectState,
)

HOLD_FRAMES = 8


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("[INFO] Webcam opened. Press ESC to quit.")

    with HandTracker(max_hands=2, detection_confidence=0.7) as tracker:
        effect_state = EffectState()
        prev_time = time.time()
        gesture_label = None
        gesture_hold = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Failed to read frame.")
                break

            frame = cv2.flip(frame, 1)
            all_landmarks = tracker.find_hands(frame, draw=True)

            current_gesture = None
            hand_center = None

            if all_landmarks:
                landmarks = all_landmarks[0]
                current_gesture = detect_gesture(landmarks)
                pixel_lms = tracker.get_pixel_landmarks(landmarks, frame.shape)
                hand_center = tracker.get_hand_center(pixel_lms)

            if current_gesture:
                gesture_label = current_gesture
                gesture_hold = HOLD_FRAMES
            else:
                if gesture_hold > 0:
                    gesture_hold -= 1
                else:
                    gesture_label = None

            if gesture_label and hand_center:
                draw_gesture_effects(frame, gesture_label, hand_center, effect_state)
                draw_gesture_label(frame, gesture_label, effect_state)
            else:
                if gesture_hold == 0:
                    effect_state.radius = 0
                    effect_state.alpha = 1.0
                draw_idle_hint(frame)

            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now
            draw_fps(frame, fps)

            cv2.putText(
                frame,
                "JJK Gesture System  |  ESC to quit",
                (frame.shape[1] - 340, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (180, 180, 180), 1,
            )

            cv2.imshow("JJK - Anime Hand Gesture Recognition", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            elif key == ord("r"):
                effect_state = EffectState()

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()