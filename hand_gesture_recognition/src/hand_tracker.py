import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Tuple


class HandTracker:

    def __init__(
            self,
            static_image_mode: bool = False,
            max_num_hands: int = 1,
            min_detection_confidence: float = 0.7,
            min_tracking_confidence: float = 0.7
    ):

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def detect_hands(self, frame: np.ndarray) -> Tuple[Optional[any], np.ndarray]:

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        return results, rgb_frame

    def draw_landmarks(
            self,
            frame: np.ndarray,
            hand_landmarks,
            draw_connections: bool = True
    ) -> np.ndarray:

        if draw_connections:
            self.mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
        else:
            self.mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                None,
                self.mp_drawing_styles.get_default_hand_landmarks_style()
            )

        return frame

    def get_landmark_array(self, hand_landmarks) -> np.ndarray:

        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.append([landmark.x, landmark.y])

        return np.array(landmarks)

    def close(self):

        self.hands.close()

    def __enter__(self):

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):

        self.close()


if __name__ == "__main__":
    print("Testing Hand Tracker...")
    print("Press 'q' to quit")

    tracker = HandTracker()
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        results, _ = tracker.detect_hands(frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                tracker.draw_landmarks(frame, hand_landmarks)

        cv2.imshow('Hand Tracker Demo', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    tracker.close()