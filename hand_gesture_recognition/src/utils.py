import cv2
import numpy as np
from collections import deque
from typing import Optional, Tuple


class FPSTracker:

    def __init__(self, buffer_size: int = 30):
        self.buffer = deque(maxlen=buffer_size)
        self.prev_time = None

    def update(self, current_time: float) -> float:

        if self.prev_time is not None:
            fps = 1.0 / (current_time - self.prev_time + 1e-6)
            self.buffer.append(fps)

        self.prev_time = current_time
        return self.get_fps()

    def get_fps(self) -> float:
        if len(self.buffer) == 0:
            return 0.0
        return sum(self.buffer) / len(self.buffer)

    def reset(self):
        self.buffer.clear()
        self.prev_time = None


class UIDrawer:

    @staticmethod
    def draw_text(
        frame: np.ndarray,
        text: str,
        position: Tuple[int, int],
        font_scale: float = 1.0,
        color: Tuple[int, int, int] = (255, 255, 255),
        thickness: int = 2,
        bg_color: Optional[Tuple[int, int, int]] = None,
        padding: int = 5
    ) -> np.ndarray:

        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        x, y = position

        if bg_color is not None:
            cv2.rectangle(
                frame,
                (x - padding, y - text_height - padding),
                (x + text_width + padding, y + baseline + padding),
                bg_color,
                -1
            )

        cv2.putText(frame, text, position, font, font_scale, color, thickness)
        return frame

    @staticmethod
    def draw_panel(
        frame: np.ndarray,
        position: Tuple[int, int],
        size: Tuple[int, int],
        color: Tuple[int, int, int] = (0, 0, 0),
        opacity: float = 0.6
    ) -> np.ndarray:
        """Draw semi-transparent panel on the frame."""
        x, y = position
        w, h = size
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
        cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)
        return frame

    @staticmethod
    def draw_progress_bar(
        frame: np.ndarray,
        position: Tuple[int, int],
        width: int,
        height: int,
        progress: float,
        bg_color: Tuple[int, int, int] = (50, 50, 50),
        fg_color: Tuple[int, int, int] = (0, 255, 0)
    ) -> np.ndarray:
        x, y = position
        progress = np.clip(progress, 0.0, 1.0)

        cv2.rectangle(frame, (x, y), (x + width, y + height), bg_color, -1)

        progress_width = int(width * progress)
        if progress_width > 0:
            cv2.rectangle(frame, (x, y), (x + progress_width, y + height), fg_color, -1)

        return frame

    @staticmethod
    def draw_gesture_ui(
        frame: np.ndarray,
        gesture_name: str,
        confidence: Optional[float] = None,
        fps: Optional[float] = None
    ) -> np.ndarray:
        h, w, _ = frame.shape
        frame = UIDrawer.draw_panel(frame, (10, 10), (w - 20, 140))

        frame = UIDrawer.draw_text(
            frame,
            f"Gesture: {gesture_name.upper()}",
            (20, 50),
            font_scale=1.2,
            color=(0, 255, 255),
            thickness=3
        )


        if confidence is not None:
            frame = UIDrawer.draw_text(
                frame,
                f"Confidence: {confidence:.1%}",
                (20, 90),
                font_scale=0.7,
                color=(255, 255, 255),
                thickness=2
            )
            frame = UIDrawer.draw_progress_bar(
                frame,
                (20, 100),
                300,
                20,
                confidence
            )

        # FPS
        if fps is not None:
            frame = UIDrawer.draw_text(
                frame,
                f"FPS: {fps:.1f}",
                (20, 140),
                font_scale=0.6,
                color=(255, 255, 255),
                thickness=2
            )


        frame = UIDrawer.draw_text(
            frame,
            "Press 'q' to quit",
            (w - 200, h - 20),
            font_scale=0.5,
            color=(200, 200, 200),
            thickness=1
        )

        return frame


def validate_webcam(camera_id: int = 0) -> bool:
    """Check if the webcam is available."""
    cap = cv2.VideoCapture(camera_id)
    is_available = cap.isOpened()
    cap.release()
    return is_available


#DEMO

if __name__ == "__main__":
    import time

    print("Utilities Demo")

    # Check webcam
    print(f"Webcam available: {validate_webcam()}")

    # Test FPS tracker
    fps_tracker = FPSTracker()
    print("\nTesting FPS tracker...")
    for _ in range(10):
        time.sleep(0.033)  # simulate ~30 FPS
        fps_tracker.update(time.time())
    print(f"Average FPS: {fps_tracker.get_fps():.1f}")

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame_with_ui = UIDrawer.draw_gesture_ui(
        dummy_frame,
        gesture_name="Open Hand",
        confidence=0.87,
        fps=fps_tracker.get_fps()
    )
    cv2.imshow("UI Demo", frame_with_ui)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()

    print("\nAll utilities are working!")