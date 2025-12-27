import cv2
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / 'src'))

from hand_tracker import HandTracker
from feature_extractor import HandFeatureExtractor
from gesture_classifier import GestureClassifier
from utils import FPSTracker, UIDrawer, validate_webcam


class GestureRecognitionApp:

    def __init__(
            self,
            model_path: str,
            scaler_path: str,
            metadata_path: str,
            camera_id: int = 0,
            smooth_window: int = 5
    ):

        print("Initializing Gesture Recognition System...")


        if not validate_webcam(camera_id):
            raise RuntimeError(f"Camera {camera_id} not available")

        self.camera_id = camera_id
        self.tracker = HandTracker(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.feature_extractor = HandFeatureExtractor()
        self.classifier = GestureClassifier(
            model_path=model_path,
            scaler_path=scaler_path,
            metadata_path=metadata_path,
            smooth_window=smooth_window
        )


        self.fps_tracker = FPSTracker(buffer_size=30)

        model_info = self.classifier.get_model_info()
        print(f"\nSystem initialized!")
        print(f"   Model: {model_info['model_name']}")
        print(f"   Accuracy: {model_info['accuracy']:.2%}")
        print(f"   Gestures: {', '.join(model_info['gestures'])}")
        print()

    def run(self):
        cap = cv2.VideoCapture(self.camera_id)
        print("HAND GESTURE RECOGNITION")
        print("\nShow your hand and make gestures!")
        print("Press 'q' to quit\n")

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read from camera")
                    break

                # Update FPS
                fps = self.fps_tracker.update(time.time())

                # Flip for mirror view
                frame = cv2.flip(frame, 1)

                # Detect hands
                results, _ = self.tracker.detect_hands(frame)

                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]

                    self.tracker.draw_landmarks(frame, hand_landmarks)

                    features = self.feature_extractor.extract_features(
                        hand_landmarks
                    )

                    # Predict gesture
                    _, gesture_name, confidence = self.classifier.predict(
                        features, use_smoothing=True
                    )

                    frame = UIDrawer.draw_gesture_ui(
                        frame, gesture_name, confidence, fps
                    )

                else:

                    frame = UIDrawer.draw_text(
                        frame,
                        "No hand detected",
                        (20, 50),
                        font_scale=1.0,
                        color=(0, 0, 255),
                        thickness=2
                    )

                    # Reset smoothing when no hand
                    self.classifier.reset_smoothing()

                    frame = UIDrawer.draw_text(
                        frame,
                        f"FPS: {fps:.1f}",
                        (20, 90),
                        font_scale=0.6,
                        color=(255, 255, 255),
                        thickness=2
                    )

                # Display frame
                cv2.imshow('Gesture Recognition', frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n Exiting")
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.tracker.close()

            avg_fps = self.fps_tracker.get_fps()
            print(f"\nSession Statistics:")
            print(f"   Average FPS: {avg_fps:.1f}")
            print("\n Application closed successfully")


def main():

    project_root = Path(__file__).parent.parent
    model_dir = project_root / 'models'

    model_path = model_dir / 'gesture_classifier.pkl'
    scaler_path = model_dir / 'scaler.pkl'
    metadata_path = model_dir / 'model_metadata.json'

    if not model_path.exists():
        print(" Error: Model files not found!")
        print(f"\nPlease ensure these files exist:")
        print(f"  - {model_path}")
        print(f"  - {scaler_path}")
        print(f"  - {metadata_path}")
        print("\nTrain the model first by running notebook 04")
        return

    try:
        app = GestureRecognitionApp(
            model_path=str(model_path),
            scaler_path=str(scaler_path),
            metadata_path=str(metadata_path),
            camera_id=0,
            smooth_window=5
        )

        app.run()

    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()