import cv2
import sys
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from hand_tracker import HandTracker
from feature_extractor import HandFeatureExtractor
from gesture_classifier import GestureClassifier
from utils import FPSTracker, UIDrawer, validate_webcam


class ISLRecognitionApp:

    def __init__(self, model_path, scaler_path, metadata_path, camera_id=0, smooth_window=7):
        """Initialize ISL recognition app"""
        print(" Initializing Indian Sign Language Recognition System...")

        if not validate_webcam(camera_id):
            raise RuntimeError(f"Camera {camera_id} not available")

        self.camera_id = camera_id
        self.tracker = HandTracker(static_image_mode=False,
                                   max_num_hands=1,
                                   min_detection_confidence=0.7,
                                   min_tracking_confidence=0.7)
        self.feature_extractor = HandFeatureExtractor()
        self.classifier = GestureClassifier(model_path=model_path,
                                            scaler_path=scaler_path,
                                            metadata_path=metadata_path,
                                            smooth_window=smooth_window)
        self.fps_tracker = FPSTracker(buffer_size=30)
        self.sentence = []
        self.last_gesture = None
        self.gesture_hold_time = 0
        self.gesture_hold_threshold = 1.5  # seconds

        model_info = self.classifier.get_model_info()
        print(f"\nSystem initialized!")
        print(f"   Model: {model_info['model_name']}")
        print(f"   Accuracy: {model_info['accuracy']:.2%}")
        print(f"   Classes: {len(model_info['gestures'])} ISL signs")
        print(f"   Signs: {', '.join(model_info['gestures'][:10])}...\n")

    def draw_isl_ui(self, frame, gesture_name, confidence, fps, sentence):
        """Draw ISL-specific UI"""
        h, w, _ = frame.shape
        UIDrawer.draw_panel(frame, (10, 10), (w - 20, 180), opacity=0.6)

        UIDrawer.draw_text(frame, f"Sign: {gesture_name.upper()}", (20, 50), font_scale=1.5, color=(0, 255, 255),
                           thickness=3)
        if confidence is not None:
            UIDrawer.draw_text(frame, f"Confidence: {confidence:.1%}", (20, 95), font_scale=0.7, color=(255, 255, 255),
                               thickness=2)
            UIDrawer.draw_progress_bar(frame, (20, 105), 300, 20, confidence)

        hold_progress = min(self.gesture_hold_time / self.gesture_hold_threshold, 1.0)
        if hold_progress > 0:
            UIDrawer.draw_text(frame, f"Hold: {self.gesture_hold_time:.1f}s", (20, 145), font_scale=0.6,
                               color=(255, 255, 0), thickness=2)
            UIDrawer.draw_progress_bar(frame, (20, 155), 150, 15, hold_progress, fg_color=(255, 255, 0))

        UIDrawer.draw_text(frame, f"FPS: {fps:.1f}", (w - 150, 50), font_scale=0.7, color=(255, 255, 255), thickness=2)

        if sentence:
            sentence_text = " ".join(sentence)
            UIDrawer.draw_panel(frame, (10, h - 100), (w - 20, 90), opacity=0.7)
            UIDrawer.draw_text(frame, "Sentence:", (20, h - 75), font_scale=0.6, color=(200, 200, 200), thickness=1)
            max_chars = int((w - 40) / 12)
            if len(sentence_text) > max_chars:
                sentence_text = sentence_text[-max_chars:]
            UIDrawer.draw_text(frame, sentence_text, (20, h - 40), font_scale=0.8, color=(0, 255, 0), thickness=2)

        UIDrawer.draw_text(frame, "SPACE: Add | BACKSPACE: Delete | C: Clear | Q: Quit", (20, h - 15), font_scale=0.5,
                           color=(180, 180, 180), thickness=1)
        return frame

    def run(self):
        """Run real-time ISL recognition"""
        cap = cv2.VideoCapture(self.camera_id)
        print("=" * 70)
        print(" INDIAN SIGN LANGUAGE RECOGNITION")
        print("=" * 70)

        prev_time = time.time()

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read from camera")
                    break

                curr_time = time.time()
                delta_time = curr_time - prev_time
                fps = self.fps_tracker.update(curr_time)
                prev_time = curr_time
                frame = cv2.flip(frame, 1)

                results, _ = self.tracker.detect_hands(frame)

                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    self.tracker.draw_landmarks(frame, hand_landmarks)
                    features = self.feature_extractor.extract_features(hand_landmarks)
                    _, gesture_name, confidence = self.classifier.predict(features, use_smoothing=True)

                    if gesture_name == self.last_gesture:
                        self.gesture_hold_time += delta_time
                    else:
                        self.gesture_hold_time = 0
                        self.last_gesture = gesture_name

                    frame = self.draw_isl_ui(frame, gesture_name, confidence, fps, self.sentence)

                else:
                    UIDrawer.draw_text(frame, "No hand detected", (20, 50), font_scale=1.0, color=(0, 0, 255),
                                       thickness=2)
                    self.classifier.reset_smoothing()
                    self.last_gesture = None
                    self.gesture_hold_time = 0
                    if self.sentence:
                        h = frame.shape[0]
                        sentence_text = " ".join(self.sentence)
                        UIDrawer.draw_panel(frame, (10, h - 100), (frame.shape[1] - 20, 90), opacity=0.7)
                        UIDrawer.draw_text(frame, f"Sentence: {sentence_text}", (20, h - 40), font_scale=0.8,
                                           color=(0, 255, 0), thickness=2)
                    UIDrawer.draw_text(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 150, 50), font_scale=0.7,
                                       color=(255, 255, 255), thickness=2)

                cv2.imshow('ISL Recognition', frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord(' ') and self.last_gesture:
                    self.sentence.append(self.last_gesture)
                    self.gesture_hold_time = 0
                    print(f"Added: {self.last_gesture} → Sentence: {' '.join(self.sentence)}")
                elif key == 8 and self.sentence:
                    removed = self.sentence.pop()
                    print(f"Removed: {removed} → Sentence: {' '.join(self.sentence)}")
                elif key in [ord('c'), ord('C')]:
                    self.sentence.clear()
                    print("Sentence cleared")
                elif key in [ord('q'), ord('Q')]:
                    print("\n Exiting...")
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.tracker.close()
            avg_fps = self.fps_tracker.get_fps()
            print(f"\nSession Statistics:\n   Average FPS: {avg_fps:.1f}")
            if self.sentence:
                print(f"   Final sentence: {' '.join(self.sentence)}")
            print("\nApplication closed successfully")


def main():
    """Main entry point"""
    project_root = Path(__file__).parent.parent
    model_dir = project_root / 'models'

    model_path = model_dir / 'isl_gesture_classifier.pkl'
    scaler_path = model_dir / 'isl_scaler.pkl'
    metadata_path = model_dir / 'isl_model_metadata.json'

    if not model_path.exists():
        print("Error: ISL model files not found!")
        print(f"  Ensure these files exist:\n    - {model_path}\n    - {scaler_path}\n    - {metadata_path}")
        return

    try:
        app = ISLRecognitionApp(
            model_path=str(model_path),
            scaler_path=str(scaler_path),
            metadata_path=str(metadata_path),
            camera_id=0,
            smooth_window=7
        )
        app.run()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()