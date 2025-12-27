import json
from collections import deque
from pathlib import Path
from typing import Tuple, Dict, Optional

import joblib
import numpy as np


class GestureClassifier:

    def __init__(
            self,
            model_path: str,
            scaler_path: str,
            metadata_path: str,
            smooth_window: int = 5
    ):

        self.model = self._load_model(model_path)
        self.scaler = self._load_scaler(scaler_path)
        self.metadata = self._load_metadata(metadata_path)


        self.gesture_classes = {
            int(k): v for k, v in self.metadata['gesture_classes'].items()
        }

        self.smooth_window = smooth_window
        self.prediction_queue = deque(maxlen=smooth_window)

    def _load_model(self, model_path: str):

        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        return joblib.load(model_path)

    def _load_scaler(self, scaler_path: str):
        if not Path(scaler_path).exists():
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")
        return joblib.load(scaler_path)

    def _load_metadata(self, metadata_path: str) -> Dict:
        if not Path(metadata_path).exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        with open(metadata_path, 'r') as f:
            return json.load(f)

    def predict(
            self,
            features: np.ndarray,
            use_smoothing: bool = True
    ) -> Tuple[int, str, Optional[float]]:

        if features.shape[0] != self.metadata['num_features']:
            raise ValueError(
                f"Expected {self.metadata['num_features']} features, "
                f"got {features.shape[0]}"
            )

        features_scaled = self.scaler.transform(features.reshape(1, -1))


        prediction = self.model.predict(features_scaled)[0]


        if use_smoothing:
            prediction = self._smooth_prediction(prediction)

        gesture_name = self.gesture_classes[prediction]

        confidence = None
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(features_scaled)[0]
            confidence = float(probabilities[prediction])

        return prediction, gesture_name, confidence

    def _smooth_prediction(self, prediction: int) -> int:

        self.prediction_queue.append(prediction)

        if len(self.prediction_queue) > 0:
            # Return most common prediction
            return max(set(self.prediction_queue),
                       key=self.prediction_queue.count)

        return prediction

    def reset_smoothing(self):
        self.prediction_queue.clear()

    def get_model_info(self) -> Dict:
        return {
            'model_name': self.metadata['model_name'],
            'accuracy': self.metadata['accuracy'],
            'f1_score': self.metadata['f1_score'],
            'num_features': self.metadata['num_features'],
            'gestures': list(self.gesture_classes.values()),
            'trained_date': self.metadata['trained_date']
        }

    def get_all_probabilities(self, features: np.ndarray) -> Dict[str, float]:

        if not hasattr(self.model, 'predict_proba'):
            return {}

        features_scaled = self.scaler.transform(features.reshape(1, -1))
        probs = self.model.predict_proba(features_scaled)[0]

        return {
            self.gesture_classes[i]: float(prob)
            for i, prob in enumerate(probs)
        }


if __name__ == "__main__":
    import os

    print("Gesture Classifier Demo")

    # Check if models exist
    model_dir = Path(__file__).parent.parent / 'models'
    model_path = model_dir / 'gesture_classifier.pkl'
    scaler_path = model_dir / 'scaler.pkl'
    metadata_path = model_dir / 'model_metadata.json'

    if not model_path.exists():
        print(f"  Model not found: {model_path}")
        print("Please train the model first (run notebook 04)")
    else:
        try:
            classifier = GestureClassifier(
                str(model_path),
                str(scaler_path),
                str(metadata_path)
            )

            print("\n Classifier loaded successfully!")
            print("\nModel Info:")
            info = classifier.get_model_info()
            for key, value in info.items():
                print(f"  {key}: {value}")

        except Exception as e:
            print(f" Error loading classifier: {e}")