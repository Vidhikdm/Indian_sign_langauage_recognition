import numpy as np
from typing import List

class HandFeatureExtractor:

    def __init__(self):
        # Landmark indices for each finger
        self.finger_tips = [4, 8, 12, 16, 20]
        self.finger_pips = [2, 6, 10, 14, 18]
        self.finger_mcps = [1, 5, 9, 13, 17]
        self.wrist_idx = 0

    def extract_features(self, hand_landmarks) -> np.ndarray:
        # Extract landmark coordinates
        landmarks = self._get_landmark_array(hand_landmarks)

        # Feature components
        normalized = self._normalize_landmarks(landmarks)
        distances = self._calculate_distances(landmarks)
        angles = self._calculate_angles(landmarks)
        finger_states = self._get_finger_states(landmarks)

        # 42 features (21 landmarks × 2 coords)
        features = np.concatenate([
            normalized.flatten(),
            distances,
            angles,
            finger_states
        ])

        return features

    def _get_landmark_array(self, hand_landmarks) -> np.ndarray:

        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.append([lm.x, lm.y])
        return np.array(landmarks)

    def _normalize_landmarks(self, landmarks: np.ndarray) -> np.ndarray:

        wrist = landmarks[self.wrist_idx]
        normalized = landmarks - wrist

        # Scale to unit size
        hand_size = np.max(np.linalg.norm(normalized, axis=1))
        if hand_size > 0:
            normalized = normalized / hand_size

        return normalized

    def _calculate_distances(self, landmarks: np.ndarray) -> np.ndarray:
        # Calculate Euclidean distances from fingertips to wrist
        wrist = landmarks[self.wrist_idx]
        distances = []

        for tip_idx in self.finger_tips:
            tip = landmarks[tip_idx]
            dist = np.linalg.norm(tip - wrist)
            distances.append(dist)

        return np.array(distances)

    def _calculate_angles(self, landmarks: np.ndarray) -> np.ndarray:

        angles = []

        for tip_idx, pip_idx, mcp_idx in zip(
                self.finger_tips, self.finger_pips, self.finger_mcps
        ):

            v1 = landmarks[pip_idx] - landmarks[mcp_idx]
            v2 = landmarks[tip_idx] - landmarks[pip_idx]

            angle = self._angle_between_vectors(v1, v2)
            angles.append(angle)

        return np.array(angles)

    def _angle_between_vectors(self, v1: np.ndarray, v2: np.ndarray) -> float:
        # Normalize vectors
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)

        if v1_norm == 0 or v2_norm == 0:
            return 0.0

        cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)

        angle = np.arccos(cos_angle)
        return np.degrees(angle)

    def _get_finger_states(self, landmarks: np.ndarray) -> np.ndarray:

        wrist = landmarks[self.wrist_idx]
        states = []

        for tip_idx, pip_idx in zip(self.finger_tips, self.finger_pips):
            tip = landmarks[tip_idx]
            pip = landmarks[pip_idx]

            tip_dist = np.linalg.norm(tip - wrist)
            pip_dist = np.linalg.norm(pip - wrist)

            is_extended = 1 if tip_dist > pip_dist * 1.1 else 0
            states.append(is_extended)

        return np.array(states)

    def get_finger_names(self) -> List[str]:
        return ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']

    def get_feature_dimension(self) -> int:
        return 57  # 42 + 5 + 5 + 5


if __name__ == "__main__":
    print("Feature Extractor Demo")
    print(f"Feature dimension: {HandFeatureExtractor().get_feature_dimension()}")
    print("\nFeature breakdown:")
    print("  - Normalized coordinates: 42")
    print("  - Fingertip distances: 5")
    print("  - Finger angles: 5")
    print("  - Finger states: 5")
    print("  - Total: 57")