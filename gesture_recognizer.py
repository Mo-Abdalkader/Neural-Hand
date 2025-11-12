"""
Advanced Gesture Control System - Gesture Recognizer Module
Analyzes hand landmarks to classify and recognize gestures.

This module provides sophisticated gesture recognition using distance-based
detection, finger state analysis, and confidence scoring.
"""

import numpy as np
import logging
import time
from typing import Optional, List, Tuple, Dict
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)


class GestureType(Enum):
    """Enumeration of supported gesture types."""
    NONE = "none"
    INDEX_EXTENDED = "index_extended"
    THUMB_INDEX_PINCH = "thumb_index_pinch"
    THUMB_MIDDLE_PINCH = "thumb_middle_pinch"
    TWO_FINGER_SCROLL = "two_finger_scroll"
    THREE_FINGER_CLOSE = "three_finger_close"
    THREE_FINGER_SPREAD = "three_finger_spread"
    CLOSED_FIST = "closed_fist"
    OPEN_PALM = "open_palm"
    THUMB_DOWN = "thumb_down"
    THUMB_PINKY_EXTENDED = "thumb_pinky_extended"


class GestureRecognizer:
    """
    Advanced gesture recognition from hand landmarks.

    Analyzes hand landmark positions to classify gestures with confidence scores,
    temporal consistency, and anti-jitter mechanisms.
    """

    def __init__(self):
        """Initialize gesture recognizer with default settings."""
        # Detection thresholds (configurable)
        self.pinch_threshold = 0.05  # Distance for pinch detection
        self.finger_extended_threshold = 0.6  # Ratio for finger extension
        self.hold_time = 0.3  # Seconds to hold gesture for activation
        self.cooldown_time = 0.5  # Seconds between gesture activations

        # Gesture state management
        self.current_gesture = GestureType.NONE
        self.gesture_start_time: Optional[float] = None
        self.last_activation_time: Dict[GestureType, float] = {}

        # Gesture history for smoothing
        self.gesture_history = deque(maxlen=5)

        # Landmark indices for easy reference
        self.WRIST = 0
        self.THUMB_TIP = 4
        self.INDEX_TIP = 8
        self.MIDDLE_TIP = 12
        self.RING_TIP = 16
        self.PINKY_TIP = 20

        self.INDEX_MCP = 5
        self.MIDDLE_MCP = 9
        self.RING_MCP = 13
        self.PINKY_MCP = 17

        logger.info("GestureRecognizer initialized")

    def recognize_gesture(
            self,
            landmarks: List[Tuple[float, float, float]]
    ) -> Tuple[GestureType, float]:
        """
        Recognize gesture from hand landmarks.

        Args:
            landmarks: List of 21 hand landmarks as (x, y, z) tuples

        Returns:
            tuple: (GestureType, confidence_score)
        """
        if not landmarks or len(landmarks) != 21:
            return GestureType.NONE, 0.0

        try:
            # Convert to numpy array for easier manipulation
            landmarks_array = np.array(landmarks)

            # Detect all gesture types with confidence scores
            gestures = {
                GestureType.THUMB_INDEX_PINCH: self._detect_pinch(
                    landmarks_array, self.THUMB_TIP, self.INDEX_TIP
                ),
                GestureType.THUMB_MIDDLE_PINCH: self._detect_pinch(
                    landmarks_array, self.THUMB_TIP, self.MIDDLE_TIP
                ),
                GestureType.CLOSED_FIST: self._detect_closed_fist(landmarks_array),
                GestureType.OPEN_PALM: self._detect_open_palm(landmarks_array),
                GestureType.TWO_FINGER_SCROLL: self._detect_two_finger_scroll(landmarks_array),
                GestureType.THREE_FINGER_CLOSE: self._detect_three_finger_close(landmarks_array),
                GestureType.THREE_FINGER_SPREAD: self._detect_three_finger_spread(landmarks_array),
                GestureType.THUMB_DOWN: self._detect_thumb_down(landmarks_array),
                GestureType.THUMB_PINKY_EXTENDED: self._detect_thumb_pinky_extended(landmarks_array),
                GestureType.INDEX_EXTENDED: self._detect_index_extended(landmarks_array),
            }

            # Find gesture with highest confidence above threshold
            best_gesture = max(gestures.items(), key=lambda x: x[1])
            gesture_type, confidence = best_gesture

            # Apply confidence threshold
            if confidence < 0.5:
                gesture_type = GestureType.NONE
                confidence = 0.0

            # Add to history for smoothing
            self.gesture_history.append((gesture_type, confidence))

            # Get smoothed gesture (most common in recent history)
            smoothed_gesture = self._get_smoothed_gesture()

            return smoothed_gesture

        except Exception as e:
            logger.error(f"Error recognizing gesture: {e}")
            return GestureType.NONE, 0.0

    def _detect_pinch(
            self,
            landmarks: np.ndarray,
            finger1_idx: int,
            finger2_idx: int
    ) -> float:
        """
        Detect pinch gesture between two fingers.

        Args:
            landmarks: Array of landmark positions
            finger1_idx: Index of first finger tip
            finger2_idx: Index of second finger tip

        Returns:
            float: Confidence score (0-1)
        """
        try:
            pos1 = landmarks[finger1_idx][:2]  # Only x, y coordinates
            pos2 = landmarks[finger2_idx][:2]

            distance = np.linalg.norm(pos1 - pos2)

            # Confidence inversely proportional to distance
            if distance < self.pinch_threshold:
                confidence = 1.0 - (distance / self.pinch_threshold)
                return max(0.0, min(1.0, confidence))

            return 0.0

        except Exception as e:
            logger.error(f"Error detecting pinch: {e}")
            return 0.0

    def _is_finger_extended(
            self,
            landmarks: np.ndarray,
            tip_idx: int,
            mcp_idx: int
    ) -> bool:
        """
        Check if a finger is extended based on tip and MCP positions.

        Args:
            landmarks: Array of landmark positions
            tip_idx: Index of finger tip
            mcp_idx: Index of finger MCP (metacarpophalangeal) joint

        Returns:
            bool: True if finger is extended
        """
        try:
            tip = landmarks[tip_idx]
            mcp = landmarks[mcp_idx]
            wrist = landmarks[self.WRIST]

            # Calculate distances
            tip_to_wrist = np.linalg.norm(tip[:2] - wrist[:2])
            mcp_to_wrist = np.linalg.norm(mcp[:2] - wrist[:2])

            # Finger is extended if tip is significantly farther from wrist than MCP
            ratio = tip_to_wrist / (mcp_to_wrist + 1e-6)
            # Ensure the comparison result is a single boolean value
            return bool(ratio > self.finger_extended_threshold)

        except Exception as e:
            logger.error(f"Error checking finger extension: {e}")
            return False

    def _detect_closed_fist(self, landmarks: np.ndarray) -> float:
        """
        Detect closed fist gesture (all fingers closed).

        Args:
            landmarks: Array of landmark positions

        Returns:
            float: Confidence score (0-1)
        """
        try:
            # Check if all fingers are NOT extended
            fingers_extended = [
                self._is_finger_extended(landmarks, self.INDEX_TIP, self.INDEX_MCP),
                self._is_finger_extended(landmarks, self.MIDDLE_TIP, self.MIDDLE_MCP),
                self._is_finger_extended(landmarks, self.RING_TIP, self.RING_MCP),
                self._is_finger_extended(landmarks, self.PINKY_TIP, self.PINKY_MCP),
            ]

            # Confidence based on how many fingers are closed
            closed_count = sum(not extended for extended in fingers_extended)
            confidence = closed_count / 4.0

            # High confidence only if all fingers closed
            return confidence if confidence > 0.75 else 0.0

        except Exception as e:
            logger.error(f"Error detecting closed fist: {e}")
            return 0.0

    def _detect_open_palm(self, landmarks: np.ndarray) -> float:
        """
        Detect open palm gesture (all fingers extended).

        Args:
            landmarks: Array of landmark positions

        Returns:
            float: Confidence score (0-1)
        """
        try:
            # Check if all fingers are extended
            fingers_extended = [
                self._is_finger_extended(landmarks, self.INDEX_TIP, self.INDEX_MCP),
                self._is_finger_extended(landmarks, self.MIDDLE_TIP, self.MIDDLE_MCP),
                self._is_finger_extended(landmarks, self.RING_TIP, self.RING_MCP),
                self._is_finger_extended(landmarks, self.PINKY_TIP, self.PINKY_MCP),
            ]

            # Confidence based on how many fingers are extended
            extended_count = sum(fingers_extended)
            confidence = extended_count / 4.0

            # High confidence only if all fingers extended
            return confidence if confidence > 0.75 else 0.0

        except Exception as e:
            logger.error(f"Error detecting open palm: {e}")
            return 0.0

    def _detect_index_extended(self, landmarks: np.ndarray) -> float:
        """
        Detect index finger extended for cursor control.

        Args:
            landmarks: Array of landmark positions

        Returns:
            float: Confidence score (0-1)
        """
        try:
            index_extended = self._is_finger_extended(landmarks, self.INDEX_TIP, self.INDEX_MCP)
            middle_extended = self._is_finger_extended(landmarks, self.MIDDLE_TIP, self.MIDDLE_MCP)
            ring_extended = self._is_finger_extended(landmarks, self.RING_TIP, self.RING_MCP)
            pinky_extended = self._is_finger_extended(landmarks, self.PINKY_TIP, self.PINKY_MCP)

            # Index extended, others closed
            if index_extended and not middle_extended and not ring_extended and not pinky_extended:
                return 0.9

            # Lower confidence if multiple fingers extended
            if index_extended:
                return 0.5

            return 0.0

        except Exception as e:
            logger.error(f"Error detecting index extended: {e}")
            return 0.0

    def _detect_two_finger_scroll(self, landmarks: np.ndarray) -> float:
        """
        Detect two-finger scroll gesture (index and middle extended).

        Args:
            landmarks: Array of landmark positions

        Returns:
            float: Confidence score (0-1)
        """
        try:
            index_extended = self._is_finger_extended(landmarks, self.INDEX_TIP, self.INDEX_MCP)
            middle_extended = self._is_finger_extended(landmarks, self.MIDDLE_TIP, self.MIDDLE_MCP)
            ring_extended = self._is_finger_extended(landmarks, self.RING_TIP, self.RING_MCP)
            pinky_extended = self._is_finger_extended(landmarks, self.PINKY_TIP, self.PINKY_MCP)

            # Index and middle extended, others closed
            if index_extended and middle_extended and not ring_extended and not pinky_extended:
                return 0.9

            return 0.0

        except Exception as e:
            logger.error(f"Error detecting two-finger scroll: {e}")
            return 0.0

    def _detect_three_finger_close(self, landmarks: np.ndarray) -> float:
        """
        Detect three fingers close together for drag start.

        Args:
            landmarks: Array of landmark positions

        Returns:
            float: Confidence score (0-1)
        """
        try:
            # Check if index, middle, and ring tips are close together
            index_pos = landmarks[self.INDEX_TIP][:2]
            middle_pos = landmarks[self.MIDDLE_TIP][:2]
            ring_pos = landmarks[self.RING_TIP][:2]

            dist_index_middle = np.linalg.norm(index_pos - middle_pos)
            dist_middle_ring = np.linalg.norm(middle_pos - ring_pos)

            threshold = 0.08
            if dist_index_middle < threshold and dist_middle_ring < threshold:
                confidence = 1.0 - (dist_index_middle + dist_middle_ring) / (2 * threshold)
                return max(0.0, min(1.0, confidence))

            return 0.0

        except Exception as e:
            logger.error(f"Error detecting three-finger close: {e}")
            return 0.0

    def _detect_three_finger_spread(self, landmarks: np.ndarray) -> float:
        """
        Detect three fingers spread apart for drag end.

        Args:
            landmarks: Array of landmark positions

        Returns:
            float: Confidence score (0-1)
        """
        try:
            # Check if index, middle, and ring are extended and spread
            index_extended = self._is_finger_extended(landmarks, self.INDEX_TIP, self.INDEX_MCP)
            middle_extended = self._is_finger_extended(landmarks, self.MIDDLE_TIP, self.MIDDLE_MCP)
            ring_extended = self._is_finger_extended(landmarks, self.RING_TIP, self.RING_MCP)

            if not (index_extended and middle_extended and ring_extended):
                return 0.0

            # Check spread distance
            index_pos = landmarks[self.INDEX_TIP][:2]
            middle_pos = landmarks[self.MIDDLE_TIP][:2]
            ring_pos = landmarks[self.RING_TIP][:2]

            dist_index_middle = np.linalg.norm(index_pos - middle_pos)
            dist_middle_ring = np.linalg.norm(middle_pos - ring_pos)

            # Fingers should be reasonably spread apart
            min_spread = 0.1
            if dist_index_middle > min_spread and dist_middle_ring > min_spread:
                return 0.8

            return 0.0

        except Exception as e:
            logger.error(f"Error detecting three-finger spread: {e}")
            return 0.0

    def _detect_thumb_down(self, landmarks: np.ndarray) -> float:
        """
        Detect thumb pointing down gesture.

        Args:
            landmarks: Array of landmark positions

        Returns:
            float: Confidence score (0-1)
        """
        try:
            thumb_tip = landmarks[self.THUMB_TIP]
            wrist = landmarks[self.WRIST]

            # Thumb tip should be below wrist
            if thumb_tip[1] > wrist[1]:  # Y increases downward
                # Calculate confidence based on how far below
                distance = thumb_tip[1] - wrist[1]
                confidence = min(1.0, distance / 0.3)
                return confidence

            return 0.0

        except Exception as e:
            logger.error(f"Error detecting thumb down: {e}")
            return 0.0

    def _detect_thumb_pinky_extended(self, landmarks: np.ndarray) -> float:
        """
        Detect thumb and pinky extended for volume control.

        Args:
            landmarks: Array of landmark positions

        Returns:
            float: Confidence score (0-1)
        """
        try:
            # Check thumb and pinky distances from wrist
            thumb_tip = landmarks[self.THUMB_TIP]
            pinky_tip = landmarks[self.PINKY_TIP]
            wrist = landmarks[self.WRIST]

            thumb_dist = np.linalg.norm(thumb_tip[:2] - wrist[:2])
            pinky_dist = np.linalg.norm(pinky_tip[:2] - wrist[:2])

            # Check other fingers are closed
            index_extended = self._is_finger_extended(landmarks, self.INDEX_TIP, self.INDEX_MCP)
            middle_extended = self._is_finger_extended(landmarks, self.MIDDLE_TIP, self.MIDDLE_MCP)
            ring_extended = self._is_finger_extended(landmarks, self.RING_TIP, self.RING_MCP)

            # Thumb and pinky should be far from wrist, others closed
            if thumb_dist > 0.15 and pinky_dist > 0.15 and not index_extended and not middle_extended and not ring_extended:
                return 0.85

            return 0.0

        except Exception as e:
            logger.error(f"Error detecting thumb-pinky extended: {e}")
            return 0.0

    def _get_smoothed_gesture(self) -> Tuple[GestureType, float]:
        """
        Get smoothed gesture from recent history.

        Returns:
            tuple: (GestureType, average_confidence)
        """
        if not self.gesture_history:
            return GestureType.NONE, 0.0

        # Count occurrences of each gesture
        gesture_counts: Dict[GestureType, List[float]] = {}
        for gesture, confidence in self.gesture_history:
            if gesture not in gesture_counts:
                gesture_counts[gesture] = []
            gesture_counts[gesture].append(confidence)

        # Find most common gesture with average confidence
        best_gesture = GestureType.NONE
        best_score = 0.0

        for gesture, confidences in gesture_counts.items():
            avg_confidence = sum(confidences) / len(confidences)
            score = len(confidences) * avg_confidence

            if score > best_score:
                best_score = score
                best_gesture = gesture

        # Get average confidence for best gesture
        if best_gesture in gesture_counts:
            avg_confidence = sum(gesture_counts[best_gesture]) / len(gesture_counts[best_gesture])
        else:
            avg_confidence = 0.0

        return best_gesture, avg_confidence

    def can_activate_gesture(self, gesture: GestureType) -> bool:
        """
        Check if gesture can be activated based on cooldown.

        Args:
            gesture: Gesture type to check

        Returns:
            bool: True if gesture can be activated
        """
        if gesture == GestureType.NONE:
            return False

        current_time = time.time()
        last_time = self.last_activation_time.get(gesture, 0)

        return (current_time - last_time) > self.cooldown_time

    def activate_gesture(self, gesture: GestureType):
        """
        Mark gesture as activated and update cooldown.

        Args:
            gesture: Gesture type that was activated
        """
        self.last_activation_time[gesture] = time.time()
        logger.debug(f"Gesture activated: {gesture.value}")

    def get_cursor_position(self, landmarks: List[Tuple[float, float, float]]) -> Optional[Tuple[float, float]]:
        """
        Get normalized cursor position from index finger tip.

        Args:
            landmarks: List of hand landmarks

        Returns:
            tuple: (x, y) normalized coordinates or None
        """
        if not landmarks or len(landmarks) < 9:
            return None

        try:
            index_tip = landmarks[self.INDEX_TIP]
            return (index_tip[0], index_tip[1])
        except Exception as e:
            logger.error(f"Error getting cursor position: {e}")
            return None


if __name__ == "__main__":
    """Test the gesture recognizer module."""
    print("Gesture Recognizer Test")
    print("This module requires hand landmarks from hand_tracker.py")
    print("Run main.py or hand_tracker.py for visual testing")

    recognizer = GestureRecognizer()
    print(f"Initialized with settings:")
    print(f"  Pinch threshold: {recognizer.pinch_threshold}")
    print(f"  Hold time: {recognizer.hold_time}s")
    print(f"  Cooldown time: {recognizer.cooldown_time}s")