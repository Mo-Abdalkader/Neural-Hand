"""
Advanced Gesture Control System - Hand Tracker Module
Detects and tracks hands using MediaPipe.
"""

import cv2
import mediapipe as mp
import logging
import time
from typing import Optional, Tuple, List, Any

logger = logging.getLogger(__name__)

# MediaPipe Hands setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


class HandTracker:
    """
    A class to handle hand detection and tracking using MediaPipe.
    """

    def __init__(self, mode=False, max_hands=1, model_complexity=1, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        """
        Initializes the HandTracker with MediaPipe settings.
        """
        self.mode = mode
        self.max_hands = max_hands
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.hands: Optional[mp_hands.Hands] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.results: Optional[Any] = None
        self.last_time = 0
        self.current_fps = 0.0 # New attribute to store FPS

        logger.info("HandTracker initialized")

    def start(self, camera_id=0) -> bool:
        """
        Starts the video capture and MediaPipe Hands model.

        Args:
            camera_id: The ID of the camera to use (default is 0).

        Returns:
            bool: True if the camera and model started successfully, False otherwise.
        """
        try:
            self.cap = cv2.VideoCapture(camera_id)
            if not self.cap.isOpened():
                logger.error(f"Failed to open video capture with ID {camera_id}")
                return False

            self.hands = mp_hands.Hands(
                self.mode,
                self.max_hands,
                self.model_complexity,
                self.min_detection_confidence,
                self.min_tracking_confidence
            )
            logger.info("MediaPipe Hands model started")
            return True
        except Exception as e:
            logger.error(f"Error starting HandTracker: {e}")
            return False

    def stop(self):
        """
        Releases the video capture and closes the MediaPipe Hands model.
        """
        if self.cap:
            self.cap.release()
            self.cap = None
            logger.info("Video capture released")
        if self.hands:
            self.hands.close()
            self.hands = None
            logger.info("MediaPipe Hands model closed")

    def get_frame(self) -> Optional[cv2.Mat]:
        """
        Reads a frame from the video capture.

        Returns:
            Optional[cv2.Mat]: The captured frame or None if failed.
        """
        if self.cap and self.cap.isOpened():
            success, frame = self.cap.read()
            if success:
                # Flip the image horizontally for a selfie-view display
                frame = cv2.flip(frame, 1)
                return frame
        return None

    def process_frame(self, frame: cv2.Mat) -> cv2.Mat:
        """
        Processes a frame to detect hands and draws landmarks.

        Args:
            frame: The OpenCV image frame.

        Returns:
            cv2.Mat: The frame with hand landmarks drawn.
        """
        # Convert the BGR image to RGB and process it with MediaPipe Hands.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(rgb_frame)

        # Draw hand landmarks on the frame.
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

        # Calculate FPS
        current_time = time.time()
        if self.last_time:
            self.current_fps = 1 / (current_time - self.last_time)
        self.last_time = current_time
        cv2.putText(frame, f'FPS: {int(self.current_fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return frame

    def get_hand_landmarks(self) -> List[List[Tuple[float, float, float]]]:
        """
        Extracts normalized hand landmarks from the last processed frame.

        Returns:
            List[List[Tuple[float, float, float]]]: A list of hands, where each hand is a list of 21 (x, y, z) landmarks.
        """
        landmarks_list = []
        if self.results and self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                hand_landmarks_normalized = []
                for landmark in hand_landmarks.landmark:
                    # Normalized coordinates (x, y, z)
                    hand_landmarks_normalized.append((landmark.x, landmark.y, landmark.z))
                landmarks_list.append(hand_landmarks_normalized)
        return landmarks_list

    def get_hand_info(self) -> List[Tuple[str, List[Tuple[float, float, float]]]]:
        """
        Extracts hand type (left/right) and normalized landmarks.

        Returns:
            List[Tuple[str, List[Tuple[float, float, float]]]]: A list of tuples (hand_type, landmarks).
        """
        hand_info_list = []
        if self.results and self.results.multi_hand_landmarks:
            for hand_index, hand_landmarks in enumerate(self.results.multi_hand_landmarks):
                # Determine hand type (left/right)
                hand_type = "Unknown"
                if self.results.multi_handedness and hand_index < len(self.results.multi_handedness):
                    handedness = self.results.multi_handedness[hand_index].classification[0]
                    hand_type = handedness.label

                # Extract normalized landmarks
                hand_landmarks_normalized = []
                for landmark in hand_landmarks.landmark:
                    hand_landmarks_normalized.append((landmark.x, landmark.y, landmark.z))

                hand_info_list.append((hand_type, hand_landmarks_normalized))
        return hand_info_list

    def get_average_fps(self) -> float:
        """
        Returns the most recently calculated FPS.
        """
        return self.current_fps


if __name__ == '__main__':
    # Simple test to ensure the module can be run independently
    print("Running HandTracker test. Press 'q' to quit.")
    tracker = HandTracker()
    if tracker.start():
        while True:
            frame = tracker.get_frame()
            if frame is None:
                continue

            processed_frame = tracker.process_frame(frame)
            hand_info = tracker.get_hand_info()

            # Display hand info in console
            if hand_info:
                print(f"Detected {len(hand_info)} hand(s):")
                for hand_type, landmarks in hand_info:
                    print(f"  - {hand_type} Hand (Wrist: {landmarks[0][0]:.2f}, {landmarks[0][1]:.2f})")

            cv2.imshow('Hand Tracking Test', processed_frame)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

        tracker.stop()
        cv2.destroyAllWindows()
    else:
        print("Failed to start HandTracker. Check camera connection.")
