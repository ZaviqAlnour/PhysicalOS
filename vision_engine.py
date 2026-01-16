"""
Vision Engine - MediaPipe Pose Detection Layer
Extracts 33 skeletal landmarks from video frames.
Compatible with MediaPipe 0.10.8 on Windows 11 / Python 3.11.9
"""

import cv2
import numpy as np

# Standard MediaPipe import pattern
import mediapipe as mp
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


class PoseDetector:
    """
    Wrapper for MediaPipe Pose detection.
    Extracts 33 body landmarks from video frames.
    """
    
    def __init__(self, 
                 model_complexity=1,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        """
        Initialize MediaPipe Pose detector.
        
        Args:
            model_complexity: 0=Lite, 1=Full, 2=Heavy (default: 1 for Intel i5)
            min_detection_confidence: Minimum confidence for initial detection
            min_tracking_confidence: Minimum confidence for tracking
        """
        self.pose = mp_pose.Pose(
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            static_image_mode=False  # Video mode for better tracking
        )
        
        # MediaPipe landmark indices (33 total)
        # Key points we use:
        # 11: LEFT_SHOULDER, 12: RIGHT_SHOULDER
        # 13: LEFT_ELBOW, 14: RIGHT_ELBOW
        # 15: LEFT_WRIST, 16: RIGHT_WRIST
        # 23: LEFT_HIP, 24: RIGHT_HIP
        
    def detect_pose(self, frame):
        """
        Detect pose landmarks in a single frame.
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            numpy array of shape (33, 4) with [x, y, z, visibility] or None if no pose
        """
        # Convert BGR to RGB (MediaPipe requirement)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = self.pose.process(rgb_frame)
        
        if results.pose_landmarks is None:
            return None
        
        # Extract landmarks to numpy array
        landmarks = []
        for landmark in results.pose_landmarks.landmark:
            landmarks.append([
                landmark.x,           # Normalized x [0, 1]
                landmark.y,           # Normalized y [0, 1]
                landmark.z,           # Depth (relative to hips)
                landmark.visibility   # Confidence [0, 1]
            ])
        
        return np.array(landmarks)
    
    def draw_skeleton(self, frame, landmarks):
        """
        Draw skeleton overlay on frame.
        
        Args:
            frame: Original BGR frame
            landmarks: Array of shape (33, 4)
            
        Returns:
            Frame with skeleton drawn
        """
        if landmarks is None:
            return frame
        
        # Convert landmarks back to MediaPipe format for drawing
        h, w, _ = frame.shape
        annotated_frame = frame.copy()
        
        # Draw connections (bones)
        connections = mp_pose.POSE_CONNECTIONS
        
        for connection in connections:
            start_idx, end_idx = connection
            
            # Get pixel coordinates
            start = landmarks[start_idx]
            end = landmarks[end_idx]
            
            start_px = (int(start[0] * w), int(start[1] * h))
            end_px = (int(end[0] * w), int(end[1] * h))
            
            # Draw line (bone)
            cv2.line(annotated_frame, start_px, end_px, (0, 255, 0), 2)
        
        # Draw landmarks (joints)
        for i, landmark in enumerate(landmarks):
            px = (int(landmark[0] * w), int(landmark[1] * h))
            
            # Color based on visibility
            color = (0, 255, 255) if landmark[3] > 0.5 else (0, 0, 255)
            cv2.circle(annotated_frame, px, 3, color, -1)
        
        return annotated_frame
    
    def get_landmark_pixel_coords(self, landmarks, frame_shape, landmark_idx):
        """
        Convert normalized landmark to pixel coordinates.
        
        Args:
            landmarks: Array of shape (33, 4)
            frame_shape: (height, width, channels)
            landmark_idx: Index of landmark (0-32)
            
        Returns:
            (x_pixel, y_pixel) or None
        """
        if landmarks is None or landmark_idx >= len(landmarks):
            return None
        
        h, w = frame_shape[:2]
        landmark = landmarks[landmark_idx]
        
        x_px = int(landmark[0] * w)
        y_px = int(landmark[1] * h)
        
        return (x_px, y_px)
    
    def close(self):
        """Release MediaPipe resources."""
        self.pose.close()


def test_vision_engine():
    """
    Standalone test function for vision engine.
    Tests with webcam if no video specified.
    """
    import sys
    
    print("[TEST] Vision Engine - MediaPipe Pose Detection")
    print("[INFO] Press 'q' to quit")
    
    # Try to open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam")
        print("Place a video in input_videos/ and modify this test function")
        sys.exit(1)
    
    detector = PoseDetector()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect pose
        landmarks = detector.detect_pose(frame)
        
        if landmarks is not None:
            # Draw skeleton
            frame = detector.draw_skeleton(frame, landmarks)
            
            # Print Right Elbow coordinates (landmark index 14)
            right_elbow = landmarks[14]
            pixel_coords = detector.get_landmark_pixel_coords(
                landmarks, frame.shape, 14
            )
            
            print(f"[RIGHT ELBOW] Pixel: {pixel_coords} | "
                  f"Normalized: ({right_elbow[0]:.3f}, {right_elbow[1]:.3f}, {right_elbow[2]:.3f})")
            
            # Display info on frame
            cv2.putText(frame, "POSE DETECTED", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "NO POSE", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow('Vision Engine Test', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    detector.close()
    cv2.destroyAllWindows()
    
    print("[TEST COMPLETE]")


if __name__ == "__main__":
    test_vision_engine()