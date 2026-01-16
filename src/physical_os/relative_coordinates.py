"""
Relative Coordinates Module - 3D Position Calculations
Computes joint positions relative to Mid-Hip (root joint) for camera-independent data.
Enables intent-based motion representation for inverse kinematics.
"""

import numpy as np
from utils import vector_from_points


# MediaPipe Pose Landmark Indices
LANDMARK_INDICES = {
    # Core body
    'NOSE': 0,
    'LEFT_EYE_INNER': 1,
    'LEFT_EYE': 2,
    'LEFT_EYE_OUTER': 3,
    'RIGHT_EYE_INNER': 4,
    'RIGHT_EYE': 5,
    'RIGHT_EYE_OUTER': 6,
    'LEFT_EAR': 7,
    'RIGHT_EAR': 8,
    'MOUTH_LEFT': 9,
    'MOUTH_RIGHT': 10,
    
    # Upper body
    'LEFT_SHOULDER': 11,
    'RIGHT_SHOULDER': 12,
    'LEFT_ELBOW': 13,
    'RIGHT_ELBOW': 14,
    'LEFT_WRIST': 15,
    'RIGHT_WRIST': 16,
    
    # Hands
    'LEFT_PINKY': 17,
    'RIGHT_PINKY': 18,
    'LEFT_INDEX': 19,
    'RIGHT_INDEX': 20,
    'LEFT_THUMB': 21,
    'RIGHT_THUMB': 22,
    
    # Lower body
    'LEFT_HIP': 23,
    'RIGHT_HIP': 24,
    'LEFT_KNEE': 25,
    'RIGHT_KNEE': 26,
    'LEFT_ANKLE': 27,
    'RIGHT_ANKLE': 28,
    'LEFT_HEEL': 29,
    'RIGHT_HEEL': 30,
    'LEFT_FOOT_INDEX': 31,
    'RIGHT_FOOT_INDEX': 32,
}

# End effectors - critical points for intent-based motion
END_EFFECTORS = [
    'LEFT_WRIST', 'RIGHT_WRIST',    # Hand end effectors
    'LEFT_ANKLE', 'RIGHT_ANKLE',    # Foot end effectors
]

# Key joints for full body motion
KEY_JOINTS = [
    'LEFT_SHOULDER', 'RIGHT_SHOULDER',
    'LEFT_ELBOW', 'RIGHT_ELBOW',
    'LEFT_WRIST', 'RIGHT_WRIST',
    'LEFT_HIP', 'RIGHT_HIP',
    'LEFT_KNEE', 'RIGHT_KNEE',
    'LEFT_ANKLE', 'RIGHT_ANKLE',
]


def calculate_mid_hip(landmarks):
    """
    Calculate the Mid-Hip position (root joint).
    
    Args:
        landmarks: Array of shape (33, 4) with [x, y, z, visibility]
        
    Returns:
        numpy array: [x, y, z] of mid-hip position
    """
    left_hip = landmarks[LANDMARK_INDICES['LEFT_HIP']][:3]
    right_hip = landmarks[LANDMARK_INDICES['RIGHT_HIP']][:3]
    
    mid_hip = (left_hip + right_hip) / 2.0
    return mid_hip


def get_relative_positions(landmarks, reference_point='mid_hip'):
    """
    Convert all landmarks to positions relative to a reference point.
    
    Args:
        landmarks: Array of shape (33, 4) with [x, y, z, visibility]
        reference_point: 'mid_hip' or custom 3D point [x, y, z]
        
    Returns:
        dict: {joint_name: [x, y, z] relative position}
    """
    # Calculate reference point
    if reference_point == 'mid_hip':
        ref = calculate_mid_hip(landmarks)
    else:
        ref = np.array(reference_point[:3])
    
    # Calculate relative positions for all landmarks
    relative_positions = {}
    
    for joint_name, idx in LANDMARK_INDICES.items():
        joint_pos = landmarks[idx][:3]
        relative_pos = joint_pos - ref
        relative_positions[joint_name] = relative_pos
    
    return relative_positions


def get_end_effector_positions(landmarks):
    """
    Get 3D positions of end effectors relative to Mid-Hip.
    
    This is critical for intent-based motion - these positions define
    WHERE the hands and feet should be, allowing downstream systems
    to solve inverse kinematics for their own joint angles.
    
    Args:
        landmarks: Array of shape (33, 4) with [x, y, z, visibility]
        
    Returns:
        dict: {effector_name: [x, y, z] relative position}
    """
    relative_positions = get_relative_positions(landmarks, 'mid_hip')
    
    end_effector_positions = {}
    for effector_name in END_EFFECTORS:
        end_effector_positions[effector_name] = relative_positions[effector_name]
    
    return end_effector_positions


def get_key_joint_positions(landmarks):
    """
    Get 3D positions of all key joints relative to Mid-Hip.
    
    Provides complete body pose in 3D space for motion retargeting.
    
    Args:
        landmarks: Array of shape (33, 4) with [x, y, z, visibility]
        
    Returns:
        dict: {joint_name: [x, y, z] relative position}
    """
    relative_positions = get_relative_positions(landmarks, 'mid_hip')
    
    key_positions = {}
    for joint_name in KEY_JOINTS:
        key_positions[joint_name] = relative_positions[joint_name]
    
    # Also include Mid-Hip itself (always at origin in relative space)
    key_positions['MID_HIP'] = np.array([0.0, 0.0, 0.0])
    
    return key_positions


def scale_to_body_height(relative_positions, landmarks, height_meters=None):
    """
    Scale relative positions to match a specific body height.
    
    Useful for retargeting motion to different body sizes.
    
    Args:
        relative_positions: Dict of relative 3D positions
        landmarks: Original landmarks for reference
        height_meters: Target height in meters (None = use normalized units)
        
    Returns:
        dict: Scaled relative positions
    """
    if height_meters is None:
        return relative_positions
    
    # Estimate body height from landmarks (nose to mid-ankle)
    nose = landmarks[LANDMARK_INDICES['NOSE']][:3]
    left_ankle = landmarks[LANDMARK_INDICES['LEFT_ANKLE']][:3]
    right_ankle = landmarks[LANDMARK_INDICES['RIGHT_ANKLE']][:3]
    mid_ankle = (left_ankle + right_ankle) / 2.0
    
    current_height = np.linalg.norm(nose - mid_ankle)
    
    # Calculate scale factor
    if current_height < 1e-6:
        return relative_positions
    
    scale_factor = height_meters / current_height
    
    # Scale all positions
    scaled_positions = {}
    for joint_name, pos in relative_positions.items():
        scaled_positions[joint_name] = pos * scale_factor
    
    return scaled_positions


def format_positions_for_csv(positions_dict):
    """
    Flatten 3D positions into individual X, Y, Z columns for CSV export.
    
    Args:
        positions_dict: {joint_name: [x, y, z]}
        
    Returns:
        dict: {joint_name_X: x, joint_name_Y: y, joint_name_Z: z}
    """
    flattened = {}
    
    for joint_name, pos in positions_dict.items():
        flattened[f"{joint_name}_X"] = pos[0]
        flattened[f"{joint_name}_Y"] = pos[1]
        flattened[f"{joint_name}_Z"] = pos[2]
    
    return flattened


def calculate_velocity(current_pos, prev_pos, dt):
    """
    Calculate velocity of a joint (for advanced intent representation).
    
    Args:
        current_pos: Current 3D position [x, y, z]
        prev_pos: Previous 3D position [x, y, z]
        dt: Time difference in seconds
        
    Returns:
        numpy array: Velocity vector [vx, vy, vz] in units/second
    """
    if dt <= 0:
        return np.array([0.0, 0.0, 0.0])
    
    velocity = (np.array(current_pos) - np.array(prev_pos)) / dt
    return velocity


class PositionTracker:
    """
    Track positions over time for velocity and acceleration calculations.
    """
    
    def __init__(self):
        """Initialize position tracker."""
        self.history = {}  # {joint_name: [(timestamp, position), ...]}
        self.max_history = 5  # Keep last N frames
    
    def update(self, joint_name, position, timestamp):
        """
        Update position history for a joint.
        
        Args:
            joint_name: Name of joint
            position: 3D position [x, y, z]
            timestamp: Time in seconds
        """
        if joint_name not in self.history:
            self.history[joint_name] = []
        
        self.history[joint_name].append((timestamp, np.array(position)))
        
        # Keep only recent history
        if len(self.history[joint_name]) > self.max_history:
            self.history[joint_name].pop(0)
    
    def get_velocity(self, joint_name):
        """
        Calculate current velocity for a joint.
        
        Args:
            joint_name: Name of joint
            
        Returns:
            numpy array: Velocity [vx, vy, vz] or None
        """
        if joint_name not in self.history or len(self.history[joint_name]) < 2:
            return None
        
        # Use last two positions
        prev_t, prev_pos = self.history[joint_name][-2]
        curr_t, curr_pos = self.history[joint_name][-1]
        
        dt = curr_t - prev_t
        
        return calculate_velocity(curr_pos, prev_pos, dt)
    
    def reset(self):
        """Clear all history."""
        self.history.clear()


def test_relative_coordinates():
    """
    Test relative coordinate calculations with synthetic data.
    """
    print("[TEST] Relative Coordinates Module")
    
    # Create synthetic landmarks (33 points)
    landmarks = np.zeros((33, 4))
    
    # Set up test positions
    landmarks[LANDMARK_INDICES['LEFT_HIP']] = [0.4, 0.5, 0.0, 1.0]
    landmarks[LANDMARK_INDICES['RIGHT_HIP']] = [0.6, 0.5, 0.0, 1.0]
    landmarks[LANDMARK_INDICES['LEFT_WRIST']] = [0.3, 0.3, 0.1, 1.0]
    landmarks[LANDMARK_INDICES['RIGHT_WRIST']] = [0.7, 0.3, 0.1, 1.0]
    landmarks[LANDMARK_INDICES['LEFT_ANKLE']] = [0.4, 0.8, 0.0, 1.0]
    landmarks[LANDMARK_INDICES['RIGHT_ANKLE']] = [0.6, 0.8, 0.0, 1.0]
    
    # Test mid-hip calculation
    mid_hip = calculate_mid_hip(landmarks)
    print(f"Mid-Hip: {mid_hip} (expected: [0.5, 0.5, 0.0])")
    
    # Test relative positions
    relative = get_relative_positions(landmarks)
    print(f"\nRelative position of LEFT_WRIST: {relative['LEFT_WRIST']}")
    print(f"(expected: [-0.2, -0.2, 0.1])")
    
    # Test end effector positions
    end_effectors = get_end_effector_positions(landmarks)
    print(f"\nEnd Effectors extracted: {list(end_effectors.keys())}")
    
    # Test CSV formatting
    flattened = format_positions_for_csv(end_effectors)
    print(f"\nFlattened columns: {list(flattened.keys())[:6]}...")
    
    # Test position tracker
    tracker = PositionTracker()
    tracker.update('LEFT_WRIST', [0.0, 0.0, 0.0], 0.0)
    tracker.update('LEFT_WRIST', [0.1, 0.0, 0.0], 0.033)  # Moved 0.1 units in 0.033s
    
    velocity = tracker.get_velocity('LEFT_WRIST')
    if velocity is not None:
        print(f"\nVelocity of LEFT_WRIST: {velocity}")
        print(f"(expected: [~3.0, 0.0, 0.0])")
    
    print("[TEST COMPLETE] Relative coordinates working")


if __name__ == "__main__":
    test_relative_coordinates()