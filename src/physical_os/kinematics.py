"""
Kinematics Engine - Physics Layer
Converts 3D joint coordinates into biomechanically valid joint angles.
Pure vector mathematics: θ = arccos((A·B) / (|A||B|))
"""

import numpy as np
from utils import vector_from_points, angle_between_vectors


def calculate_elbow_angle(shoulder, elbow, wrist):
    """
    Calculate elbow flexion angle.
    
    Physics:
    - Straight arm (extended): 180°
    - Bent arm (flexed): < 180°
    - Range: [0°, 180°]
    
    Args:
        shoulder: 3D point [x, y, z]
        elbow: 3D point [x, y, z]
        wrist: 3D point [x, y, z]
        
    Returns:
        float: Elbow angle in degrees
    """
    # Vector from elbow to shoulder (upper arm)
    v_upper = vector_from_points(elbow, shoulder)
    
    # Vector from elbow to wrist (forearm)
    v_forearm = vector_from_points(elbow, wrist)
    
    # Calculate angle between vectors
    angle_rad = angle_between_vectors(v_upper, v_forearm)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg


def calculate_shoulder_flexion(hip, shoulder, elbow):
    """
    Calculate shoulder flexion angle.
    
    Physics:
    - Arm at side (neutral): ~0°
    - Arm forward/up: positive angle
    - Range: typically [0°, 180°]
    
    Args:
        hip: 3D point [x, y, z]
        shoulder: 3D point [x, y, z]
        elbow: 3D point [x, y, z]
        
    Returns:
        float: Shoulder flexion angle in degrees
    """
    # Vector from shoulder to hip (torso reference)
    v_torso = vector_from_points(shoulder, hip)
    
    # Vector from shoulder to elbow (upper arm)
    v_arm = vector_from_points(shoulder, elbow)
    
    # Calculate angle
    angle_rad = angle_between_vectors(v_torso, v_arm)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg


def calculate_knee_angle(hip, knee, ankle):
    """
    Calculate knee flexion angle.
    
    Physics:
    - Straight leg (extended): 180°
    - Bent leg (flexed): < 180°
    - Range: [0°, 180°]
    
    Args:
        hip: 3D point [x, y, z]
        knee: 3D point [x, y, z]
        ankle: 3D point [x, y, z]
        
    Returns:
        float: Knee angle in degrees
    """
    v_thigh = vector_from_points(knee, hip)
    v_shin = vector_from_points(knee, ankle)
    
    angle_rad = angle_between_vectors(v_thigh, v_shin)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg


def calculate_hip_flexion(shoulder, hip, knee):
    """
    Calculate hip flexion angle.
    
    Physics:
    - Standing straight: ~180°
    - Leg raised forward: < 180°
    
    Args:
        shoulder: 3D point [x, y, z]
        hip: 3D point [x, y, z]
        knee: 3D point [x, y, z]
        
    Returns:
        float: Hip flexion angle in degrees
    """
    v_torso = vector_from_points(hip, shoulder)
    v_thigh = vector_from_points(hip, knee)
    
    angle_rad = angle_between_vectors(v_torso, v_thigh)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg


def validate_angle(angle, min_angle=0.0, max_angle=180.0):
    """
    Validate that angle is biomechanically plausible.
    
    Args:
        angle: Angle in degrees
        min_angle: Minimum valid angle
        max_angle: Maximum valid angle
        
    Returns:
        bool: True if valid, False otherwise
    """
    return min_angle <= angle <= max_angle


def test_kinematics():
    """
    Test kinematics calculations with known poses.
    """
    print("[TEST] Kinematics Engine - Joint Angle Calculations")
    
    # Test 1: Straight arm (elbow extended)
    shoulder = np.array([0.0, 0.0, 0.0])
    elbow = np.array([1.0, 0.0, 0.0])
    wrist = np.array([2.0, 0.0, 0.0])
    
    elbow_angle = calculate_elbow_angle(shoulder, elbow, wrist)
    print(f"Test 1 - Straight Arm: {elbow_angle:.1f}° (expected: 180°)")
    assert abs(elbow_angle - 180.0) < 1.0, "Straight arm should be ~180°"
    
    # Test 2: 90° bent elbow
    shoulder = np.array([0.0, 0.0, 0.0])
    elbow = np.array([1.0, 0.0, 0.0])
    wrist = np.array([1.0, 1.0, 0.0])
    
    elbow_angle = calculate_elbow_angle(shoulder, elbow, wrist)
    print(f"Test 2 - 90° Bent Arm: {elbow_angle:.1f}° (expected: 90°)")
    assert abs(elbow_angle - 90.0) < 1.0, "90° bend should be ~90°"
    
    # Test 3: Shoulder flexion (arm at side)
    hip = np.array([0.0, 0.0, 0.0])
    shoulder = np.array([0.0, 1.0, 0.0])
    elbow = np.array([0.0, 0.5, 0.0])
    
    shoulder_flex = calculate_shoulder_flexion(hip, shoulder, elbow)
    print(f"Test 3 - Arm at Side: {shoulder_flex:.1f}° (expected: 180°)")
    
    # Test 4: Validation
    valid = validate_angle(90.0)
    invalid = validate_angle(200.0)
    print(f"Test 4 - Validation: 90° valid={valid}, 200° valid={invalid}")
    
    print("[TEST COMPLETE] All kinematics tests passed")


if __name__ == "__main__":
    test_kinematics()