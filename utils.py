"""
Utilities - Pure Math Functions
Shared vector mathematics and coordinate normalization.
No business logic, no I/O - only pure functions.
"""

import numpy as np


def vector_from_points(p1, p2):
    """
    Calculate vector from point p1 to point p2.
    
    Args:
        p1: Starting point [x, y, z]
        p2: Ending point [x, y, z]
        
    Returns:
        numpy array: Vector p2 - p1
    """
    return np.array(p2[:3]) - np.array(p1[:3])


def vector_magnitude(v):
    """
    Calculate magnitude (length) of a vector.
    
    Args:
        v: Vector [x, y, z]
        
    Returns:
        float: |v| = sqrt(x² + y² + z²)
    """
    return np.linalg.norm(v)


def angle_between_vectors(v1, v2):
    """
    Calculate angle between two vectors using dot product.
    
    Physics: θ = arccos((A·B) / (|A||B|))
    
    Args:
        v1: First vector [x, y, z]
        v2: Second vector [x, y, z]
        
    Returns:
        float: Angle in radians [0, π]
    """
    v1 = np.array(v1[:3])
    v2 = np.array(v2[:3])
    
    # Handle zero-length vectors
    mag1 = vector_magnitude(v1)
    mag2 = vector_magnitude(v2)
    
    if mag1 < 1e-6 or mag2 < 1e-6:
        # Degenerate case: one or both vectors have near-zero length
        return 0.0
    
    # Normalize vectors
    v1_normalized = v1 / mag1
    v2_normalized = v2 / mag2
    
    # Dot product
    dot_product = np.dot(v1_normalized, v2_normalized)
    
    # Clamp to [-1, 1] to avoid numerical errors with arccos
    dot_product = np.clip(dot_product, -1.0, 1.0)
    
    # Calculate angle
    angle_rad = np.arccos(dot_product)
    
    return angle_rad


def normalize_to_torso(landmarks, left_hip_idx=23, right_hip_idx=24):
    """
    Normalize all landmarks relative to torso center.
    
    This centers the coordinate system at the midpoint between hips
    and scales based on torso width, making motion data independent
    of camera distance and body position in frame.
    
    Args:
        landmarks: Array of shape (33, 4) with [x, y, z, visibility]
        left_hip_idx: Index of left hip landmark (default: 23)
        right_hip_idx: Index of right hip landmark (default: 24)
        
    Returns:
        numpy array: Normalized landmarks (33, 4)
    """
    if landmarks is None:
        return None
    
    # Calculate torso center (midpoint between hips)
    left_hip = landmarks[left_hip_idx][:3]
    right_hip = landmarks[right_hip_idx][:3]
    
    torso_center = (left_hip + right_hip) / 2.0
    
    # Calculate torso width (distance between hips)
    torso_width = vector_magnitude(right_hip - left_hip)
    
    # Avoid division by zero
    if torso_width < 1e-6:
        torso_width = 1.0
    
    # Normalize all landmarks
    normalized = landmarks.copy()
    
    for i in range(len(landmarks)):
        # Translate to torso-centered coordinates
        normalized[i][:3] = (landmarks[i][:3] - torso_center) / torso_width
        # Keep visibility unchanged
        normalized[i][3] = landmarks[i][3]
    
    return normalized


def calculate_distance_3d(p1, p2):
    """
    Calculate Euclidean distance between two 3D points.
    
    Args:
        p1: Point [x, y, z]
        p2: Point [x, y, z]
        
    Returns:
        float: Distance
    """
    v = vector_from_points(p1, p2)
    return vector_magnitude(v)


def project_to_2d(point_3d, plane='xy'):
    """
    Project 3D point to 2D plane.
    
    Args:
        point_3d: [x, y, z]
        plane: 'xy', 'xz', or 'yz'
        
    Returns:
        numpy array: 2D point
    """
    point = np.array(point_3d[:3])
    
    if plane == 'xy':
        return point[:2]  # [x, y]
    elif plane == 'xz':
        return np.array([point[0], point[2]])  # [x, z]
    elif plane == 'yz':
        return np.array([point[1], point[2]])  # [y, z]
    else:
        raise ValueError(f"Unknown plane: {plane}")


def cross_product(v1, v2):
    """
    Calculate cross product of two vectors.
    
    Args:
        v1: First vector [x, y, z]
        v2: Second vector [x, y, z]
        
    Returns:
        numpy array: v1 × v2
    """
    v1 = np.array(v1[:3])
    v2 = np.array(v2[:3])
    
    return np.cross(v1, v2)


def test_utils():
    """
    Test utility functions with known cases.
    """
    print("[TEST] Utilities - Math Functions")
    
    # Test 1: Vector from points
    p1 = np.array([0, 0, 0])
    p2 = np.array([1, 0, 0])
    v = vector_from_points(p1, p2)
    print(f"Test 1 - Vector: {v} (expected: [1, 0, 0])")
    assert np.allclose(v, [1, 0, 0]), "Vector calculation failed"
    
    # Test 2: Vector magnitude
    v = np.array([3, 4, 0])
    mag = vector_magnitude(v)
    print(f"Test 2 - Magnitude: {mag:.1f} (expected: 5.0)")
    assert abs(mag - 5.0) < 0.01, "Magnitude calculation failed"
    
    # Test 3: Angle between vectors (90 degrees)
    v1 = np.array([1, 0, 0])
    v2 = np.array([0, 1, 0])
    angle_rad = angle_between_vectors(v1, v2)
    angle_deg = np.degrees(angle_rad)
    print(f"Test 3 - Angle: {angle_deg:.1f}° (expected: 90°)")
    assert abs(angle_deg - 90.0) < 0.1, "Angle calculation failed"
    
    # Test 4: Angle between parallel vectors (0 degrees)
    v1 = np.array([1, 0, 0])
    v2 = np.array([2, 0, 0])
    angle_rad = angle_between_vectors(v1, v2)
    angle_deg = np.degrees(angle_rad)
    print(f"Test 4 - Parallel: {angle_deg:.1f}° (expected: 0°)")
    assert abs(angle_deg - 0.0) < 0.1, "Parallel angle failed"
    
    # Test 5: Angle between opposite vectors (180 degrees)
    v1 = np.array([1, 0, 0])
    v2 = np.array([-1, 0, 0])
    angle_rad = angle_between_vectors(v1, v2)
    angle_deg = np.degrees(angle_rad)
    print(f"Test 5 - Opposite: {angle_deg:.1f}° (expected: 180°)")
    assert abs(angle_deg - 180.0) < 0.1, "Opposite angle failed"
    
    # Test 6: Normalization
    fake_landmarks = np.array([
        [0.5, 0.5, 0.0, 1.0],   # Landmark 0
        [0.6, 0.5, 0.0, 1.0],   # Landmark 1
        # ... (would have 33 total)
    ])
    # Add more landmarks to reach index 24
    while len(fake_landmarks) < 25:
        fake_landmarks = np.vstack([fake_landmarks, [0.5, 0.5, 0.0, 1.0]])
    
    # Set hips at specific positions
    fake_landmarks[23] = [0.4, 0.5, 0.0, 1.0]  # Left hip
    fake_landmarks[24] = [0.6, 0.5, 0.0, 1.0]  # Right hip
    
    normalized = normalize_to_torso(fake_landmarks, 23, 24)
    print(f"Test 6 - Normalization: Torso center at origin")
    # After normalization, hip midpoint should be at origin
    hip_midpoint = (normalized[23][:3] + normalized[24][:3]) / 2.0
    print(f"  Hip midpoint: {hip_midpoint} (expected: ~[0, 0, 0])")
    
    print("[TEST COMPLETE] All utility tests passed")


if __name__ == "__main__":
    test_utils()