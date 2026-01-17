"""
Constraints System - Safety Layer
Defines biomechanical limits for human joints and ensures physical plausibility.

Purpose: Prevent physically impossible or destructive joint commands in robotics.
"""

import numpy as np

# Human-safe joint angle ranges (degrees)
# These represent typical range of motion (ROM)
JOINT_LIMITS = {
    'R_Elbow': (0.0, 160.0),            # Extension to Flexion
    'L_Elbow': (0.0, 160.0),
    'R_Shoulder_Flex': (0.0, 180.0),    # Neutral to overhead
    'L_Shoulder_Flex': (0.0, 180.0),
    'R_Knee': (0.0, 140.0),             # Extension to Flexion
    'L_Knee': (0.0, 140.0),
    'R_Hip_Flex': (0.0, 120.0),
    'L_Hip_Flex': (0.0, 120.0),
}

class MotionConstraintSystem:
    """
    Applies safety constraints to joint angles.
    """
    
    def __init__(self, limits=None):
        """
        Initialize the constraint system.
        
        Args:
            limits: Dictionary of joint limits. If None, uses default JOINT_LIMITS.
        """
        self.limits = limits if limits is not None else JOINT_LIMITS
        
    def apply_constraints(self, joints_dict):
        """
        Clamp joint angles to their allowed ranges.
        
        Args:
            joints_dict: Dictionary of {joint_name: angle_value}
            
        Returns:
            dict: Clamped joint angles
        """
        clamped_joints = joints_dict.copy()
        
        for joint, angle in joints_dict.items():
            if joint in self.limits:
                min_val, max_val = self.limits[joint]
                clamped_joints[joint] = np.clip(angle, min_val, max_val)
                
                # Log warning if clamping occurred (optional, for debugging)
                # if clamped_joints[joint] != angle:
                #    print(f"[CONSTRAINT] Clamped {joint}: {angle:.1f} -> {clamped_joints[joint]:.1f}")
                    
        return clamped_joints

def test_constraints():
    """Unit test for joint constraints."""
    system = MotionConstraintSystem()
    
    test_data = {
        'R_Elbow': 175.0,  # Should be clamped to 160
        'L_Elbow': -10.0,  # Should be clamped to 0
        'R_Shoulder_Flex': 90.0, # Valid
        'Unknown_Joint': 45.0 # Should be ignored
    }
    
    clamped = system.apply_constraints(test_data)
    
    print("[TEST] Constraint System")
    print(f"R_Elbow: {test_data['R_Elbow']} -> {clamped['R_Elbow']} (Expected: 160.0)")
    assert clamped['R_Elbow'] == 160.0
    
    print(f"L_Elbow: {test_data['L_Elbow']} -> {clamped['L_Elbow']} (Expected: 0.0)")
    assert clamped['L_Elbow'] == 0.0
    
    print(f"R_Shoulder_Flex: {test_data['R_Shoulder_Flex']} -> {clamped['R_Shoulder_Flex']} (Expected: 90.0)")
    assert clamped['R_Shoulder_Flex'] == 90.0
    
    print("[TEST] All tests passed.")

if __name__ == "__main__":
    test_constraints()
