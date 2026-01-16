"""
Enhanced Data Logger - Extended CSV Export
Exports both smoothed joint angles AND 3D relative coordinates.
Supports intent-based motion representation for inverse kinematics.
"""

import csv
from pathlib import Path


class EnhancedDataLogger:
    """
    Enhanced logger that exports:
    1. Smoothed joint angles (forward kinematics)
    2. 3D relative coordinates (intent-based / inverse kinematics)
    """
    
    def __init__(self, output_path, joint_names, position_names):
        """
        Initialize enhanced CSV logger.
        
        Args:
            output_path: Full path to output CSV file
            joint_names: List of joint angle column names (e.g., ['R_Elbow', 'L_Elbow'])
            position_names: List of position column names (e.g., ['LEFT_WRIST_X', 'LEFT_WRIST_Y'])
        """
        self.output_path = Path(output_path)
        self.joint_names = joint_names
        self.position_names = position_names
        
        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Open CSV file for writing
        self.file = open(self.output_path, 'w', newline='')
        self.writer = csv.writer(self.file)
        
        # Write header
        # Format: timestamp, angle1, angle2, ..., pos1_X, pos1_Y, pos1_Z, pos2_X, ...
        header = ['timestamp'] + joint_names + position_names
        self.writer.writerow(header)
        
        self.frame_count = 0
        
        print(f"[ENHANCED LOGGER] Initialized: {self.output_path}")
        print(f"[ENHANCED LOGGER] Joint Angles: {joint_names}")
        print(f"[ENHANCED LOGGER] 3D Positions: {len(position_names)} coordinates")
    
    def log_frame(self, timestamp, angles_dict, positions_dict):
        """
        Log a single frame with both angles and positions.
        
        Args:
            timestamp: Time in seconds
            angles_dict: {joint_name: angle_value} - smoothed angles
            positions_dict: {position_name: value} - 3D coordinates (flattened)
                          Example: {'LEFT_WRIST_X': 0.1, 'LEFT_WRIST_Y': 0.2, ...}
        """
        # Build row: [timestamp, angles..., positions...]
        row = [f"{timestamp:.3f}"]
        
        # Add joint angles
        for joint_name in self.joint_names:
            angle = angles_dict.get(joint_name, None)
            if angle is not None:
                row.append(f"{angle:.2f}")
            else:
                row.append("")
        
        # Add 3D positions
        for pos_name in self.position_names:
            pos_value = positions_dict.get(pos_name, None)
            if pos_value is not None:
                row.append(f"{pos_value:.4f}")  # Higher precision for coordinates
            else:
                row.append("")
        
        self.writer.writerow(row)
        self.frame_count += 1
    
    def close(self):
        """Close CSV file and finalize logging."""
        self.file.close()
        print(f"[ENHANCED LOGGER] Closed: {self.output_path}")
        print(f"[ENHANCED LOGGER] Total frames logged: {self.frame_count}")
    
    def __enter__(self):
        """Context manager support."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup."""
        self.close()


class DualLogger:
    """
    Dual logger that writes to TWO CSV files:
    1. Original format (angles only) - for backward compatibility
    2. Enhanced format (angles + positions) - for new features
    """
    
    def __init__(self, base_output_path, joint_names, position_names):
        """
        Initialize dual logger.
        
        Args:
            base_output_path: Base path (e.g., "output_data/video_20250116_motion.csv")
            joint_names: List of joint angle names
            position_names: List of 3D position coordinate names
        """
        base_path = Path(base_output_path)
        
        # Original CSV (angles only)
        self.original_path = base_path
        
        # Enhanced CSV (angles + positions)
        enhanced_name = base_path.stem + "_enhanced" + base_path.suffix
        self.enhanced_path = base_path.parent / enhanced_name
        
        # Initialize both loggers
        from data_logger import DataLogger
        
        self.original_logger = DataLogger(str(self.original_path), joint_names)
        self.enhanced_logger = EnhancedDataLogger(
            str(self.enhanced_path), joint_names, position_names
        )
        
        print(f"[DUAL LOGGER] Writing to:")
        print(f"  Original: {self.original_path}")
        print(f"  Enhanced: {self.enhanced_path}")
    
    def log_frame(self, timestamp, angles_dict, positions_dict=None):
        """
        Log to both CSV files.
        
        Args:
            timestamp: Time in seconds
            angles_dict: {joint_name: angle_value}
            positions_dict: {position_name: value} - optional for enhanced logger
        """
        # Log to original (angles only)
        self.original_logger.log_frame(timestamp, angles_dict)
        
        # Log to enhanced (angles + positions)
        if positions_dict is not None:
            self.enhanced_logger.log_frame(timestamp, angles_dict, positions_dict)
    
    def close(self):
        """Close both loggers."""
        self.original_logger.close()
        self.enhanced_logger.close()
    
    def __enter__(self):
        """Context manager support."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup."""
        self.close()


def test_enhanced_logger():
    """
    Test enhanced logging functionality.
    """
    import os
    import tempfile
    
    print("[TEST] Enhanced Data Logger")
    
    # Create temporary output directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = os.path.join(tmpdir, "test_motion.csv")
        
        # Initialize logger
        joint_names = ['R_Shoulder_Flex', 'R_Elbow', 'L_Shoulder_Flex', 'L_Elbow']
        position_names = [
            'LEFT_WRIST_X', 'LEFT_WRIST_Y', 'LEFT_WRIST_Z',
            'RIGHT_WRIST_X', 'RIGHT_WRIST_Y', 'RIGHT_WRIST_Z',
        ]
        
        logger = EnhancedDataLogger(tmp_path, joint_names, position_names)
        
        # Log test frames
        test_data = [
            (0.033, 
             {'R_Shoulder_Flex': 45.2, 'R_Elbow': 92.8, 'L_Shoulder_Flex': 40.1, 'L_Elbow': 88.3},
             {'LEFT_WRIST_X': -0.15, 'LEFT_WRIST_Y': -0.20, 'LEFT_WRIST_Z': 0.05,
              'RIGHT_WRIST_X': 0.15, 'RIGHT_WRIST_Y': -0.20, 'RIGHT_WRIST_Z': 0.05}),
            (0.067,
             {'R_Shoulder_Flex': 46.1, 'R_Elbow': 91.5, 'L_Shoulder_Flex': 39.8, 'L_Elbow': 89.1},
             {'LEFT_WRIST_X': -0.16, 'LEFT_WRIST_Y': -0.21, 'LEFT_WRIST_Z': 0.06,
              'RIGHT_WRIST_X': 0.16, 'RIGHT_WRIST_Y': -0.21, 'RIGHT_WRIST_Z': 0.06}),
        ]
        
        for timestamp, angles, positions in test_data:
            logger.log_frame(timestamp, angles, positions)
        
        logger.close()
        
        # Verify output
        print(f"\n[TEST] Enhanced CSV Contents:")
        with open(tmp_path, 'r') as f:
            content = f.read()
            print(content)
        
        # Test dual logger
        print("\n[TEST] Testing Dual Logger...")
        dual_path = os.path.join(tmpdir, "test_dual.csv")
        
        dual_logger = DualLogger(dual_path, joint_names, position_names)
        
        for timestamp, angles, positions in test_data:
            dual_logger.log_frame(timestamp, angles, positions)
        
        dual_logger.close()
        
        print("[TEST COMPLETE] Enhanced logger tests passed")


if __name__ == "__main__":
    test_enhanced_logger()