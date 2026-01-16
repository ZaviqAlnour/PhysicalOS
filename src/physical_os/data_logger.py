"""
Data Logger - Memory / CSV Persistence Layer
Saves frame-by-frame joint angles to CSV for robot control systems.
"""

import csv
import os
from pathlib import Path


class DataLogger:
    """
    Logs motion data to CSV with frame-accurate timestamps.
    One row per video frame.
    """
    
    def __init__(self, output_path, joint_names):
        """
        Initialize CSV logger.
        
        Args:
            output_path: Full path to output CSV file
            joint_names: List of joint angle column names
        """
        self.output_path = Path(output_path)
        self.joint_names = joint_names
        
        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Open CSV file for writing
        self.file = open(self.output_path, 'w', newline='')
        self.writer = csv.writer(self.file)
        
        # Write header
        header = ['timestamp'] + self.joint_names
        self.writer.writerow(header)
        
        self.frame_count = 0
        
        print(f"[LOGGER] Initialized: {self.output_path}")
        print(f"[LOGGER] Columns: {header}")
    
    def log_frame(self, timestamp, angles_dict):
        """
        Log a single frame's joint angles.
        
        Args:
            timestamp: Time in seconds (frame_number / fps)
            angles_dict: Dictionary mapping joint names to angles
                        Example: {'R_Elbow': 92.5, 'L_Elbow': 88.3}
        """
        # Build row: [timestamp, angle1, angle2, ...]
        row = [f"{timestamp:.3f}"]  # Timestamp with millisecond precision
        
        for joint_name in self.joint_names:
            angle = angles_dict.get(joint_name, None)
            
            if angle is not None:
                row.append(f"{angle:.2f}")  # Angle with 2 decimal places
            else:
                row.append("")  # Missing data
        
        self.writer.writerow(row)
        self.frame_count += 1
    
    def close(self):
        """
        Close CSV file and finalize logging.
        """
        self.file.close()
        print(f"[LOGGER] Closed: {self.output_path}")
        print(f"[LOGGER] Total frames logged: {self.frame_count}")
    
    def __enter__(self):
        """Context manager support."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup."""
        self.close()


class JSONLogger:
    """
    Alternative logger for JSON output (future expansion).
    """
    
    def __init__(self, output_path, joint_names):
        """
        Initialize JSON logger.
        
        Args:
            output_path: Full path to output JSON file
            joint_names: List of joint angle names
        """
        import json
        
        self.output_path = Path(output_path)
        self.joint_names = joint_names
        self.data = {
            'metadata': {
                'joint_names': joint_names,
                'frame_count': 0
            },
            'frames': []
        }
    
    def log_frame(self, timestamp, angles_dict):
        """Log frame to JSON structure."""
        frame_data = {
            'timestamp': round(timestamp, 3),
            'angles': {name: round(angles_dict.get(name, 0.0), 2) 
                      for name in self.joint_names}
        }
        self.data['frames'].append(frame_data)
        self.data['metadata']['frame_count'] += 1
    
    def close(self):
        """Write JSON to file."""
        import json
        
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.output_path, 'w') as f:
            json.dump(self.data, f, indent=2)
        
        print(f"[JSON LOGGER] Saved: {self.output_path}")
        print(f"[JSON LOGGER] Frames: {self.data['metadata']['frame_count']}")


def test_data_logger():
    """
    Test CSV logging functionality.
    """
    import os
    import tempfile
    
    print("[TEST] Data Logger - CSV Persistence")
    
    # Create temporary output file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        # Initialize logger
        joint_names = ['R_Shoulder_Flex', 'R_Elbow', 'L_Shoulder_Flex', 'L_Elbow']
        logger = DataLogger(tmp_path, joint_names)
        
        # Log some test frames
        test_data = [
            (0.033, {'R_Shoulder_Flex': 45.2, 'R_Elbow': 92.8, 
                     'L_Shoulder_Flex': 40.1, 'L_Elbow': 88.3}),
            (0.067, {'R_Shoulder_Flex': 46.1, 'R_Elbow': 91.5, 
                     'L_Shoulder_Flex': 39.8, 'L_Elbow': 89.1}),
            (0.100, {'R_Shoulder_Flex': 47.3, 'R_Elbow': 90.2, 
                     'L_Shoulder_Flex': 38.5, 'L_Elbow': 90.4}),
        ]
        
        for timestamp, angles in test_data:
            logger.log_frame(timestamp, angles)
        
        logger.close()
        
        # Verify output
        print(f"\n[TEST] CSV Contents:")
        with open(tmp_path, 'r') as f:
            content = f.read()
            print(content)
        
        # Check file exists and has content
        assert os.path.exists(tmp_path), "CSV file not created"
        assert os.path.getsize(tmp_path) > 0, "CSV file is empty"
        
        print("[TEST COMPLETE] Data logger test passed")
        
    finally:
        # Cleanup
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    test_data_logger()