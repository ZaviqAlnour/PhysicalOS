"""
Ground Reaction Force (GRF) Estimator
Determines which foot is in contact with the ground.

Purpose: Prevent simultaneous leg lifting and maintain balance logic.

Logic: The foot with the lowest ANKLE_Y value is considered grounded.
"""

import numpy as np
import pandas as pd
from pathlib import Path


class GRFEstimator:
    """
    Estimate ground contact based on ankle positions.
    
    Uses vertical position (Y-coordinate) to determine which foot is planted.
    The foot closest to the ground (lowest Y value) is considered in contact.
    """
    
    def __init__(self, threshold=0.05, transition_frames=3):
        """
        Initialize GRF estimator.
        
        Args:
            threshold: Vertical distance threshold (meters) for considering both feet grounded
            transition_frames: Minimum frames before allowing support leg switch (prevents jitter)
        """
        self.threshold = threshold
        self.transition_frames = transition_frames
        
    def determine_support_leg(self, left_ankle_y, right_ankle_y):
        """
        Determine which leg is supporting based on ankle heights.
        
        Args:
            left_ankle_y: Y-coordinate of left ankle (scalar or array)
            right_ankle_y: Y-coordinate of right ankle (scalar or array)
            
        Returns:
            str or array of str: 'left', 'right', 'both', or 'none'
            
        Logic:
            - If |left_y - right_y| < threshold: 'both' (double support)
            - Else if left_y < right_y: 'left' (left foot grounded)
            - Else: 'right' (right foot grounded)
            - If both feet very high (> 0.3m): 'none' (jumping/falling)
        """
        left_ankle_y = np.asarray(left_ankle_y)
        right_ankle_y = np.asarray(right_ankle_y)
        
        diff = np.abs(left_ankle_y - right_ankle_y)
        
        # Initialize result
        if left_ankle_y.ndim == 0:  # Scalar
            # Check if jumping/falling (both feet elevated)
            if left_ankle_y > 0.3 and right_ankle_y > 0.3:
                return 'none'
            
            # Strict single support enforcement: lowest foot is pivot
            return 'left' if left_ankle_y < right_ankle_y else 'right'
        
        else:  # Array
            result = np.empty(len(left_ankle_y), dtype=object)
            
            # Vectorized logic
            for i in range(len(left_ankle_y)):
                left_y = left_ankle_y[i]
                right_y = right_ankle_y[i]
                diff_i = diff[i]
                
                # Check for airborne
                if left_y > 0.3 and right_y > 0.3:
                    result[i] = 'none'
                # Single support (strict enforcement)
                else:
                    result[i] = 'left' if left_y < right_y else 'right'
            
            return result
    
    def smooth_support_transitions(self, support_sequence):
        """
        Apply temporal smoothing to prevent rapid support leg switching.
        
        Args:
            support_sequence: Array of support leg labels
            
        Returns:
            Smoothed support sequence
            
        Purpose: Real human gait has stable support phases. This prevents
                 noise-induced jitter in support leg detection.
        """
        support_sequence = np.asarray(support_sequence, dtype=object)
        smoothed = support_sequence.copy()
        
        # Hold support leg for minimum transition frames
        current_support = support_sequence[0]
        hold_count = 0
        
        for i in range(1, len(support_sequence)):
            if support_sequence[i] == current_support:
                hold_count += 1
            else:
                if hold_count < self.transition_frames:
                    # Too rapid, maintain previous support
                    smoothed[i] = current_support
                    hold_count += 1
                else:
                    # Valid transition
                    current_support = support_sequence[i]
                    hold_count = 0
        
        return smoothed
    
    def calculate_grf_percentage(self, left_ankle_y, right_ankle_y):
        """
        Calculate weight distribution percentage between feet.
        
        Args:
            left_ankle_y: Left ankle height
            right_ankle_y: Right ankle height
            
        Returns:
            tuple: (left_percentage, right_percentage) where sum = 100
            
        Logic: Inverse relationship - lower foot bears more weight
        """
        left_ankle_y = np.asarray(left_ankle_y)
        right_ankle_y = np.asarray(right_ankle_y)
        
        # Invert heights (lower = more weight)
        # Add small epsilon to avoid division by zero
        eps = 0.001
        left_inv = 1.0 / (left_ankle_y + eps)
        right_inv = 1.0 / (right_ankle_y + eps)
        
        total = left_inv + right_inv
        
        left_pct = (left_inv / total) * 100
        right_pct = (right_inv / total) * 100
        
        return left_pct, right_pct
    
    def process_csv_data(self, csv_path, output_path=None, apply_smoothing=True):
        """
        Process CSV file to add support leg and GRF columns.
        
        Args:
            csv_path: Path to input CSV
            output_path: Optional output path
            apply_smoothing: Whether to smooth support transitions
            
        Returns:
            pandas.DataFrame with additional columns:
                - Support_Leg: 'left', 'right', 'both', or 'none'
                - GRF_Left_Pct: Percentage of weight on left foot
                - GRF_Right_Pct: Percentage of weight on right foot
        """
        # Read CSV
        df = pd.read_csv(csv_path)
        
        # Check for required columns
        if 'LEFT_ANKLE_Y' not in df.columns or 'RIGHT_ANKLE_Y' not in df.columns:
            raise ValueError("CSV must contain LEFT_ANKLE_Y and RIGHT_ANKLE_Y columns")
        
        print(f"[GRF] Processing {Path(csv_path).name}")
        
        # Get ankle data
        left_ankle_y = df['LEFT_ANKLE_Y'].values
        right_ankle_y = df['RIGHT_ANKLE_Y'].values
        
        # Determine support leg
        support_leg = self.determine_support_leg(left_ankle_y, right_ankle_y)
        
        # Apply smoothing if requested
        if apply_smoothing:
            support_leg = self.smooth_support_transitions(support_leg)
            print(f"[GRF] Applied temporal smoothing ({self.transition_frames} frame minimum)")
        
        # Calculate weight distribution
        left_pct, right_pct = self.calculate_grf_percentage(left_ankle_y, right_ankle_y)
        
        # Add columns to dataframe
        df['Support_Leg'] = support_leg
        df['GRF_Left_Pct'] = left_pct
        df['GRF_Right_Pct'] = right_pct
        
        # Add binary columns for easy filtering
        df['Is_Left_Support'] = (support_leg == 'left').astype(int)
        df['Is_Right_Support'] = (support_leg == 'right').astype(int)
        df['Is_Double_Support'] = (support_leg == 'both').astype(int)
        df['Is_Airborne'] = (support_leg == 'none').astype(int)
        
        # Statistics
        print(f"[GRF] Support distribution:")
        print(f"  Left:   {np.sum(support_leg == 'left')} frames ({np.sum(support_leg == 'left')/len(support_leg)*100:.1f}%)")
        print(f"  Right:  {np.sum(support_leg == 'right')} frames ({np.sum(support_leg == 'right')/len(support_leg)*100:.1f}%)")
        print(f"  Both:   {np.sum(support_leg == 'both')} frames ({np.sum(support_leg == 'both')/len(support_leg)*100:.1f}%)")
        print(f"  None:   {np.sum(support_leg == 'none')} frames ({np.sum(support_leg == 'none')/len(support_leg)*100:.1f}%)")
        
        # Save output
        if output_path is None:
            input_path = Path(csv_path)
            output_path = input_path.parent / f"{input_path.stem}_grf{input_path.suffix}"
        
        df.to_csv(output_path, index=False)
        print(f"[GRF] Saved to: {output_path}")
        
        return df


if __name__ == "__main__":
    """
    Standalone test/usage example.
    Usage: python grf_estimator.py path/to/csv_file.csv
    """
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python grf_estimator.py <csv_file_path>")
        print("\nExample:")
        print("  python grf_estimator.py output_data/motion_20260116_motion_enhanced.csv")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    
    if not Path(csv_path).exists():
        print(f"[ERROR] File not found: {csv_path}")
        sys.exit(1)
    
    print("=" * 60)
    print("Ground Reaction Force (GRF) Estimator")
    print("=" * 60)
    
    estimator = GRFEstimator(threshold=0.05, transition_frames=3)
    df = estimator.process_csv_data(csv_path, apply_smoothing=True)
    
    print("\n" + "=" * 60)
