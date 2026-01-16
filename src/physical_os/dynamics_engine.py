"""
Dynamics Engine - Velocity and Acceleration Calculation
Computes motion dynamics from position time-series data.

Purpose: Enable torque calculations (τ = Iα) and ensure motion strength
         is physically meaningful ("not ghost-like").

Mathematical Foundation:
    Velocity: v = dx/dt (first derivative of position)
    Acceleration: a = dv/dt (second derivative of position)

Uses central difference method for numerical derivatives.
"""

import numpy as np
import pandas as pd
from pathlib import Path


class DynamicsEngine:
    """
    Calculate velocity and acceleration from position time-series data.
    
    Implements numerical differentiation using central differences for
    interior points and forward/backward differences for endpoints.
    """
    
    def __init__(self, fps=30.0):
        """
        Initialize dynamics engine.
        
        Args:
            fps: Frames per second of input data (default: 30.0)
        """
        self.fps = fps
        self.dt = 1.0 / fps  # Time delta between frames
        
    def calculate_velocity(self, positions, timestamps=None):
        """
        Calculate velocity using numerical differentiation.
        
        Args:
            positions: numpy array of shape (N,) or (N, 3) for position data
            timestamps: Optional numpy array of timestamps. If None, uses uniform dt
            
        Returns:
            numpy array of velocities with same shape as positions
            
        Method:
            - Central difference for interior points: v[i] = (x[i+1] - x[i-1]) / (2*dt)
            - Forward difference for first point: v[0] = (x[1] - x[0]) / dt
            - Backward difference for last point: v[-1] = (x[-1] - x[-2]) / dt
        """
        positions = np.asarray(positions)
        original_shape = positions.shape
        
        # Handle both 1D and multi-dimensional data
        if positions.ndim == 1:
            positions = positions.reshape(-1, 1)
        
        n_frames = positions.shape[0]
        velocities = np.zeros_like(positions)
        
        # Calculate time deltas
        if timestamps is not None:
            timestamps = np.asarray(timestamps)
            dt = np.diff(timestamps)
        else:
            dt = np.full(n_frames - 1, self.dt)
        
        # Forward difference for first point
        velocities[0] = (positions[1] - positions[0]) / dt[0]
        
        # Central difference for interior points
        for i in range(1, n_frames - 1):
            dt_central = timestamps[i+1] - timestamps[i-1] if timestamps is not None else 2 * self.dt
            velocities[i] = (positions[i+1] - positions[i-1]) / dt_central
        
        # Backward difference for last point
        velocities[-1] = (positions[-1] - positions[-2]) / dt[-1]
        
        # Restore original shape
        return velocities.reshape(original_shape)
    
    def calculate_acceleration(self, velocities, timestamps=None):
        """
        Calculate acceleration using numerical differentiation of velocity.
        
        Args:
            velocities: numpy array of velocity data
            timestamps: Optional numpy array of timestamps
            
        Returns:
            numpy array of accelerations
            
        Note: This is the second derivative of position (a = d²x/dt²)
        """
        # Acceleration is just the derivative of velocity
        return self.calculate_velocity(velocities, timestamps)
    
    def calculate_dynamics_from_positions(self, positions, timestamps=None):
        """
        Calculate both velocity and acceleration from position data.
        
        Args:
            positions: numpy array of position time-series
            timestamps: Optional timestamps
            
        Returns:
            tuple: (velocities, accelerations)
        """
        velocities = self.calculate_velocity(positions, timestamps)
        accelerations = self.calculate_acceleration(velocities, timestamps)
        return velocities, accelerations
    
    def process_csv_data(self, csv_path, output_path=None):
        """
        Process CSV file to add velocity and acceleration columns for all joints.
        
        Args:
            csv_path: Path to input CSV with position data
            output_path: Optional path for output CSV. If None, creates new file
            
        Returns:
            pandas.DataFrame with additional velocity and acceleration columns
            
        Expected CSV format:
            - Must have 'timestamp' column
            - Position columns named like: JOINT_NAME_X, JOINT_NAME_Y, JOINT_NAME_Z
            - Will create: JOINT_NAME_VX, JOINT_NAME_VY, JOINT_NAME_VZ (velocity)
            - Will create: JOINT_NAME_AX, JOINT_NAME_AY, JOINT_NAME_AZ (acceleration)
        """
        # Read CSV
        df = pd.read_csv(csv_path)
        
        if 'timestamp' not in df.columns:
            raise ValueError("CSV must contain 'timestamp' column")
        
        timestamps = df['timestamp'].values
        
        # Find all position columns (assume format: JOINT_X, JOINT_Y, JOINT_Z)
        position_cols = [col for col in df.columns if col.endswith(('_X', '_Y', '_Z'))]
        
        # Group by joint name
        joints = set()
        for col in position_cols:
            # Extract joint name (everything before the last underscore)
            joint_name = '_'.join(col.split('_')[:-1])
            joints.add(joint_name)
        
        print(f"[DYNAMICS] Processing {len(joints)} joints from {Path(csv_path).name}")
        
        # Calculate dynamics for each joint
        for joint in sorted(joints):
            x_col = f"{joint}_X"
            y_col = f"{joint}_Y"
            z_col = f"{joint}_Z"
            
            if all(col in df.columns for col in [x_col, y_col, z_col]):
                # Get position data
                pos_x = df[x_col].values
                pos_y = df[y_col].values
                pos_z = df[z_col].values
                
                # Calculate velocities
                vel_x, acc_x = self.calculate_dynamics_from_positions(pos_x, timestamps)
                vel_y, acc_y = self.calculate_dynamics_from_positions(pos_y, timestamps)
                vel_z, acc_z = self.calculate_dynamics_from_positions(pos_z, timestamps)
                
                # Add to dataframe
                df[f"{joint}_VX"] = vel_x
                df[f"{joint}_VY"] = vel_y
                df[f"{joint}_VZ"] = vel_z
                df[f"{joint}_AX"] = acc_x
                df[f"{joint}_AY"] = acc_y
                df[f"{joint}_AZ"] = acc_z
        
        # Calculate magnitude columns for key joints (useful for analysis)
        key_joints = ['RIGHT_WRIST', 'LEFT_WRIST', 'RIGHT_ANKLE', 'LEFT_ANKLE', 'CoM']
        for joint in key_joints:
            if all(f"{joint}_{axis}" in df.columns for axis in ['VX', 'VY', 'VZ']):
                # Velocity magnitude
                df[f"{joint}_V_MAG"] = np.sqrt(
                    df[f"{joint}_VX"]**2 + 
                    df[f"{joint}_VY"]**2 + 
                    df[f"{joint}_VZ"]**2
                )
                # Acceleration magnitude
                df[f"{joint}_A_MAG"] = np.sqrt(
                    df[f"{joint}_AX"]**2 + 
                    df[f"{joint}_AY"]**2 + 
                    df[f"{joint}_AZ"]**2
                )
        
        # Save output
        if output_path is None:
            input_path = Path(csv_path)
            output_path = input_path.parent / f"{input_path.stem}_dynamics{input_path.suffix}"
        
        df.to_csv(output_path, index=False)
        print(f"[DYNAMICS] Saved to: {output_path}")
        print(f"[DYNAMICS] Added {len(df.columns) - len(pd.read_csv(csv_path).columns)} new columns")
        
        return df
    
    def get_angular_velocity(self, angles, timestamps=None):
        """
        Calculate angular velocity from angle time-series.
        
        Args:
            angles: numpy array of angles in degrees
            timestamps: Optional timestamps
            
        Returns:
            Angular velocity in degrees/second
        """
        return self.calculate_velocity(angles, timestamps)
    
    def get_angular_acceleration(self, angles, timestamps=None):
        """
        Calculate angular acceleration (α) from angle time-series.
        
        This is critical for torque calculation: τ = I * α
        
        Args:
            angles: numpy array of angles in degrees
            timestamps: Optional timestamps
            
        Returns:
            Angular acceleration (α) in degrees/second²
        """
        angular_vel = self.get_angular_velocity(angles, timestamps)
        return self.calculate_velocity(angular_vel, timestamps)


if __name__ == "__main__":
    """
    Standalone test/usage example.
    Usage: python dynamics_engine.py path/to/csv_file.csv
    """
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python dynamics_engine.py <csv_file_path>")
        print("\nExample:")
        print("  python dynamics_engine.py output_data/motion_20260116_motion_enhanced.csv")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    
    if not Path(csv_path).exists():
        print(f"[ERROR] File not found: {csv_path}")
        sys.exit(1)
    
    print("=" * 60)
    print("Dynamics Engine - Velocity & Acceleration Analysis")
    print("=" * 60)
    
    engine = DynamicsEngine(fps=30.0)
    df = engine.process_csv_data(csv_path)
    
    print("\n[SUMMARY]")
    print(f"Total frames: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    
    # Show sample statistics
    if 'RIGHT_WRIST_V_MAG' in df.columns:
        print(f"\nRight Wrist Velocity (magnitude):")
        print(f"  Mean: {df['RIGHT_WRIST_V_MAG'].mean():.4f} m/s")
        print(f"  Max:  {df['RIGHT_WRIST_V_MAG'].max():.4f} m/s")
    
    if 'CoM_A_MAG' in df.columns:
        print(f"\nCenter of Mass Acceleration (magnitude):")
        print(f"  Mean: {df['CoM_A_MAG'].mean():.4f} m/s²")
        print(f"  Max:  {df['CoM_A_MAG'].max():.4f} m/s²")
    
    print("\n" + "=" * 60)
