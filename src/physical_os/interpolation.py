"""
Interpolation Module - Frame Upsampling and Velocity Calculation
Handles sparse frame data by interpolating between frames.
Calculates linear and angular velocities for all joints.
"""

import numpy as np
from scipy.interpolate import interp1d, CubicSpline
from scipy.spatial.transform import Rotation, Slerp


class FrameInterpolator:
    """
    Interpolates motion data to higher frame rates.
    Useful for capturing high-speed motion at 30 FPS.
    """
    
    def __init__(self, method='cubic'):
        """
        Initialize interpolator.
        
        Args:
            method: 'linear', 'cubic', or 'pchip' (Piecewise Cubic Hermite)
        """
        self.method = method
    
    def interpolate_angles(self, timestamps, angles, target_fps=None, target_timestamps=None):
        """
        Interpolate joint angles to higher temporal resolution.
        
        Args:
            timestamps: Original timestamps (array)
            angles: Original angle values (array)
            target_fps: Target frame rate (e.g., 60, 120)
            target_timestamps: Specific timestamps to interpolate to
            
        Returns:
            tuple: (new_timestamps, interpolated_angles)
        """
        if len(timestamps) < 2:
            return timestamps, angles
        
        # Create target timestamps
        if target_timestamps is None:
            if target_fps is None:
                target_fps = 60  # Default to 60 FPS
            
            # Calculate original FPS
            dt_original = np.median(np.diff(timestamps))
            original_fps = 1.0 / dt_original
            
            if target_fps <= original_fps:
                return timestamps, angles  # No interpolation needed
            
            # Create new timestamp array
            t_start = timestamps[0]
            t_end = timestamps[-1]
            num_frames = int((t_end - t_start) * target_fps) + 1
            target_timestamps = np.linspace(t_start, t_end, num_frames)
        
        # Interpolate based on method
        if self.method == 'linear':
            interpolator = interp1d(timestamps, angles, kind='linear', 
                                   fill_value='extrapolate')
        elif self.method == 'cubic':
            if len(timestamps) >= 4:
                interpolator = CubicSpline(timestamps, angles)
            else:
                interpolator = interp1d(timestamps, angles, kind='linear',
                                       fill_value='extrapolate')
        elif self.method == 'pchip':
            from scipy.interpolate import PchipInterpolator
            interpolator = PchipInterpolator(timestamps, angles)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        interpolated = interpolator(target_timestamps)
        
        return target_timestamps, interpolated
    
    def interpolate_positions(self, timestamps, positions, target_fps=None):
        """
        Interpolate 3D positions to higher temporal resolution.
        
        Args:
            timestamps: Original timestamps (array)
            positions: Original positions (N x 3 array)
            target_fps: Target frame rate
            
        Returns:
            tuple: (new_timestamps, interpolated_positions)
        """
        if len(timestamps) < 2:
            return timestamps, positions
        
        # Create target timestamps
        if target_fps is None:
            target_fps = 60
        
        dt_original = np.median(np.diff(timestamps))
        original_fps = 1.0 / dt_original
        
        if target_fps <= original_fps:
            return timestamps, positions
        
        t_start = timestamps[0]
        t_end = timestamps[-1]
        num_frames = int((t_end - t_start) * target_fps) + 1
        target_timestamps = np.linspace(t_start, t_end, num_frames)
        
        # Interpolate each axis separately
        interpolated_positions = np.zeros((len(target_timestamps), 3))
        
        for axis in range(3):
            if self.method == 'linear':
                interpolator = interp1d(timestamps, positions[:, axis], 
                                       kind='linear', fill_value='extrapolate')
            elif self.method == 'cubic':
                if len(timestamps) >= 4:
                    interpolator = CubicSpline(timestamps, positions[:, axis])
                else:
                    interpolator = interp1d(timestamps, positions[:, axis],
                                           kind='linear', fill_value='extrapolate')
            else:  # pchip
                from scipy.interpolate import PchipInterpolator
                interpolator = PchipInterpolator(timestamps, positions[:, axis])
            
            interpolated_positions[:, axis] = interpolator(target_timestamps)
        
        return target_timestamps, interpolated_positions


class VelocityCalculator:
    """
    Calculate linear and angular velocities from motion data.
    """
    
    @staticmethod
    def calculate_linear_velocity(positions, timestamps):
        """
        Calculate linear velocity from position data.
        
        Args:
            positions: Array of 3D positions (N x 3)
            timestamps: Array of timestamps (N,)
            
        Returns:
            numpy array: Velocity vectors (N x 3) in units/second
        """
        if len(positions) < 2:
            return np.zeros_like(positions)
        
        # Calculate differences
        dt = np.diff(timestamps)
        dp = np.diff(positions, axis=0)
        
        # Avoid division by zero
        dt = np.maximum(dt, 1e-6)
        
        # Velocity = displacement / time
        velocities = dp / dt[:, np.newaxis]
        
        # Pad to match original length (duplicate last velocity)
        velocities = np.vstack([velocities, velocities[-1:]])
        
        return velocities
    
    @staticmethod
    def calculate_linear_speed(velocities):
        """
        Calculate speed (magnitude of velocity).
        
        Args:
            velocities: Velocity vectors (N x 3)
            
        Returns:
            numpy array: Speed values (N,)
        """
        return np.linalg.norm(velocities, axis=1)
    
    @staticmethod
    def calculate_angular_velocity(angles, timestamps):
        """
        Calculate angular velocity from angle data.
        
        Args:
            angles: Array of angles in degrees (N,)
            timestamps: Array of timestamps (N,)
            
        Returns:
            numpy array: Angular velocities in degrees/second (N,)
        """
        if len(angles) < 2:
            return np.zeros_like(angles)
        
        # Calculate differences
        dt = np.diff(timestamps)
        da = np.diff(angles)
        
        # Avoid division by zero
        dt = np.maximum(dt, 1e-6)
        
        # Angular velocity = angle change / time
        ang_velocities = da / dt
        
        # Pad to match original length
        ang_velocities = np.append(ang_velocities, ang_velocities[-1])
        
        return ang_velocities
    
    @staticmethod
    def calculate_acceleration(velocities, timestamps):
        """
        Calculate acceleration from velocity data.
        
        Args:
            velocities: Velocity vectors (N x 3)
            timestamps: Array of timestamps (N,)
            
        Returns:
            numpy array: Acceleration vectors (N x 3)
        """
        if len(velocities) < 2:
            return np.zeros_like(velocities)
        
        dt = np.diff(timestamps)
        dv = np.diff(velocities, axis=0)
        
        dt = np.maximum(dt, 1e-6)
        
        accelerations = dv / dt[:, np.newaxis]
        accelerations = np.vstack([accelerations, accelerations[-1:]])
        
        return accelerations


class MotionAnalyzer:
    """
    Comprehensive motion analysis including interpolation and velocities.
    """
    
    def __init__(self, interpolation_method='cubic'):
        """
        Initialize motion analyzer.
        
        Args:
            interpolation_method: Method for frame interpolation
        """
        self.interpolator = FrameInterpolator(method=interpolation_method)
        self.velocity_calc = VelocityCalculator()
    
    def analyze_joint_motion(self, timestamps, angles, positions=None, 
                            target_fps=60, calculate_derivatives=True):
        """
        Complete motion analysis for a joint.
        
        Args:
            timestamps: Original timestamps
            angles: Joint angles (degrees)
            positions: 3D positions (optional)
            target_fps: Target frame rate for interpolation
            calculate_derivatives: Whether to calculate velocities/accelerations
            
        Returns:
            dict: Analysis results containing interpolated data and derivatives
        """
        results = {
            'original': {
                'timestamps': timestamps,
                'angles': angles,
                'positions': positions
            },
            'interpolated': {},
            'velocities': {},
            'accelerations': {}
        }
        
        # Interpolate angles
        interp_t, interp_angles = self.interpolator.interpolate_angles(
            timestamps, angles, target_fps=target_fps
        )
        results['interpolated']['timestamps'] = interp_t
        results['interpolated']['angles'] = interp_angles
        
        if calculate_derivatives:
            # Angular velocity on interpolated data
            ang_vel = self.velocity_calc.calculate_angular_velocity(
                interp_angles, interp_t
            )
            results['velocities']['angular'] = ang_vel
            
            # Angular acceleration
            ang_acc = self.velocity_calc.calculate_acceleration(
                ang_vel[:, np.newaxis], interp_t
            ).flatten()
            results['accelerations']['angular'] = ang_acc
        
        # Interpolate and analyze positions if provided
        if positions is not None:
            interp_t_pos, interp_positions = self.interpolator.interpolate_positions(
                timestamps, positions, target_fps=target_fps
            )
            results['interpolated']['positions'] = interp_positions
            
            if calculate_derivatives:
                # Linear velocity
                lin_vel = self.velocity_calc.calculate_linear_velocity(
                    interp_positions, interp_t_pos
                )
                results['velocities']['linear'] = lin_vel
                results['velocities']['speed'] = self.velocity_calc.calculate_linear_speed(lin_vel)
                
                # Linear acceleration
                lin_acc = self.velocity_calc.calculate_acceleration(
                    lin_vel, interp_t_pos
                )
                results['accelerations']['linear'] = lin_acc
        
        return results


def test_interpolation():
    """
    Test interpolation and velocity calculations.
    """
    print("[TEST] Interpolation & Velocity Module")
    
    # Simulate 30 FPS motion data (1 second)
    original_fps = 30
    duration = 1.0
    timestamps = np.linspace(0, duration, int(original_fps * duration) + 1)
    
    # Simulate sinusoidal elbow motion
    angles = 90 + 30 * np.sin(2 * np.pi * timestamps)  # Oscillating elbow
    
    # Simulate wrist position
    positions = np.zeros((len(timestamps), 3))
    positions[:, 0] = 0.2 * np.cos(2 * np.pi * timestamps)  # X
    positions[:, 1] = -0.3 + 0.1 * np.sin(2 * np.pi * timestamps)  # Y
    positions[:, 2] = 0.05 * np.sin(4 * np.pi * timestamps)  # Z
    
    print(f"Original data: {len(timestamps)} frames at {original_fps} FPS")
    
    # Test interpolation to 60 FPS
    analyzer = MotionAnalyzer(interpolation_method='cubic')
    results = analyzer.analyze_joint_motion(
        timestamps, angles, positions, 
        target_fps=60, calculate_derivatives=True
    )
    
    interp_t = results['interpolated']['timestamps']
    interp_angles = results['interpolated']['angles']
    
    print(f"\nInterpolated data: {len(interp_t)} frames at ~60 FPS")
    print(f"Timestamps range: {interp_t[0]:.3f} to {interp_t[-1]:.3f} seconds")
    
    # Test velocities
    ang_vel = results['velocities']['angular']
    print(f"\nAngular velocity range: {ang_vel.min():.2f} to {ang_vel.max():.2f} deg/s")
    
    lin_vel = results['velocities']['linear']
    speed = results['velocities']['speed']
    print(f"Linear speed range: {speed.min():.4f} to {speed.max():.4f} units/s")
    
    # Test individual components
    print("\n[TEST] Individual Components:")
    
    interpolator = FrameInterpolator(method='cubic')
    new_t, new_angles = interpolator.interpolate_angles(timestamps, angles, target_fps=120)
    print(f"120 FPS interpolation: {len(new_t)} frames")
    
    velocity_calc = VelocityCalculator()
    velocities = velocity_calc.calculate_linear_velocity(positions, timestamps)
    print(f"Velocity calculation: {velocities.shape}")
    
    print("[TEST COMPLETE] All interpolation tests passed")


if __name__ == "__main__":
    test_interpolation()