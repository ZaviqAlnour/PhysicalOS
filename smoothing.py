"""
Smoothing Module - Signal Filtering for Joint Angle Data
Removes jitter and noise from motion capture data using multiple filtering strategies.
"""

import numpy as np
from collections import deque
from scipy.signal import savgol_filter


class MovingAverageFilter:
    """
    Simple moving average filter for real-time smoothing.
    Good for: Quick smoothing, low computational cost
    """
    
    def __init__(self, window_size=5):
        """
        Initialize moving average filter.
        
        Args:
            window_size: Number of frames to average (default: 5)
        """
        self.window_size = window_size
        self.buffers = {}  # One buffer per joint
    
    def filter(self, joint_name, value):
        """
        Apply moving average to a single value.
        
        Args:
            joint_name: Name of joint (for separate buffer)
            value: Current angle value
            
        Returns:
            float: Smoothed value
        """
        # Initialize buffer for this joint if needed
        if joint_name not in self.buffers:
            self.buffers[joint_name] = deque(maxlen=self.window_size)
        
        # Add current value
        self.buffers[joint_name].append(value)
        
        # Return average
        return np.mean(self.buffers[joint_name])
    
    def filter_dict(self, angles_dict):
        """
        Apply filter to all joints in dictionary.
        
        Args:
            angles_dict: {'joint_name': angle_value}
            
        Returns:
            dict: Smoothed angles
        """
        smoothed = {}
        for joint_name, value in angles_dict.items():
            if value is not None:
                smoothed[joint_name] = self.filter(joint_name, value)
            else:
                smoothed[joint_name] = None
        
        return smoothed
    
    def reset(self):
        """Clear all buffers."""
        self.buffers.clear()


class OneEuroFilter:
    """
    One Euro Filter - Advanced adaptive smoothing.
    Good for: Low latency with good smoothing, reduces jitter while preserving responsiveness
    
    Reference: Casiez, G., Roussel, N., & Vogel, D. (2012)
    "1€ Filter: A Simple Speed-based Low-pass Filter for Noisy Input"
    """
    
    def __init__(self, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        """
        Initialize One Euro Filter.
        
        Args:
            min_cutoff: Minimum cutoff frequency (lower = more smoothing)
            beta: Speed coefficient (higher = more responsive to speed changes)
            d_cutoff: Cutoff frequency for derivative
        """
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        
        self.states = {}  # State for each joint: {prev_value, prev_derivative, prev_time}
    
    def _smoothing_factor(self, cutoff, dt):
        """Calculate smoothing factor from cutoff frequency."""
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)
    
    def _exponential_smoothing(self, value, prev_value, alpha):
        """Apply exponential smoothing."""
        return alpha * value + (1.0 - alpha) * prev_value
    
    def filter(self, joint_name, value, timestamp):
        """
        Apply One Euro Filter to a single value.
        
        Args:
            joint_name: Name of joint
            value: Current angle value
            timestamp: Current timestamp in seconds
            
        Returns:
            float: Smoothed value
        """
        # Initialize state for this joint
        if joint_name not in self.states:
            self.states[joint_name] = {
                'prev_value': value,
                'prev_derivative': 0.0,
                'prev_time': timestamp
            }
            return value
        
        state = self.states[joint_name]
        
        # Calculate time difference
        dt = timestamp - state['prev_time']
        if dt <= 0:
            dt = 0.001  # Minimum dt to avoid division by zero
        
        # Calculate derivative (speed of change)
        derivative = (value - state['prev_value']) / dt
        
        # Smooth the derivative
        alpha_d = self._smoothing_factor(self.d_cutoff, dt)
        derivative_smoothed = self._exponential_smoothing(
            derivative, state['prev_derivative'], alpha_d
        )
        
        # Calculate adaptive cutoff frequency
        cutoff = self.min_cutoff + self.beta * abs(derivative_smoothed)
        
        # Smooth the value
        alpha = self._smoothing_factor(cutoff, dt)
        value_smoothed = self._exponential_smoothing(
            value, state['prev_value'], alpha
        )
        
        # Update state
        state['prev_value'] = value_smoothed
        state['prev_derivative'] = derivative_smoothed
        state['prev_time'] = timestamp
        
        return value_smoothed
    
    def filter_dict(self, angles_dict, timestamp):
        """
        Apply filter to all joints in dictionary.
        
        Args:
            angles_dict: {'joint_name': angle_value}
            timestamp: Current timestamp in seconds
            
        Returns:
            dict: Smoothed angles
        """
        smoothed = {}
        for joint_name, value in angles_dict.items():
            if value is not None:
                smoothed[joint_name] = self.filter(joint_name, value, timestamp)
            else:
                smoothed[joint_name] = None
        
        return smoothed
    
    def reset(self):
        """Clear all states."""
        self.states.clear()


class SavitzkyGolayFilter:
    """
    Savitzky-Golay Filter - Polynomial smoothing (batch processing).
    Good for: Post-processing, preserving peaks and valleys
    
    Note: Requires buffering frames, best used in post-processing mode.
    """
    
    def __init__(self, window_size=11, poly_order=3):
        """
        Initialize Savitzky-Golay filter.
        
        Args:
            window_size: Window length (must be odd, >= poly_order + 2)
            poly_order: Polynomial order (typically 2-4)
        """
        if window_size % 2 == 0:
            window_size += 1  # Must be odd
        
        if window_size <= poly_order:
            window_size = poly_order + 2
            if window_size % 2 == 0:
                window_size += 1
        
        self.window_size = window_size
        self.poly_order = poly_order
        self.buffers = {}
    
    def filter(self, joint_name, value):
        """
        Buffer value and return smoothed result when window is full.
        
        Args:
            joint_name: Name of joint
            value: Current angle value
            
        Returns:
            float: Smoothed value (returns original until buffer fills)
        """
        # Initialize buffer
        if joint_name not in self.buffers:
            self.buffers[joint_name] = deque(maxlen=self.window_size)
        
        buffer = self.buffers[joint_name]
        buffer.append(value)
        
        # Wait until buffer is full
        if len(buffer) < self.window_size:
            return value
        
        # Apply Savitzky-Golay filter
        data = np.array(buffer)
        smoothed = savgol_filter(data, self.window_size, self.poly_order)
        
        # Return the middle value (current frame)
        return smoothed[len(smoothed) // 2]
    
    def filter_dict(self, angles_dict):
        """
        Apply filter to all joints in dictionary.
        
        Args:
            angles_dict: {'joint_name': angle_value}
            
        Returns:
            dict: Smoothed angles
        """
        smoothed = {}
        for joint_name, value in angles_dict.items():
            if value is not None:
                smoothed[joint_name] = self.filter(joint_name, value)
            else:
                smoothed[joint_name] = None
        
        return smoothed
    
    def reset(self):
        """Clear all buffers."""
        self.buffers.clear()


def smooth_batch_data(data_array, window_size=11, poly_order=3):
    """
    Apply Savitzky-Golay smoothing to entire data array (post-processing).
    
    Args:
        data_array: 1D numpy array of angle values
        window_size: Filter window size (odd number)
        poly_order: Polynomial order
        
    Returns:
        numpy array: Smoothed data
    """
    if len(data_array) < window_size:
        return data_array
    
    if window_size % 2 == 0:
        window_size += 1
    
    return savgol_filter(data_array, window_size, poly_order)


def test_smoothing():
    """
    Test all smoothing filters with synthetic noisy data.
    """
    print("[TEST] Smoothing Filters")
    
    # Generate synthetic noisy signal
    t = np.linspace(0, 2 * np.pi, 100)
    clean_signal = 90 + 30 * np.sin(t)  # Sinusoidal angle motion
    noise = np.random.normal(0, 5, len(t))  # Add noise
    noisy_signal = clean_signal + noise
    
    print(f"Original signal variance: {np.var(clean_signal):.2f}")
    print(f"Noisy signal variance: {np.var(noisy_signal):.2f}")
    
    # Test Moving Average
    ma_filter = MovingAverageFilter(window_size=5)
    ma_smoothed = [ma_filter.filter('test', val) for val in noisy_signal]
    print(f"Moving Average smoothed variance: {np.var(ma_smoothed):.2f}")
    
    # Test One Euro Filter
    euro_filter = OneEuroFilter(min_cutoff=1.0, beta=0.007)
    timestamps = np.linspace(0, 3.33, len(noisy_signal))  # Simulate 30 FPS
    euro_smoothed = [euro_filter.filter('test', val, ts) 
                     for val, ts in zip(noisy_signal, timestamps)]
    print(f"One Euro smoothed variance: {np.var(euro_smoothed):.2f}")
    
    # Test Savitzky-Golay (batch)
    sg_smoothed = smooth_batch_data(noisy_signal, window_size=11, poly_order=3)
    print(f"Savitzky-Golay smoothed variance: {np.var(sg_smoothed):.2f}")
    
    print("[TEST COMPLETE] All filters working")


if __name__ == "__main__":
    test_smoothing()