"""
Advanced Visualizer - Multi-Plot Real-Time Feedback
Extends the stickman view with real-time graphs for metrics analysis.

Features:
1. 3D Stickman Animation (Real-time skeleton)
2. CoM_Y Graph (Vertical Center of Mass height)
3. Right Elbow Angle Graph (Joint angle time-series)

Purpose: Visually detect noise, instability, or biomechanical spikes.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import cv2
from pathlib import Path
from collections import deque

class AdvancedVisualizer:
    def __init__(self, history_size=100):
        """
        Initialize the advanced visualizer.
        
        Args:
            history_size: Number of previous frames to show in graphs
        """
        self.history_size = history_size
        
        # Data history for graphs
        self.time_history = deque(maxlen=history_size)
        self.com_y_history = deque(maxlen=history_size)
        self.elbow_angle_history = deque(maxlen=history_size)
        
        # Setup the figure and subplots
        self.fig = plt.figure(figsize=(12, 8))
        self.fig.canvas.manager.set_window_title('PhysicalOS V1 - Advanced Motion Analysis')
        
        # 1. 3D Stickman Plot (Large, left side)
        self.ax_3d = self.fig.add_subplot(121, projection='3d')
        
        # 2. CoM_Y Graph (Top right)
        self.ax_com = self.fig.add_subplot(222)
        self.ax_com.set_title("Center of Mass (Y-Height)")
        self.ax_com.set_ylabel("Height (normalized)")
        self.line_com, = self.ax_com.plot([], [], 'b-', label='CoM_Y')
        self.ax_com.grid(True)
        
        # 3. Right Elbow Graph (Bottom right)
        self.ax_elbow = self.fig.add_subplot(224)
        self.ax_elbow.set_title("Right Elbow Angle")
        self.ax_elbow.set_ylabel("Degrees")
        self.ax_elbow.set_xlabel("Time (s)")
        self.line_elbow, = self.ax_elbow.plot([], [], 'r-', label='R_Elbow')
        self.ax_elbow.grid(True)
        
        # Stickman connections (MediaPipe indices)
        self.connections = [
            (11, 12), (11, 23), (12, 24), (23, 24), # Torso
            (11, 13), (13, 15),                     # Left Arm
            (12, 14), (14, 16),                     # Right Arm
            (23, 25), (25, 27),                     # Left Leg
            (24, 26), (26, 28)                      # Right Leg
        ]
        
        plt.tight_layout()
        
    def update(self, landmarks, com_y, elbow_angle, timestamp):
        """
        Update all plots with new data.
        
        Args:
            landmarks: 33 body landmarks (x, y, z)
            com_y: Numeric CoM height
            elbow_angle: Numeric elbow angle in degrees
            timestamp: Current time in seconds
        """
        # 1. Update History
        self.time_history.append(timestamp)
        self.com_y_history.append(com_y)
        self.elbow_angle_history.append(elbow_angle)
        
        # 2. Clear and Update 3D Stickman
        self.ax_3d.clear()
        self.ax_3d.set_title("3D Intent Representation")
        
        # Set bounds (normalized space)
        self.ax_3d.set_xlim(-1, 1)
        self.ax_3d.set_ylim(-1, 1)
        self.ax_3d.set_zlim(-1, 1)
        
        if landmarks is not None:
            # Format: In MediaPipe, Y is down. In 3D plots, we usually want Y up.
            # We map: MP_X -> Plot_X, MP_Z -> Plot_Y (depth), MP_Y -> Plot_Z (height, inverted)
            xs = landmarks[:, 0]
            ys = landmarks[:, 2] # Depth
            zs = -landmarks[:, 1] # Inverted Height
            
            # Draw joints
            self.ax_3d.scatter(xs, ys, zs, c='g', s=20)
            
            # Draw bones
            for connection in self.connections:
                start_idx, end_idx = connection
                self.ax_3d.plot(
                    [xs[start_idx], xs[end_idx]],
                    [ys[start_idx], ys[end_idx]],
                    [zs[start_idx], zs[end_idx]],
                    'g-', linewidth=2
                )
            
            # Highlight CoM if available as a virtual joint (usually calculated separately)
            # For now, we just plot the skeletal data.
            
        # 3. Update CoM_Y Graph
        if len(self.time_history) > 1:
            self.line_com.set_data(list(self.time_history), list(self.com_y_history))
            self.ax_com.relim()
            self.ax_com.autoscale_view()
            
            # 4. Update Elbow Graph
            self.line_elbow.set_data(list(self.time_history), list(self.elbow_angle_history))
            self.ax_elbow.relim()
            self.ax_elbow.autoscale_view()
            
        plt.pause(0.001) # Allow figure to update

    def close(self):
        plt.close(self.fig)

if __name__ == "__main__":
    # Dummy test
    viz = AdvancedVisualizer()
    for t in np.linspace(0, 10, 100):
        # Generate some dummy sine wave data
        dummy_landmarks = np.random.normal(0, 0.1, (33, 3))
        dummy_com = 0.5 + 0.1 * np.sin(t)
        dummy_elbow = 90 + 30 * np.cos(t)
        viz.update(dummy_landmarks, dummy_com, dummy_elbow, t)
    viz.close()
