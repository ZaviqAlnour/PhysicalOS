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
from matplotlib.widgets import Button, TextBox

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
        
        # Recording state
        self.is_recording = False
        self.task_label = "unlabeled"
        self.recording_start_time = 0
        self.recording_duration = 0
        self.on_record_toggle = None # Callback function
        self.on_label_change = None  # Callback function
        
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
        
        # 4. Recording Controls UI
        # Add space at bottom for buttons
        plt.subplots_adjust(bottom=0.2)
        
        # Record Button
        ax_rec = self.fig.add_axes([0.1, 0.05, 0.1, 0.075])
        self.btn_rec = Button(ax_rec, 'Start REC', color='lightgrey', hovercolor='red')
        self.btn_rec.on_clicked(self._toggle_recording)
        
        # Label Input
        ax_label = self.fig.add_axes([0.3, 0.05, 0.25, 0.075])
        self.txt_label = TextBox(ax_label, 'Label: ', initial='unlabeled')
        self.txt_label.on_submit(self._set_label)
        
        # Duration indicator
        self.rec_status_text = self.fig.text(0.6, 0.06, 'Status: IDLE', fontsize=12, fontweight='bold')
        
        plt.tight_layout(rect=[0, 0.15, 1, 1])
        
    def update(self, landmarks, com_3d, elbow_angle, timestamp):
        """
        Update all plots with new data.
        
        Args:
            landmarks: 33 body landmarks (x, y, z)
            com_3d: 3D point [x, y, z] representing Center of Mass
            elbow_angle: Numeric elbow angle in degrees
            timestamp: Current time in seconds
        """
        # 1. Update History
        com_y = com_3d[1] if com_3d is not None else 0
        self.time_history.append(timestamp)
        self.com_y_history.append(com_y)
        self.elbow_angle_history.append(elbow_angle)
        
        # 2. Clear and Update 3D Stickman
        self.ax_3d.clear()
        self.ax_3d.set_title("PhysicalOS - Balance Analysis")
        
        # Set bounds (normalized space)
        self.ax_3d.set_xlim(-1, 1)
        self.ax_3d.set_ylim(-1, 1)
        self.ax_3d.set_zlim(-1, 1)
        
        if landmarks is not None:
            # Format: In MediaPipe, Y is down. In 3D plots, we usually want Y up.
            xs = landmarks[:, 0]
            ys = landmarks[:, 2] # Depth
            zs = -landmarks[:, 1] # Inverted Height
            
            # Balance Proof Logic:
            # Check if CoM (x, z) is within support area (between feet)
            stickman_color = 'g' # Healthy / Balanced
            
            if com_3d is not None:
                com_x, com_z = com_3d[0], com_3d[2]
                left_foot_x, left_foot_z = landmarks[31, 0], landmarks[31, 2] # Left Foot index
                right_foot_x, right_foot_z = landmarks[32, 0], landmarks[32, 2] # Right Foot index
                
                # Simple 1D support area (X-axis) for visual demonstration
                min_x = min(left_foot_x, right_foot_x) - 0.05
                max_x = max(left_foot_x, right_foot_x) + 0.05
                
                if not (min_x <= com_x <= max_x):
                    stickman_color = 'r' # UNSTABLE
            
            # Draw joints
            self.ax_3d.scatter(xs, ys, zs, c=stickman_color, s=20)
            
            # Draw bones
            for connection in self.connections:
                start_idx, end_idx = connection
                self.ax_3d.plot(
                    [xs[start_idx], xs[end_idx]],
                    [ys[start_idx], ys[end_idx]],
                    [zs[start_idx], zs[end_idx]],
                    color=stickman_color, linewidth=2
                )
            
            # Draw CoM Dot (Bright Red)
            if com_3d is not None:
                cx, cz, cy = com_3d[0], com_3d[2], -com_3d[1]
                self.ax_3d.scatter([cx], [cz], [cy], c='red', s=100, label='CoM', edgecolor='white')
            
        # 3. Update Status and Recording Info
        if self.is_recording:
            elapsed = timestamp - self.recording_start_time
            self.rec_status_text.set_text(f'REC: [{self.task_label}] {elapsed:.1f}s')
            self.rec_status_text.set_color('red')
            if elapsed >= 10.0: # Auto-stop at 10s as requested
                self._toggle_recording(None)
        else:
            self.rec_status_text.set_text(f'IDLE: {self.task_label}')
            self.rec_status_text.set_color('black')

        # 4. Update CoM_Y Graph
        if len(self.time_history) > 1:
            self.line_com.set_data(list(self.time_history), list(self.com_y_history))
            self.ax_com.relim()
            self.ax_com.autoscale_view()
            
            # 4. Update Elbow Graph
            self.line_elbow.set_data(list(self.time_history), list(self.elbow_angle_history))
            self.ax_elbow.relim()
            self.ax_elbow.autoscale_view()
            
        plt.pause(0.001) # Allow figure to update

    def _toggle_recording(self, event):
        self.is_recording = not self.is_recording
        if self.is_recording:
            self.btn_rec.label.set_text('Stop REC')
            from time import time
            # We'll use the timestamp from update loop for precision, 
            # but need a trigger flag for main.py
        else:
            self.btn_rec.label.set_text('Start REC')
        
        if self.on_record_toggle:
            self.on_record_toggle(self.is_recording)

    def _set_label(self, text):
        self.task_label = text.strip().replace(" ", "_")
        if self.on_label_change:
            self.on_label_change(self.task_label)

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
