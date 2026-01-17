# PhysicalOS: Technical Specifications & Dynamics Layer

## 1. Mathematical Foundation of Motion Derivatives

PhysicalOS implements numerical differentiation to derive velocity ($v$) and acceleration ($a$) from positional time-series data. Unlike naive implementations that rely on forward or backward differences, PhysicalOS enforces **Central Difference** for all interior points to minimize truncation error and ensure phase alignment.

### Numerical Differentiation
For a position vector $x$ at frame $i$ with a constant time delta $dt$:

$$v_i = \frac{x_{i+1} - x_{i-1}}{2dt}$$
$$a_i = \frac{v_{i+1} - v_{i-1}}{2dt}$$

This approach yields $O(dt^2)$ accuracy, providing the stability required for high-frequency motor control and torque estimation ($\tau = I\alpha$).

## 2. Temporal Upsampling & Signal Continuity

To bridge the gap between computer vision (typically 30 FPS) and robotics control loops (typically 400Hz - 1kHz), PhysicalOS employs a **Cubic Spline Interpolation** pipeline.

- **Input Frequency**: ~30 Hz (Raw MediaPipe Landmarks)
- **Target Output**: 400 Hz
- **Smoothing**: One-Euro Filter (adaptive cutoff) applied pre-interpolation to eliminate jitter while preserving high-speed transients.
- **Continuity**: Enforces $C^2$ continuity across frame boundaries, preventing "stair-step" artifacts that could induce resonance in physical actuators.

## 3. Biomechanical Constraint System (Safety Layer)

The `constraints.py` module acts as a runtime safety governor. It enforces hard limits on joint articulation to prevent physically impossible or self-destructive commands.

| Joint | Lower Limit | Upper Limit | Description |
| :--- | :--- | :--- | :--- |
| **Elbow** | 0° | 160° | Full extension to deep flexion |
| **Shoulder Flexion** | 0° | 180° | Neutral to overhead reach |
| **Knee** | 0° | 140° | Full extension to flexion |
| **Hip Flexion** | 0° | 120° | Standing to seated/squat |

**Implementation**: Clamping is applied via `numpy.clip` immediately after the smoothing phase and before any downstream simulation or logging.

## 4. Dynamics & Balance (Ground Reaction Force)

PhysicalOS calculates stability through a real-time Center of Mass (CoM) projection relative to the Support Base.

### Support Leg Enforcement
The system utilizes a **Single Pivot Logic** for Ground Reaction Force (GRF) estimation. Specifically:
- $\text{Pivot} = \min(L_{ankle\_y}, R_{ankle\_y})$
- The foot with the lowest vertical coordinate is treated as the fixed anchoring point for the skeletal frame.
- Simultaneous double-support is intentionally suppressed to provide deterministic pivot points for torque calculations.

### Balance Proof
A real-time projection of the CoM is validated against the support polygon. If $\text{Proj}(\text{CoM})$ shifts outside the convex hull of the grounded feet, the system triggers an instability flag (UI state: Red), signaling a loss of static balance.

## 5. Data Architecture (Enhanced CSV)

The pipeline produces an **Enhanced CSV** format designed for Large-Scale Learning (LSL) and imitation learning.

- **Header Schema**: `timestamp`, `joint_angles`, `3d_relative_positions`, `velocities`, `accelerations`, `grf_labels`.
- **Coordinate System**: Mid-Hip relative, normalized to torso length.
- **Labeling**: Every sequence is tagged with a task-specific label (e.g., `sand_shoveling_001`) to support supervised training of motion primitives.
