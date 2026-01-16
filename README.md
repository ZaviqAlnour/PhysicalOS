# PhysicalOS: The "Photons to Intent" Engine

**PhysicalOS** is a professional-grade, offline motion capture pipeline designed to extract high-fidelity physical intent from simple video inputs. It bridges the gap between raw computer vision (photons) and robotic control (physics), proving that you don't need a million-dollar mocap studio to get useful motion data.

> **Philosophy**: Photons $\to$ Geometry $\to$ Physics $\to$ Memory $\to$ Intent

---

## ⚡ Key Features

- **Privacy First**: Runs 100% offline. No cloud APIs, no data leaks.
- **Physics-Based**: Computes **Velocity**, **Acceleration**, and **Center of Mass** (CoM).
- **Robot Ready**: Exports camera-independent relative coordinates (e.g., "Hand relative to Hip").
- **Signal Processing**:
  - **One Euro Filter** for adaptive jitter reduction (smooth slow motion, fast reflexes).
  - **Dynamics Engine** for $\tau = I\alpha$ torque-ready data.
- **Advanced Tools**:
  - **Ghost Predictor**: Upsample 30 FPS video to **400 FPS** for smooth robot control.
  - **GRF Estimator**: Detects ground contact to prevent "floating" physics.
- **Visual Intelligence**: Real-time 3D visualization with live metric graphs.

---

## 📦 Installation

**Requirements**: Windows 10/11, Python 3.10+

1.  **Clone/Download** the repository.
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: Requires `opencv-python`, `mediapipe`, `numpy`, `pandas`, `scipy`, `matplotlib`)*

---

## 🚀 Usage

Run the main orchestrator to access the interactive menu:

```bash
python run.py
```

You will be presented with 4 options:

### 1. Live Video (Webcam)
- **Best for**: Rapid prototyping and testing.
- Opens your webcam and records motion in real-time.
- Shows live **3D Stickman**, **CoM Graph**, and **Elbow Angle Graph**.
- Saves data continuously to `output_data/`.

### 2. Recorded Video (Single File)
- **Best for**: Analyzing a specific movement.
- Place your video in `input_videos/`.
- Enter the filename when prompted (e.g., `jump.mp4`).
- Processes with full visual feedback.

### 3. All Input Files (Batch Processing)
- **Best for**: Processing large datasets overnight.
- Automatically processes **EVERY** video in `input_videos/`.
- Runs in "Headless Mode" (no GUI) for maximum speed.

### 4. Training Data Library
- **Best for**: Machine Learning dataset creation.
- Processes videos from `Training_Data/raw_videos/`.
- Aggregates all results into a single master CSV (`aggregated_data.csv`).

---

## 🛠️ Advanced Tools

### Motion Interpolation (The "Ghost Predictor")
Convert standard 30 FPS webcam footage into high-frequency 400 FPS control data.

```bash
python scripts/interpolate_motion.py output_data/my_movement.csv my_movement_400fps.csv
```

### Dynamics & Force Analysis
Every time you process a video, the system automatically runs the **Dynamics Engine**:
- Calculates **Velocity** (`_VX`, `_VY`, `_VZ`) and **Acceleration** (`_AX`, `_AY`, `_AZ`) for all joints.
- Estimates **Ground Reaction Forces** and adds a `Support_Leg` column (`left`, `right`, `both`, `none`).

---

## 📂 Project Structure

```
physicalOs/
├── input_videos/          # Place your source MP4s here
├── output_data/           # Generated CSVs live here
├── Training_Data/         # Infrastructure for ML datasets
│   ├── raw_videos/
│   └── processed_csv/
├── scripts/
│   ├── interpolate_motion.py  # 30->400 FPS Upsampler
│   └── aggregate_dataset.py   # Dataset builder
├── src/physical_os/       # Core Engine Source
│   ├── main.py            # Orchestrator
│   ├── vision_engine.py   # MediaPipe Wrapper
│   ├── dynamics_engine.py # Velocity/Acceleration Logic
│   ├── grf_estimator.py   # Ground Contact Logic
│   ├── visualizer_live.py # Real-time Graphs
│   └── ... (physics, kinematics, smoothing)
└── run.py                 # Entry point
```

## 📊 Data Format

The **Enhanced CSV** output contains:
- **Time**: Timestamp in seconds.
- **Angles**: Biomechanical joint angles (Degrees).
- **Positions**: X/Y/Z coordinates relative to the user's hips (Meters).
- **Physics**: CoM Position, Velocity Magnitude, Acceleration Magnitude.
- **Logic**: Support Leg state/Boolean flags.

---

**PhysicalOS** — *Give your robot a sense of self.*