# PhysicalOS

**Advanced Biomechanical Motion Capture & Robotics Intent Pipeline**

PhysicalOS is a high-performance, offline-first motion analysis engine designed to convert standard RGB video input into high-fidelity, robot-ready motion primitives. It bridges the gap between raw computer vision and physical robotic control by enforcing mathematical integrity and biomechanical safety.

## 🚀 Key Features

- **Real-Time 400Hz Pipeline**: Upsamples sparse vision data (30 FPS) to 400Hz using Cubic Spline Interpolation for high-frequency motor control compatibility.
- **Mathematical Integrity**: Enforces Central Difference calculations for velocity and acceleration to ensure precise motion derivatives.
- **Biomechanical Safety Layer**: Hard-coded joint constraints (e.g., Elbow 0°-160°) prevent physically impossible robot commands.
- **Balance & CoM Analysis**: Real-time Center of Mass projection and Ground Reaction Force (GRF) estimation for stability validation.
- **Industry-Standard Logging**: Generates labeled, enhanced CSV datasets optimized for imitation learning and robotics simulation.

## 🛠 Project Structure

```text
physical_os/
├── src/
│   ├── main.py                 # Orchestrator
│   ├── constraints.py          # Safety Layer (Joint Limits)
│   ├── dynamics_engine.py      # Numerical Derivatives (Central Difference)
│   ├── interpolation.py        # 400Hz Spline Upsampling
│   ├── grf_estimator.py        # Pivot Point & Support Detection
│   ├── visualizer_live.py      # Real-time Analysis & Recording UI
│   └── ...                     # Core Vision/Kinematics Modules
├── output_data/                # Labeled Enhanced CSVs
└── input_videos/               # Raw MP4 Input Source
```

## 🚥 Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Orchestrator**:
   ```bash
   python run.py
   ```

3. **Select Mode**:
   - `Option 1`: Live Camera (with Recording/Labeling UI)
   - `Option 2`: Single Video Processing
   - `Option 3/4`: Batch Processing & Dataset Aggregation

## 📊 Recording Workflow

The system includes a dedicated UI for generating labeled motion datasets:
1. Launch **Live Video** mode.
2. Enter a **Task Label** (e.g., `reaching_high`).
3. Click **Start REC** (captures 10-second high-fidelity sequences).
4. Output files are automatically post-processed with full dynamics and GRF data.

## 📄 Documentation

For deep-dive engineering specifications, mathematical foundations, and joint limit definitions, see [TECHNICAL_DETAILS.md](TECHNICAL_DETAILS.md).

---
*Optimized for Windows 11 | Python 3.11.9 | MediaPipe | NumPy | Matplotlib*