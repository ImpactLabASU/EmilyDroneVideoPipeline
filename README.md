# 🚁 EMILY Drone: Physics-Informed Parameter Estimation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Research](https://img.shields.io/badge/License-Research-green.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.48550/arXiv.xxxx.xxxxx-blue)](https://proceedings.mlr.press/v255/banerjee24a.html)

**EMILY** (Extracting sparse Model from ImpLicit dYnamics) is an end-to-end pipeline that estimates physical drone parameters from video data using physics-informed neural networks. The system combines computer vision, audio processing, and 6DOF physics simulation to learn 12 physical parameters of a quadcopter drone.

## ✨ Features

- **🎯 Advanced Drone Detection**: Multi-modal tracking with video stabilization
- **🎵 Audio Analysis**: Rotor sound extraction and motor signal estimation  
- **🧠 Physics-Informed Learning**: 6DOF simulation integrated into loss function
- **📊 Parameter Estimation**: 12 physical drone parameters (inertia, thrust, drag, etc.)
- **🎬 3D Visualization**: Animated trajectory simulation with Z-up orientation
- **🔧 Multiple YOLO Models**: Support for YOLOv8, YOLO11 (n/s/m/l variants)

## 🚀 Quick Start

### Prerequisites
- Python 3.10–3.12
- ffmpeg (for audio processing)
  - **macOS**: `brew install ffmpeg`
  - **Ubuntu**: `sudo apt install ffmpeg`
  - **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/)

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd Automate_EMILY_Drone

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### One-Command Execution
```bash
# Complete pipeline: video → detection → training → simulation
python3 -m venv .venv && source .venv/bin/activate && \
  pip install --upgrade pip && pip install -r requirements.txt && \
  [ -f yolo11n.pt ] || curl -L -o yolo11n.pt https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo11n.pt && \
  EMILY_AUTOMATE_MODE=1 EMILY_RUN_ORCHESTRATOR=1 python3 new_run.py --video DroneVideo.mp4 --weights yolo11n.pt && \
  EMILY_RUN_ORCHESTRATOR=0 MPLBACKEND=Agg python3 new_run.py --epochs 3 --size 32 --model ltc --log 1
```

## 📁 Input Files

Place these files in the project root:
- **`DroneVideo.mp4`** - Input video of drone flight
- **YOLO weights** (auto-downloaded if missing):
  - `yolo11n.pt` - YOLO11 nano (recommended, fastest)
  - `yolo11s.pt` - YOLO11 small (balanced)
  - `yolo11m.pt` - YOLO11 medium (best accuracy)
  - `yolo11l.pt` - YOLO11 large (highest accuracy)

## 🎮 Usage Examples

### Video Processing Only
```bash
# Process video with YOLO11n (recommended)
EMILY_ENABLE_VISION=1 python3 new_run.py \
  --video DroneVideo.mp4 \
  --weights yolo11n.pt \
  --out_video annotated.mp4 \
  --out_csv data/trajectory.csv \
  --conf 0.25

# Or use YOLO11m for better accuracy
EMILY_ENABLE_VISION=1 python3 new_run.py \
  --video DroneVideo.mp4 \
  --weights yolo11m.pt \
  --out_video annotated.mp4 \
  --out_csv data/trajectory.csv \
  --conf 0.25
```

### Training & Simulation Only
```bash
# Run EMILY training and 3D simulation
EMILY_RUN_ORCHESTRATOR=0 MPLBACKEND=TkAgg python3 new_run.py \
  --epochs 5 \
  --size 64 \
  --model ltc \
  --log 1
```

### Live 3D Simulation
```bash
# Show live 3D animation window
EMILY_RUN_ORCHESTRATOR=0 MPLBACKEND=TkAgg python3 new_run.py \
  --epochs 3 --size 32 --model ltc --log 1
```

## 📊 Output Files

### Generated Data
- **`data/trajectory.csv`** - Frame-by-frame tracking data (x,y,z,confidence)
- **`data/trajectory_with_audio.csv`** - Includes audio features (RMS, frequency)
- **`data/xData.txt`, `yData.txt`, `zData.txt`** - Position matrices (165×1701)
- **`data/uMotor1.txt` - `uMotor4.txt`** - Motor control signals from audio
- **`data/minMotor.txt`, `maxMotor.txt`** - Motor speed constraints

### Visualizations
- **`annotated.mp4`** - Video with tracking annotations
- **`Output/drone_animation.gif`** - 3D trajectory animation
- **`Output/theta_simulation_full.png`** - Static trajectory plot

### Results
- **`outputV2GPUV3_torch_ltc.csv`** - Estimated parameters and performance metrics

## 🔬 Estimated Parameters

The system estimates 12 physical drone parameters:

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| **dxm, dym** | X/Y arm lengths | 0.1-0.3 m |
| **dzm** | Z arm length | 0.02-0.1 m |
| **IBxx, IByy, IBzz** | Moments of inertia | 0.001-0.1 kg⋅m² |
| **Cd** | Drag coefficient | 0.01-1.0 |
| **kTh** | Thrust coefficient | 1e-6 - 1e-4 |
| **kTo** | Torque coefficient | 1e-8 - 1e-6 |
| **tau2** | Motor time constant | 0.01-0.1 s |
| **kp** | Motor gain | 0.1-10.0 |
| **damp** | Motor damping | 0.1-2.0 |

## 🏗️ Architecture

### 1. Video Processing Pipeline
- **Video Stabilization**: Optical flow-based frame stabilization
- **Multi-modal Tracking**: Template → Feature → Detection fallback
- **ROI Management**: Dynamic search region around expected position
- **Depth Estimation**: Size-based or MiDaS depth estimation

### 2. Audio Processing
- **Feature Extraction**: RMS energy, spectral centroid, peak frequency
- **Motor Signal Estimation**: Audio-to-motor mapping
- **Temporal Alignment**: Audio features synchronized with video frames

### 3. Physics-Informed Learning
- **6DOF Simulation**: Complete quadcopter dynamics with quaternions
- **Custom Loss Function**: Physics-based loss integrating simulation
- **LTC Neural Network**: Liquid Time-Constant network for parameter estimation

## ⚙️ Configuration

### Environment Variables
- `EMILY_RUN_ORCHESTRATOR=1` - Run full pipeline (video + audio + training)
- `EMILY_ENABLE_VISION=1` - Enable video processing only
- `EMILY_AUTOMATE_MODE=1` - Automation mode for scripting
- `MPLBACKEND=TkAgg` - GUI backend for live visualization
- `MPLBACKEND=Agg` - Headless backend for servers
- `MIDAS_ONNX=path` - Path to MiDaS depth estimation model

### Command Line Options
```bash
# Video processing options
--video PATH              # Input video file
--weights PATH            # YOLO model weights
--out_video PATH          # Output annotated video
--out_csv PATH            # Output trajectory CSV
--conf FLOAT              # Detection confidence threshold (0.1-0.9)
--imgsz INT               # Input image size (640, 1280, etc.)

# Training options
--epochs INT              # Number of training epochs
--size INT                # Hidden layer size (16, 32, 64, 128)
--model STR               # Model type (ltc, lstm, gru)
--log INT                 # Logging frequency
```

## 🔧 Troubleshooting

### Common Issues

**1. Missing Dependencies**
```bash
# Install ffmpeg
brew install ffmpeg  # macOS
sudo apt install ffmpeg  # Ubuntu

# Reinstall Python packages
pip install --upgrade pip
pip install -r requirements.txt
```

**2. YOLO Model Issues**
```bash
# Download YOLO11 weights manually
curl -L -o yolo11n.pt https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo11n.pt
curl -L -o yolo11s.pt https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo11s.pt
curl -L -o yolo11m.pt https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo11m.pt
curl -L -o yolo11l.pt https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo11l.pt
```

**3. Display Issues**
```bash
# Try different backends
MPLBACKEND=TkAgg python3 new_run.py ...  # GUI
MPLBACKEND=MacOSX python3 new_run.py ... # macOS
MPLBACKEND=Agg python3 new_run.py ...    # Headless
```

**4. Depth Estimation Problems**
- Z values too small: Calibrate camera focal length
- Z values unrealistic: Check drone size estimation
- Use `--focal_px` and `--real_width_m` parameters

### Performance Tips

- **Faster Processing**: Use YOLO11n (nano)
- **Balanced Performance**: Use YOLO11s (small)
- **Better Accuracy**: Use YOLO11m (medium)
- **Highest Accuracy**: Use YOLO11l (large)
- **Memory Issues**: Reduce `--imgsz` or `--size`
- **Training Speed**: Reduce `--epochs` or `--size`

## 📈 Performance Benchmarks

| Model | Speed (FPS) | Accuracy | Memory | Use Case |
|-------|-------------|----------|---------|----------|
| YOLO11n | 40+ | Good | Low | Quick testing, real-time |
| YOLO11s | 35+ | Better | Low-Medium | Balanced performance |
| YOLO11m | 25+ | Best | Medium | Recommended for most cases |
| YOLO11l | 15+ | Excellent | High | Research, highest accuracy |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@InProceedings{pmlr-v255-banerjee24a,
  title = {EMILY: Extracting sparse Model from ImpLicit dYnamics},
  author = {Banerjee, Ayan and Gupta, Sandeep},
  booktitle = {Proceedings of the 1st ECAI Workshop on "Machine Learning Meets Differential Equations: From Theory to Applications"},
  pages = {1--11},
  year = {2024},
  editor = {Coelho, Cec{\i}lia and Zimmering, Bernd and Costa, M. Fernanda P. and Ferr{\'a}s, Lu{\'\i}s L. and Niggemann, Oliver},
  volume = {255},
  series = {Proceedings of Machine Learning Research},
  month = {20 Oct},
  publisher = {PMLR},
  pdf = {https://raw.githubusercontent.com/mlresearch/v255/main/assets/banerjee24a/banerjee24a.pdf},
  url = {https://proceedings.mlr.press/v255/banerjee24a.html},
  abstract = {Sparse model recovery requires us to extract model coefficients of ordinary differential equations (ODE) with few nonlinear terms from data. This problem has been effectively solved in recent literature for the case when all state variables of the ODE are measured. In practical deployments, measurements of all the state variables of the underlying ODE model of a process are not available, resulting in implicit (unmeasured) dynamics. In this paper, we propose EMILY, that can extract the underlying ODE of a dynamical process even if much of the dynamics is implicit. We show the utility of EMILY on four baseline examples and compare with the state-of-the-art techniques such as SINDY-MPC. Results show that unlike SINDY-MPC, EMILY can recover model coefficients accurately under implicit dynamics.}
}
```

## 👥 Credits

- **Created by**: Farhat Shaikh
- **Original Research**: Ayan Banerjee, Sandeep Gupta (IMPACT Lab ASU)
- **Institution**: Arizona State University, IMPACT Lab

## 📄 License

**Property of IMPACT Lab ASU. For research use only.**

---

**Happy Flying! 🚁✨**
