# Automate EMILY Drone

End-to-end pipeline to:
- Detect and track a drone in a video (x, y, z)
- Extract rotor audio and estimate motor signals (uMotor1..4)
- Train a physics-informed model (LTC) briefly
- Run an automated 3D (Z-up) simulation and save a GIF

## Requirements
- Python 3.10–3.12
- ffmpeg (for audio via moviepy)
  - macOS: `brew install ffmpeg`

Install Python deps (recommended pinned):
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Files to provide
- `DroneVideo.mp4` in repo root (input video)
- YOLO weights `yolov8n.pt` in repo root (auto-downloaded by the one-liner below if missing)

## Quick start (headless, saves GIF only)
Runs: extract → audio → data → quick train → simulation (saves `Output/drone_animation.gif`).
```bash
python3 -m venv .venv && source .venv/bin/activate && \
  pip install --upgrade pip && pip install -r requirements.txt && \
  [ -f yolov8n.pt ] || curl -L -o yolov8n.pt https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt && \
  EMILY_AUTOMATE_MODE=1 EMILY_RUN_ORCHESTRATOR=1 python3 run.py --video DroneVideo.mp4 --weights yolov8n.pt && \
  EMILY_RUN_ORCHESTRATOR=0 MPLBACKEND=Agg python3 run.py --epochs 1 --size 32 --model ltc --log 1
```

## Live simulation window (local)
After data is generated, run the 3D (Z-up) animation window:
```bash
EMILY_RUN_ORCHESTRATOR=0 MPLBACKEND=TkAgg python3 run.py --epochs 1 --size 32 --model ltc --log 1
```
- On macOS, if `TkAgg` has issues, try: `MPLBACKEND=MacOSX`.

## Outputs
- `data/trajectory.csv`, `data/trajectory_with_audio.csv`
- `data/xData.txt`, `yData.txt`, `zData.txt`
- `data/uMotor1..4.txt`, `minMotor.txt`, `maxMotor.txt`
- `annotated.mp4`, `annotated_audio.mp4`
- `Output/drone_animation.gif` (final simulation)

## Env variables
- `EMILY_RUN_ORCHESTRATOR=1` to run the full extract→audio→data pipeline
- `EMILY_AUTOMATE_MODE=1` used to gate internal mains when orchestrating
- `MPLBACKEND=Agg` for headless CI; `TkAgg`/`MacOSX` for local GUI
- `MIDAS_ONNX=path/to/midas_small.onnx` if using `--depth_mode midas` (optional)

## Troubleshooting
- Missing YOLO weights: the quick-start downloads `yolov8n.pt` automatically
- Missing ffmpeg: install with your OS package manager
- Live window doesn’t show: switch backend (`TkAgg` ↔ `MacOSX`) or run headless (`Agg`)
- Data not found: make sure the end-to-end command ran before the simulation-only command

## License
Property of IMPACT Lab ASU. For research use.
