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

## Credits
Github Created by: Farhat Shaikh

## Citation
If you use this repository, please cite the EMILY paper:

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

## License
Property of IMPACT Lab ASU. For research use.
