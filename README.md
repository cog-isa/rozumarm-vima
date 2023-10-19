# Code for the "Fine-tuning Multimodal Transformer Models for Generating Actions in Virtual and Real Environments" paper

How to install:
1. Clone to the project root or install the following repos: [cv-utils](https://github.com/andrey1908/kas_utils), [cam-utils](https://github.com/andrey1908/kas_camera_utils), [cv-repo](https://github.com/andrey1908/rozumarm_vima_cv), [ultralytics](http://github.com/andrey1908/ultralytics), and [arm-utils](https://github.com/ag-cdsl/rozumarm-vima-utils).
1. Install this package and dependencies from `requirements.txt`.
1. Add to `PYTHONPATH` the paths to `rozumarm_vima_cv`, `utils`, and `camera_utils` directories that you cloned.
1. Download all missing VIMA checkpoints from https://github.com/vimalabs/VIMA

How to run:
- to start (cube detector -> sim -> oracle -> arm) pipeline, run `scripts/run_aruco2sim_loop.py`
- to start (cube detector -> sim -> ML model -> arm) pipeline, run `scripts/run_model_loop.py`
- to start (cam image -> ML model -> arm) pipeline, set `USE_OBS_FROM_SIM=False` in `scripts/run_model_loop.py` and run it 


Links to datasets:
- [Sweep-Plan](https://disk.yandex.ru/d/R_YkzOZDwpxs_g)
- [Sweep-Seg](https://disk.yandex.ru/d/hvBoKFwiCq-M2g)