# MuskDeer Monitor

MuskDeer Monitor is a PySide6 desktop application for automatic musk deer behavior detection, monitoring, alerting, and result export. It uses a YOLO model to identify four behavior classes in enclosure video: `Egestion`, `Feeding`, `Stand/Walk`, and `Lie Down`.

This repository is a minimized SoftwareX review package. It contains the runnable application code, one PyTorch detection model, one bundled test video for reviewer reproduction, and one short system demonstration video. Build folders, packaged executables, cached wheels, local databases, generated outputs, and TensorRT/ONNX exports are intentionally excluded to keep the repository small and reproducible.

## Software Metadata

| Item | Description |
| --- | --- |
| Software name | MuskDeer Monitor |
| Current version | 0.1.0 PySide6 minimal review package |
| Programming language | Python 3.11+ |
| Main interface | PySide6 desktop GUI |
| Core ML dependency | Ultralytics YOLO |
| Database | SQLite by default; PostgreSQL can be configured for managed deployments |
| License | MIT, see `LICENSE.md` |
| Operating system tested | Windows 11 |
| Repository contents | Source code, runtime requirements, YOLO `.pt` model, bundled test video, demo video |

## Repository Layout

```text
deerui/                  Application source code
scripts/                 Benchmark and diagnostic scripts
weights/best.pt          Bundled YOLO behavior-detection model
benchmark_assets/        Reviewer test video used by smoke/benchmark commands
demo/                    Short system demonstration videos
requirements.txt         Runtime Python dependencies
pyproject.toml           Project metadata
.env.example             Optional runtime configuration template
LICENSE.md               MIT license
```

The following large or local-only artifacts are excluded: `dist/`, `build/`, `UI_result/`, `offline_wheels/`, `deployment/`, `packaging/`, local databases, cache folders, TensorRT `.engine` files, ONNX exports, and test/development notes.

## Hardware And Software Requirements

- Windows 10/11 or a Linux system with GUI support.
- Python 3.11 or newer.
- Conda or Miniconda for environment management.
- CPU execution is supported for functional review.
- NVIDIA GPU plus a matching PyTorch CUDA build is recommended for real-time detection.
- A local camera, RTSP stream, or video file can be used as an input source.

## Conda Installation

Create a clean Conda environment. The environment name below is only a suggestion for reviewers; it does not assume that an existing `yolov8n` environment is present.

```powershell
git clone https://github.com/17qingxinflower/MDM.git
cd MDM

conda create -n muskdeer-monitor python=3.11 -y
conda activate muskdeer-monitor

python -m pip install --upgrade pip setuptools wheel
```

Install PyTorch first. For NVIDIA GPUs with CUDA 12.8:

```powershell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

For CPU-only review:

```powershell
python -m pip install torch torchvision torchaudio
```

Then install the remaining application dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Configuration

The application works without editing configuration files. By default it uses:

- model: `weights/best.pt`
- database: `sqlite+pysqlite:///./deerui_dev.db`
- output directory: `UI_result/`
- benchmark video: `benchmark_assets/053358f40cf7b5cfe5d8620a94d621b9.mp4`

To override settings, copy `.env.example` to `deerui.env` and edit the values:

```powershell
Copy-Item .env.example deerui.env
```

For managed installations, `DEERUI_DATABASE_URL` can be changed to a PostgreSQL connection string.

## Quick Verification

Run a smoke test to validate imports and configuration:

```powershell
python -m deerui.app --smoke-test
```

Initialize the local SQLite database:

```powershell
python -m deerui.app --init-db
```

Start the GUI:

```powershell
python -m deerui.app
```

In the GUI, load `weights/best.pt`, select the bundled test video from `benchmark_assets/`, assign a deer ID such as `2.1.6`, and start detection. Results are written under `UI_result/`.

## Optional Benchmark

The packaged benchmark command can exercise multiple input streams against the bundled test video. For a short reviewer check:

```powershell
python -m deerui.app --capacity-benchmark --benchmark-streams 1 --benchmark-duration 5
```

Benchmark results are saved under `UI_result/benchmarks/`.

## Demonstration Videos

- [Watch the complete system workflow demo](https://github.com/17qingxinflower/MDM/raw/main/demo/system_full_feature_demo.mp4) (`demo/system_full_feature_demo.mp4`) demonstrates the complete system workflow and major functions.
- [Watch the multi-threaded detection and data-recording demo](https://github.com/17qingxinflower/MDM/raw/main/demo/system_demo_2026-07-13.mp4) (`demo/system_demo_2026-07-13.mp4`) demonstrates multi-threaded detection and data-recording behavior.

These videos are included for SoftwareX reviewers who want a quick visual confirmation of the workflow before running the application locally.

## Main Functions

- Real-time or video-file behavior detection with YOLO.
- Multi-camera monitoring matrix with shared inference workers.
- Batch processing for historical videos.
- Deer profile registration and task history management.
- Alert rules for long behavior duration and high-frequency behavior transitions.
- Export of raw behavior records and alert records.
- Behavior timeline, frequency, and time-share visualizations.

## Notes For Reviewers

This repository is intentionally smaller than the development workspace. The excluded packaging outputs can be regenerated from the source code and requirements, while the included `.pt` model and test video are sufficient to verify the main detection workflow.
