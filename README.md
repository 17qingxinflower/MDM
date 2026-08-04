# MuskDeerMonitor

MuskDeerMonitor is a desktop application for automated Forest Musk Deer
(*Moschus berezovskii*) behavior monitoring. This repository is the minimal
runnable package prepared for GitHub and SoftwareX review: the GUI program, one
trained YOLO model, and two test videos.

This release follows the March 16, 2026 realtime acceleration and seek-control
snapshot selected by the maintainers. It uses `customtkinter`, OpenCV,
Ultralytics YOLO, SQLite, pandas, and matplotlib. Training scripts, experimental
notebooks, intermediate datasets, generated outputs, and later UI-modernization
branches are intentionally not included in this minimal repository.

## SoftwareX Metadata

| Item | Value |
| --- | --- |
| Current code version | `2026-03-16-realtime-minimal` |
| Repository | <https://github.com/17qingxinflower/MDM> |
| License | MIT, see `LICENSE.md` |
| Version control | Git / GitHub |
| Language | Python |
| Main dependencies | customtkinter, OpenCV, Pillow, Ultralytics YOLO, pandas, openpyxl, matplotlib |
| Main program | `main.py` |
| Included model | `weight/Demo.pt` |
| Included test videos | `video/ch01_20160707002310.mp4`, `video/ch01_20160708011933.mp4` |
| Runtime outputs | `UI_result/`, `history.db` |

## Included Files

```text
main.py                         # Desktop GUI application
weight/Demo.pt                  # YOLO behavior detection model
video/ch01_20160707002310.mp4   # Test video 1
video/ch01_20160708011933.mp4   # Test video 2
requirements.txt                # Python runtime dependencies
LICENSE.md                      # MIT license
README.md                       # User and reviewer documentation
```

## Behavior Classes

The included model and code use seven Forest Musk Deer behavior labels:

| Model label | English display |
| --- | --- |
| `hunt` | Foraging |
| `egestion` | Excretion |
| `Standing_feeding` | Standing Feeding |
| `Stand_or_walk` | Walking |
| `Lie_down_and_rest` | Resting |
| `Licking_the_pussy` | Anogenital Licking |
| `Comb_and_lick` | Grooming |

## Features

- English desktop GUI for reviewer-facing and user-facing operation.
- Real-time single-camera or video-file behavior detection.
- Video progress control with seeking and playback-speed adjustment.
- Frame-queue based processing for smoother real-time operation.
- Multi-view monitoring wall for several camera or video sources.
- Offline historical-video batch analysis.
- Behavior duration and frequency recording.
- Deer ID management and task-history browsing with SQLite.
- Expert threshold settings for abnormal-behavior review.
- Excel export for raw behavior streams and abnormal-review records.
- Interactive behavior charts for timeline, frequency, and proportion review.

## Installation

Use Python 3.10 or 3.11. Windows is recommended because the current desktop
workflow uses Tkinter file dialogs and Windows-style local file opening.

With conda:

```powershell
git clone https://github.com/17qingxinflower/MDM.git
cd MDM

conda create -n mdm python=3.11 -y
conda activate mdm

python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

With `venv`:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

If your GPU, driver, or CUDA version is different, install the PyTorch command
recommended by the official PyTorch selector instead of the CUDA 12.8 example
above. CPU-only PyTorch can also run the test videos, but inference will be
slower.

## Run

```powershell
python main.py
```

In the GUI:

1. Click `Load AI Model` and select `weight/Demo.pt`.
2. Click `Connect Video Source`.
3. Select a deer ID, then choose `Yes` for local video.
4. Select either `video/ch01_20160707002310.mp4` or
   `video/ch01_20160708011933.mp4`.
5. Click `Start Detection`.
6. Use the progress and speed controls for seeking and accelerated playback.
7. Use `Behavior Alert Settings` to tune abnormal-behavior review thresholds.

For offline processing, click `Video Batch` and select a video file.

## Outputs

Runtime outputs are generated in the working directory:

```text
UI_result/YYYYMMDD/
history.db
```

Typical generated files include behavior-record Excel files, abnormal-review
Excel files, timeline charts, frequency charts, and proportion charts. These
runtime outputs are ignored by Git and are not committed to the repository.

## Validation

The selected source file can be checked with:

```powershell
python -m py_compile main.py
```

The program is GUI-based, so full functional validation requires launching
`python main.py`, loading `weight/Demo.pt`, and selecting one of the included
test videos.

## Citation

Before the SoftwareX paper receives a DOI, cite the repository and code version:

```bibtex
@software{muskdeermonitor_realtime_2026,
  title = {MuskDeerMonitor: A desktop system for automated Forest Musk Deer behavior monitoring},
  author = {{MuskDeerMonitor contributors}},
  year = {2026},
  version = {2026-03-16-realtime-minimal},
  url = {https://github.com/17qingxinflower/MDM}
}
```
