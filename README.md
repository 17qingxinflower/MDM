# MuskDeerMonitor

MuskDeerMonitor is a desktop application for automated Forest Musk Deer
(*Moschus berezovskii*) behavior monitoring. This repository is the minimal
runnable package prepared for GitHub and SoftwareX review: the GUI program, one
trained YOLO model, and one small test video.

This release is based on:

```text
检测与统计各版本备份/26-3-16-2 实时模式下的加速，跳播/main.py
```

It uses `customtkinter`, OpenCV, Ultralytics YOLO, SQLite, pandas, and
matplotlib. Training scripts, experimental notebooks, intermediate datasets,
generated outputs, and later UI-modernization branches are intentionally not
included in this minimal repository.

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
| Included test video | `video/test_video.mp4` |
| Runtime outputs | `UI_result/`, `history.db` |

## Included Files

```text
main.py                # Desktop GUI application
weight/Demo.pt         # YOLO behavior detection model
video/test_video.mp4   # Small test video for users and reviewers
requirements.txt       # Python runtime dependencies
LICENSE.md             # MIT license
README.md              # User and reviewer documentation
```

## Behavior Classes

The included model and code use seven Forest Musk Deer behavior labels:

| Model label | Chinese display | English display |
| --- | --- | --- |
| `hunt` | 搜寻 | Foraging |
| `egestion` | 排遗 | Excretion |
| `Standing_feeding` | 站立取食 | Standing Feeding |
| `Stand_or_walk` | 行走 | Walking |
| `Lie_down_and_rest` | 卧息 | Resting |
| `Licking_the_pussy` | 舔阴 | Anogenital Licking |
| `Comb_and_lick` | 梳舔 | Grooming |

## Features

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
above. CPU-only PyTorch can also run the demo video, but inference will be
slower.

## Run

```powershell
python main.py
```

In the GUI:

1. Click `1. 载入 AI 权重模型` and select `weight/Demo.pt`.
2. Click `2. 选择主监控流接入`.
3. Choose `Yes` for local video and select `video/test_video.mp4`, or choose `No`
   to use camera index `0`.
4. Click `开始单路实时检测`.
5. Use the progress and speed controls for seeking and accelerated playback.
6. Use `专家过滤规则调整` to tune abnormal-behavior review thresholds.

For offline processing, click `历史录像离线批处理` and select a video file.

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
`python main.py`, loading `weight/Demo.pt`, and selecting `video/test_video.mp4`.

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
