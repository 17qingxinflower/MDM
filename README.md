# MuskDeerMonitor 🦌

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8-green)](https://github.com/ultralytics/ultralytics)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-lightgrey)]()

**MuskDeerMonitor** is a multi-threaded automated behavior analysis system designed for precision farming of Forest Musk Deer (*Moschus berezovskii*).

Built with **Python**, **YOLOv8**, and a **Producer-Consumer multi-threaded architecture**, this desktop application addresses the challenges of quantifying animal behavior in complex breeding environments. It provides real-time monitoring, multi-source parallel analysis, and high-throughput offline batch processing to support ethological research and animal welfare assessment.

------

## ✨ Key Features

- **⚡ Real-time Individual Monitoring**:
  - Millisecond-latency detection with "What You See Is What You Get" (WYSIWYG) visual feedback.
  - Live status panel updating behavior types (e.g., Foraging, Rumination) and durations instantly.
- **🎥 Multi-source Parallel Analysis**:
  - Supports simultaneous processing of multiple camera feeds (Camera Arrays) to overcome visual blind spots.
  - Independent rendering and inference threads for each stream to ensure stability.
- **🚀 High-throughput Offline Batch Processing**:
  - Accelerated processing for historical video data (faster than real-time).
  - Ideal for analyzing nocturnal activity rhythms and long-term behavioral patterns.
- **🧠 Heuristic Anomaly Detection**:
  - **Expert Rule-based Filtering**: Automatically flags abnormal events based on user-defined thresholds (e.g., prolonged immobility, high-frequency state switching).
  - **Estrus Identification**: Detects "high-frequency switching" patterns associated with mating behaviors.
- **📊 Automatic Visualization & Reporting**:
  - Generates Gantt charts (Time-Action timelines), Frequency Histograms, and Proportion Pie Charts.
  - Exports structured `.xlsx` data with millisecond-level timestamps.

------

## 🛠️ System Architecture

MuskDeerMonitor implements a robust **Producer-Consumer** architecture to decouple video decoding from AI inference:

- **Producer Thread**: Handles video capture and buffering (thread-safe queue) to prevent frame loss.
- **Consumer Thread**: Executes YOLO inference and business logic mapping.
- **Resource-Aware Scheduler**: Dynamically balances frame rates based on CPU/GPU load during multi-video analysis.

------

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (Recommended for real-time performance)
- GPU: NVIDIA GPU with CUDA support (Highly Recommended for real-time analysis)

### 1. Clone the Repository

```bash
git clone [https://github.com/17qingxinflower/MDM.git]
```

------

### 2.Install PyTorch (CUDA Version) ⚠️ Important

**To enable GPU acceleration, you MUST install PyTorch separately before other dependencies.** Default pip installation usually provides the CPU-only version, which is too slow for real-time monitoring.

1. Visit the **PyTorch Official Website**.

2. Select your **OS**, **Package** (Pip), and **Compute Platform** (e.g., CUDA 11.8 or 12.1).

3. Run the generated command.

   *Example command (check website for latest):*

   ```
   pip3 install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
   ```

### 3. Install Remaining Dependencies

After installing PyTorch, run the following command to install the rest of the required libraries (YOLOv8, Pandas, OpenCV, etc.):

```
pip install -r requirements.txt
```

## 🚀 Usage

### 1. Start the Application

Run the main entry script:

```bash
python main.py
```

### 2. Load Model

- Click **File** -> **Select Model File**.

- Choose your trained YOLO weights file (e.g., `best.pt`).

  Pre-trained demonstration model is located in DeerUI/weight/best.pt.*

### 3. Select Input Source

- **Real-time**: Click **File** -> **Camera Settings** to select USB/RTSP camera index (Default: 0).
- **Offline**: Click **File** -> **Select Video File** to load a local `.mp4` or `.avi` file.

### 4. Configure Thresholds (Optional)

- Click **Operations** -> **Set Filter Thresholds**.
- Set minimum duration (e.g., 5s) or maximum transition rate.

### 5. Start Analysis

- **Single Mode**: Click **Start Single Detection** in the GUI.
- **Multi-Mode**: Click **Operations** -> **Start Multi-Video Analysis**.

### 6. View Results

- Upon completion (or stopping), the system automatically generates analysis charts and Excel reports in the `UI_result/` directory.

------

## 📂 Project Structure

```
MuskDeerMonitor/
├── DeerUI/                 # Core source code
│   ├── main.py             # Main application entry
│   ├── weight/             # Model weights directory
│   │   └── best.pt         # Pre-trained demo model
│   └── ...
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── UI_result/              # Output directory (Charts & Excel)
```

------

## 📜 Citation

If you use this software in your research, please cite our paper:

```
@article{MuskDeerMonitor2026,
  title={MuskDeerMonitor: A Multi-threaded Automated Behavior Analysis System for Precision Farming of Forest Musk Deer},
  author={Your Name and Co-authors},
  journal={Journal Name (e.g., Software Impacts)},
  year={2026},
  doi={10.xxxx/xxxxx}
}
```

------

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

------

## 🤝 Acknowledgments

- **Ultralytics YOLO** for the state-of-the-art object detection framework.
- **Dujiangyan Forest Musk Deer Breeding Center** for providing experimental environments and data support.
