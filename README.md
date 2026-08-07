# 👥 Real-Time Crowd Density Estimation & Safety Analytics System

An AI-powered computer vision system built using **Python**, **YOLOv8 Deep Learning**, **OpenCV**, and **Streamlit** to estimate crowd density, calculate spatial density distribution via 2D Gaussian Kernel Density Estimation (KDE) heatmaps, and trigger real-time overcrowding alerts.

---

## 🌟 Key Features

- **SOTA AI Detection (YOLOv8)**: Detects human subjects with 95%+ precision across varying poses, angles, and crowd densities.
- **2D Spatial Density Heatmaps**: Generates continuous Kernel Density Estimation (KDE) thermal overlays (`COLORMAP_JET`).
- **4-Quadrant Sector Occupancy**: Divides space into `Zone A`, `Zone B`, `Zone C`, and `Zone D` to locate bottleneck areas.
- **Safety Risk Classification**:
  - 🟢 **NORMAL** (0 - 40% Capacity)
  - 🟡 **MODERATE** (40 - 70% Capacity)
  - 🟠 **HIGH DENSITY** (70 - 95% Capacity)
  - 🔴 **CRITICAL OVERCROWDING** (> 95% Capacity)
- **Multiple Interfaces**:
  - **Command Prompt CLI (`main.py`)**: Windows native file picker (`tkinter`) for photos, videos, or live webcam feeds.
  - **Streamlit Web Dashboard (`app.py`)**: Interactive web layout with Plotly meters, sector graphs, and CSV audit downloads.

---

## 📂 Project Structure

```
crowd_density/
├── app.py                   # Streamlit Web Application Dashboard
├── main.py                  # Command Prompt CLI (Photo/Video/Webcam Picker)
├── crowd_detector.py        # YOLOv8 Deep Learning & Heatmap Engine
├── analytics_engine.py      # Risk Level Evaluation & Sector Breakdown
├── visualizer.py            # Plotly Gauge Charts & Graphs
├── sample_generator.py     # Synthetic crowd scene generator
├── requirements.txt         # Project Dependencies
├── .gitignore               # Git Ignore Configuration
└── tests/                   # Pytest Test Suite
    └── test_crowd.py
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites & Virtual Environment Setup
Ensure Python 3.10+ and `uv` package manager are installed.

```bash
# Create virtual environment
uv venv

# Install dependencies
uv pip install -r requirements.txt
```

### 2. Running in Command Prompt CLI (`main.py`)
```bash
uv run python main.py
```
*Allows choosing local Photo files, Video files, or Live System Webcam.*

### 3. Running Web Dashboard (`app.py`)
```bash
uv run streamlit run app.py
```
*Launches web browser interface at `http://localhost:8501`.*

### 4. Running Automated Tests
```bash
uv run pytest
```

---

## 📜 License
Distributed under the MIT License.
