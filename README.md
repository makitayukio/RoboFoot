# robofoot-tracker

Robot football tracker using color tag detection — a final project for **Processamento Digital de Imagens** at **UFABC**.

## Group

| Name | RA |
|---|---|
| Igor Ladeia de Freitas | 11201922180 |
| Gustavo Fernandes do Nascimento | 11202021700 |
| Ryan Lucas da Silva | 11202522362 |
| Eduardo Yukio Makita | 11202020221 |

## Description

A computer vision system that detects and tracks the **position and orientation of robots** on a robot football (RoboCup/VSS) field using overhead camera footage. Each robot carries a color tag (team color + two ID colors), and the ball is detected by its orange color.

## Prerequisites

- **Python >= 3.10**
- **pip** (Python package manager)

### Required Libraries

| Library | Version | Purpose |
|---|---|---|
| `opencv-python` | >= 4.5 | All image processing (color conversion, CLAHE, morphological ops, contour finding, homography, undistortion, drawing) |
| `numpy` | >= 1.21 | Array operations and perspective transforms |
| `jupyter` / `notebook` | Opening and running the `.ipynb` notebooks |

## Installation

### 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install the package

```bash
pip install -e .
```

### 4. Install Jupyter to run the notebooks

```bash
pip install notebook
```

## Project Structure

```
trabalho/
├── robofoot_tracker/       # Core Python package
│   ├── __init__.py
│   ├── models.py           # Dataclasses + COLOR_PAIR_TO_ID
│   ├── color_config.py     # HSV color ranges
│   ├── geometry.py         # Bezier curve utility
│   ├── calibration.py      # Homography + distortion + interactive calibration
│   ├── detector.py         # Robot + ball detection via color segmentation
│   ├── preprocessing.py    # CLAHE + Gaussian blur
│   ├── tracker.py          # Main Tracker class
│   └── viz.py              # Annotation drawing
├── tests/                  # Test suite
├── frames/                 # Extracted video frames
├── Trabalho_Final.ipynb    # Main report notebook (theory + pipeline demo)
├── Cenario_Aplicacao.ipynb # Application scenario notebook
├── pyproject.toml
└── README.md
```

## How to Run

### Option 1: Run the Notebooks (Recommended)

The notebooks contain the full pipeline demonstration with explanations and visual outputs.

1. **Start Jupyter:**

   ```bash
   jupyter notebook
   ```

2. **Open `Trabalho_Final.ipynb`** in your browser.

3. **Run all cells in order:**
   - Go to **Cell → Run All** (or press `Ctrl+Shift+Enter` / `Cmd+Shift+Enter`).
   - The notebook will:
     1. Import the package and set up calibration points.
     2. Load a frame from the `frames/` directory.
     3. Apply preprocessing (CLAHE + Gaussian blur) and show before/after with histograms.
     4. Compute the homography matrix.
     5. Run detection on a range of frames and display annotated results.

> **Note:** Cells must be executed in order — later cells depend on imports and variables defined earlier.

### Option 2: Use the Package Programmatically

```python
from robofoot_tracker import Tracker, FieldDimensions

tracker = Tracker(
    video_path="video.mp4",
    field_dims=FieldDimensions(width_cm=150, height_cm=130),
)

for frame_result in tracker:
    print(f"Frame {frame_result.frame_idx}:")
    for robot in frame_result.robots:
        print(f"  Robot {robot.team}-{robot.robot_id}: pos=({robot.x_cm:.1f}, {robot.y_cm:.1f}), angle={robot.angle_deg:.1f}")
    if frame_result.ball:
        print(f"  Ball: pos=({frame_result.ball.x_cm:.1f}, {frame_result.ball.y_cm:.1f})")
```

## Configuration

### Calibration Points

The default calibration maps pixel coordinates to field coordinates (cm). The corners are specified in **TL, TR, BR, BL** order (top-left, top-right, bottom-right, bottom-left):

```python
CALIBRATION_POINTS = [(100, 80), (540, 80), (540, 400), (100, 400)]
```

### Field Dimensions

Default: **150 × 130 cm** (width × height).

### Color Calibration

HSV ranges for each tag color are pre-configured in `ColorConfig`. For interactive color calibration, use:

```python
from robofoot_tracker import calibrate_colors_interactive
```

## Troubleshooting

| Issue | Solution |
|---|---|
| **`ModuleNotFoundError: No module named 'robofoot_tracker'`** | Make sure the virtual environment is activated (`source .venv/bin/activate`) and the package is installed (`pip install -e .`). |
| **OpenCV `cv2.imshow` windows don't appear** | This is expected in headless environments or Jupyter. The notebooks use `matplotlib` for display instead. For local GUI mode, run the `Tracker.run_live()` method from a standard Python script. |
| **No detections found in a frame** | Check that the frame contains visible color tags. Dark/blank frames are skipped by default (`blank_threshold=5.0`). Adjust HSV ranges via `ColorConfig.set_range()` if colors aren't being detected. |
| **`video.mp4` not found** | The project uses extracted frames in the `frames/` directory. If you only have frames, modify the notebook to load from there instead of a video file. |
| **Jupyter cell outputs are missing** | Run cells sequentially from the top. Some cells depend on variables (e.g., `VIDEO_PATH`, `CALIBRATION_POINTS`) defined earlier. Use **Cell → Run All** to ensure correct order. |
| **Red color not detected** | Red spans the hue wraparound at 0/180. The config already handles this with two HSV ranges — if issues persist, adjust them via `ColorConfig.set_range("red", lower, upper)`. |
