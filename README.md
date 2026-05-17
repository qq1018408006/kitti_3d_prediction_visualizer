# KITTI 3D Detection Prediction Visualizer

[中文](./README_zh.md)

This repository provides a standalone script, [`visualize_kitti_predictions.py`](./visualize_kitti_predictions.py), for visualizing KITTI-format 3D object detection predictions in:

- an Open3D point cloud viewer
- the RGB image plane with projected 3D bounding boxes

Ground truth and predictions are rendered together for quick qualitative comparison.

Color convention:

- Ground truth: green
- Prediction: red

## Features

- Visualize KITTI-format predictions, ground-truth labels, LiDAR, calibration, and RGB images
- Support three modes: `pcd`, `image`, and `both`
- Render 3D boxes in Open3D with tube-style mesh edges so box thickness is actually visible
- Project 3D boxes onto `image_2`
- Colorize LiDAR points by sampling RGB values from the image
- Preserve the Open3D camera view as much as possible when moving across frames
- Adjust point size interactively with an OpenCV trackbar
- Save projected RGB visualizations to disk

## Expected Directory Layout

The script accepts either:

1. a dataset root directory, or
2. the `pred/` directory directly

Expected KITTI-style layout:

```text
DATA_ROOT/
├── calib/
├── image_2/
├── label_2/
├── pred/
└── velodyne/
```

Required files per sample:

- `pred/<sample_id>.txt`: prediction results in KITTI detection format
- `label_2/<sample_id>.txt`: ground-truth labels
- `calib/<sample_id>.txt`: calibration file
- `velodyne/<sample_id>.bin`: LiDAR point cloud
- `image_2/<sample_id>.png`: RGB image

## Label Format

The script parses KITTI detection labels with either:

- 15 fields: standard label format
- 16 fields: label format with score

## Installation

```bash
pip install numpy open3d opencv-python
```

## Environment Notes

Before importing GUI libraries, the script sets these defaults for better X11 compatibility:

```bash
QT_QPA_PLATFORM=xcb
GDK_BACKEND=x11
SDL_VIDEODRIVER=x11
CLUTTER_BACKEND=x11
```

This is mainly useful on Wayland desktops where Open3D/OpenCV windows can otherwise be less stable. If you already manage your own GUI environment variables, your shell environment can still override these defaults.

## Usage

Visualize a single sample in both point cloud and image views:

```bash
python3 visualize_kitti_predictions.py \
  --input_path /path/to/DATA_ROOT \
  --sample-id 002425 \
  --mode both
```

Point cloud view only:

```bash
python3 visualize_kitti_predictions.py \
  --input_path /path/to/DATA_ROOT \
  --sample-id 002425 \
  --mode pcd
```

Image projection only:

```bash
python3 visualize_kitti_predictions.py \
  --input_path /path/to/DATA_ROOT \
  --sample-id 002425 \
  --mode image
```

Save the RGB projection result:

```bash
python3 visualize_kitti_predictions.py \
  --input_path /path/to/DATA_ROOT \
  --sample-id 002425 \
  --mode image \
  --image-output ./vis_002425.png
```

If `--sample-id` is omitted, the script uses the first `.txt` file in `pred/` after sorting.

## Arguments

```text
--input_path
```

- Dataset root containing `pred/`, or the `pred/` directory itself
- The current script contains a local default path, so in practice you should usually pass this explicitly

```text
--sample-id
```

- Frame id to visualize, for example `002425`

```text
--mode {pcd,image,both}
```

- `pcd`: point cloud view only
- `image`: RGB projection only
- `both`: show both viewers

```text
--image-output
```

- Output path for saving the projected RGB visualization
- In the current script logic, this save-and-exit path is intended for `--mode image`

```text
--point-size
```

- Initial Open3D point size
- Can still be adjusted after launch with the trackbar window

```text
--box-thickness
```

- Line thickness for projected 3D boxes in the image view

```text
--open3d-box-radius
```

- Tube radius for 3D boxes in Open3D

## Keyboard Controls

In `image` or `both` mode:

- `n` or `d`: next frame
- `p` or `a`: previous frame
- `q` or `Esc`: quit

In `pcd` or `both` mode:

- Use the mouse in the Open3D window to rotate, pan, and zoom
- The camera view is preserved as much as possible when switching frames

Extra window:

- `Open3D Point Size`: interactive trackbar for point size

## Example Configurations

Thicker Open3D boxes and thicker image lines:

```bash
python3 visualize_kitti_predictions.py \
  --input_path /path/to/DATA_ROOT \
  --sample-id 002425 \
  --mode both \
  --box-thickness 4 \
  --open3d-box-radius 0.10
```

Smaller points and thinner Open3D boxes:

```bash
python3 visualize_kitti_predictions.py \
  --input_path /path/to/DATA_ROOT \
  --sample-id 002425 \
  --mode both \
  --point-size 1.0 \
  --open3d-box-radius 0.04
```

## Notes

- 3D boxes in Open3D are rendered as cylinder meshes rather than `LineSet`, so thickness is visually meaningful
- LiDAR colors are sampled from the RGB image; points projected outside the image are shown in light gray
- If required files are missing for a sample, the script raises a file-not-found error directly
