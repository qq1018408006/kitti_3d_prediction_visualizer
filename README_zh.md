# KITTI 3D 检测预测结果可视化工具

[English](./README.md)

这个仓库提供了一个独立脚本 [`visualize_kitti_predictions.py`](./visualize_kitti_predictions.py)，用于可视化 KITTI 格式的 3D 目标检测预测结果，支持：

- Open3D 点云视图
- RGB 图像上的 3D 框投影视图

脚本会同时显示真值标注和预测结果，方便做定性对比。

颜色约定：

- 真值 `gt`：绿色
- 预测 `pred`：红色

## 功能简介

- 可视化 KITTI 格式的预测结果、真值标注、LiDAR 点云、标定文件和 RGB 图像
- 支持三种模式：`pcd`、`image` 和 `both`
- 在 Open3D 中使用管状 mesh 边渲染 3D 框，框粗细可以真实体现
- 支持将 3D 框投影到 `image_2`
- 支持使用 RGB 图像对 LiDAR 点云进行着色
- 切换帧时尽量保留 Open3D 当前视角
- 支持通过 OpenCV 滑条实时调节点大小
- 支持将 RGB 投影视图保存到本地

## 目录约定

脚本支持两种输入方式：

1. 传入数据根目录
2. 直接传入 `pred/` 目录

默认按 KITTI 风格目录组织：

```text
DATA_ROOT/
├── calib/
├── image_2/
├── label_2/
├── pred/
└── velodyne/
```

每一帧需要以下文件：

- `pred/<sample_id>.txt`：预测结果，KITTI 检测格式
- `label_2/<sample_id>.txt`：真值标注
- `calib/<sample_id>.txt`：标定文件
- `velodyne/<sample_id>.bin`：点云
- `image_2/<sample_id>.png`：RGB 图像

## 标注格式

脚本支持解析两种 KITTI 检测标注行格式：

- 15 列：标准标签格式
- 16 列：额外包含 `score`

## 安装依赖

```bash
pip install numpy open3d opencv-python
```

`open3d` 目前无法在 Python 3.13 及更新版本上通过 `pip` 安装，运行这个脚本请使用 Python 3.12 或更低版本。

## 图形环境说明

脚本在导入图形库之前会自动设置以下环境变量，以提升 X11 兼容性：

```bash
XDG_SESSION_TYPE=x11
QT_QPA_PLATFORM=xcb
GDK_BACKEND=x11
SDL_VIDEODRIVER=x11
CLUTTER_BACKEND=x11
```

这主要是为了在使用 Wayland 桌面会话的 Ubuntu 上更稳定地拉起 Open3D 和 OpenCV 窗口。尤其是 Open3D 窗口显示异常时，强制使用 `XDG_SESSION_TYPE=x11`、`GDK_BACKEND=x11` 和 `QT_QPA_PLATFORM=xcb` 往往能让它走 X11 兼容路径。

现在 Python 脚本会在导入图形库前强制覆盖这些环境变量。如果直接运行 `python3 visualize_kitti_predictions.py` 仍然因为 Wayland 出错，建议改用仓库里提供的启动脚本，让 shell 在 Python 进程启动前先导出环境变量：

```bash
./run_visualizer_x11.sh --input_path /path/to/DATA_ROOT --mode both
```

## 基本用法

同时显示点云和图像：

```bash
python3 visualize_kitti_predictions.py \
  --input_path /path/to/DATA_ROOT \
  --sample-id 002425 \
  --mode both
```

如果是在 Wayland Ubuntu 上运行失败，可以改用启动脚本：

```bash
./run_visualizer_x11.sh \
  --input_path /path/to/DATA_ROOT \
  --sample-id 002425 \
  --mode both
```

只看点云：

```bash
python3 visualize_kitti_predictions.py \
  --input_path /path/to/DATA_ROOT \
  --sample-id 002425 \
  --mode pcd
```

只看图像投影：

```bash
python3 visualize_kitti_predictions.py \
  --input_path /path/to/DATA_ROOT \
  --sample-id 002425 \
  --mode image
```

将 RGB 投影视图保存到本地：

```bash
python3 visualize_kitti_predictions.py \
  --input_path /path/to/DATA_ROOT \
  --sample-id 002425 \
  --mode image \
  --image-output ./vis_002425.png
```

如果不传 `--sample-id`，脚本会默认使用 `pred/` 目录下排序后的第一帧。

## 参数说明

```text
--input_path
```

- 数据根目录，或者直接传 `pred/` 目录
- 当前脚本里带了一个本地默认路径，因此实际使用时通常建议显式传入

```text
--sample-id
```

- 指定要可视化的帧编号，例如 `002425`

```text
--mode {pcd,image,both}
```

- `pcd`：只显示点云
- `image`：只显示图像投影
- `both`：同时显示两种视图

```text
--image-output
```

- 输出保存路径，用于保存 2D RGB 投影视图
- 按当前脚本逻辑，这个保存并退出的分支主要用于 `--mode image`

```text
--point-size
```

- Open3D 中的初始点大小
- 启动后仍可通过滑条窗口实时调整

```text
--box-thickness
```

- 图像视图中 3D 投影框的线宽

```text
--open3d-box-radius
```

- Open3D 中 3D 框管状边的半径

## 交互按键

在 `image` 或 `both` 模式下：

- `n` 或 `d`：下一帧
- `p` 或 `a`：上一帧
- `q` 或 `Esc`：退出

在 `pcd` 或 `both` 模式下：

- 可在 Open3D 窗口中使用鼠标旋转、平移、缩放视角
- 切换帧时会尽量保留当前视角

额外窗口：

- `Open3D Point Size`：用于实时调节点大小的滑条窗口

## 示例

更粗的 Open3D 3D 框和图像投影框：

```bash
python3 visualize_kitti_predictions.py \
  --input_path /path/to/DATA_ROOT \
  --sample-id 002425 \
  --mode both \
  --box-thickness 4 \
  --open3d-box-radius 0.10
```

更小的点和更细的 Open3D 框：

```bash
python3 visualize_kitti_predictions.py \
  --input_path /path/to/DATA_ROOT \
  --sample-id 002425 \
  --mode both \
  --point-size 1.0 \
  --open3d-box-radius 0.04
```

## 补充说明

- Open3D 中的 3D 框不是 `LineSet`，而是圆柱 mesh，因此粗细参数会真实生效
- 点云颜色来自图像投影采样；投影到图像范围外的点会显示为浅灰色
- 如果某一帧缺少必要文件，脚本会直接抛出缺失文件错误
