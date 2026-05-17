#!/usr/bin/env python3
"""Visualize KITTI-format prediction results in point cloud and RGB image views."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
os.environ.setdefault("GDK_BACKEND", "x11")
os.environ.setdefault("SDL_VIDEODRIVER", "x11")
os.environ.setdefault("CLUTTER_BACKEND", "x11")

try:
    import cv2
except ImportError:  # pragma: no cover - import guard
    cv2 = None

try:
    import open3d as o3d
except ImportError:  # pragma: no cover - import guard
    o3d = None


BOX_EDGE_INDICES: Tuple[Tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)

POINT_SIZE_TRACKBAR_WINDOW = "Open3D Point Size"
POINT_SIZE_TRACKBAR_NAME = "Point Size"
POINT_SIZE_TRACKBAR_SCALE = 10
POINT_SIZE_TRACKBAR_MAX = 100
IMAGE_WINDOW_NAME = "KITTI 3D Boxes on RGB"
IMAGE_WINDOW_SCALE = 1.5


@dataclass
class ObjectLabel:
    cls_type: str
    truncation: float
    occlusion: int
    alpha: float
    bbox: np.ndarray
    h: float
    w: float
    l: float
    x: float
    y: float
    z: float
    ry: float
    score: Optional[float] = None

    @property
    def dimensions(self) -> np.ndarray:
        return np.array([self.h, self.w, self.l], dtype=np.float64)

    @property
    def location(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float64)


class Calibration:
    def __init__(self, calib_path: Path) -> None:
        calib = self._read_calib_file(calib_path)
        self.P2 = self._reshape_or_default(calib, ["P2"], (3, 4))
        self.R0 = self._reshape_or_default(
            calib, ["R0_rect", "R_rect", "R_rect_00"], (3, 3), np.eye(3)
        )
        self.V2C = self._reshape_or_default(
            calib,
            ["Tr_velo_to_cam", "Tr_velo_cam", "Tr_velo_to_cam_0"],
            (3, 4),
        )
        self.C2V = self._inverse_rigid_transform(self.V2C)

    @staticmethod
    def _read_calib_file(calib_path: Path) -> Dict[str, np.ndarray]:
        values: Dict[str, np.ndarray] = {}
        with calib_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or ":" not in line:
                    continue
                key, raw = line.split(":", 1)
                raw = raw.strip()
                if not raw:
                    continue
                values[key] = np.array([float(x) for x in raw.split()], dtype=np.float64)
        return values

    @staticmethod
    def _reshape_or_default(
        calib: Dict[str, np.ndarray],
        keys: Sequence[str],
        shape: Tuple[int, int],
        default: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        for key in keys:
            if key in calib:
                return calib[key].reshape(shape)
        if default is not None:
            return default.astype(np.float64)
        joined = ", ".join(keys)
        raise KeyError(f"Missing calibration field. Expected one of: {joined}")

    @staticmethod
    def _inverse_rigid_transform(transform: np.ndarray) -> np.ndarray:
        inv = np.zeros((3, 4), dtype=np.float64)
        rotation = transform[:, :3]
        translation = transform[:, 3]
        inv[:, :3] = rotation.T
        inv[:, 3] = -rotation.T @ translation
        return inv

    @staticmethod
    def cart_to_hom(points: np.ndarray) -> np.ndarray:
        return np.hstack([points, np.ones((points.shape[0], 1), dtype=points.dtype)])

    def lidar_to_rect(self, points_lidar: np.ndarray) -> np.ndarray:
        pts_hom = self.cart_to_hom(points_lidar[:, :3])
        pts_cam = pts_hom @ self.V2C.T
        return pts_cam @ self.R0.T

    def rect_to_lidar(self, points_rect: np.ndarray) -> np.ndarray:
        r0_inv = np.linalg.inv(self.R0)
        pts_ref = points_rect @ r0_inv.T
        pts_ref_hom = self.cart_to_hom(pts_ref)
        return pts_ref_hom @ self.C2V.T

    def rect_to_img(self, points_rect: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pts_hom = self.cart_to_hom(points_rect)
        proj = pts_hom @ self.P2.T
        depth = proj[:, 2]
        pixels = proj[:, :2] / np.clip(depth[:, None], 1e-6, None)
        return pixels, depth


def parse_kitti_label_file(label_path: Path) -> List[ObjectLabel]:
    labels: List[ObjectLabel] = []
    if not label_path.exists():
        return labels

    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) not in (15, 16):
                raise ValueError(f"Unexpected label format in {label_path}: {line.strip()}")
            labels.append(
                ObjectLabel(
                    cls_type=parts[0],
                    truncation=float(parts[1]),
                    occlusion=int(float(parts[2])),
                    alpha=float(parts[3]),
                    bbox=np.array([float(x) for x in parts[4:8]], dtype=np.float64),
                    h=float(parts[8]),
                    w=float(parts[9]),
                    l=float(parts[10]),
                    x=float(parts[11]),
                    y=float(parts[12]),
                    z=float(parts[13]),
                    ry=float(parts[14]),
                    score=float(parts[15]) if len(parts) == 16 else None,
                )
            )
    return labels


def rotation_y_matrix(ry: float) -> np.ndarray:
    c = np.cos(ry)
    s = np.sin(ry)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def camera_box_corners_3d(label: ObjectLabel) -> np.ndarray:
    h, w, l = label.h, label.w, label.l
    x_corners = np.array([l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2])
    y_corners = np.array([0.0, 0.0, 0.0, 0.0, -h, -h, -h, -h])
    z_corners = np.array([w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2])
    corners = np.vstack([x_corners, y_corners, z_corners])
    rotated = rotation_y_matrix(label.ry) @ corners
    rotated += label.location.reshape(3, 1)
    return rotated.T


def rotation_matrix_from_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = source / np.linalg.norm(source)
    target = target / np.linalg.norm(target)
    cross = np.cross(source, target)
    dot = np.clip(np.dot(source, target), -1.0, 1.0)
    if np.isclose(dot, 1.0):
        return np.eye(3, dtype=np.float64)
    if np.isclose(dot, -1.0):
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if np.allclose(source, axis):
            axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = axis - source * np.dot(source, axis)
        axis = axis / np.linalg.norm(axis)
        return o3d.geometry.get_rotation_matrix_from_axis_angle(axis * np.pi)
    skew = np.array(
        [[0.0, -cross[2], cross[1]], [cross[2], 0.0, -cross[0]], [-cross[1], cross[0], 0.0]],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + skew + skew @ skew * ((1.0 - dot) / (np.linalg.norm(cross) ** 2))


def create_cylinder_segment(
    start: np.ndarray,
    end: np.ndarray,
    radius: float,
    color: Sequence[float],
) -> "o3d.geometry.TriangleMesh":
    if o3d is None:
        raise ImportError("open3d is required for point cloud visualization.")
    segment = end - start
    length = np.linalg.norm(segment)
    if length < 1e-6:
        return o3d.geometry.TriangleMesh()
    mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length)
    rotation = rotation_matrix_from_vectors(
        np.array([0.0, 0.0, 1.0], dtype=np.float64),
        segment / length,
    )
    mesh.rotate(rotation, center=np.zeros(3, dtype=np.float64))
    mesh.translate((start + end) / 2.0)
    mesh.paint_uniform_color(np.asarray(color, dtype=np.float64))
    mesh.compute_vertex_normals()
    return mesh


def create_box_tube_geometries(
    points: np.ndarray,
    color: Sequence[float],
    radius: float,
) -> List["o3d.geometry.TriangleMesh"]:
    return [
        create_cylinder_segment(points[start], points[end], radius=radius, color=color)
        for start, end in BOX_EDGE_INDICES
    ]


def create_axis_geometries(axis_length: float, axis_radius: float) -> List["o3d.geometry.TriangleMesh"]:
    origin = np.zeros(3, dtype=np.float64)
    return [
        create_cylinder_segment(origin, np.array([axis_length, 0.0, 0.0]), axis_radius, (1.0, 0.0, 0.0)),
        create_cylinder_segment(origin, np.array([0.0, axis_length, 0.0]), axis_radius, (0.0, 1.0, 0.0)),
        create_cylinder_segment(origin, np.array([0.0, 0.0, axis_length]), axis_radius, (0.0, 0.0, 1.0)),
    ]


def resolve_dataset_paths(input_path: Path) -> Dict[str, Path]:
    root = input_path.resolve()
    pred_dir = root / "pred" if (root / "pred").is_dir() else root
    if pred_dir.name != "pred":
        raise FileNotFoundError(
            "Input path must be either a dataset root containing pred/ or the pred directory itself."
        )
    dataset_root = pred_dir.parent
    return {
        "dataset_root": dataset_root,
        "pred_dir": pred_dir,
        "label_dir": dataset_root / "label_2",
        "calib_dir": dataset_root / "calib",
        "velodyne_dir": dataset_root / "velodyne",
        "image_dir": dataset_root / "image_2",
    }


def load_point_cloud(bin_path: Path) -> np.ndarray:
    points = np.fromfile(bin_path, dtype=np.float32)
    if points.size % 4 != 0:
        raise ValueError(f"Unexpected point cloud format: {bin_path}")
    return points.reshape(-1, 4)


def draw_projected_box3d(
    image: np.ndarray,
    corners_rect: np.ndarray,
    calib: Calibration,
    color: Tuple[int, int, int],
    thickness: int = 2,
) -> None:
    pixels, depth = calib.rect_to_img(corners_rect)
    if np.any(depth <= 0.1):
        return
    pixels = np.round(pixels).astype(np.int32)
    for start, end in BOX_EDGE_INDICES:
        pt1 = tuple(pixels[start])
        pt2 = tuple(pixels[end])
        cv2.line(image, pt1, pt2, color, thickness, lineType=cv2.LINE_AA)


def make_point_cloud_visualization(
    lidar_points: np.ndarray,
    gt_labels: Iterable[ObjectLabel],
    pred_labels: Iterable[ObjectLabel],
    calib: Calibration,
    point_size: float,
) -> None:
    if o3d is None:
        raise ImportError("open3d is not installed. Please install open3d first.")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(lidar_points[:, :3])

    intensities = lidar_points[:, 3]
    if intensities.size > 0:
        normalized = intensities - intensities.min()
        denom = normalized.max()
        normalized = normalized / denom if denom > 1e-6 else np.ones_like(normalized) * 0.5
        colors = np.stack([normalized, normalized, normalized], axis=1)
        pcd.colors = o3d.utility.Vector3dVector(colors)

    geometries: List[object] = [pcd, *create_axis_geometries(axis_length=1.5, axis_radius=0.025)]

    for label in gt_labels:
        corners_cam = camera_box_corners_3d(label)
        corners_lidar = calib.rect_to_lidar(corners_cam)
        geometries.extend(create_box_tube_geometries(corners_lidar, color=(0.0, 1.0, 0.0), radius=0.06))

    for label in pred_labels:
        corners_cam = camera_box_corners_3d(label)
        corners_lidar = calib.rect_to_lidar(corners_cam)
        geometries.extend(create_box_tube_geometries(corners_lidar, color=(1.0, 0.0, 0.0), radius=0.06))

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="KITTI Prediction Visualization")
    for geom in geometries:
        vis.add_geometry(geom)
    render_option = vis.get_render_option()
    render_option.point_size = point_size
    render_option.background_color = np.array([1.0, 1.0, 1.0], dtype=np.float64)

    if cv2 is None:
        vis.run()
        vis.destroy_window()
        return

    cv2.namedWindow(POINT_SIZE_TRACKBAR_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(POINT_SIZE_TRACKBAR_WINDOW, 480, 80)
    initial_trackbar_value = max(1, min(POINT_SIZE_TRACKBAR_MAX, int(round(point_size * POINT_SIZE_TRACKBAR_SCALE))))
    cv2.createTrackbar(
        POINT_SIZE_TRACKBAR_NAME,
        POINT_SIZE_TRACKBAR_WINDOW,
        initial_trackbar_value,
        POINT_SIZE_TRACKBAR_MAX,
        lambda _value: None,
    )

    while True:
        trackbar_value = cv2.getTrackbarPos(POINT_SIZE_TRACKBAR_NAME, POINT_SIZE_TRACKBAR_WINDOW)
        render_option.point_size = max(1.0 / POINT_SIZE_TRACKBAR_SCALE, trackbar_value / POINT_SIZE_TRACKBAR_SCALE)
        if not vis.poll_events():
            break
        vis.update_renderer()
        if cv2.waitKey(10) & 0xFF == 27:
            break

    cv2.destroyWindow(POINT_SIZE_TRACKBAR_WINDOW)
    vis.destroy_window()


def make_image_visualization(
    image_path: Path,
    gt_labels: Iterable[ObjectLabel],
    pred_labels: Iterable[ObjectLabel],
    calib: Calibration,
    output_path: Optional[Path],
) -> None:
    if cv2 is None:
        raise ImportError("opencv-python is required for image visualization.")

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    for label in gt_labels:
        draw_projected_box3d(image, camera_box_corners_3d(label), calib, color=(0, 255, 0))
    for label in pred_labels:
        draw_projected_box3d(image, camera_box_corners_3d(label), calib, color=(0, 0, 255))

    legend_y = 30
    cv2.putText(image, "GT", (20, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(
        image, "Prediction", (90, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
    )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), image)
        print(f"Saved image visualization to: {output_path}")
        return

    cv2.imshow("KITTI 3D Boxes on RGB", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def choose_sample_id(paths: Dict[str, Path], sample_id: Optional[str]) -> str:
    if sample_id:
        return sample_id
    pred_files = sorted(paths["pred_dir"].glob("*.txt"))
    if not pred_files:
        raise FileNotFoundError(f"No prediction txt files found in {paths['pred_dir']}")
    return pred_files[0].stem


def list_sample_ids(paths: Dict[str, Path]) -> List[str]:
    sample_ids = [path.stem for path in sorted(paths["pred_dir"].glob("*.txt"))]
    if not sample_ids:
        raise FileNotFoundError(f"No prediction txt files found in {paths['pred_dir']}")
    return sample_ids


def validate_required_files(
    paths: Dict[str, Path],
    sample_id: str,
    need_pcd: bool,
    need_image: bool,
) -> Dict[str, Path]:
    file_map = {
        "pred": paths["pred_dir"] / f"{sample_id}.txt",
        "gt": paths["label_dir"] / f"{sample_id}.txt",
        "calib": paths["calib_dir"] / f"{sample_id}.txt",
    }
    if need_pcd:
        file_map["velodyne"] = paths["velodyne_dir"] / f"{sample_id}.bin"
        file_map["image_for_color"] = paths["image_dir"] / f"{sample_id}.png"
    if need_image:
        file_map["image"] = paths["image_dir"] / f"{sample_id}.png"

    missing = [str(path) for path in file_map.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))
    return file_map


def colorize_lidar_points(
    lidar_points: np.ndarray,
    image_bgr: np.ndarray,
    calib: Calibration,
) -> np.ndarray:
    points_xyz = lidar_points[:, :3]
    points_rect = calib.lidar_to_rect(points_xyz)
    pixels, depth = calib.rect_to_img(points_rect)

    image_h, image_w = image_bgr.shape[:2]
    pixel_xy = np.round(pixels).astype(np.int32)
    valid = (
        (depth > 0.1)
        & (pixel_xy[:, 0] >= 0)
        & (pixel_xy[:, 0] < image_w)
        & (pixel_xy[:, 1] >= 0)
        & (pixel_xy[:, 1] < image_h)
    )

    colors = np.full((points_xyz.shape[0], 3), 0.65, dtype=np.float64)
    sampled_bgr = image_bgr[pixel_xy[valid, 1], pixel_xy[valid, 0]].astype(np.float64) / 255.0
    colors[valid] = sampled_bgr[:, ::-1]
    return colors


def prepare_point_cloud_geometries(
    lidar_points: np.ndarray,
    gt_labels: Iterable[ObjectLabel],
    pred_labels: Iterable[ObjectLabel],
    calib: Calibration,
    image_bgr: np.ndarray,
    box_radius: float,
) -> List[object]:
    if o3d is None:
        raise ImportError("open3d is not installed. Please install open3d first.")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(lidar_points[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(colorize_lidar_points(lidar_points, image_bgr, calib))

    geometries: List[object] = [pcd, *create_axis_geometries(axis_length=1.5, axis_radius=0.025)]
    for label in gt_labels:
        corners_cam = camera_box_corners_3d(label)
        corners_lidar = calib.rect_to_lidar(corners_cam)
        geometries.extend(create_box_tube_geometries(corners_lidar, color=(0.0, 1.0, 0.0), radius=box_radius))

    for label in pred_labels:
        corners_cam = camera_box_corners_3d(label)
        corners_lidar = calib.rect_to_lidar(corners_cam)
        geometries.extend(create_box_tube_geometries(corners_lidar, color=(1.0, 0.0, 0.0), radius=box_radius))
    return geometries


def render_image_visualization(
    image_path: Path,
    gt_labels: Iterable[ObjectLabel],
    pred_labels: Iterable[ObjectLabel],
    calib: Calibration,
    box_thickness: int,
) -> np.ndarray:
    if cv2 is None:
        raise ImportError("opencv-python is required for image visualization.")

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    for label in gt_labels:
        draw_projected_box3d(
            image,
            camera_box_corners_3d(label),
            calib,
            color=(0, 255, 0),
            thickness=box_thickness,
        )
    for label in pred_labels:
        draw_projected_box3d(
            image,
            camera_box_corners_3d(label),
            calib,
            color=(0, 0, 255),
            thickness=box_thickness,
        )

    legend_y = 30
    cv2.putText(image, "GT", (20, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(
        image, "Prediction", (90, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
    )
    return image


def load_scene(
    paths: Dict[str, Path],
    sample_id: str,
    need_pcd: bool,
    need_image: bool,
    box_thickness: int,
) -> Dict[str, object]:
    file_map = validate_required_files(paths, sample_id, need_pcd=need_pcd, need_image=need_image)
    calib = Calibration(file_map["calib"])
    gt_labels = parse_kitti_label_file(file_map["gt"])
    pred_labels = parse_kitti_label_file(file_map["pred"])
    scene: Dict[str, object] = {
        "sample_id": sample_id,
        "calib": calib,
        "gt_labels": gt_labels,
        "pred_labels": pred_labels,
        "file_map": file_map,
    }
    if need_pcd:
        scene["lidar_points"] = load_point_cloud(file_map["velodyne"])
        raw_image = cv2.imread(str(file_map["image_for_color"])) if cv2 is not None else None
        if raw_image is None:
            raise FileNotFoundError(f"Failed to read image: {file_map['image_for_color']}")
        scene["raw_image_bgr"] = raw_image
    if need_image:
        scene["image"] = render_image_visualization(
            file_map["image"],
            gt_labels=gt_labels,
            pred_labels=pred_labels,
            calib=calib,
            box_thickness=box_thickness,
        )
    return scene


def set_initial_front_view(vis: "o3d.visualization.Visualizer", lidar_points: np.ndarray) -> None:
    view_control = vis.get_view_control()
    center = lidar_points[:, :3].mean(axis=0)
    view_control.set_lookat(center.tolist())
    view_control.set_front([-1.0, 0.0, 0.0])
    view_control.set_up([0.0, 0.0, 1.0])
    view_control.set_zoom(0.08)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize KITTI prediction results with Open3D point cloud and/or RGB projection."
    )
    parser.add_argument(
        "--input_path",
        type=Path,
        default='/home/xiao/data/mid360/kitti_final/training',
        help="Dataset root containing pred/ or the pred directory itself.",
    )
    parser.add_argument(
        "--sample-id",
        type=str,
        default=None,
        help="Frame id to visualize, e.g. 002425. Defaults to the first txt file in pred/.",
    )
    parser.add_argument(
        "--mode",
        choices=("pcd", "image", "both"),
        default="both",
        help="Visualization mode.",
    )
    parser.add_argument(
        "--image-output",
        type=Path,
        default=None,
        help="Optional output path for the 2D projected RGB visualization.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=3.0,
        help="Point size used in the Open3D viewer.",
    )
    parser.add_argument(
        "--box-thickness",
        type=int,
        default=2,
        help="Bounding box line thickness for 2D image boxes.",
    )
    parser.add_argument(
        "--open3d-box-radius",
        type=float,
        default=0.06,
        help="Tube radius used to render 3D boxes in Open3D.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    paths = resolve_dataset_paths(args.input_path)
    sample_ids = list_sample_ids(paths)
    sample_id = choose_sample_id(paths, args.sample_id)
    sample_index = sample_ids.index(sample_id)

    need_pcd = args.mode in ("pcd", "both")
    need_image = args.mode in ("image", "both")
    scene = load_scene(
        paths,
        sample_id,
        need_pcd=need_pcd,
        need_image=need_image,
        box_thickness=args.box_thickness,
    )
    current_point_size = args.point_size * 2.0

    if args.image_output is not None and args.mode == "image":
        output_image = scene["image"]
        args.image_output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.image_output), output_image)
        print(f"Saved image visualization to: {args.image_output}")
        return

    vis = None
    render_option = None
    if need_pcd:
        if o3d is None:
            raise ImportError("open3d is not installed. Please install open3d first.")
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="KITTI Prediction Visualization")
        render_option = vis.get_render_option()
        render_option.point_size = current_point_size
        render_option.background_color = np.array([1.0, 1.0, 1.0], dtype=np.float64)

    if cv2 is not None:
        if need_pcd:
            cv2.namedWindow(POINT_SIZE_TRACKBAR_WINDOW, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(POINT_SIZE_TRACKBAR_WINDOW, 480, 80)
            initial_trackbar_value = max(
                1, min(POINT_SIZE_TRACKBAR_MAX, int(round(current_point_size * POINT_SIZE_TRACKBAR_SCALE)))
            )
            cv2.createTrackbar(
                POINT_SIZE_TRACKBAR_NAME,
                POINT_SIZE_TRACKBAR_WINDOW,
                initial_trackbar_value,
                POINT_SIZE_TRACKBAR_MAX,
                lambda _value: None,
            )
        if need_image:
            cv2.namedWindow(IMAGE_WINDOW_NAME, cv2.WINDOW_NORMAL)
            image_h, image_w = scene["image"].shape[:2]
            cv2.resizeWindow(
                IMAGE_WINDOW_NAME,
                int(image_w * IMAGE_WINDOW_SCALE),
                int(image_h * IMAGE_WINDOW_SCALE),
            )

    first_scene = True

    def refresh_scene(
        current_scene: Dict[str, object],
        saved_camera_params: Optional["o3d.camera.PinholeCameraParameters"] = None,
    ) -> None:
        nonlocal first_scene
        title = f"KITTI 3D Boxes on RGB - {current_scene['sample_id']}"
        if need_pcd and vis is not None:
            vis.clear_geometries()
            for geom in prepare_point_cloud_geometries(
                current_scene["lidar_points"],
                current_scene["gt_labels"],
                current_scene["pred_labels"],
                current_scene["calib"],
                current_scene["raw_image_bgr"],
                args.open3d_box_radius,
            ):
                vis.add_geometry(geom)
            view_control = vis.get_view_control()
            if saved_camera_params is not None:
                view_control.convert_from_pinhole_camera_parameters(
                    saved_camera_params, allow_arbitrary=True
                )
            elif first_scene:
                set_initial_front_view(vis, current_scene["lidar_points"])
                first_scene = False
            vis.poll_events()
            vis.update_renderer()
        if need_image and cv2 is not None:
            image = current_scene["image"].copy()
            cv2.putText(
                image,
                f"sample: {current_scene['sample_id']}  [n/d]: next  [p/a]: prev  [q/esc]: quit",
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (20, 20, 20),
                2,
            )
            cv2.imshow(IMAGE_WINDOW_NAME, image)
            cv2.setWindowTitle(IMAGE_WINDOW_NAME, title)

    refresh_scene(scene)

    while True:
        if need_pcd and render_option is not None and cv2 is not None:
            trackbar_value = cv2.getTrackbarPos(POINT_SIZE_TRACKBAR_NAME, POINT_SIZE_TRACKBAR_WINDOW)
            current_point_size = max(
                1.0 / POINT_SIZE_TRACKBAR_SCALE, trackbar_value / POINT_SIZE_TRACKBAR_SCALE
            )
            render_option.point_size = current_point_size

        if need_pcd and vis is not None:
            if not vis.poll_events():
                break
            vis.update_renderer()

        key = cv2.waitKey(10) & 0xFF if cv2 is not None else -1
        if key in (27, ord("q")):
            break
        if key in (ord("n"), ord("d")):
            saved_camera_params = None
            if need_pcd and vis is not None:
                saved_camera_params = vis.get_view_control().convert_to_pinhole_camera_parameters()
            sample_index = min(sample_index + 1, len(sample_ids) - 1)
            scene = load_scene(
                paths,
                sample_ids[sample_index],
                need_pcd=need_pcd,
                need_image=need_image,
                box_thickness=args.box_thickness,
            )
            refresh_scene(scene, saved_camera_params=saved_camera_params)
        elif key in (ord("p"), ord("a")):
            saved_camera_params = None
            if need_pcd and vis is not None:
                saved_camera_params = vis.get_view_control().convert_to_pinhole_camera_parameters()
            sample_index = max(sample_index - 1, 0)
            scene = load_scene(
                paths,
                sample_ids[sample_index],
                need_pcd=need_pcd,
                need_image=need_image,
                box_thickness=args.box_thickness,
            )
            refresh_scene(scene, saved_camera_params=saved_camera_params)

    if cv2 is not None:
        if need_pcd:
            cv2.destroyWindow(POINT_SIZE_TRACKBAR_WINDOW)
        if need_image:
            cv2.destroyWindow(IMAGE_WINDOW_NAME)
    if vis is not None:
        vis.destroy_window()


if __name__ == "__main__":
    main()
