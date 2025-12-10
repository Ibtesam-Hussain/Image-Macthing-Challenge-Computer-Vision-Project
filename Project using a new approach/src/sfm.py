import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import pycolmap
except ImportError:
    pycolmap = None


def run_colmap(image_dir: Path, work_dir: Path, colmap_bin: str = "colmap") -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    db_path = work_dir / "database.db"
    sparse_dir = work_dir / "sparse"
    sparse_dir.mkdir(exist_ok=True)
    subprocess.run(
        [
            colmap_bin,
            "feature_extractor",
            "--database_path",
            str(db_path),
            "--image_path",
            str(image_dir),
            "--ImageReader.single_camera",
            "1",
        ],
        check=True,
    )
    subprocess.run(
        [colmap_bin, "exhaustive_matcher", "--database_path", str(db_path)],
        check=True,
    )
    subprocess.run(
        [
            colmap_bin,
            "mapper",
            "--database_path",
            str(db_path),
            "--image_path",
            str(image_dir),
            "--output_path",
            str(sparse_dir),
        ],
        check=True,
    )
    return sparse_dir


def _parse_images_txt(images_txt: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    poses = {}
    if not images_txt.exists():
        return poses
    with open(images_txt, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith("#") or not line.strip():
            continue
        parts = line.strip().split()
        if len(parts) < 10:
            continue
        # line format: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        name = parts[9]
        # convert quaternion to rotation matrix
        q = np.array([qw, qx, qy, qz], dtype=np.float64)
        # normalize
        q = q / (np.linalg.norm(q) + 1e-8)
        w, x, y, z = q
        rot = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float32,
        )
        t = np.array([tx, ty, tz], dtype=np.float32)
        poses[name] = (rot, t)
    return poses


def run_pycolmap(image_dir: Path, work_dir: Path) -> Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    if pycolmap is None:
        return None
    work_dir.mkdir(parents=True, exist_ok=True)
    reconstruction = pycolmap.incremental_mapping(database_path=None, image_dir=image_dir, output_dir=work_dir)
    poses = {}
    for image_id, image in reconstruction.images.items():
        name = image.name
        r = image.rotmat()
        t = image.tvec
        poses[name] = (np.array(r), np.array(t))
    return poses


def extract_colmap_poses(colmap_dir: Path, colmap_bin: str = "colmap") -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Convert COLMAP binary model to TXT and parse poses. Returns name -> (R, t)."""
    poses = {}
    model_dir = colmap_dir / "sparse"
    if not model_dir.exists():
        return poses
    txt_dir = colmap_dir / "sparse_txt"
    txt_dir.mkdir(exist_ok=True)
    try:
        subprocess.run(
            [
                colmap_bin,
                "model_converter",
                "--input_path",
                str(model_dir / "0"),
                "--output_path",
                str(txt_dir),
                "--output_type",
                "TXT",
            ],
            check=True,
        )
    except subprocess.CalledProcessError:
        return poses
    images_txt = txt_dir / "images.txt"
    poses = _parse_images_txt(images_txt)
    return poses

