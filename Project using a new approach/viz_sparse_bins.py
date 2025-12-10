import argparse
import math
import random
import subprocess
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def quat_to_rot(qw, qx, qy, qz) -> np.ndarray:
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    q = q / (np.linalg.norm(q) + 1e-8)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def run_model_converter(model_dir: Path, colmap_bin: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            colmap_bin,
            "model_converter",
            "--input_path",
            str(model_dir),
            "--output_path",
            str(out_dir),
            "--output_type",
            "TXT",
        ],
        check=True,
    )


def load_points(points_txt: Path, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    xyz = []
    colors = []
    with open(points_txt, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            x, y, z = map(float, parts[1:4])
            r, g, b = map(int, parts[4:7])
            xyz.append((x, y, z))
            colors.append((r / 255.0, g / 255.0, b / 255.0))
    if not xyz:
        return np.zeros((0, 3)), np.zeros((0, 3))
    if len(xyz) > max_points:
        idx = random.sample(range(len(xyz)), max_points)
        xyz = [xyz[i] for i in idx]
        colors = [colors[i] for i in idx]
    return np.array(xyz, dtype=np.float32), np.array(colors, dtype=np.float32)


def load_cameras(images_txt: Path, max_cams: int = 500) -> np.ndarray:
    cams = []
    with open(images_txt, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) < 10:
                continue
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            rot = quat_to_rot(qw, qx, qy, qz)
            t = np.array([tx, ty, tz], dtype=np.float64)
            center = -rot.T @ t
            cams.append(center)
            if len(cams) >= max_cams:
                break
    if not cams:
        return np.zeros((0, 3))
    return np.array(cams, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser(description="Visualize COLMAP sparse bin directly via model_converter + matplotlib")
    ap.add_argument("--model_dir", required=True, help="Path to COLMAP model dir (e.g., outputs/ETs/cluster_0/sparse/0)")
    ap.add_argument("--colmap_binary", default="colmap", help="Path to colmap executable")
    ap.add_argument("--max_points", type=int, default=50000, help="Max points to plot for speed")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise SystemExit(f"Model dir not found: {model_dir}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        run_model_converter(model_dir, args.colmap_binary, tmp_path)
        pts, colors = load_points(tmp_path / "points3D.txt", args.max_points)
        cams = load_cameras(tmp_path / "images.txt")

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    if len(pts) > 0:
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=colors, s=1, alpha=0.8)
    if len(cams) > 0:
        ax.scatter(cams[:, 0], cams[:, 1], cams[:, 2], c="red", s=10, marker="^", label="cameras")
        ax.legend()
    ax.set_title(f"COLMAP sparse: {model_dir}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

