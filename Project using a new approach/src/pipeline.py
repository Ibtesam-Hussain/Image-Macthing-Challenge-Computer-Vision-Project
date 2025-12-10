import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.clustering import build_graph, cosine_similarity_matrix, louvain_clusters, low_degree_outliers
from src.config import PipelineConfig
from src.data_utils import discover_images, load_thresholds, set_seed
from src.features import embed_images, l2_normalize
from src.geo_verification import verify_edges
from src.sfm import run_colmap, run_pycolmap
from src.submission import write_submission


def parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--test_split", type=str, default="test")
    parser.add_argument("--train_labels", type=Path, default=None)
    parser.add_argument("--thresholds_csv", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--clip_model", type=str, default="ViT-B-32")
    parser.add_argument("--clip_pretrain", type=str, default="openai")
    parser.add_argument("--dino_model", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--sim_threshold", type=float, default=0.25)
    parser.add_argument("--k_neighbors", type=int, default=20)
    parser.add_argument("--lightglue_pairs", type=int, default=200)
    parser.add_argument("--lightglue_min_matches", type=int, default=15)
    parser.add_argument("--outlier_degree", type=int, default=1)
    parser.add_argument("--colmap_binary", type=str, default="colmap")
    parser.add_argument("--work_dir", type=Path, default=Path("outputs"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return PipelineConfig(
        data_root=args.data_root,
        output_csv=args.output_csv,
        train_split=args.train_split,
        test_split=args.test_split,
        train_labels=args.train_labels,
        thresholds_csv=args.thresholds_csv,
        device=args.device,
        clip_model=args.clip_model,
        clip_pretrain=args.clip_pretrain,
        dino_model=args.dino_model,
        batch_size=args.batch_size,
        max_images=args.max_images,
        sim_threshold=args.sim_threshold,
        k_neighbors=args.k_neighbors,
        lightglue_pairs=args.lightglue_pairs,
        lightglue_min_matches=args.lightglue_min_matches,
        outlier_degree=args.outlier_degree,
        colmap_binary=args.colmap_binary,
        work_dir=args.work_dir,
        seed=args.seed,
        use_clip=bool(args.clip_model),
        use_dino=bool(args.dino_model),
    )


def select_threshold(thresholds: Dict, dataset: str, scene: str, fallback: float) -> float:
    if (dataset, scene) in thresholds:
        arr = thresholds[(dataset, scene)]
        return float(np.median(arr))
    return fallback


def cluster_dataset(
    dataset: str,
    images: List[Path],
    cfg: PipelineConfig,
    thresholds: Optional[Dict],
) -> Dict[str, int]:
    if cfg.max_images:
        images = images[: cfg.max_images]
    feats = embed_images(
        images,
        device=cfg.device,
        batch_size=cfg.batch_size,
        clip_model_name=cfg.clip_model if cfg.use_clip else None,
        clip_pretrained=cfg.clip_pretrain,
        dino_model_name=cfg.dino_model if cfg.use_dino else None,
    )
    feats = l2_normalize(feats)
    names, sims = cosine_similarity_matrix(feats)
    sim_thresh = cfg.sim_threshold
    g = build_graph(names, sims, k=cfg.k_neighbors, sim_threshold=sim_thresh)

    # geometric verification on top edges
    edges = sorted(g.edges(data=True), key=lambda x: x[2]["weight"], reverse=True)
    top_edges = [(a, b) for a, b, _ in edges[: cfg.lightglue_pairs]]
    name_to_path = {p.name: p for p in images}
    verified = verify_edges(
        name_to_path,
        top_edges,
        min_inliers=cfg.lightglue_min_matches,
        device=cfg.device,
    )
    gv = g.copy()
    gv.remove_edges_from(list(gv.edges()))
    gv.add_nodes_from(g.nodes())
    gv.add_edges_from(verified)

    clusters = louvain_clusters(gv)
    outliers = set(low_degree_outliers(gv, cfg.outlier_degree))
    assignments: Dict[str, int] = {}
    cid = 0
    for c_nodes in clusters.values():
        for n in c_nodes:
            if n in outliers:
                continue
            assignments[n] = cid
        cid += 1
    for n in outliers:
        assignments[n] = -1
    return assignments


def estimate_poses(
    dataset: str,
    images: List[Path],
    assignments: Dict[str, int],
    cfg: PipelineConfig,
) -> Dict[str, tuple]:
    poses = {}
    image_by_name = {p.name: p for p in images}
    for cid in sorted({c for c in assignments.values() if c >= 0}):
        cluster_imgs = [image_by_name[n] for n, c in assignments.items() if c == cid]
        if len(cluster_imgs) < 3:
            continue
        out_dir = cfg.work_dir / dataset / f"cluster_{cid}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for img in cluster_imgs:
            target = out_dir / img.name
            if not target.exists():
                # On Windows symlinks often require admin; fallback to copy.
                import shutil
                shutil.copy2(img, target)
        try:
            pose_dict = run_pycolmap(out_dir, out_dir)
            if pose_dict is None:
                run_colmap(out_dir, out_dir, cfg.colmap_binary)
                from src.sfm import extract_colmap_poses

                pose_dict = extract_colmap_poses(out_dir, cfg.colmap_binary)
            if pose_dict:
                poses.update(pose_dict)
        except Exception:
            continue
    return poses


def main(cfg: PipelineConfig) -> None:
    set_seed(cfg.seed)
    thresholds = load_thresholds(cfg.thresholds_csv) if cfg.thresholds_csv else None
    datasets = (cfg.data_root / cfg.test_split).iterdir()
    submissions = []
    for ds_dir in datasets:
        if not ds_dir.is_dir():
            continue
        ds_name = ds_dir.name
        images = discover_images(ds_dir)
        print(f"[{ds_name}] images: {len(images)}")
        assign = cluster_dataset(ds_name, images, cfg, thresholds)
        poses = estimate_poses(ds_name, images, assign, cfg)
        write_submission(cfg.output_csv, ds_name, assign, poses)
        submissions.append(ds_name)
    print(f"Done. Wrote {cfg.output_csv} for datasets: {submissions}")


if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)

