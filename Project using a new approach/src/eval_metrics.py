from collections import defaultdict
from typing import Dict, Tuple

import pandas as pd


def _greedy_assign(scenes: Dict[str, set], clusters: Dict[str, set]):
    """Greedily assign each GT scene to the user cluster maximizing mAA, tie-breaking with clustering score."""
    assignment = {}
    for scene, imgs in scenes.items():
        best_cluster = None
        best_maa = -1.0
        best_clust = -1.0
        for cid, cimgs in clusters.items():
            inter = len(imgs & cimgs)
            maa = inter / max(1, len(imgs))
            clust_score = inter / max(1, len(cimgs))
            if maa > best_maa or (maa == best_maa and clust_score > best_clust):
                best_maa = maa
                best_clust = clust_score
                best_cluster = cid
        assignment[scene] = (best_cluster, best_maa, best_clust)
    return assignment


def evaluate_dataset(gt: pd.DataFrame, sub: pd.DataFrame) -> Tuple[float, float, float]:
    """Compute mAA, clustering score, and harmonic mean for one dataset."""
    # Build GT scene -> images
    scenes = defaultdict(set)
    for _, r in gt.iterrows():
        scenes[str(r["scene"])].add(str(r["image"]))

    # Build user clusters -> images (exclude outliers scene=='-1')
    clusters = defaultdict(set)
    for _, r in sub.iterrows():
        scene_name = str(r["scene"])
        if scene_name == "-1":
            continue
        clusters[scene_name].add(str(r["image"]))

    if not scenes or not clusters:
        return 0.0, 0.0, 0.0

    assignment = _greedy_assign(scenes, clusters)

    maa_vals = []
    clust_vals = []
    for scene, (cid, maa, clust_score) in assignment.items():
        if cid is None:
            maa_vals.append(0.0)
            clust_vals.append(0.0)
        else:
            maa_vals.append(maa)
            clust_vals.append(clust_score)

    mean_maa = sum(maa_vals) / len(maa_vals)
    mean_clust = sum(clust_vals) / len(clust_vals)
    if mean_maa + mean_clust == 0:
        harmonic = 0.0
    else:
        harmonic = 2 * mean_maa * mean_clust / (mean_maa + mean_clust)
    return mean_maa, mean_clust, harmonic


def evaluate_all(train_labels_csv: str, submission_csv: str) -> Dict[str, Tuple[float, float, float]]:
    gt_df = pd.read_csv(train_labels_csv)
    sub_df = pd.read_csv(submission_csv)

    results = {}
    for dataset, gtd in gt_df.groupby("dataset"):
        # Filter submission to same dataset
        subs = sub_df[sub_df["dataset"] == dataset]
        m1, m2, f1 = evaluate_dataset(gtd, subs)
        results[dataset] = (m1, m2, f1)
    # overall average
    if results:
        avg = (
            sum(v[0] for v in results.values()) / len(results),
            sum(v[1] for v in results.values()) / len(results),
            sum(v[2] for v in results.values()) / len(results),
        )
        results["overall"] = avg
    return results

