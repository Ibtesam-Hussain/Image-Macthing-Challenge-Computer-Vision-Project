from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
from community import community_louvain


def cosine_similarity_matrix(feats: Dict[str, np.ndarray]) -> Tuple[List[str], np.ndarray]:
    names = list(feats.keys())
    mat = np.stack([feats[n] for n in names])
    sims = mat @ mat.T
    return names, sims


def build_graph(
    names: List[str],
    sims: np.ndarray,
    k: int = 20,
    sim_threshold: float = 0.25,
) -> nx.Graph:
    g = nx.Graph()
    for i, name in enumerate(names):
        g.add_node(name)
        row = sims[i]
        top_idx = np.argsort(row)[::-1][: k + 1]
        for j in top_idx:
            if i == j:
                continue
            if row[j] < sim_threshold:
                continue
            g.add_edge(name, names[j], weight=float(row[j]))
    return g


def louvain_clusters(graph: nx.Graph) -> Dict[int, List[str]]:
    if graph.number_of_nodes() == 0:
        return {}
    part = community_louvain.best_partition(graph, weight="weight")
    clusters: Dict[int, List[str]] = {}
    for node, cid in part.items():
        clusters.setdefault(cid, []).append(node)
    return clusters


def low_degree_outliers(graph: nx.Graph, min_degree: int) -> List[str]:
    return [n for n, deg in graph.degree() if deg <= min_degree]

