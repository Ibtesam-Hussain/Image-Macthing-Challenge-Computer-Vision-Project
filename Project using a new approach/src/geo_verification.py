from typing import Dict, Iterable, List, Tuple

# LightGlue removed; passthrough verification that keeps provided edges.


def verify_edges(
    image_lookup: Dict[str, str],
    edges: Iterable[Tuple[str, str]],
    min_inliers: int = 0,
    device: str = "cpu",
) -> List[Tuple[str, str]]:
    return list(edges)

