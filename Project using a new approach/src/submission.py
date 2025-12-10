from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


def format_matrix(mat: np.ndarray) -> str:
    return ";".join([f"{x:.9f}" for x in mat.flatten()])


def write_submission(
    output_csv: Path,
    dataset: str,
    assignments: Dict[str, int],
    poses: Dict[str, Tuple[np.ndarray, np.ndarray]],
) -> None:
    rows = []
    for image_name, cluster_id in assignments.items():
        rot, trans = poses.get(image_name, (np.eye(3), np.zeros(3)))
        rows.append(
            {
                "image_id": f"{dataset}_{image_name}_public",
                "dataset": dataset,
                "scene": f"cluster{cluster_id}",
                "image": image_name,
                "rotation_matrix": format_matrix(rot),
                "translation_vector": format_matrix(trans),
            }
        )
    df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

