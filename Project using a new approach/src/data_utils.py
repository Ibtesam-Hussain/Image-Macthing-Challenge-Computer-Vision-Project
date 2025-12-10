import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


def set_seed(seed: int) -> None:
    random.seed(seed)


def load_train_labels(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected_cols = {"dataset", "scene", "image"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"train_labels missing columns: {missing}")
    return df


def load_thresholds(path: Path) -> Dict[Tuple[str, str], List[float]]:
    df = pd.read_csv(path)
    thresholds: Dict[Tuple[str, str], List[float]] = {}
    for _, row in df.iterrows():
        thresholds[(row["dataset"], row["scene"])] = [
            float(x) for x in str(row["thresholds"]).split(";")
        ]
    return thresholds


def discover_images(root: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sorted([p for p in root.glob("**/*") if p.suffix.lower() in exts])


def group_by_dataset(root: Path, split: str) -> Dict[str, List[Path]]:
    split_dir = root / split
    datasets = {}
    for ds in split_dir.iterdir():
        if ds.is_dir():
            datasets[ds.name] = discover_images(ds)
    return datasets


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

