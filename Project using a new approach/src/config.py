from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class PipelineConfig:
    data_root: Path
    output_csv: Path
    train_split: str = "train"
    test_split: str = "test"
    train_labels: Optional[Path] = None
    thresholds_csv: Optional[Path] = None
    device: str = "cuda"
    clip_model: str = "ViT-B-32"
    clip_pretrain: str = "openai"
    dino_model: str = ""
    batch_size: int = 8
    num_workers: int = 0
    max_images: Optional[int] = None
    sim_threshold: float = 0.25
    k_neighbors: int = 20
    lightglue_pairs: int = 200
    lightglue_min_matches: int = 15
    outlier_degree: int = 1
    colmap_binary: str = "colmap"
    work_dir: Path = Path("outputs")
    seed: int = 42
    use_dino: bool = True
    use_clip: bool = True
    save_embeddings: bool = True
    embedding_dir: Optional[Path] = None
    verbose: bool = True
    scenes_to_process: Optional[List[str]] = field(default_factory=list)

