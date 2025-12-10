# CV Scene Clustering & SfM (Lightweight)

Lightweight pipeline to cluster images into scenes, reject outliers, and reconstruct camera poses. Uses global descriptors (CLIP + DINOv2), geometric verification (LightGlue), and COLMAP/pycolmap for Structure-from-Motion.

## Python version
Tested with Python 3.10. Use a matching PyTorch build for your CUDA version (or CPU-only).

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt  # install matching torch build if needed
```

### COLMAP
- Preferred: install the COLMAP binary and ensure `colmap` is on PATH.
- Alternative: `pycolmap` (included in requirements) if the environment supports it; otherwise comment out COLMAP calls and use any available SfM backend.

## Project layout
- `src/config.py` – configuration defaults.
- `src/data_utils.py` – load labels/thresholds, image discovery.
- `src/features.py` – CLIP + DINOv2 embedding extraction.
- `src/clustering.py` – graph building, thresholding, community detection.
- `src/geo_verification.py` – LightGlue matching and RANSAC filtering.
- `src/sfm.py` – COLMAP/pycolmap helpers for pose estimation.
- `src/submission.py` – submission writer in the competition format.
- `src/pipeline.py` – orchestrates end-to-end inference.
- `run_pipeline.py` – entry point.

## Quickstart (inference on test/)
```bash
python run_pipeline.py ^
  --data_root "D:\K228727\CV PROJECT\NEW PROJECT CV" ^
  --output_csv "submission.csv"
```

This will:
1) load thresholds/metadata if present, 2) extract global embeddings, 3) build a similarity graph, 4) refine edges with LightGlue on a subset, 5) cluster images and tag low-support nodes as outliers, 6) run SfM per cluster (if COLMAP available), 7) write CSV.

## Training-time validation
Point `--train_split train` and `--train_labels train_labels.csv` to evaluate clustering against provided labels. Pose errors can be computed if training poses are available.

## Notes
- For speed, reduce image count with `--max_images` or lower `--lightglue_pairs`.
- Swap CLIP backbone via `--clip_model ViT-B-32` or DINO via `--dino_model dinov2_vits14`.
- If GPU memory is tight, set `--device cpu`.

## Outputs
- `submission.csv` with columns matching `sample_submission.csv`.
- Intermediate working directories in `./outputs/<dataset>/cluster_*`.
- Run `python run_eval.py --train_labels train_labels.csv --submission submission.csv` to compute mAA, clustering score, and harmonic mean (F1-style) using the provided train labels.

## Caveats
- pycolmap/COLMAP availability varies on Windows; if unavailable, the pipeline will skip pose estimation and still produce cluster assignments. You can post-process with any SfM tool and update the submission CSV.

