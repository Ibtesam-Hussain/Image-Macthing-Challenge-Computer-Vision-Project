Image Matching Challenge 2025: A Computer Vision Semester Project
Project Overview

This project explores classical and modern image matching architectures in computer vision, aiming to establish reliable correspondences between image pairs for tasks like image registration, 3D reconstruction, and SLAM. The inspiration came from the Image Matching Challenge 2025
 hosted by Czech Technical University in Prague.

We implemented and evaluated:

Classical pipelines using ORB and SIFT descriptors with Brute-Force (BF) and FLANN matchers.

Transformer-based pipelines using LightGlue (SuperPoint + transformer-based feature matching).

Custom architectures integrating global descriptors (CLIP + DINOv2), geometric verification, and 3D reconstruction using COLMAP and a Sparse Feature-based Local Mapping (SFLM) module.

Authors

Ibtesam Hussain – K224125@nu.edu.pk

Shaheer Uddin – K228719@nu.edu.pk

Safey Ahmed – K224039@nu.edu.pk

Sajjad Ali – K228729@nu.edu.pk

Department of Artificial Intelligence, FAST National University of Computer and Emerging Sciences

Key Contributions

Classical Matching Pipeline: ORB/SIFT descriptors with BFMatcher/FLANN and homography-based geometric verification.

LightGlue Transformer Framework: SuperPoint-based keypoints with transformer-based feature matching.

Custom Architecture: Combines global descriptors (CLIP + DINOv2) with LightGlue and 3D reconstruction using COLMAP/pycolmap.

Sparse Feature-based Local Mapping (SFLM): Efficient local-scale mapping using PnP and bundle adjustment for real-time applications.

Methodology
1. Classical Pipeline

Keypoints extracted using ORB and SIFT.

Feature correspondence computed via BFMatcher (ORB) or FLANN (SIFT).

Matches filtered using distance ratio tests and homography-based geometric verification.

2. LightGlue Framework

Keypoints extracted using SuperPoint.

Features matched using LightGlue transformers.

Evaluation performed using number of matches, inlier ratio, precision, recall, and F1-score.

3. CLIP + COLMAP Architecture

Images clustered using global descriptors (CLIP + DINOv2).

Feature matches verified with LightGlue.

Camera poses and sparse/dense reconstructions obtained using COLMAP/pycolmap.

Evaluation includes F1 score, recall, and mAA.

4. Sparse Feature-based Local Mapping (SFLM)

Uses SuperPoint keypoints and PnP pose estimation.

Triangulation and refinement of 3D points via bundle adjustment.

Efficient for real-time or local mapping scenarios.

Experiments and Evaluation

We evaluated the pipelines on ETs and Stairs datasets. Metrics included:

Number of feature matches

Inlier ratio after RANSAC

Precision, Recall, and F1-score

Method	Avg Matches	Precision	Recall	F1
SIFT + FLANN	743	0.76	1.00	1.00
ORB + BF	1000	1.00	1.00	1.00
LightGlue (SuperPoint)	1036.75	0.88	0.82	0.85
CLIP + COLMAP	3381.2	0.90	0.85	0.87

Observations:

LightGlue improved robustness and inlier ratio compared to classical methods.

CLIP + COLMAP achieved higher match density and better overall 3D reconstruction.

SFLM offers fast, local mapping suitable for real-time scenarios.

Project Architecture

The overall architecture combines classical and learned pipelines with geometric verification and reconstruction modules.

Pipeline Steps:

Input Image Pairs → Classical (ORB/SIFT) & Modern (LightGlue, CLIP+DINOv2) Extractors

Feature Matching → BFMatcher/FLANN or LightGlue

Geometric Verification → RANSAC + Homography

Evaluation → mAA, Precision, Recall, F1

3D Reconstruction → COLMAP / SFLM

Output → Matched keypoints and reconstructed 3D models

(See figures in the LaTeX report for detailed architecture visualization)

Future Work

Integrating geometric priors to improve correspondence accuracy.

Optimizing transformer inference speed for real-time deployment.

Combining dense depth estimation with learned correspondences for better scene understanding.

References

Lowe, D.G., "Distinctive image features from scale-invariant keypoints," IJCV, 2004.

Rublee, E., et al., "ORB: An efficient alternative to SIFT or SURF," ICCV, 2011.

DeTone, D., Malisiewicz, T., Rabinovich, A., "SuperPoint: Self-supervised interest point detection and description," CVPR Workshops, 2018.

Sarlin, P.-E., et al., "SuperGlue: Learning feature matching with graph neural networks," CVPR, 2020.

Lü, J., Sarlin, P.-E., et al., "LightGlue: Local Feature Matching at Light Speed," CVPR, 2023.

How to Run

Install dependencies:

pip install opencv-python numpy torch torchvision pycolmap lightglue


Prepare datasets (ETs and Stairs images).

Run classical pipelines with OpenCV scripts.

Run LightGlue + SuperPoint pipelines for transformer-based matching.

Run CLIP + COLMAP pipelines for 3D reconstruction.

Evaluate metrics and visualize results.

License

This project is developed for academic purposes under FAST National University of Computer and Emerging Sciences.
