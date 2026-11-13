# Computer Vision Congestion Forecasting Project

## Person 1 — Object Detection
Final notebook: notebooks/CV_end_sem_project.ipynb.

A tuned YOLOv8s baseline is compared with an identity-initialized ECA
enhancement at three neck locations. Classes: 0=car, 1=truck, 2=bus.
ECA adds 9 parameters.

| Validation metric | Baseline | ECA |
|---|---:|---:|
| Precision | 0.572 | 0.619 |
| Recall | 0.510 | 0.511 |
| mAP@50 | 0.533 | 0.553 |
| mAP@50–95 | 0.340 | 0.347 |
| Car AP@50 | 0.767 | 0.785 |
| Truck AP@50 | 0.204 | 0.210 |
| Bus AP@50 | 0.628 | 0.664 |

The improvement is 2.0 percentage points in validation mAP@50.
These are single-run validation results; test evaluation remains pending.
Statistical significance has not been established.

## Dataset and preparation
The raw dataset contains 995 images with Supervisely JSON annotations.
Detection uses 312 training, 79 validation and 243 test images.
361 generic-vehicle test images are excluded from three-class detection.
The saved split keeps sequences separate.
Truck/bus-containing training images are repeated twice:
530 training entries from 312 unique images.

The final notebook retains the original Colab paths.
For retraining, place data/raw/Dataset in MyDrive/Dataset and copy
data/splits/three_class_v1.json into
MyDrive/Dataset/experiment_splits/three_class_v1.json.
Do not recreate or randomly change the split.
Existing run folders intentionally prevent accidental overwriting.
The notebook retains the Coordinate Attention experiment; ECA is the
selected enhanced model.

## Person 2
Read docs/PERSON2_HANDOFF.md for loading the detector and implementing
tracking, temporal feature extraction and congestion forecasting.
No detector retraining is required to begin integration.

## Attribution
Dataset: UAVDT, Du et al., ECCV 2018.
Dataset and dependencies retain their upstream licenses and terms.
