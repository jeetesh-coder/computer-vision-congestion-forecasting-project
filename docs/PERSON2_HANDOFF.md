# Person 2 — Tracking, Feature Extraction and Congestion Forecasting

## What Person 1 completed
- Image/annotation audit and YOLO conversion.
- Fixed sequence-separated detection splits.
- Minority-containing training-image oversampling.
- Tuned standard YOLOv8s and identity-initialized ECA architecture.
- Saved models, configurations and validation comparisons.

Selected enhanced model: models/eca/best.pt.
Baseline for comparison: models/baseline/best.pt.
Required custom module: src/detection/eca_neck.py.
Class mapping: 0=car, 1=truck, 2=bus.
Ultralytics version: 8.4.210.
Final test evaluation is pending.

## Load the enhanced detector
From the repository root:

    import sys
    sys.path.insert(0, "src/detection")
    from eca_neck import ECANeckBlock
    from ultralytics import YOLO

    detector = YOLO("models/eca/best.pt")
    result = detector.predict("example.jpg", imgsz=640, conf=0.25)[0]

    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)

Boxes are in original-image pixel coordinates.
The library handles resizing and letterboxing.
0.25 is an initial inference threshold, not a tuned tracking threshold.
Validation AP used conf=0.001 to build precision-recall curves.
Import eca_neck before loading the custom checkpoint.

## Person 2 tasks
1. Check image sequence continuity, original frame numbers and timing.
2. Integrate a tracker such as ByteTrack.
3. Preserve track IDs within each sequence; reset between sequences.
4. Extract timestamped features such as ROI counts, class counts,
   line-crossing flow and motion features.
5. Define congestion ground truth and the forecasting horizon.
6. Build past-only temporal windows and evaluate on held-out sequences
   or time blocks without overlapping-window leakage.
7. Document tracking and forecasting evaluation separately from detector AP.

## Required interface
For each detection, preserve:
sequence_id, original_frame_number, timestamp_seconds,
x1, y1, x2, y2, confidence, class_id.
Tracking adds track_id.

## Important data limitations
The 995-image subset is sampled; it may be insufficient for reliable
tracking and forecasting. Additional consecutive frames or source videos
may be required. Verify this before interpreting trajectories or flow.

Do not assume frame spacing is uniform.
Do not calculate timestamps from subset position when frames are missing.
Real-world speed requires calibration and UAV camera-motion handling.
Congestion forecasting requires temporal targets/labels; detections alone
do not supply congestion ground truth.

Raw annotation target IDs are ground truth, not tracker predictions.
Generic `vehicle` annotations do not validate car/truck/bus classification.

## Results and scope
Baseline validation mAP@50: 0.533.
ECA validation mAP@50: 0.553.
Improvement: 2.0 percentage points in this run.
No statistical significance or final test improvement is established.
Tracking, feature extraction and forecasting are not implemented yet.

## Deliverables expected from Person 2
- Tracking code and a sequence demonstration.
- Timestamped feature table with definitions and units.
- Forecasting model, horizon, ground truth and evaluation.
- End-to-end execution instructions and limitations.
