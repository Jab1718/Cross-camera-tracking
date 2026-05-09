# Cross-Camera Tracking & Re-Identification

A real-time multi-camera person tracking system that detects, tracks, and re-identifies individuals across different camera feeds using appearance and pose-based features.

## How It Works

1. **Detection & Tracking** — YOLOv8n-pose detects people and ByteTrack assigns consistent IDs within each camera
2. **Feature Extraction** — OSNet extracts 512-d appearance embeddings per person
3. **Pose Features** — Biometric ratios (hip/shoulder, leg/torso, shoulder/height) extracted from keypoints as a dark-scene fallback
4. **Re-ID Matching** — Camera 2 queries against Camera 1's gallery using cosine similarity (appearance) or Euclidean distance (pose)
5. **Brightness Fallback** — On dark frames, OSNet is skipped and pose-based matching is used instead

## Project Structure

```text
CrossCam_ReID/
├── data/
│   ├── raw_videos/         # Input videos (cam1.mp4, cam2.mp4)
│   └── gallery/            # Person crop storage
├── models/
│   ├── checkpoints/
│   │   └── osnet_x1_0_imagenet.pth
│   └── yolov8n-pose.pt
├── src/
│   ├── detector.py         # YOLOv8 + ByteTrack
│   ├── features_extract.py # OSNet embeddings + pose ratios
│   ├── matcher.py          # Gallery management + matching logic
│   ├── brightness.py       # Dark frame detection
│   └── main.py             # Pipeline controller
├── requirements.txt
├── Dockerfile
└── README.md
```

## Requirements

- Python 3.13.2
- CPU-only supported (no GPU required)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourname/cross-camera-tracking.git
cd cross-camera-tracking
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download models — YOLOv8n-pose downloads automatically on first run. OSNet downloads automatically to `models/checkpoints/` on first run.

## Usage

1. Place your video files in `data/raw_videos/`
2. Update `CAM1_PATH` and `CAM2_PATH` in `src/main.py`
3. Run:
```bash
cd src
python main.py
```

Press `q` to quit.

## Configuration

| Parameter | Location | Default | Description |
|---|---|---|---|
| `conf` | `detector.py` | `0.25` | Detection confidence threshold |
| `app_threshold` | `matcher.py` | `0.3` | Appearance match threshold (cosine distance) |
| `pose_threshold` | `matcher.py` | `0.1` | Pose match threshold (Euclidean distance) |
| `DARK_THRESHOLD` | `brightness.py` | `80` | Frame brightness cutoff |
| `frame_count % 3` | `main.py` | `3` | Gallery update frequency (every N frames) |

## Display

- **Camera 1** — Green box with local track ID
- **Camera 2** — Red label showing match result:
  - `ReID: 1 (appearance) Dist: 0.24` — matched by OSNet
  - `ReID: 1 (pose) Dist: 0.08` — matched by biometric ratios
  - `ID: 7` — no match found

## Tech Stack

- [YOLOv8](https://github.com/ultralytics/ultralytics) — Detection + Pose Estimation
- [ByteTrack](https://github.com/ifzhang/ByteTrack) — Multi-object tracking
- [OSNet](https://github.com/KaiyangZhou/deep-person-reid) — Person Re-ID feature extraction
- [OpenCV](https://opencv.org/) — Video processing
