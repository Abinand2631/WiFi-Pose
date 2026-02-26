# WiFi-Pose

Real-time human pose estimation using WiFi Channel State Information (CSI) from 2 ESP32 receivers.

**GT Model:** RTMPose-x (MMPose) — highest accuracy 2D pose estimation  
**CSI Model:** TEDNet 2RX — CNN Encoder + Transformer  
**Output:** 17 COCO keypoints (x, y) normalised to [0, 1]

---

## Pipeline

```
1_capture.py       →  Record CSI + video  (Person + Empty sessions)
2_generate_gt.py   →  RTMPose-x GT generation  (run ONCE, offline)
3_extraction.py    →  Timestamp-synced CSI windowing
4_train.py         →  Train WiFiPoseNet on your data
5_test_live.py     →  Real-time skeleton from WiFi
```

---

## Hardware

| Component | Details |
|-----------|---------|
| 2× ESP32 | CSI receivers (COM ports) |
| 1× ESP32 | CSI transmitter |
| Webcam | For training data recording |
| GPU | RTX 2050 or better (for GT generation + training) |

---

## Installation

```bash
# 1. Clone
git clone https://github.com/Abinand2631/WiFi-Pose.git
cd WiFi-Pose

# 2. Install base requirements
pip install -r requirements.txt

# 3. Install MMPose (for GT generation only)
pip install openmim
mim install mmengine "mmcv>=2.0.0" "mmdet>=3.0.0" "mmpose>=1.0.0"
```

---

## Step-by-Step Usage

### Step 1 — Record Data

Edit `1_capture.py`:
```python
SERIAL_PORTS = {1: "COM7", 2: "COM3"}   # your ports
SESSION_LABEL = "Person"                 # or "Empty"
```

Run twice:
```bash
# First: record WITH person in room
python 1_capture.py   # SESSION_LABEL = "Person"

# Then: record empty room
python 1_capture.py   # SESSION_LABEL = "Empty"
```

Outputs in `data/`:
```
data/Person_RX1.csv   data/Person_RX2.csv   data/video_person.avi
data/Empty_RX1.csv    data/Empty_RX2.csv    data/video_empty.avi
```

---

### Step 2 — Generate Ground Truth (runs once)

```bash
python 2_generate_gt.py
```

- Uses **RTMPose-x** (best accuracy pose model)
- Downloads ~80 MB weights automatically on first run
- Takes ~2–4 hours for 180s video on RTX 2050
- Outputs: `data/gt/skeleton_gt.npy`  shape: `(N_frames, 17, 3)`

---

### Step 3 — Extract & Sync CSI Windows

```bash
python 3_extraction.py
```

**Key fix vs old code:** Uses actual `timestamp` column for CSI↔video alignment — not a fake uniform mapping.

Outputs in `data/processed/`:
```
csi_person.npy   (N, 2, 128, 30)
gt_person.npy    (N, 17, 2)
csi_empty.npy    (M, 2, 128, 30)
gt_empty.npy     (M, 17, 2)
```

---

### Step 4 — Train

```bash
python 4_train.py
```

- **Architecture:** CNNEncoder → PoseTransformer → MLP head
- **Loss:** Wing loss with validity mask (ignores undetected frames)
- **Optimizer:** AdamW + Cosine LR schedule
- **Output:** `models/best_model.pth`
- Typical training time: ~1–2 hours on RTX 2050

---

### Step 5 — Live Inference

Edit `5_test_live.py`:
```python
SERIAL_PORTS = {1: "COM3", 2: "COM4"}   # your live inference ports
```

```bash
python 5_test_live.py
```

Displays:
- Live skeleton overlay (17 coloured joints + limb connections)
- Person detection status
- Inference FPS
- Joint position heatmap

---

## Key Improvements vs Old Repo

| Old (WiFi-Densepose) | New (WiFi-Pose) |
|---|---|
| DensePose → 3,072 outputs | RTMPose-x → 34 outputs |
| Fake uniform timestamp sync | Real timestamp-based sync |
| Model collapses to -1 | Masked Wing loss ignores bad GT |
| Only MSELoss | Wing loss (better keypoint regression) |
| Window size = 10 | Window size = 30 (more context) |
| 100 epochs, SGD-like | 150 epochs, AdamW + Cosine LR |

---

## Folder Structure

```
WiFi-Pose/
├── 1_capture.py
├── 2_generate_gt.py
├── 3_extraction.py
├── 4_train.py
├── 5_test_live.py
├── requirements.txt
├── README.md
├── data/
│   ├── Person_RX1.csv
│   ├── Person_RX2.csv
│   ├── video_person.avi
│   ├── Empty_RX1.csv
│   ├── Empty_RX2.csv
│   ├── video_empty.avi
│   ├── gt/
│   │   └── skeleton_gt.npy
│   └── processed/
│       ├── csi_person.npy
│       ├── gt_person.npy
│       ├── csi_empty.npy
│       └── gt_empty.npy
└── models/
    └── best_model.pth
```

---

## Tips for Better Accuracy

1. **Record more data** — 3+ sessions with different activities (standing, sitting, walking, raising arms)
2. **Good lighting** — RTMPose needs clear video to generate accurate GT
3. **Stay in frame** — person should be fully visible throughout recording
4. **Consistent setup** — same room, same antenna positions at training and inference time
