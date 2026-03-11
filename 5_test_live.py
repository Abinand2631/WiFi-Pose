"""
STEP 5 — Live CSI → Skeleton Inference (2 RX, ESP32 CSI)
=========================================================
Press 'q' to quit.
"""

import serial
import numpy as np
import torch
import torch.nn as nn
import cv2
import re
from collections import deque

# ================= CONFIG =================

COM_PORT_RX1 = "COM3"
COM_PORT_RX2 = "COM7"
BAUD_RATE = 115200

WINDOW_SIZE = 100
SUBCARRIERS = 128
N_FEATURES = 256      # 128 RX1 + 128 RX2
OUTPUT_DIM = 34

MODEL_PATH = "models/tednet_best.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# ================= MODEL (MATCHES TRAINING) =================

class CNNEncoder(nn.Module):
    def __init__(self, in_channels, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, d_model, 3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, 3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model * 2, 3, padding=1),
            nn.BatchNorm1d(d_model * 2),
            nn.GELU(),
            nn.Conv1d(d_model * 2, d_model, 3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.net(x)
        x = x.permute(0, 2, 1)
        return x


class TEDNet(nn.Module):
    def __init__(self):
        super().__init__()

        d_model = 128
        n_heads = 8
        n_layers = 4
        dim_ff = 512
        dropout = 0.1

        self.cnn_encoder = CNNEncoder(N_FEATURES, d_model)

        self.pos_embed = nn.Parameter(
            torch.randn(1, WINDOW_SIZE, d_model) * 0.02
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            enc_layer,
            num_layers=n_layers
        )

        self.regressor = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, OUTPUT_DIM),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.cnn_encoder(x)
        x = x + self.pos_embed
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.regressor(x)
        return x


# ================= LOAD MODEL =================

print("Loading model...")
model = TEDNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("Model loaded.")

# ================= CSI PARSER =================

CSI_PATTERN = re.compile(r'\[([^\]]+)\]')

def parse_csi(line):
    if "CSI_DATA" not in line:
        return None

    match = CSI_PATTERN.search(line)
    if not match:
        return None

    try:
        csi_str = match.group(1)
        values = np.fromstring(csi_str, sep=',', dtype=np.float32)

        # Your firmware sends 384 values
        if len(values) < 384:
            return None

        # Convert real/imag pairs → magnitude
        real = values[0::2][:SUBCARRIERS]
        imag = values[1::2][:SUBCARRIERS]
        magnitude = np.sqrt(real**2 + imag**2)

        return magnitude.astype(np.float32)

    except:
        return None


# ================= SERIAL =================

ser1 = serial.Serial(COM_PORT_RX1, BAUD_RATE, timeout=1)
ser2 = serial.Serial(COM_PORT_RX2, BAUD_RATE, timeout=1)

print("[RX1] Connected:", COM_PORT_RX1)
print("[RX2] Connected:", COM_PORT_RX2)
print("Waiting for CSI buffers to fill...")

buffer_rx1 = deque(maxlen=WINDOW_SIZE)
buffer_rx2 = deque(maxlen=WINDOW_SIZE)

started = False

# ================= MAIN LOOP =================

while True:

    # RX1
    line1 = ser1.readline().decode(errors="ignore")
    csi1 = parse_csi(line1)
    if csi1 is not None:
        buffer_rx1.append(csi1)

    # RX2
    line2 = ser2.readline().decode(errors="ignore")
    csi2 = parse_csi(line2)
    if csi2 is not None:
        buffer_rx2.append(csi2)

    print(f"\rRX1: {len(buffer_rx1)}/{WINDOW_SIZE}   "
          f"RX2: {len(buffer_rx2)}/{WINDOW_SIZE}", end="")

    if len(buffer_rx1) < WINDOW_SIZE or len(buffer_rx2) < WINDOW_SIZE:
        continue

    if not started:
        print("\nBuffers full. Starting inference...\n")
        started = True

    rx1_arr = np.array(buffer_rx1)
    rx2_arr = np.array(buffer_rx2)

    x = np.concatenate([rx1_arr, rx2_arr], axis=1)  # (100,256)

    # Normalization (same style as training)
    mean = x.mean()
    std = x.std() + 1e-6
    x = (x - mean) / std

    x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(x_t).cpu().numpy()[0]

    # ================= DRAW SKELETON & RECOGNISE ACTION =================
    kps = pred.reshape(17, 2)
    
    # Define COCO connections (0: Nose, 1: LEye, 2: REye, 3: LEar, 4: REar, 
    # 5: LShoulder, 6: RShoulder, 7: LElbow, 8: RElbow, 9: LWrist, 10: RWrist,
    # 11: LHip, 12: RHip, 13: LKnee, 14: RKnee, 15: LAnkle, 16: RAnkle)
    SKELETON = [
        (15, 13), (13, 11), (16, 14), (14, 12),  # Legs
        (11, 12), (5, 11), (6, 12), (5, 6),      # Torso/Pelvis
        (5, 7), (7, 9), (6, 8), (8, 10),         # Arms
        (1, 2), (0, 1), (0, 2), (1, 3), (2, 4),  # Face
        (3, 5), (4, 6)                           # Neck/Shoulders
    ]

    canvas = np.zeros((480, 640, 3), dtype=np.uint8)

    # Convert normalized (0..1) to pixel coords
    kps_pixels = []
    for x_norm, y_norm in kps:
        x_pix = int(np.clip(x_norm, 0, 1) * 640)
        y_pix = int(np.clip(y_norm, 0, 1) * 480)
        kps_pixels.append((x_pix, y_pix))

    # Draw Connections (Lines)
    for p1, p2 in SKELETON:
        pt1 = kps_pixels[p1]
        pt2 = kps_pixels[p2]
        
        # Don't draw if the point is predicted exactly at (0,0) which means missing GT
        if pt1 == (0,0) or pt2 == (0,0):
             continue

        cv2.line(canvas, pt1, pt2, (255, 100, 0), 3) # Blue skeleton lines

    # Draw Joints (Dots)
    for x_pix, y_pix in kps_pixels:
        if (x_pix, y_pix) == (0,0):
            continue
        cv2.circle(canvas, (x_pix, y_pix), 5, (0, 255, 0), -1) # Green dots

    # Simple Heuristic Action Recognition
    action = "Unknown"
    
    try:
        # Get coordinates for heuristics
        # Note: Y-axis goes DOWN in OpenCV (0 is top of screen, 480 is bottom)
        left_hip_y, right_hip_y = kps_pixels[11][1], kps_pixels[12][1]
        left_knee_y, right_knee_y = kps_pixels[13][1], kps_pixels[14][1]
        left_ankle_y, right_ankle_y = kps_pixels[15][1], kps_pixels[16][1]
        left_ankle_x, right_ankle_x = kps_pixels[15][0], kps_pixels[16][0]

        avg_hip_y = (left_hip_y + right_hip_y) / 2
        avg_knee_y = (left_knee_y + right_knee_y) / 2
        
        # Distance between ankles for walking heuristics
        ankle_dist_x = abs(left_ankle_x - right_ankle_x)

        # 1. Check if sitting (hips are close to knees vertically)
        if abs(avg_hip_y - avg_knee_y) < 40: # threshold in pixels
            action = "Sitting"
        
        # 2. Check if walking (legs wide apart horizontally)
        elif ankle_dist_x > 90:
            action = "Walking"
            
        # 3. Otherwise standing (hips significantly above knees and legs together)
        elif avg_hip_y < avg_knee_y - 40: 
            action = "Standing"

    except Exception:
        pass # If points are missing or out of bounds, keep "Unknown"

    cv2.putText(canvas, f"Action: {action}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)


    cv2.imshow("WiFi-Pose Live Inference (q to quit)", canvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

ser1.close()
ser2.close()
cv2.destroyAllWindows()
print("\nDone.")