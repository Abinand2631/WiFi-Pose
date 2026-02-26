"""
STEP 5 — Live Inference & Visualisation
========================================
Streams CSI from 2 ESP32 receivers in real-time,
runs TEDNet 2RX, and draws predicted skeleton on screen.

Controls:
  q  — quit
  s  — save current frame
"""

import serial
import numpy as np
import torch
import torch.nn as nn
import cv2
import threading
import collections
import time
import os
import re
from train import TEDNet   # reuse model definition from 4_train.py

# ===================== CONFIGURATION =====================
SERIAL_PORTS = {
    1: "COM7",   # ← update to your RX1 port
    2: "COM3",   # ← update to your RX2 port
}
BAUD_RATE    = 115200
MODEL_PATH   = os.path.join("models", "tednet_best.pth")
CSI_MEAN     = os.path.join("data", "processed", "csi_mean.npy")
CSI_STD      = os.path.join("data", "processed", "csi_std.npy")

WINDOW_SIZE  = 100
N_SUB        = 128
N_FEATURES   = 256    # RX1 + RX2
N_JOINTS     = 17
CSI_PATTERN  = re.compile(r'CSI_DATA,\d+,[^\"]*,"\[([^\[\]]*)\]"")

DISPLAY_W    = 640
DISPLAY_H    = 480
SAVE_DIR     = "output_frames"
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ===================== COCO SKELETON =====================
# 17 COCO keypoints:
# 0=nose, 1=l_eye, 2=r_eye, 3=l_ear, 4=r_ear,
# 5=l_shoulder, 6=r_shoulder, 7=l_elbow, 8=r_elbow,
# 9=l_wrist, 10=r_wrist, 11=l_hip, 12=r_hip,
# 13=l_knee, 14=r_knee, 15=l_ankle, 16=r_ankle

SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),            # head
    (5, 6),                                       # shoulders
    (5, 7), (7, 9),                               # left arm
    (6, 8), (8, 10),                              # right arm
    (5, 11), (6, 12), (11, 12),                   # torso
    (11, 13), (13, 15),                           # left leg
    (12, 14), (14, 16),                           # right leg
]

JOINT_COLOUR  = (0, 255, 0)        # green
BONE_COLOUR   = (255, 100, 0)      # blue-orange
POINT_RADIUS  = 6
LINE_THICK    = 2


# ===================== CSI BUFFER =====================

class CSIBuffer:
    def __init__(self, maxlen):
        self.buf  = collections.deque(maxlen=maxlen)
        self.lock = threading.Lock()

    def push(self, row):
        with self.lock:
            self.buf.append(row)

    def get_window(self):
        with self.lock:
            if len(self.buf) < self.buf.maxlen:
                return None
            return np.array(list(self.buf), dtype=np.float32)


# ===================== SERIAL READER THREAD =====================

def csi_reader_thread(dev_id, port, buffer, stop_event):
    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=1)
        print(f"[RX{dev_id}] Connected: {port}")
    except Exception as e:
        print(f"[RX{dev_id}] FAILED to open {port}: {e}")
        stop_event.set()
        return

    while not stop_event.is_set():
        try:
            if ser.in_waiting:
                line = ser.readline().decode("utf-8", errors="ignore").rstrip("\x00")
                if "CSI_DATA" not in line:
                    continue
                match = CSI_PATTERN.search(line)
                if not match:
                    continue
                clean = re.sub(r"[^\d,\-]", "", match.group(1).strip())
                vals  = [int(x) for x in clean.split(",") if x]
                if len(vals) < 2 * N_SUB:
                    continue
                # Compute amplitude from I/Q
                I   = np.array(vals[0:2*N_SUB:2], dtype=np.float32)
                Q   = np.array(vals[1:2*N_SUB:2], dtype=np.float32)
                amp = np.sqrt(I**2 + Q**2)
                buffer.push(amp)
            else:
                time.sleep(0.001)
        except Exception as ex:
            print(f"[RX{dev_id}] Error: {ex}")
            break

    ser.close()


# ===================== LOAD MODEL =====================

def load_model(path):
    model = TEDNet().to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    print(f"✅ Model loaded: {path}")
    return model


# ===================== INFERENCE =====================

def infer(model, rx1_window, rx2_window, mean, std):
    """
    rx1_window, rx2_window : (WINDOW_SIZE, N_SUB)
    Returns: keypoints (17, 2) in pixel coords for DISPLAY_W × DISPLAY_H
    """
    x = np.concatenate([rx1_window, rx2_window], axis=1)  # (W, 256)
    x = (x - mean) / std
    x_t = torch.tensor(x[np.newaxis], dtype=torch.float32).to(DEVICE)  # (1, W, 256)

    with torch.no_grad():
        pred = model(x_t).cpu().numpy()[0]   # (34,)

    kps = pred.reshape(17, 2)
    kps[:, 0] *= DISPLAY_W
    kps[:, 1] *= DISPLAY_H
    return kps.astype(int)


# ===================== DRAW SKELETON =====================

def draw_skeleton(frame, keypoints):
    for (a, b) in SKELETON_EDGES:
        pt1 = tuple(keypoints[a])
        pt2 = tuple(keypoints[b])
        cv2.line(frame, pt1, pt2, BONE_COLOUR, LINE_THICK)
    for kp in keypoints:
        cv2.circle(frame, tuple(kp), POINT_RADIUS, JOINT_COLOUR, -1)
    return frame


# ===================== MAIN =====================

if __name__ == "__main__":
    print("=" * 55)
    print("WiFi-Pose — Live Inference")
    print("=" * 55)

    # Load normalisation stats
    mean = np.load(CSI_MEAN) if os.path.exists(CSI_MEAN) else 0.0
    std  = np.load(CSI_STD)  if os.path.exists(CSI_STD)  else 1.0

    # Load model
    model = load_model(MODEL_PATH)

    # CSI buffers (one per receiver)
    buf1 = CSIBuffer(maxlen=WINDOW_SIZE)
    buf2 = CSIBuffer(maxlen=WINDOW_SIZE)

    stop_event = threading.Event()

    # Start reader threads
    t1 = threading.Thread(target=csi_reader_thread, args=(1, SERIAL_PORTS[1], buf1, stop_event), daemon=True)
    t2 = threading.Thread(target=csi_reader_thread, args=(2, SERIAL_PORTS[2], buf2, stop_event), daemon=True)
    t1.start(); t2.start()

    print("Waiting for CSI buffers to fill...")
    frame_count = 0
    fps_time    = time.time()

    while not stop_event.is_set():
        win1 = buf1.get_window()
        win2 = buf2.get_window()

        # Create blank canvas
        canvas = np.zeros((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8)

        if win1 is None or win2 is None:
            # Not enough data yet
            cv2.putText(canvas, "Collecting CSI data...",
                        (20, DISPLAY_H // 2), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (255, 255, 255), 2)
        else:
            try:
                kps = infer(model, win1, win2, mean, std)
                canvas = draw_skeleton(canvas, kps)

                # FPS counter
                frame_count += 1
                elapsed = time.time() - fps_time
                if elapsed >= 1.0:
                    fps = frame_count / elapsed
                    frame_count = 0
                    fps_time    = time.time()
                else:
                    fps = frame_count / max(elapsed, 1e-6)

                cv2.putText(canvas, f"FPS: {fps:.1f}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (200, 200, 200), 2)
                cv2.putText(canvas, "WiFi-Pose Live",
                            (10, DISPLAY_H - 15), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (100, 200, 255), 1)
            except Exception as ex:
                cv2.putText(canvas, f"Inference error: {ex}",
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 255), 1)

        cv2.imshow("WiFi-Pose — Live Inference (q to quit)", canvas)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("Quitting...")
            stop_event.set()
        elif key == ord("s"):
            fname = os.path.join(SAVE_DIR, f"frame_{int(time.time())}.png")
            cv2.imwrite(fname, canvas)
            print(f"Saved: {fname}")

    cv2.destroyAllWindows()
    stop_event.set()
    print("Done.")
