"""
STEP 1 — CSI + Video Capture
=============================
Records CSI from 2 ESP32 receivers simultaneously with webcam video.
Run TWICE:
  1. SESSION_LABEL = "Person"  (person in room)
  2. SESSION_LABEL = "Empty"   (empty room)
"""

import serial
import cv2
import threading
import csv
import time
import os
import re
from datetime import datetime

# ===================== CONFIGURATION =====================
SERIAL_PORTS = {
    1: "COM7",   # ← update to your RX1 port
    2: "COM3",   # ← update to your RX2 port
}
BAUD_RATE     = 115200
SESSION_LABEL = "Person"          # "Person" or "Empty"
DURATION_SEC  = 180               # recording duration in seconds
OUTPUT_DIR    = "data"
N_SUB         = 128               # number of subcarriers
CSI_PATTERN   = re.compile(r'CSI_DATA,\d+,[^\"]*,,"\[([^\[\]]*)\]"")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===================== CSI READER THREAD =====================

def csi_reader(dev_id, port, stop_event):
    filename = os.path.join(OUTPUT_DIR, f"{SESSION_LABEL}_RX{dev_id}.csv")
    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=1)
        print(f"[RX{dev_id}] Connected: {port}")
    except Exception as e:
        print(f"[RX{dev_id}] FAILED to open {port}: {e}")
        return

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        # header: timestamp + subcarrier columns
        writer.writerow(["timestamp"] + [f"sub_{i}" for i in range(N_SUB)])

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
                    if len(vals) < N_SUB:
                        continue
                    ts = time.time()
                    writer.writerow([ts] + vals[:N_SUB])
                else:
                    time.sleep(0.001)
            except Exception as ex:
                print(f"[RX{dev_id}] Read error: {ex}")
                break

    ser.close()
    print(f"[RX{dev_id}] Saved → {filename}")


# ===================== VIDEO CAPTURE =====================

def video_capture(stop_event):
    filename = os.path.join(OUTPUT_DIR, f"video_{SESSION_LABEL.lower()}.avi")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[CAM] ERROR: Cannot open webcam")
        return

    fps    = 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out    = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    print(f"[CAM] Recording {width}x{height} @ {fps}fps → {filename}")

    while not stop_event.is_set():
        ret, frame = cap.read()
        if ret:
            out.write(frame)
            cv2.imshow(f"Recording — {SESSION_LABEL} (press q to stop early)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                stop_event.set()
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"[CAM] Saved → {filename}")


# ===================== MAIN =====================

if __name__ == "__main__":
    print("=" * 55)
    print(f"WiFi-Pose Capture  |  Session: {SESSION_LABEL}")
    print(f"Duration          : {DURATION_SEC}s")
    print(f"Output dir        : {OUTPUT_DIR}/")
    print("=" * 55)
    print("Starting in 3 seconds... get ready!\n")
    time.sleep(3)

    stop_event = threading.Event()

    # Start CSI readers
    threads = []
    for dev_id, port in SERIAL_PORTS.items():
        t = threading.Thread(target=csi_reader, args=(dev_id, port, stop_event), daemon=True)
        t.start()
        threads.append(t)
        time.sleep(0.2)

    # Start video in separate thread
    vid_thread = threading.Thread(target=video_capture, args=(stop_event,), daemon=True)
    vid_thread.start()

    # Run for DURATION_SEC then stop
    try:
        time.sleep(DURATION_SEC)
    except KeyboardInterrupt:
        print("\n[MAIN] Interrupted by user")

    print("\n[MAIN] Stopping recording...")
    stop_event.set()

    for t in threads:
        t.join(timeout=3)
    vid_thread.join(timeout=3)

    print("\n✅ Recording complete!")
    print(f"   data/{{SESSION_LABEL}}_RX1.csv")
    print(f"   data/{{SESSION_LABEL}}_RX2.csv")
    print(f"   data/video_{{SESSION_LABEL.lower()}}.avi")
