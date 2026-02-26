'"""
STEP 3 — CSI Feature Extraction & Synchronisation
===================================================
1. Loads raw CSI CSVs from both receivers
2. Applies Hampel filter + Butterworth bandpass (0.1–10 Hz)
3. Computes amplitude from I/Q pairs → (N_sub,) per sample
4. Synchronises CSI timestamps with video frame timestamps
5. Builds sliding windows: (window_size, N_sub*2) per sample
6. Saves aligned dataset: X.npy (CSI windows) + Y.npy (GT keypoints)

Output:
  data/processed/X.npy   shape: (N_samples, window_size, 256)
  data/processed/Y.npy   shape: (N_samples, 34)   [17 joints × (x,y)]
"""

import numpy as np
import pandas as pd
import os
from scipy.signal import butter, filtfilt
from tqdm import tqdm

# ===================== CONFIGURATION =====================
DATA_DIR     = "data"
OUTPUT_DIR   = os.path.join(DATA_DIR, "processed")
GT_PATH      = os.path.join(DATA_DIR, "gt", "skeleton_gt.npy")

PERSON_CSI   = [
    os.path.join(DATA_DIR, "Person_RX1.csv"),
    os.path.join(DATA_DIR, "Person_RX2.csv"),
]
EMPTY_CSI    = [
    os.path.join(DATA_DIR, "Empty_RX1.csv"),
    os.path.join(DATA_DIR, "Empty_RX2.csv"),
]

N_SUB        = 128       # subcarriers per receiver
WINDOW_SIZE  = 100       # temporal window (samples)
STEP_SIZE    = 10        # stride between windows
VIDEO_FPS    = 30.0      # must match 1_capture.py

# Bandpass filter
LOWCUT       = 0.1       # Hz
HIGHCUT      = 10.0      # Hz
SAMPLE_RATE  = 100.0     # estimated CSI sample rate (Hz)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===================== SIGNAL PROCESSING =====================
def hampel_filter(series, window=10, n_sigma=3.0):
    """Hampel identifier: replace outliers with local median."""
    s = series.copy()
    k = 1.4826  # consistent with normal distribution
    for i in range(window, len(s) - window):
        win   = s[i - window: i + window + 1]
        med   = np.median(win)
        mad   = k * np.median(np.abs(win - med))
        if np.abs(s[i] - med) > n_sigma * mad:
            s[i] = med
    return s

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq  = 0.5 * fs
    low  = lowcut / nyq
    high = min(highcut / nyq, 0.99)
    b, a = butter(order, [low, high], btype="band")
    return b, a

def process_csi(csv_path, n_sub=N_SUB):
    """
    Load CSV, apply Hampel + bandpass filter.
    Returns:
      timestamps : np.ndarray (N,)
      amplitude  : np.ndarray (N, n_sub)
    """
    print(f"  Loading {csv_path} ...")
    df = pd.read_csv(csv_path)

    # Auto-detect timestamp column
    ts_col = None
    for c in df.columns:
        if "time" in c.lower() or "ts" in c.lower():
            ts_col = c
            break
    if ts_col is None:
        # Assume first column is timestamp
        ts_col = df.columns[0]
    print(f"  Timestamp column: '{ts_col}'")

    timestamps = df[ts_col].values.astype(np.float64)

    # Select numeric CSI columns (exclude timestamp)
    csi_cols = [c for c in df.columns if c != ts_col]
    raw = df[csi_cols].values.astype(np.float32)   # (N, n_col)

    # If I/Q interleaved (2*n_sub cols) → compute amplitude
    if raw.shape[1] >= 2 * n_sub:
        I = raw[:, 0:2*n_sub:2]    # even indices
        Q = raw[:, 1:2*n_sub:2]    # odd  indices
        amp = np.sqrt(I**2 + Q**2)
    elif raw.shape[1] >= n_sub:
        amp = raw[:, :n_sub]
    else:
        raise ValueError(f"Expected >= {n_sub} CSI columns, got {raw.shape[1]}")

    print(f"  Shape before filter : {amp.shape}")

    # Estimate sample rate
    dt = np.median(np.diff(timestamps))
    fs = 1.0 / dt if dt > 0 else SAMPLE_RATE
    fs = min(max(fs, 10.0), 2000.0)   # clamp to sane range
    print(f"  Estimated sample rate: {fs:.1f} Hz")

    # Apply Hampel + bandpass per subcarrier
    b, a = butter_bandpass(LOWCUT, HIGHCUT, fs)
    filtered = np.zeros_like(amp)
    for i in tqdm(range(amp.shape[1]), desc="  Filtering", leave=False):
        col = hampel_filter(amp[:, i])
        filtered[:, i] = filtfilt(b, a, col)

    print(f"  Shape after filter  : {filtered.shape}")
    return timestamps, filtered

def subtract_empty(person_amp, empty_amp):
    """Subtract empty-room background (min-length aligned)."""
    n = min(len(person_amp), len(empty_amp))
    return person_amp[:n] - empty_amp[:n]

# ===================== SYNCHRONISATION =====================
def sync_csi_to_video(csi_timestamps, csi_data, video_fps, n_frames):
    """
    Resample CSI to align with video frames using linear interpolation.
    Returns: csi_at_frames  (n_frames, n_sub)
    """
    # Build video frame timestamps relative to CSI start
    t0 = csi_timestamps[0]
    t_end = csi_timestamps[-1]
    video_times = np.linspace(t0, t_end, n_frames)

    n_sub = csi_data.shape[1]
    csi_at_frames = np.zeros((n_frames, n_sub), dtype=np.float32)

    for i in range(n_sub):
        csi_at_frames[:, i] = np.interp(video_times, csi_timestamps, csi_data[:, i])

    return csi_at_frames

# ===================== SLIDING WINDOW =====================
def build_windows(csi_rx1, csi_rx2, gt_kps,
                  window_size=WINDOW_SIZE, step=STEP_SIZE):
    """
    Combine RX1 + RX2 CSI → sliding windows aligned with GT.

    csi_rx1  : (N_frames, N_sub)
    csi_rx2  : (N_frames, N_sub)
    gt_kps   : (N_frames, 17, 3)

    Returns:
      X : (M, window_size, 2*N_sub)
      Y : (M, 34)   normalised x,y only (confidence dropped)
    """
    n = min(len(csi_rx1), len(csi_rx2), len(gt_kps))
    csi = np.concatenate([csi_rx1[:n], csi_rx2[:n]], axis=1)  # (N, 256)
    gt  = gt_kps[:n]

    X, Y = [], []
    for start in range(0, n - window_size, step):
        end     = start + window_size
        window  = csi[start:end]                   # (window_size, 256)
        # Use GT at centre frame of window
        mid     = (start + end) // 2
        label   = gt[mid, :, :2].flatten()         # (34,) x,y only

        # Skip frames where all keypoints are zero (no detection)
        if np.all(label == 0):
            continue

        X.append(window)
        Y.append(label)

    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

# ===================== NORMALISATION =====================
def normalise(X, Y):
    """Zero-mean unit-variance normalisation on CSI (per-subcarrier)."""
    mean = X.mean(axis=(0, 1), keepdims=True)
    std  = X.std(axis=(0, 1), keepdims=True) + 1e-8
    X_n  = (X - mean) / std
    np.save(os.path.join(OUTPUT_DIR, "csi_mean.npy"), mean)
    np.save(os.path.join(OUTPUT_DIR, "csi_std.npy"),  std)
    return X_n, Y

# ===================== MAIN =====================
if __name__ == "__main__":
    print("=" * 55)
    print("WiFi-Pose — CSI Extraction & Sync")
    print("=" * 55)

    # ── Load GT ──────────────────────────────────────────────
    if not os.path.exists(GT_PATH):
        raise FileNotFoundError(
            f"GT not found: {GT_PATH}\n"
            "Run 2_generate_gt.py first."
        )
    gt = np.load(GT_PATH)           # (N_frames, 17, 3)
    n_frames = len(gt)
    print(f"GT loaded: {gt.shape}")

    # ── Process Person CSI ───────────────────────────────────
    print("\n[RX1] Person CSI")
    ts1_p, amp1_p = process_csi(PERSON_CSI[0])
    print("\n[RX2] Person CSI")
    ts2_p, amp2_p = process_csi(PERSON_CSI[1])

    # ── Process Empty CSI ────────────────────────────────────
    print("\n[RX1] Empty CSI")
    ts1_e, amp1_e = process_csi(EMPTY_CSI[0])
    print("\n[RX2] Empty CSI")
    ts2_e, amp2_e = process_csi(EMPTY_CSI[1])

    # ── Background Subtraction ───────────────────────────────
    print("\nBackground subtraction...")
    diff1 = subtract_empty(amp1_p, amp1_e)
    diff2 = subtract_empty(amp2_p, amp2_e)

    # Use timestamps from person session
    min_len1 = min(len(ts1_p), len(diff1))
    min_len2 = min(len(ts2_p), len(diff2))
    ts1_p = ts1_p[:min_len1]; diff1 = diff1[:min_len1]
    ts2_p = ts2_p[:min_len2]; diff2 = diff2[:min_len2]

    # ── Sync to Video ────────────────────────────────────────
    print(f"\nSyncing CSI to {n_frames} video frames...")
    rx1_synced = sync_csi_to_video(ts1_p, diff1, VIDEO_FPS, n_frames)
    rx2_synced = sync_csi_to_video(ts2_p, diff2, VIDEO_FPS, n_frames)
    print(f"RX1 synced: {rx1_synced.shape}")
    print(f"RX2 synced: {rx2_synced.shape}")

    # ── Build Windows ────────────────────────────────────────
    print(f"\nBuilding sliding windows (size={WINDOW_SIZE}, step={STEP_SIZE})...")
    X, Y = build_windows(rx1_synced, rx2_synced, gt)
    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")

    # ── Normalise ─────────────────────────────────────────────
    print("\nNormalising CSI...")
    X, Y = normalise(X, Y)

    # ── Save ──────────────────────────────────────────────────
    np.save(os.path.join(OUTPUT_DIR, "X.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, "Y.npy"), Y)

    print(f"\n✅ Saved:")
    print(f"   {OUTPUT_DIR}/X.npy  {X.shape}")
    print(f"   {OUTPUT_DIR}/Y.npy  {Y.shape}")
    print(f"\nReady for training! Run 4_train.py")
