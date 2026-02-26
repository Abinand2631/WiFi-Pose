"""
STEP 2 — Ground Truth Generation (RTMPose-x)
=============================================
Runs RTMPose-x on the recorded person video to generate
17 COCO keypoints per frame as ground truth for training.

Run ONCE offline — takes ~2-4 hours on RTX 2050 for 180s video.
Output: data/gt/skeleton_gt.npy  shape: (N_frames, 17, 3)
        columns: [x_norm, y_norm, confidence]
"""

import cv2
import numpy as np
import os
import sys
import time
from tqdm import tqdm

# ===================== CONFIGURATION =====================
VIDEO_PATH  = os.path.join("data", "video_person.avi")
OUTPUT_DIR  = os.path.join("data", "gt")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "skeleton_gt.npy")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===================== LOAD RTMPOSE =====================
def load_rtmpose():
    """Load RTMPose-x via MMPose inference API."""
    try:
        from mmpose.apis import init_model, inference_topdown
        from mmpose.utils import adapt_mmdet_pipeline
        import mmdet
        from mmdet.apis import init_detector, inference_detector
        print("✅ MMPose + MMDet loaded")
        return "mmpose", None
    except (ImportError, ValueError) as e:
        print(f"[WARNING] MMPose not available: {e}")
        print("Falling back to MediaPipe...")
        return "mediapipe", None

def init_rtmpose():
    """Initialise RTMPose-x detector + pose model."""
    from mmpose.apis import init_model
    from mmdet.apis import init_detector

    # RTMDet-nano for person detection
    det_config = "https://raw.githubusercontent.com/open-mmlab/mmdetection/main/configs/rtmdet/rtmdet_nano_320-8xb32_coco-person.py"
    det_ckpt   = "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_nano_8xb32-300e_coco/rtmdet_nano_8xb32-300e_coco_20220615_230143-f477c511.pth"

    # RTMPose-x for pose estimation (highest accuracy)
    pose_config = "https://raw.githubusercontent.com/open-mmlab/mmpose/main/configs/body_2d_keypoint/rtmpose/coco/rtmpose-x_8xb256-420e_coco-256x192.py"
    pose_ckpt   = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-x_simcc-coco_pt-body7_420e-256x192-e2462712_20230224.pth"

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading RTMDet detector...")
    det_model  = init_detector(det_config, det_ckpt, device=device)

    print("Loading RTMPose-x model...")
    pose_model = init_model(pose_config, pose_ckpt, device=device)

    return det_model, pose_model

def run_mmpose(video_path, output_file):
    """Run RTMPose-x on video, save keypoints."""
    from mmpose.apis import inference_topdown
    from mmpose.utils import adapt_mmdet_pipeline
    from mmdet.apis import inference_detector

    det_model, pose_model = init_rtmpose()
    adapt_mmdet_pipeline(det_model)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    print(f"Video: {total_frames} frames @ {fps:.1f}fps  ({width}x{height})")

    all_kps = []   # list of (17, 3) arrays

    for _ in tqdm(range(total_frames), desc="RTMPose-x inference"):
        ret, frame = cap.read()
        if not ret:
            break

        # Detect persons
        det_result  = inference_detector(det_model, frame)
        pred_inst   = det_result.pred_instances
        person_bboxes = pred_inst.bboxes[
            pred_inst.labels == 0].cpu().numpy()  # class 0 = person

        if len(person_bboxes) == 0:
            # No person detected → store zeros
            all_kps.append(np.zeros((17, 3), dtype=np.float32))
            continue

        # Use largest bbox (closest person)
        areas = (
            (person_bboxes[:, 2] - person_bboxes[:, 0]) *
            (person_bboxes[:, 3] - person_bboxes[:, 1])
        )
        best  = person_bboxes[np.argmax(areas)]

        # Run pose
        pose_results = inference_topdown(pose_model, frame, [best])
        if not pose_results:
            all_kps.append(np.zeros((17, 3), dtype=np.float32))
            continue

        kps = pose_results[0].pred_instances.keypoints[0]      # (17, 2)
        scr = pose_results[0].pred_instances.keypoint_scores[0] # (17,)

        # Normalise to [0, 1]
        kps_norm = kps.copy()
        kps_norm[:, 0] /= width
        kps_norm[:, 1] /= height
        kps_norm = np.clip(kps_norm, 0.0, 1.0)

        frame_kps = np.column_stack([kps_norm, scr])   # (17, 3)
        all_kps.append(frame_kps.astype(np.float32))

    cap.release()

    gt_array = np.array(all_kps, dtype=np.float32)   # (N, 17, 3)
    np.save(output_file, gt_array)
    print(f"\n✅ GT saved → {output_file}  shape: {gt_array.shape}")
    return gt_array

def _install_mediapipe():
    """Install mediapipe<=0.10.14 (last version with solutions.pose API).
    Falls back to --user install if access is denied."""
    import subprocess
    # Pin to 0.10.14 — last version with mp.solutions.pose support
    PKG = "mediapipe<=0.10.14"
    print(f"Installing {PKG} ...")
    ret = subprocess.call(
        [sys.executable, "-m", "pip", "install", PKG, "--quiet"]
    )
    if ret != 0:
        print("[WARNING] Normal install failed. Trying --user install...")
        ret = subprocess.call(
            [sys.executable, "-m", "pip", "install", PKG, "--user", "--quiet"]
        )
    if ret != 0:
        raise RuntimeError(
            "Could not install mediapipe automatically.\n"
            "Please run manually in your conda env:\n"
            "    pip install \"mediapipe<=0.10.14\"\n"
            "or with user flag:\n"
            "    pip install \"mediapipe<=0.10.14\" --user"
        )

def _check_mediapipe_version():
    """Warn if mediapipe >= 0.10.15 is installed (solutions API removed)."""
    try:
        import importlib.metadata as meta
        ver = meta.version("mediapipe")
        parts = [int(x) for x in ver.split(".")[:3]]
        if parts >= [0, 10, 15]:
            print(
                f"[WARNING] mediapipe {ver} does not support mp.solutions.pose.\n"
                "  Run: pip install \"mediapipe<=0.10.14\" --force-reinstall"
            )
            return False
    except Exception:
        pass
    return True

def run_mediapipe_fallback(video_path, output_file):
    """Fallback: use MediaPipe Pose (requires mediapipe<=0.10.14)."""
    try:
        import mediapipe as mp
        if not _check_mediapipe_version():
            raise ImportError("mediapipe version too new — solutions API removed")
    except ImportError:
        _install_mediapipe()
        try:
            import mediapipe as mp
            _check_mediapipe_version()
        except ImportError:
            raise RuntimeError(
                "mediapipe could not be imported even after install.\n"
                "Please close any programs locking cv2.pyd and run:\n"
                "    pip install \"mediapipe<=0.10.14\"\n"
                "Then re-run this script."
            )

    # MediaPipe landmark indices that map to COCO 17 keypoints
    MP_TO_COCO = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

    mp_pose = mp.solutions.pose
    pose    = mp_pose.Pose(
        model_complexity=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[MediaPipe fallback] {total_frames} frames")

    all_kps = []
    for _ in tqdm(range(total_frames), desc="MediaPipe inference"):
        ret, frame = cap.read()
        if not ret:
            break

        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks is None:
            all_kps.append(np.zeros((17, 3), dtype=np.float32))
            continue

        lms = results.pose_landmarks.landmark
        kps = np.array([[lms[i].x, lms[i].y, lms[i].visibility]
                        for i in MP_TO_COCO], dtype=np.float32)
        all_kps.append(kps)

    cap.release()
    pose.close()

    gt_array = np.array(all_kps, dtype=np.float32)
    np.save(output_file, gt_array)
    print(f"\n✅ GT saved → {output_file}  shape: {gt_array.shape}")
    return gt_array


# ===================== MAIN =====================

if __name__ == "__main__":
    print("=" * 55)
    print("WiFi-Pose — GT Generation")
    print("=" * 55)

    if not os.path.exists(VIDEO_PATH):
        raise FileNotFoundError(
            f"Video not found: {VIDEO_PATH}\n"
            "Run 1_capture.py first with SESSION_LABEL = 'Person'"
        )

    backend, _ = load_rtmpose()
    t0 = time.time()

    if backend == "mmpose":
        gt = run_mmpose(VIDEO_PATH, OUTPUT_FILE)
    else:
        gt = run_mediapipe_fallback(VIDEO_PATH, OUTPUT_FILE)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} minutes")
    print(f"GT shape : {gt.shape}  (frames, 17 joints, 3 values)")
    print(f"Valid frames (conf>0.5): {(gt[:, :, 2] > 0.5).all(axis=1).sum()} / {len(gt)}")
