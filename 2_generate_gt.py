"""
STEP 2 — Ground Truth Generation (RTMPose-l)
=============================================
Generates 17 COCO keypoints per frame.

Output:
data/gt/skeleton_gt.npy
Shape: (N_frames, 17, 3)
Columns: [x_norm, y_norm, confidence]
"""

import cv2
import numpy as np
import os
import time
from tqdm import tqdm

# ===================== CONFIG =====================
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(BASE_DIR, "data", "video_person.avi")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "gt")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "skeleton_gt.npy")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =================================================
# MODEL INIT
# =================================================

def init_rtmpose():
    from mmpose.apis import init_model
    from mmdet.apis import init_detector
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    det_config = os.path.join(
        BASE_DIR,
        "mmdetection",
        "configs",
        "rtmdet",
        "rtmdet_tiny_8xb32-300e_coco.py"
    )

    pose_config = os.path.join(
        BASE_DIR,
        "mmpose",
        "configs",
        "body_2d_keypoint",
        "rtmpose",
        "coco",
        "rtmpose-l_8xb256-420e_coco-256x192.py"
    )

    det_ckpt = (
        "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/"
        "rtmdet_tiny_8xb32-300e_coco/"
        "rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth"
    )

    print("Loading RTMDet-tiny detector...")
    det_model = init_detector(det_config, det_ckpt, device=device)

    print("Loading RTMPose-l model...")
    pose_model = init_model(pose_config, None, device=device)

    return det_model, pose_model


# =================================================
# GT GENERATION
# =================================================

def run_mmpose(video_path, output_file):
    from mmdet.apis import inference_detector
    from mmpose.apis import inference_topdown
    from mmpose.utils import adapt_mmdet_pipeline

    det_model, pose_model = init_rtmpose()
    adapt_mmdet_pipeline(det_model.cfg)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video file.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {total_frames} frames")

    all_keypoints = []

    for _ in tqdm(range(total_frames), desc="RTMPose inference"):
        ret, frame = cap.read()
        if not ret:
            break

        # -------- DETECTION --------
        det_results = inference_detector(det_model, frame)
        pred = det_results.pred_instances

        person_bboxes = [
            bbox.cpu().numpy()
            for bbox, score, label in zip(pred.bboxes, pred.scores, pred.labels)
            if label == 0 and score > 0.3
        ]

        if len(person_bboxes) == 0:
            all_keypoints.append(np.zeros((17, 3), dtype=np.float32))
            continue

        # -------- POSE --------
        pose_results = inference_topdown(
            pose_model,
            frame,
            person_bboxes,
            bbox_format="xyxy"
        )

        if len(pose_results) > 0:
            inst = pose_results[0].pred_instances

            keypoints = inst.keypoints[0]
            scores = inst.keypoint_scores[0]

            # Safe conversion
            if hasattr(keypoints, "cpu"):
                keypoints = keypoints.cpu().numpy()
            if hasattr(scores, "cpu"):
                scores = scores.cpu().numpy()

            keypoints = keypoints.astype(np.float32)
            scores = scores.astype(np.float32)

            # Normalize coordinates
            keypoints[:, 0] /= width
            keypoints[:, 1] /= height

            # Clamp to valid range
            keypoints[:, 0] = np.clip(keypoints[:, 0], 0.0, 1.0)
            keypoints[:, 1] = np.clip(keypoints[:, 1], 0.0, 1.0)

            # Convert logits → probability
            scores = 1 / (1 + np.exp(-scores))

            # Combine (17,3)
            keypoints = np.concatenate(
                [keypoints, scores[:, None]],
                axis=1
            )

            all_keypoints.append(keypoints)
        else:
            all_keypoints.append(np.zeros((17, 3), dtype=np.float32))

    cap.release()

    gt_array = np.array(all_keypoints, dtype=np.float32)
    np.save(output_file, gt_array)

    print(f"\n✅ GT saved → {output_file}")
    print(f"Shape: {gt_array.shape}")

    return gt_array


# =================================================
# MAIN
# =================================================

if __name__ == "__main__":
    print("=" * 55)
    print("WiFi-Pose — GT Generation")
    print("=" * 55)

    if not os.path.exists(VIDEO_PATH):
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

    t0 = time.time()
    gt = run_mmpose(VIDEO_PATH, OUTPUT_FILE)
    elapsed = time.time() - t0

    print(f"\nDone in {elapsed/60:.1f} minutes")
    print(f"GT shape : {gt.shape}")
    print(f"Valid frames (conf>0.5): {(gt[:, :, 2] > 0.5).all(axis=1).sum()} / {len(gt)}")