import os
os.environ["TORCH_HOME"] = "../models"
import torch
import numpy as np
import torchreid
import cv2


class features_extract:
    def __init__(self, model_name="osnet_x1_0", device="cpu"):
        self.device = device
        self.extractor = torchreid.utils.FeatureExtractor(
            model_name=model_name,
            device=device)

    def extract_features(self, frames: np.ndarray, box: tuple) -> np.ndarray | None:
        x1, y1, x2, y2 = box
        crop = frames[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        embedding = self.extractor(crop)
        return embedding[0].cpu().numpy()

    def _dist(self, a: list, b: list) -> float:
        return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def extract_pose_features(self, kp: list) -> np.ndarray | None:
        CONF_THRESHOLD = 0.5
        MIN_WIDTH_PX = 10

        nose = kp[0]
        l_shoulder = kp[5]
        r_shoulder = kp[6]
        l_hip = kp[11]
        r_hip = kp[12]
        l_ankle = kp[15]
        r_ankle = kp[16]

        shoulder = l_shoulder if l_shoulder[2] > r_shoulder[2] else r_shoulder
        hip = l_hip if l_hip[2] > r_hip[2] else r_hip
        ankle = l_ankle if l_ankle[2] > r_ankle[2] else r_ankle

        core = [shoulder, hip]
        if any(k[2] < CONF_THRESHOLD for k in core):
            return None

        shoulder_w = self._dist(l_shoulder, r_shoulder)
        hip_w = self._dist(l_hip, r_hip)
        torso_len = self._dist(shoulder, hip)

        if shoulder_w < MIN_WIDTH_PX:
            return None

        ratios = []

        ratios.append(hip_w / shoulder_w)

        if ankle[2] >= CONF_THRESHOLD and torso_len > 0:
            leg_len = self._dist(hip, ankle)
            ratios.append(leg_len / torso_len)

        if nose[2] >= CONF_THRESHOLD and ankle[2] >= CONF_THRESHOLD:
            height = self._dist(nose, ankle)
            if height > 0:
                ratios.append(shoulder_w / height)

        return np.array(ratios, dtype=np.float32)
