import threading
import numpy as np
from scipy.spatial.distance import cosine


class Matcher:
    def __init__(self, app_threshold=0.3, pose_threshold=0.1):
        self.app_threshold = app_threshold
        self.pose_threshold = pose_threshold
        self.gallery = {}
        self.lock = threading.Lock()

    def update_gallery(self, track_id: int, feat_vec: np.ndarray = None, pose_vec: np.ndarray = None):
        with self.lock:
            if track_id not in self.gallery:
                self.gallery[track_id] = {
                    "feat": feat_vec,
                    "pose": pose_vec,
                    "feat_count": 1 if feat_vec is not None else 0,
                    "pose_count": 1 if pose_vec is not None else 0
                }
                return

            entry = self.gallery[track_id]

            if feat_vec is not None:
                if entry["feat"] is None:
                    entry["feat"] = feat_vec
                    entry["feat_count"] = 1
                else:
                    n = entry["feat_count"]
                    entry["feat"] = (entry["feat"] * n + feat_vec) / (n + 1)
                    entry["feat_count"] = n + 1

            if pose_vec is not None:
                if entry["pose"] is None:
                    entry["pose"] = pose_vec
                    entry["pose_count"] = 1
                else:
                    min_len = min(len(pose_vec), len(entry["pose"]))
                    n = entry["pose_count"]
                    entry["pose"] = (entry["pose"][:min_len] * n + pose_vec[:min_len]) / (n + 1)
                    entry["pose_count"] = n + 1

    def _update_feat(self, track_id: int, feat_vec: np.ndarray):
        self.gallery[track_id]["feat"] = feat_vec
        self.gallery[track_id]["feat_count"] = 1

    def _pose_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        min_len = min(len(a), len(b))
        if min_len == 0:
            return float("inf")
        return np.linalg.norm(a[:min_len] - b[:min_len])

    def match(self, feat_vec: np.ndarray = None, pose_vec: np.ndarray = None) -> tuple[int, float, str] | None:
        with self.lock:
            if not self.gallery:
                return None

            if feat_vec is not None:
                best_id, best_score = None, float("inf")
                for track_id, entry in self.gallery.items():
                    if entry["feat"] is None:
                        continue
                    score = cosine(feat_vec, entry["feat"])
                    if score < best_score:
                        best_score = score
                        best_id = track_id
                if best_id is not None and best_score < self.app_threshold:
                    return best_id, best_score, "appearance"

            if pose_vec is not None:
                best_id, best_score = None, float("inf")
                for track_id, entry in self.gallery.items():
                    if entry["pose"] is None:
                        continue
                    score = self._pose_distance(pose_vec, entry["pose"])
                    if score < best_score:
                        best_score = score
                        best_id = track_id
                if best_id is not None and best_score < self.pose_threshold:
                    if feat_vec is not None:
                        self._update_feat(best_id, feat_vec)
                    return best_id, best_score, "pose"

        return None
