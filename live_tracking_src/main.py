import cv2
import argparse
import threading
import queue
from detector import person_detector
from features_extract import features_extract
from matcher import Matcher
from brightness import is_dark

# ─────────────────────────────────────────────
# STREAM SOURCES
# ─────────────────────────────────────────────
# RTSP:    "rtsp://username:password@192.168.1.10:554/stream"
# Webcam:  0 (built-in), 1 (USB cam ), ...
CAM1_SOURCE = 0
CAM2_SOURCE = 1
# ─────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true")
args = parser.parse_args()

matcher = Matcher(app_threshold=0.3, pose_threshold=0.1)
extractor = features_extract()

cam1_queue = queue.Queue(maxsize=2)
cam2_queue = queue.Queue(maxsize=2)

stop_event = threading.Event()


def cam1_worker():
    detector = person_detector()
    frame_count = 0
    for frame, tracks in detector.process_stream(CAM1_SOURCE):
        if stop_event.is_set():
            break

        frame_count += 1
        skip = frame_count % 2 != 0  # every other frame
        dark = is_dark(frame)

        for track_id, box, kp in tracks:
            feat_vec = None
            pose_vec = None

            if not dark and not skip:
                feat_vec = extractor.extract_features(frame, box)

            if kp is not None:
                pose_vec = extractor.extract_pose_features(kp)

            if feat_vec is not None or pose_vec is not None:
                matcher.update_gallery(track_id, feat_vec, pose_vec)

            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id}", (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        try:
            cam1_queue.put_nowait(frame)
        except queue.Full:
            pass

    cam1_queue.put(None)  # sentinel


def cam2_worker():
    detector = person_detector()
    frame_count = 0
    reid_votes = {}        # {cam2_track_id: {"id": int, "count": int}}
    CONFIRM_FRAMES = 3

    for frame, tracks in detector.process_stream(CAM2_SOURCE):
        if stop_event.is_set():
            break

        frame_count += 1
        skip = frame_count % 2 != 0  # every other frame
        dark = is_dark(frame)

        for track_id, box, kp in tracks:
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"ID: {track_id}"

            feat_vec = None
            pose_vec = None

            if not dark and not skip:
                feat_vec = extractor.extract_features(frame, box)

            if kp is not None:
                pose_vec = extractor.extract_pose_features(kp)

            result = matcher.match(feat_vec, pose_vec)
            if result is not None:
                matched_id, dist, method = result
                vote = reid_votes.get(track_id, {"id": None, "count": 0})
                if vote["id"] == matched_id:
                    vote["count"] += 1
                else:
                    vote = {"id": matched_id, "count": 1}
                reid_votes[track_id] = vote

                if vote["count"] >= CONFIRM_FRAMES:
                    label = f"ReID: {matched_id} ({method}) Dist: {dist:.2f}"

            cv2.putText(frame, label, (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        try:
            cam2_queue.put_nowait(frame)
        except queue.Full:
            pass

    cam2_queue.put(None)  # sentinel


t1 = threading.Thread(target=cam1_worker, daemon=True)
t2 = threading.Thread(target=cam2_worker, daemon=True)
t1.start()
t2.start()

frame1 = None
frame2 = None

while True:
    try:
        item = cam1_queue.get_nowait()
        if item is None:
            break
        frame1 = item
    except queue.Empty:
        pass

    try:
        item = cam2_queue.get_nowait()
        if item is None:
            break
        frame2 = item
    except queue.Empty:
        pass

    if not args.headless:
        if frame1 is not None:
            cv2.imshow("Camera 1", frame1)
        if frame2 is not None:
            cv2.imshow("Camera 2", frame2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break

stop_event.set()
t1.join(timeout=3)
t2.join(timeout=3)

if not args.headless:
    cv2.destroyAllWindows()
