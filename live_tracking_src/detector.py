import cv2
from ultralytics import YOLO


class person_detector:
    def __init__(self, model_path="../models/yolov8n-pose.pt", conf=0.25):
        self.conf = conf
        self.model = YOLO(model_path)

    def process_stream(self, source):
        """
        source: RTSP URL (str) or webcam index (int)
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ConnectionError(f"Unable to open stream: {source}")
        try:
            while True:
                success, frame = cap.read()
                if not success:
                    break
                results = self.model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    classes=[0],
                    conf=self.conf,
                    verbose=False)
                tracks = []
                boxes = results[0].boxes
                keypoints = results[0].keypoints

                if boxes is not None and boxes.id is not None:
                    track_ids = boxes.id.int().tolist()
                    boxes_xyxy = boxes.xyxy.int().tolist()
                    for i, (track_id, box_coords) in enumerate(zip(track_ids, boxes_xyxy)):
                        x1, y1, x2, y2 = box_coords
                        kp = keypoints[i].data.squeeze().tolist() if keypoints is not None else None
                        tracks.append((track_id, (x1, y1, x2, y2), kp))

                yield frame, tracks
        finally:
            cap.release()
