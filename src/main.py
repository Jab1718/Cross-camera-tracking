import cv2
from detector import person_detector
from features_extract import features_extract
from matcher import Matcher
from brightness import is_dark

CAM1_PATH = "../data/raw_videos/lap.mp4" #test
CAM2_PATH = "../data/raw_videos/phone.mp4" #test

detector1 = person_detector()
detector2 = person_detector()
extractor = features_extract()
matcher = Matcher()

gen1 = detector1.process_video(CAM1_PATH)
gen2 = detector2.process_video(CAM2_PATH)

frame_count = 0

while True:
    frame_count += 1
    pair1 = next(gen1, None)
    pair2 = next(gen2, None)

    if pair1 is None and pair2 is None:
        break

    skip = frame_count % 3 != 0

#cam1
    if pair1 is not None:
        frame1, tracks1 = pair1
        dark1 = is_dark(frame1)
        for track_id, box, kp in tracks1:
            feat_vec = None
            pose_vec = None

            if not dark1 and not skip:
                feat_vec = extractor.extract_features(frame1, box)

            if kp is not None:
                pose_vec = extractor.extract_pose_features(kp)

            if feat_vec is not None or pose_vec is not None:
                matcher.update_gallery(track_id, feat_vec, pose_vec)

            x1, y1, x2, y2 = box
            cv2.rectangle(frame1, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame1, f"ID: {track_id}", (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#cam2
    if pair2 is not None:
        frame2, tracks2 = pair2
        dark2 = is_dark(frame2)
        for track_id, box, kp in tracks2:
            x1, y1, x2, y2 = box
            cv2.rectangle(frame2, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"ID: {track_id}"

            feat_vec = None
            pose_vec = None

            if not dark2:
                feat_vec = extractor.extract_features(frame2, box)
            print(f"dark2: {dark2}, kp is None: {kp is None}")
            if kp is not None:
                pose_vec = extractor.extract_pose_features(kp)
                print(f"pose_vec result: {pose_vec}")

            result = matcher.match(feat_vec, pose_vec)
            if result is not None:
                matched_id, dist, method = result
                label = f"ReID: {matched_id} ({method}) Dist: {dist:.2f}"

            cv2.putText(frame2, label, (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    if pair1 is not None:
        cv2.imshow("Camera 1", frame1)
    if pair2 is not None:
        cv2.imshow("Camera 2", frame2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()