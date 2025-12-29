import cv2
import os
from retinaface import RetinaFace

def detect_videos(video_dir, min_detection_confidence=0.99, frame_skip=1, save_dir=None):
    if save_dir is None:
        save_dir = os.path.join(video_dir, "crops_retinaface")
    os.makedirs(save_dir, exist_ok=True)

    for file_name in os.listdir(video_dir):
        if not file_name.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            continue

        full_path = os.path.join(video_dir, file_name)
        cap = cv2.VideoCapture(full_path)
        if not cap.isOpened():
            print(f"[Error] Cannot open video: {file_name}")
            continue

        base = os.path.splitext(file_name)[0]
        print(f"Processing video: {file_name}...")
        frame_count = 0
        saved_count = 0

        try:
            while True:
                success, frame = cap.read()
                if not success:
                    break

                frame_count += 1
                if frame_skip > 1 and (frame_count % frame_skip != 0):
                    continue
                
                # print(f"Frame: {frame_count}")
                inp = frame
                resp = RetinaFace.detect_faces(inp)
                if not isinstance(resp, dict) or len(resp) == 0: continue
                h, w = frame.shape[:2]

                for idx, key in enumerate(sorted(resp.keys())):
                    face_info = resp[key]
                    if 'facial_area' not in face_info: continue

                    x1, y1, x2, y2 = face_info['facial_area']
                    score = face_info.get("score", 0.0) or 0.0
                    if score < min_detection_confidence:
                        continue

                    x1, y1, x2, y2 = max(0, min(int(x1), w - 1)), max(0, min(int(y1), h - 1)), max(0, min(int(x2), w)), max(0, min(int(y2), h))
                    if x2 <= x1 or y2 <= y1:
                        continue

                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0: continue
                    
                    save_name = f"{base}_frame{frame_count}_face{idx}.png"
                    img_saved_path = os.path.join(save_dir, base)
                    os.makedirs(img_saved_path, exist_ok=True)

                    save_path = os.path.join(img_saved_path, save_name)
                    ok = cv2.imwrite(save_path, crop)
                    if not ok:
                        print(f"[Error] cv2.imwrite failed: {save_path}")
                    else:
                        saved_count += 1

        finally:
            cap.release()

        print(f"Done: {file_name}, Saved: {saved_count}")
        
if __name__ == "__main__":
    if os.path.isdir("video"):
        detect_videos("video", min_detection_confidence=0.5, frame_skip=15)
