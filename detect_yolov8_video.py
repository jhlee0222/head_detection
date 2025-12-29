from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections
from PIL import Image
import cv2
import os

def face_detection(model, video_path, frame_skip=1, save_dir=None, min_detection_confidence=0.5):
    if save_dir is None:
        save_dir = os.path.join(video_path, "crops")
    os.makedirs(save_dir, exist_ok=True)

    for file_name in os.listdir(video_path):
        if not file_name.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            continue

        full_path = os.path.join(video_path, file_name)
        cap = cv2.VideoCapture(full_path)
        if not cap.isOpened():
            print(f"[Error] Cannot open video: {full_path}")
            continue

        base = os.path.splitext(file_name)[0]
        per_video_save_dir = os.path.join(save_dir, base)
        os.makedirs(per_video_save_dir, exist_ok=True)

        print(f"Processing video: {file_name} -> {per_video_save_dir}")
        frame_count = 0
        saved_count = 0

        try:
            while True:
                success, frame_bgr = cap.read()
                if not success:
                    break

                frame_count += 1
                if frame_skip > 1 and (frame_count % frame_skip != 0):
                    continue

                h, w = frame_bgr.shape[:2]
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(frame_rgb)

                output = model(pil_frame, verbose=False)
                det = Detections.from_ultralytics(output[0])
                if len(det) == 0: continue

                det = det[det.confidence >= min_detection_confidence]
                boxes = det.xyxy
                if boxes is None or len(boxes) == 0: continue

                for i, (x1, y1, x2, y2) in enumerate(boxes):
                    x1, y1, x2, y2 = int(round(float(x1))), int(round(float(y1))), int(round(float(x2))), int(round(float(y2)))
                    x1, y1, x2, y2 = max(0, min(x1, w - 1)), max(0, min(y1, h - 1)), max(0, min(x2, w)), max(0, min(y2, h))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    crop = frame_bgr[y1:y2, x1:x2]
                    if crop.size == 0: continue
                    save_name = f"{base}_frame{frame_count}_face{i}.png"
                    save_path = os.path.join(per_video_save_dir, save_name)
                    ok = cv2.imwrite(save_path, crop)
                    if not ok:
                        print(f"[Error] cv2.imwrite failed: {save_path}")
                    else:
                        saved_count += 1
        finally:
            cap.release()

        print(f"Done: {file_name}, Saved: {saved_count}")


if __name__ == "__main__":
    video_path = "video"
    model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection",filename="model.pt")
    model = YOLO(model_path)
    face_detection(model, video_path, frame_skip=1, min_detection_confidence=0.99)
