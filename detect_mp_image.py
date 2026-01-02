import mediapipe as mp
import cv2
import os

def face_detection(image_path, min_confidence=0.5):
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    face_detection = mp_face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=min_confidence
    )
    cropped_dir = os.path.join(image_path, "cropped_img_mp")
    os.makedirs(cropped_dir, exist_ok=True)
    
    for file_name in os.listdir(image_path):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            full_path = os.path.join(image_path, file_name)
            image = cv2.imread(full_path)
            
            if image is None:
                print(f"[Error] Cannot read image: {file_name}")
                continue
                
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_image)

            if results.detections:
                for i, detection in enumerate(results.detections):
                    bbox = detection.location_data.relative_bounding_box
                    h, w = image.shape[:2]
                    x1, y1, x2, y2 = int(bbox.xmin * w), int(bbox.ymin * h), int((bbox.xmin + bbox.width) * w), int((bbox.ymin + bbox.height) * h)
                    x1, y1, x2, y2 = max(0, min(w, x1)), max(0, min(h, y1)), max(0, min(w, x2)), max(0, min(h, y2))
                    if x2 <= x1 or y2 <= y1:
                        print("[Warn] Invalid bbox")
                        continue
                    cropped_image = image[y1:y2, x1:x2]
                    if cropped_image is None or cropped_image.size == 0:
                        print("[Warn] Empty Error")
                        continue
                    file_base = os.path.splitext(file_name)[0]
                    save_name = f"{file_base}_cropped_{i}.png"

                    save_path = os.path.join(cropped_dir, save_name)

                    cv2.imwrite(save_path, cropped_image)
                    print(f"Saved {save_name}")
            else:
                print(f"No faces found in {file_name}")
                
if __name__=="__main__":
    face_detection("img", min_confidence=0.5)       # folder name: img
