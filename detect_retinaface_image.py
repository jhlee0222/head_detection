import cv2
import os
from retinaface import RetinaFace

def face_detection(folder_path, min_detection_confidence=0.5):
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(folder_path, file_name)
            resp = RetinaFace.detect_faces(path)

            if isinstance(resp, dict):
                print(f"Detected {len(resp)} face(s) in {file_name}")
                image = cv2.imread(path)

                for idx, key in enumerate(resp):
                    face_info = resp[key]
                    x1, y1, x2, y2 = face_info['facial_area']
                    score = face_info.get("score", None)
                    if score < min_detection_confidence:
                        continue
                    
                    print(f"score: {score}")
                    cropped_image = image[y1:y2, x1:x2]
                    if cropped_image.size != 0:
                        file_name_only = os.path.splitext(file_name)[0]
                        save_name = f"{file_name_only}_cropped_{idx}.png"
                        save_path = os.path.join(folder_path, save_name)
                        cv2.imwrite(save_path, cropped_image)
                        print(f"Saved {save_name}")
            else:
                print(f"No faces found in {file_name}")

if __name__ == "__main__":
    face_detection("image", min_detection_confidence=0.99)