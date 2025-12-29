import mediapipe as mp
import cv2
import os

def face_detection(image_path):
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    face_detection = mp_face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.5
    )
    
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
                    x, y, w, h = int(bbox.xmin * image.shape[1]), int(bbox.ymin * image.shape[0]), int(bbox.width * image.shape[1]), int(bbox.height * image.shape[0])
                    cropped_image = image[y:y+h, x:x+w]
                    file_base = os.path.splitext(file_name)[0]
                    save_name = f"{file_base}_cropped_{i}.png"
                    save_path = os.path.join(image_path, save_name)
                    cv2.imwrite(save_path, cropped_image)
                    print(f"Saved {save_name}")
            else:
                print(f"No faces found in {file_name}")
                
if __name__=="__main__":
    face_detection("img")       # folder name: img
