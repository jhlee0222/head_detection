import mediapipe as mp
import cv2
import os

def face_detection(video_path):
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    face_detection = mp_face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.99
    )
    
    for file_name in os.listdir(video_path):
        if file_name.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            full_path = os.path.join(video_path, file_name)
            cap = cv2.VideoCapture(full_path)
            if not cap.isOpened():
                print(f"[Error] Cannot open video: {file_name}")
                continue
            print(f"Processing video: {file_name}...")
            frame_count = 0
            frame_skip = 1  
            while True:
                success, image = cap.read()
                if not success: break
                frame_count += 1
                if frame_count % frame_skip != 0: continue
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_detection.process(rgb_image)

                if results.detections:
                    for i, detection in enumerate(results.detections):
                        bbox = detection.location_data.relative_bounding_box
                        x, y, w, h = int(bbox.xmin * image.shape[1]), int(bbox.ymin * image.shape[0]), int(bbox.width * image.shape[1]), int(bbox.height * image.shape[0])
                        x, y, w, h = max(0, x), max(0, y), min(image.shape[1] - x, w), min(image.shape[0] - y, h)
                        if w > 0 and h > 0:
                            cropped_image = image[y:y+h, x:x+w]
                            file_base = os.path.splitext(file_name)[0]
                            save_name = f"{file_base}_frame{frame_count}_face{i}.png"
                            save_path = os.path.join(video_path, save_name)
                            cv2.imwrite(save_path, cropped_image)
                            # print(f"Saved {save_name}")
            cap.release() 

if __name__=="__main__":
    face_detection("video") 