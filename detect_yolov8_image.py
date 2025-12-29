from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections
from PIL import Image
import cv2
import os

def get_largest_face(boxes):
    max_area = 0
    max_box = None
    for box in boxes:
        area = (box[2] - box[0]) * (box[3] - box[1])
        if area > max_area:
            max_area = area
            max_box = box
    return max_box

def face_detection(model, image_path):
    for file_name in os.listdir(image_path):
        if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        
        full_path = os.path.join(image_path, file_name) 
        image = Image.open(full_path)
        output = model(image)
        results = Detections.from_ultralytics(output[0])
        boxes = results.xyxy
        if not len(boxes):
            print("NO FACES DETECTED")
            continue
        largest_box = get_largest_face(boxes)
        x1, y1, x2, y2 = largest_box
        x, y, w, h = int(x1), int(y1), int(x2-x1), int(y2-y1)
        cropped_image = image.crop((x, y, x+w, y+h))
        save_file = full_path.replace(os.path.splitext(full_path)[1], "_cropped.png")
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        cropped_image.save(save_file)
        print(f"Cropped image saved to {save_file}")
    
if __name__ == "__main__":
    image_path = "image"
    model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
    model = YOLO(model_path)
    face_detection(model, image_path)
    
"""
Requirement:
- torch
- torchvision
- ultralytics
- supervision
- huggingface_hub 

"""
# If encounter this error:RuntimeError: Couldn't load custom C++ ops. 
# This can happen if your PyTorch and torchvision versions are incompatible, 
# or if you had errors while compiling torchvision from source. 
# For further information on the compatible versions, 
# check https://github.com/pytorch/vision#installation for the compatibility matrix. 
# Please check your PyTorch version with torch.__version__ and your torchvision version with torchvision.__version__ and 
# verify if they are compatible, and if not please reinstall torchvision so that it matches your PyTorch install.