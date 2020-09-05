import base64
import cv2
import io
import random

from pydantic import BaseModel
from PIL import Image

FILE_TYPES = ["jpg", "png"]
IP_ADDRESS = "http://127.0.0.1:8000/predict"

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bfastapi moicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

class ImageInput(BaseModel):
    img_bytes: str

def base64_str_to_image(b64_str):
	b64_img_bytes = b64_str.encode('utf-8')
	b64_bytes = base64.b64decode(b64_img_bytes)
	bytes_object = io.BytesIO(b64_bytes)
	img = Image.open(bytes_object) 
	return img

def draw_bounding_box(img, boxes, pred_cls, rect_th=2, text_size=1, text_th=2):
    class_color_dict = {}
    
    #initialize some random colors for each class for better looking bounding boxes
    for cat in pred_cls:
        class_color_dict[cat] = [random.randint(0, 255) for _ in range(3)]
    
    for i in range(len(boxes)):
        print(boxes[i])
        cv2.rectangle(img, (int(boxes[i][0]), int(boxes[i][1])),
                      (int(boxes[i][2]),int(boxes[i][3])),
                      color=class_color_dict[pred_cls[i]], thickness=rect_th)
        cv2.putText(img,pred_cls[i], (int(boxes[i][0]), int(boxes[i][1])),  cv2.FONT_HERSHEY_SIMPLEX, text_size, class_color_dict[pred_cls[i]],thickness=text_th) # Write the prediction class
    
    return img

def bytes_to_PIL_image(img_bytes):
    return Image.open(io.BytesIO(img_bytes))