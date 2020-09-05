import torchvision
import io

import torchvision.transforms as transforms

from fastapi import FastAPI
from fastmaskapi.utils import COCO_INSTANCE_CATEGORY_NAMES, ImageInput, base64_str_to_image

app = FastAPI()

mask_rcnn = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
mask_rcnn.eval()

@app.put("/predict")
def get_detections(image: ImageInput):
    
    img = base64_str_to_image(image.img_bytes)
    to_tensor_transform = transforms.ToTensor()
    img = to_tensor_transform(img)

    output = mask_rcnn([img])[0]
    predicted_classes = [COCO_INSTANCE_CATEGORY_NAMES[label] for label in list(output["labels"].numpy())]
    predicted_scores = list(output["scores"].detach().numpy())
    predicted_boxes = [[float(box[0]), float(box[1]), float(box[2]), float(box[3])] for box in list(output["boxes"].detach().numpy())]
    pred_t = [predicted_scores.index(x) for x in predicted_scores if x > 0.5][-1]
    
    predicted_boxes = predicted_boxes[:pred_t+1]
    predicted_class = predicted_classes[:pred_t+1]
    
    return {'boxes': predicted_boxes, 'classes' : predicted_class}

            