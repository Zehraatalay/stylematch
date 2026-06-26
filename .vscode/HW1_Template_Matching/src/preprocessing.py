import cv2
import numpy as np
from PIL import Image

def pil_to_rgb_array(image:Image.Image) -> np.ndarray:
    return np.array(image.convert("RGB"))

def to_grayscale(image_rgb:np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_rgb,cv2.COLOR_RGBA2GRAY)
def resize_image(image:np.ndarray,size:int = 256) -> np.ndarray:
    return cv2.resize(image,(size,size),interpolation=cv2.INTER_AREA)

def preprocess_image(image:Image.Image,size:int = 256) -> np.ndarray:
    image_rgb = pil_to_rgb_array(image)
    gray = to_grayscale(image_rgb)
    resized = resize_image(gray,size)
    return resized
def scale_bbox_xyxy(bbox,original_width,original_height,new_size=256):
    x1,y1,x2,y2 = bbox

    scale_x = new_size / original_width
    scale_y = new_size/ original_height

    nx1 = int(x1*scale_x)
    ny1 = int(y1*scale_y)
    nx2 = int(x2 *scale_x)
    ny2 = int(y2 * scale_y)

    nx1 = max(0,min(nx1,new_size - 1))
    ny1 = max(0, min(ny1,new_size - 1))
    nx2 = max(nx1 + 1,min(nx2,new_size))
    ny2 = max(ny1 + 1,min(ny2,new_size))

    return [nx1,ny1,nx2,ny2]


def crop_bbox_from_pil(image: Image.Image,bbox):
    x1,y1,x2,y2 = bbox
    return image.crop((x1,y1,x2,y2))



