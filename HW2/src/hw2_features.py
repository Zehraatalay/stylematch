from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

IMAGE_SIZE = 256
WINDOW_SIZE = 128
WINDOW_STRIDE = 64
WINDOW_COORDS = [
    (x,y,x + WINDOW_SIZE,y + WINDOW_SIZE)
    for y in (0,WINDOW_STRIDE,WINDOW_STRIDE*2)
    for x in (0,WINDOW_STRIDE,WINDOW_STRIDE*2)

]


@lru_cache(maxsize=1)
def get_hog_descriptor() -> cv2.HOGDescriptor:
    return  cv2.HOGDescriptor(
        _winSize = (64,128),
        _blockSize= (16,16),
        _blockStride = (8,8),
        _nbins= 9,
    )



def resize_gray_image(gray:np.ndarray, image_size: int = IMAGE_SIZE) -> np.ndarray:
    if gray.shape[0]== image_size and gray.shape[1] == image_size:
        return gray
    return cv2.resize(gray,(image_size,image_size),interpolation=cv2.INTER_AREA)


def extract_window_feature(gray_image: np.ndarray, window_box: tuple[int, int, int, int]) -> np.ndarray:
    x1,y1,x2,y2 = window_box
    window = gray_image[y1:y2,x1:x2]
    if window.shape[:2] != (WINDOW_SIZE,WINDOW_SIZE):
        window = cv2.resize(window,(WINDOW_SIZE,WINDOW_SIZE),interpolation=cv2.INTER_AREA)
    

    left_half = window[:,:64]
    right_half = window[:,64:]
    hog = get_hog_descriptor()
    left_feature = hog.compute(left_half).reshape(-1)
    right_feature = hog.compute(right_half).reshape(-1)
    return np.concatenate([left_feature,right_feature]).astype(np.float32)


def extract_image_wimdow_features(gray_image:np.ndarray) -> list[np.ndarray]:
    return [extract_window_feature(gray_image,box) for box in WINDOW_COORDS]