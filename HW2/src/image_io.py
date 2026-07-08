from __future__ import annotations
from pathlib import Path 
import cv2 
import numpy as np 

def read_gray_image(path : str | Path) -> np.ndarray : 
    path = Path(path)
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0 : 
        raise FileNotFoundError()
    image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if image is None : 
        raise FileNotFoundError()
    return image 

def write_image(path : str | Path, image : np.ndarray) -> None :
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower() or ".png" 
    ext = suffix if suffix.startswith(".") else f".{suffix}"
    ok, encoded = cv2.imencode(ext, image)
    if not ok : 
        raise IOError()
    encoded.tofile(str(path))

    