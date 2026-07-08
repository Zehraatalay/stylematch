from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path 

SCALES =[256, 128, 64, 32]

@dataclass(frozen=True) 
class ObjectRecord :
    split_name : str 
    row_index : int 
    image_id : int 
    category_id : int 
    bbox : tuple[float, float, float, float]
    area_ratio : float
    bbox_height : float
    bbox_width : float


@dataclass
class RowRecord :
    split_name : str
    row_index : int 
    image_id : int 
    width : int 
    height : int 
    selected_boxes : list[tuple[float, float, float, float]]
    categories : set[int]
