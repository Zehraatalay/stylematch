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

def ensure_dir(path : Path) -> Path : 
    path.mkdir(parents=True, exist_ok=True)
    return path

def save_json(path : Path, payload) -> None : 
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

def load_fashionpedia() : 
    from datasets import load_dataset 
    return load_dataset("detection-datasets/fashionpedia")

def get_category_names(dataset_dict) -> dict[int, str] : 
    try : 
        feature = dataset_dict["train"].features["objects"]["category"].feature 
        return {i : n for i,n in enumerate(feature.names)} if getattr(feature, "names", None) else {}
    except Exception : 
        return {}