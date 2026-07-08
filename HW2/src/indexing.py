from __future__ import annotations

import random
from collections import defaultdict
from typing import Iterable

import cv2
import numpy as np
from tqdm import tqdm

from models import ObjectRecord, RowRecord


def normalize_bbox(bbox:Iterable[float],w: int, h: int) -> tuple[float,float,float,float]:
    x1,y1,x2,y2 = (float(v) for v in bbox)
    x1, y1 = max(0.0, min(x1, w - 1)), max(0.0, min(y1, h - 1))
    x2 = max(x1 + 1.0, min(x2, float(w)))
    y2 = max(y1 + 1.0, min(y2, float(h)))
    return x1,y1,x2,y2


def bbox_area(b) -> float:
    x1, y1, x2, y2 = b
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)

def intersect_area(a,b) -> float:
    x1,y1 = max(a[0],b[0]),max(a[1],b[1])
    x2,y2 = min(a[2],b[2]),min(a[3],b[3])
    return max(0.0,x2-x1) * max(0.0,y2-y1)


def coverage_ratio(inner,outer) -> float:
    denom = bbox_area(outer)
    return intersect_area(inner,outer) / denom if denom > 0 else 0.0


def image_to_gray(image) -> np.ndarray:
    return cv2.cvtColor(np.asarray(image.convert("RGB")),cv2.COLOR_RGB2GRAY)


def crop_square(image: np.ndarray, bbox, scale: float):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = min(max(max(bw, bh) * scale, max(bw, bh) + 2.0), float(min(w, h)))
    left = max(0.0, min(cx - side / 2, w - side))
    top = max(0.0, min(cy - side / 2, h - side))
    li, ti = int(round(left)), int(round(top))
    ri = max(li + 1, min(int(round(left + side)), w))
    bi = max(ti + 1, min(int(round(top + side)), h))
    crop = image[ti:bi, li:ri]
    return crop, (li, ti, ri, bi), (x1 - li, y1 - ti, x2 - li, y2 - ti)


def resize_with_bbox(gray_crop: np.ndarray,bbox,size: int):
    ch,cw = gray_crop.shape[:2]
    resized = cv2.resize(gray_crop,(size,size),interpolation=cv2.INTER_AREA)
    sx,sy = size / cw, size / ch
    x1, y1, x2, y2 = bbox
    rb = [int(round(x1 * sx)), int(round(y1 * sy)), int(round(x2 * sx)), int(round(y2 * sy))]
    rb[0] = max(0, min(rb[0], size - 1))
    rb[1] = max(0, min(rb[1], size - 1))
    rb[2] = max(rb[0] + 1, min(rb[2], size))
    rb[3] = max(rb[1] + 1, min(rb[3], size))
    return resized, rb


def build_indices(dataset_dict,min_object_area_ratio: float,min_bbox_size: int):
    category_index = {"train": defaultdict(list),"val":defaultdict(list)}
    row_index = {"train": [], "val" : []}


    for split_name in ( "train","val"):
        split = dataset_dict[split_name]
        for row_idx,row in enumerate(tqdm(split,desc = f"Indexing{split_name}" , leave=False)):
            w,h,image_id = int(row["width"]),int(row["height"]),int(row["image_id"])
            categories,selected_boxes = set(),[]
            for category_id, bbox in zip(row["objects"]["category"], row["objects"]["bbox"]):
                category_id = int(category_id)
                categories.add(category_id)
                nb = normalize_bbox(bbox,w,h)
                selected_boxes.append(nb)
                bw,bh = nb[2] - nb[0], nb[3]-nb[1]
                area_ratio = (bw*bh) / float(w*h)

                if area_ratio < min_object_area_ratio or min(bw,bh) < min_bbox_size:
                    continue

                category_index[split_name][category_id].append(
                    ObjectRecord(split_name,row_idx,image_id,category_id,nb,area_ratio,bw,bh)
                )
            row_index[split_name].append(
                RowRecord(split_name,row_idx,image_id,w,h,selected_boxes,categories)
            )
    return category_index, row_index



def choose_categories(
    category_index,
    category_names,
    num_classes,
    templates_per_class,
    positives_per_class,
    explicit_ids,
):
    
    if explicit_ids:
        selected = list(explicit_ids)
        num_classes = len(selected)
    else:
        candidate_ids = set(category_index["train"]) & set(category_index["val"])
        rows = []
        for cid in candidate_ids:
            tc,vc = len(category_index["train"][cid]),len(category_index["val"][cid])
            if tc < templates_per_class + positives_per_class + 20 or vc < positives_per_class:
                continue

            mean_ratio = float(np.mean([r.area_ratio for r in category_index["train"][cid][:200]]))
            rows.append((cid, min(tc, vc), mean_ratio))
        rows.sort(key=lambda r: (r[1], r[2]), reverse=True)
        selected = [r[0] for r in rows[:num_classes]]

    

    if len(selected) < num_classes:
        raise ValueError(
            f"Yeterli sınıf seçilemedi. İstenen: {num_classes}, bulunan: {len(selected)}. "
            "Eşik değerlerini gevşetin veya sınıf sayısını düşürün."
        )
    return [
        {
            "id": int(cid),
            "name": category_names.get(int(cid), f"category_{int(cid)}"),
            "train_candidates": len(category_index["train"].get(int(cid), [])),
            "test_candidates": len(category_index["val"].get(int(cid), [])),
        
        }

        for cid in selected
    
    ]


def select_unique_objects(candidates:list[ObjectRecord],count: int,exclude_image_ids: set[int],seed:int):
    rng = random.Random(seed)
    ranked = sorted(
    candidates,
    key=lambda r: (r.area_ratio, r.bbox_width * r.bbox_height),
    reverse=True
)
    head = ranked[: max(count*5,50)]
    rng.shuffle(head)
    chosen,used = [],set(exclude_image_ids)
    for item in head + ranked:
        if item.image_id in used:
            continue
        chosen.append(item)
        used.add(item.image_id)
        if len(chosen)==count:
            break
    if len(chosen) < count:
          raise ValueError(f"Yeterli benzersiz görüntü seçilemedi. İstenen {count}, bulunan {len(chosen)}")
    return chosen
    
    

                                            

             