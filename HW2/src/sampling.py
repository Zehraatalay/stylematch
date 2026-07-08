from __future__ import annotations
import random
from pathlib import Path
import cv2
import numpy as np 
from stylematch.image_io import write_image 
from models import RowRecord 
from indexing import image_to_gray, crop_square, resize_with_bbox, coverage_ratio 

def build_positive_sample(dataset_row, target_bbox, image_size : int, crop_scale : float) : 
    gray = image_to_gray(dataset_row["image"])
    crop, _, transformed_bbox = crop_square(gray, target_bbox, crop_scale) 
    return resize_with_bbox(crop, transformed_bbox, image_size)

def choose_negative_crop(gray : np.ndarray, forbidden_boxes, seed : int) -> np.ndarray :
    rng = random.Random(seed)
    h, w = gray.shape[:2]
    min_side = min(w, h)
    if min_side <= 8 : 
        return cv2.resize(gray, (256, 256), interpolation = cv2.INTER_AREA)
    sizes = [s for s in (min_side, int(min_side * 0.8), int(min_side * 0.6)) if s > 32]
    for _ in range(50) : 
        side = rng.choice(sizes)
        left, top = rng.randint(0, max(0, w - side)), rng.randint(0, max(0, h - side))
        box = (float(left), float(top), float(left + side), float(top + side))
        if not forbidden_boxes or max((coverage_ratio(box, b) for b in forbidden_boxes), default=0.0) < 0.05 :
            crop = gray[top : top + side, left : left + side]
            return cv2.resize(crop, (256, 256), interpolation=cv2.INTER_AREA)
        
    side = min_side
    left, top = max(0, (w - side) // 2), max(0, (h - side) // 2)
    crop = gray[top : top + side, left : left + side] 
    return cv2.resize(crop, (256, 256), interpolation=cv2.INTER_AREA)

def build_negative_samples(dataset_split, row_records : list[RowRecord], selected_categories : set[int], exclude_image_ids : set[int], count : int, seed : int, split_name : str, output_dir : Path) : 
    rng = random.Random(seed) 
    candidates = [r for r in row_records if r.image_id not in exclude_image_ids and r.categories.isdisjoint(selected_categories)]
    if len(candidates) < count : 
        candidates = [r for r in row_records if r.image_id not in exclude_image_ids]
    rng.shuffle(candidates)
    manifest, used_images = [], set() 
    for idx, row in enumerate(candidates) : 
        if len(manifest) == count : 
            break
        if row.image_id in used_images : 
            continue
        gray = image_to_gray(dataset_split[row.row_index]["image"])
        crop = choose_negative_crop(gray, row.selected_boxes, seed + idx) 
        sample_id = f"{split_name}_bg_{len(manifest):04d}"
        image_path = output_dir / "background" /f"{sample_id}.png"
        write_image(image_path, crop)
        manifest.append({"sample_id" : sample_id, "image_path" : str(image_path.resolve()), "label_id" : -1, "label_name" : "background", "true_bbox" : None, "source_image_id" : row.image_id, "category_id" : -1, "split" : split_name, "kind" : "background", })
        used_images.add(row.image_id)
    if len(manifest) < count : 
        raise ValueError()
    return manifest

def make_positive(row, bbox, image_size, crop_scale, out_path : Path, class_index, label_name, category_id, source_image_id, split : str | None = None) -> dict : 
    image_gray, resized_bbox = build_positive_sample(row, bbox, image_size, crop_scale)
    write_image(out_path, image_gray)
    entry = {
        "sample_id" : out_path.stem, "image_path" : str(out_path.resolve()), "label_id" : class_index, "label_name" : label_name, "true_bbox" : resized_bbox, "source_image_id" : source_image_id, "category_id" : category_id, "kind" : "template" if split is None else "positive", 
    }
    if split : 
        entry["split"] = split
    return entry 