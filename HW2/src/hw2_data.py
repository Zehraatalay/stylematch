from __future__ import annotations
import csv 
import json 
import random 
from collections import Counter 
from pathlib import Path 

import numpy as np 
from tqdm import tqdm 
from indexing  import normalize_bbox,image_to_gray
from models import ensure_dir,load_fashionpedia,get_category_names

from hw2_features import IMAGE_SIZE, WINDOW_COORDS, extract_window_feature, resize_gray_image

SPLIT_ORDER = ("train", "validation", "test")
 
def bbox_area(bbox) :
    x1, y1, x2, y2 = bbox
    return max(0.0, x2-x1) * max(0.0, y2-y1)

def iou(a, b) : 
    ax1, ay1, ax2, ay2 = a 
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1) 
    ix2, iy2 = min(ax2, bx2), min(ay2, by2) 
    inter = max(0.0, ix2-ix1)*max(0.0, iy2-iy1)
    union = bbox_area(a)+bbox_area(b)-inter 
    return 0.0 if inter <= 0.0 or union <= 0.0 else inter/union 

def resize_bbox(bbox, width, height, image_size) :
    sx, sy = image_size / width, image_size/height
    x1,y1,x2,y2 = [int(round(v)) for v in (bbox[0]*sx, bbox[1]*sy, bbox[2]*sx, bbox[3]*sy)]
    x1 = max(0, min(x1, image_size-1))
    y1 = max(0, min(y1, image_size-1))
    x2 = max(x1+1, min(x2, image_size))
    y2 = max(y1+1, min(y2, image_size))
    return [x1,y1,x2,y2]

def save_json(path: Path, payload) :
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    
def build_image_records(dataset_dict) : 
    category_names = get_category_names(dataset_dict)
    records = []

    for split_name in ("train", "val"):
        for row_index, row in enumerate(tqdm(dataset_dict[split_name], desc=f"HW2 indexing {split_name}", leave=False)) :
            width, height = int(row["width"]), int(row["height"])
            image_id = int(row["image_id"])
            objects, categories = [], set()
            for category_id, bbox in zip(row["objects"]["category"], row["objects"]["bbox"]) :
                category_id = int(category_id)
                norm_bbox = normalize_bbox(bbox, width, height)
                objects.append({"category_id" : category_id, "bbox" : norm_bbox, "area_ratio": bbox_area(norm_bbox)/float(width*height), })
                categories.add(category_id)
            records.append({"image_key" : f"{split_name}:{image_id}", "source_split" : split_name, "row_index" : row_index, "image_id" : image_id, "width" : width, "height" : height, "objects" : objects, "categories" : sorted(categories),})
    return records, category_names

def select_top_categories(records, category_names, num_classes) : 
    image_counts = Counter(
        category_id
        for record in records
        for category_id in set(record["categories"]))
    return [
        {
            "id": int(category_id),
            "name": category_names.get(int(category_id), f"category_{int(category_id)}"),
            "image_count": int(image_count),
        }
        for category_id, image_count in image_counts.most_common(num_classes) 
    ]

def assign_images_to_splits(records, selected_category_ids, train_target_per_class, val_target_per_class, test_target_per_class, seed) :
    targets = {"train" : {cid : train_target_per_class for cid in selected_category_ids}, "validation" : {cid : val_target_per_class for cid in selected_category_ids}, "test" : {cid : test_target_per_class for cid in selected_category_ids}, }
    rng = random.Random(seed)
    eligible = []
    for record in records : 
        selected = set(record["categories"]) & selected_category_ids
        if selected :
            enriched = dict(record)
            enriched["selected_categories"] = sorted(selected)
            enriched["selected_category_count"] = len(selected)
            eligible.append(enriched)

    rng.shuffle(eligible)
    eligible.sort(
    key=lambda r: (
        r["selected_category_count"],
        sum(obj["area_ratio"] for obj in r["objects"] if obj["category_id"] in selected_category_ids),
    ),
    reverse=True,
)
    assignments = {split : [] for split in SPLIT_ORDER}
    for record in eligible : 
        scores = []
        for split_name in SPLIT_ORDER : 
            categories = record["selected_categories"]
            hits = sum(targets[split_name][cid] > 0 for cid in categories)
            mass = sum(targets[split_name][cid] for cid in categories)
            if hits :
                scores.append((hits, mass, 1 if split_name == "train" else 0, 1 if split_name == "validation" else 0 , split_name))
        if not scores : 
            continue
        chosen_split = max(scores)[-1]
        assignments[chosen_split].append(record)
        for cid in record["selected_categories"] :
            if targets[chosen_split][cid] > 0 :
                targets[chosen_split][cid] -=1 
        if all(count == 0 for split_targets in targets.values() for count in split_targets.values()) :
            break
    missing = { split : {str(cid): count for cid, count in split_targets.items() if count > 0}
               for split, split_targets in targets.items()}
    if any(missing.values()) : 
        raise ValueError()
    return assignments

def save_feature_split(output_dir : Path, split_name : str, payload : dict) : 
    feature_dir = ensure_dir(output_dir / "features")
    manifest_dir = ensure_dir(output_dir / "manifests")
    np.savez_compressed(feature_dir / f"{split_name}_features.npz", features = payload["features"].astype(np.float16), labels = payload["labels"].astype(np.int16), image_indices = payload["image_indices"].astype(np.int32), window_indices = payload["window_indices"].astype(np.int16), window_boxes = payload["window_boxes"].astype(np.int16), )
    fieldnames = ["sample_id", "image_key", "source_split", "row_index", "image_id", "window_index", "window_box", "label_id", "label_name", "is_ambiguous", "matched_category_ids", "matched_ious", ]
    with (manifest_dir / f"{split_name}_windows.csv").open("w", encoding = "utf-8", newline = "") as handle : 
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(payload["rows"])

def build_window_samples(dataset_dict, image_records, class_id_to_label_id, label_order, split_name, image_size, iou_threshold) : 
    rows, features, labels, image_indices, window_indices, window_boxes = [], [], [], [], [], []
    for image_index, record in enumerate(tqdm(image_records, desc=f"HW2 windows {split_name}", leave=False)) : 
        row = dataset_dict[record["source_split"]][record["row_index"]]
        gray = resize_gray_image(image_to_gray(row["image"]), image_size = image_size)
        resized_objects = [
            {
                "category_id": obj["category_id"],
                "label_id": class_id_to_label_id[obj["category_id"]],
                "label_name": label_order[class_id_to_label_id[obj["category_id"]]],
                "bbox": resize_bbox(obj["bbox"], record["width"], record["height"], image_size),
            }
            for obj in record["objects"]
            if obj["category_id"] in class_id_to_label_id
        ]
        record["scaled_objects"] = resized_objects
        for window_index, window_box in enumerate(WINDOW_COORDS) : 
            matched = [
                (obj, score)
                for obj in resized_objects
                if (score := iou(tuple(map(float, window_box)), tuple(map(float, obj["bbox"])))) >= iou_threshold   
                ]
            if split_name in {"train", "validation"} and len(matched) >= 2:
                continue
            is_ambiguous = len(matched) >= 2 
            if not matched : 
                label_id = len(label_order) -1 
                label_name = label_order[label_id]
            elif is_ambiguous : 
                label_id = -2
                label_name = "ambiguous"
            else :
                best_obj = max(matched, key=lambda item : item[1])[0]
                label_id = int(best_obj["label_id"])
                label_name = str(best_obj["label_name"])

            features.append(extract_window_feature(gray, window_box))
            labels.append(label_id)
            image_indices.append(image_index)
            window_indices.append(window_index)
            window_boxes.append(window_box)
            rows.append(
                {
                    "sample_id": f"{split_name}_{record['image_id']}_{window_index}",
                    "image_key" : record["image_key"], 
                    "source_split" : record["source_split"],
                    "row_index" : record["row_index"],
                    "image_id" : record["image_id"],
                    "window_index" : window_index,
                    "window_box" : json.dumps(list(window_box)),
                    "label_id" : label_id, 
                    "label_name" : label_name, 
                    "is_ambiguous" : int(is_ambiguous),
                    "matched_category_ids" : json.dumps([int(item[0]["category_id"]) for item in matched]),
                    "matched_ious" : json.dumps([round(float(item[1]), 6) for item in matched]),
                                    })
            
    return {
    "features": np.asarray(features, dtype=np.float32),
    "labels": np.asarray(labels, dtype=np.int16),
    "image_indices": np.asarray(image_indices, dtype=np.int32),
    "window_indices": np.asarray(window_indices, dtype=np.int16),
    "window_boxes": np.asarray(window_boxes, dtype=np.int16),
    "rows": rows,
    "images": image_records,
}

def prepare_hw2_fashionpedia_data(output_dir:Path, num_classes:int, train_target_per_class : int, val_target_per_class : int, test_target_per_class : int, image_size : int, seed : int, iou_threshold : float,) : 
    dataset_dict = load_fashionpedia()
    records, category_names = build_image_records(dataset_dict)
    selected_classes = select_top_categories(records, category_names, num_classes)
    selected_category_ids = {int(item["id"]) for item in selected_classes}
    label_order = [item["name"] for item in selected_classes] + ["background"]
    class_id_to_label_id = {
        int(item["id"]) : index
        for index, item in enumerate(selected_classes)
            }
    
    metadata = {
        "dataset_name" : "detection-datasets/fashionpedia", "image_size" : image_size, "window_size" : 128, "window_stride" : 64, "num_windows_per_image" : 9, "hog_feature_size" : 7560, "iou_threshold" : iou_threshold, "seed" : seed, "selected_classes" : selected_classes, "label_order" : label_order, "train_target_per_class" : train_target_per_class, "validation_target_per_class" : val_target_per_class, "test_target_per_class" : test_target_per_class,
    }
    manifest_dir = ensure_dir(output_dir /"manifests")
    save_json(manifest_dir / "metadata.json", metadata)

    assignments = assign_images_to_splits(records, selected_category_ids, train_target_per_class, val_target_per_class, test_target_per_class, seed,)
    for split_name, image_records in assignments.items() : 
        save_json(manifest_dir / f"{split_name}_images.json", image_records)

    split_payloads = {}
    for split_name in SPLIT_ORDER : 
        payload = build_window_samples(dataset_dict, assignments[split_name], class_id_to_label_id, label_order, split_name, image_size, iou_threshold,)
        split_payloads[split_name] = payload
        save_feature_split(output_dir, split_name, payload)

    metadata["counts"] = {
        split_name: {
            "images": len(payload["images"]),
            "windows_total": int(len(payload["labels"])),
            "windows_ambiguous": int(np.sum(payload["labels"] == -2)),
            "label_counts": {
                label_order[label_id]: int(count)
                for label_id, count in sorted(
                    Counter(payload["labels"][payload["labels"] >= 0].tolist()).items()
                )
            },
        }
        for split_name, payload in split_payloads.items()
    }

    save_json(manifest_dir / "metadata.json", metadata)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))