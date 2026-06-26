import json 
import os 
import cv2 
from datasets import load_dataset
from preprocessing import preprocess_image, scale_bbox_xyxy

DRESS_ID = 10 
POSITIVE_COUNT = 200
NEGATIVE_COUNT = 600 

CACHE_DIR = "data/raw/fashionpedia_cache"
OUTPUT_IMAGE_DIR = "data/processed/val/images"
OUTPUT_METADATA_PATH = "data/processed/val/metadata.json"

def find_dress_bboxes(example) : 
    categories = example["objects"]["category"]
    bboxes = example["objects"]["bbox"]
    dress_bboxes = []

    for category, bbox in zip(categories, bboxes) : 
        if category == DRESS_ID :
            dress_bboxes.append(bbox)
    return dress_bboxes

def bbox_area(bbox) : 
    x1, y1, x2, y2 = bbox
    return (x2 - x1) * (y2 - y1)

def save_processed_image(example, filename) :
    image = preprocess_image(example["image"],256)
    output_path = os.path.join(OUTPUT_IMAGE_DIR, filename)
    success = cv2.imwrite(output_path, image)
    if not success : 
        raise RuntimeError()
    
def main() : 
    os.makedirs(OUTPUT_IMAGE_DIR, exist_ok = True)
    dataset = load_dataset("detection-datasets/fashionpedia", split="train", cache_dir=CACHE_DIR)
    metadata = []
    positive_saved = 0
    negative_saved = 0 
    for example in dataset : 
        dress_bboxes = find_dress_bboxes(example)
        has_dress = len(dress_bboxes) > 0 
        if has_dress and positive_saved < POSITIVE_COUNT : 
            largest_bbox = max(dress_bboxes, key = bbox_area)
            scaled_bbox = scale_bbox_xyxy(largest_bbox, example["width"], example["height"], 256)
            filename = f"pos_{positive_saved:04d}.png"
            save_processed_image(example, filename)
            metadata.append({"filename" : filename, "label" : 1, "bbox" : scaled_bbox})
            positive_saved += 1
        elif (not has_dress) and negative_saved < NEGATIVE_COUNT : 
            filename = f"neg_{negative_saved:04d}.png"
            save_processed_image(example, filename)

            metadata.append({
                "filename": filename,
                "label": 0,
                "bbox": None
            })
            negative_saved += 1
        if positive_saved >= POSITIVE_COUNT and negative_saved >= NEGATIVE_COUNT : 
            break
    with open(OUTPUT_METADATA_PATH, "w", encoding="utf-8") as f : 
        json.dump(metadata, f, indent=2)

if __name__ == "__main__" : 
    main()