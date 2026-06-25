import os 
import cv2
from datasets import load_dataset
from preprocessing import preprocess_image, crop_bbox_from_pil
DRESS_ID = 10 
TEMPLATE_COUNT = 10 

OUTPUT_DIR = "data/processed/train/templates"
CACHE_DIR = "data/raw/fashionpedia_cache"

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

def main() : 
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dataset = load_dataset("detection-datasets/fashionpedia", split="train", cache_dir = CACHE_DIR)
    saved = 0 
    for example in dataset : 
        dress_bboxes = find_dress_bboxes(example)
        if len(dress_bboxes) == 0 :
            continue
        
        largest_bbox = max(dress_bboxes, key=bbox_area)
        cropped = crop_bbox_from_pil(example["image"], largest_bbox)
        template = preprocess_image(cropped, 256)
        output_path = f"{OUTPUT_DIR}/template_{saved:02d}.png"
        success = cv2.imwrite(output_path, template)
        print("Saved :", output_path, "success", success)
        saved += 1 
        if saved == TEMPLATE_COUNT :
            break
    
if __name__ == "__main__" :
    main()
