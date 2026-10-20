from datasets import load_dataset

from preprocessing import preprocess_image, scale_bbox_xyxy, crop_bbox_from_pil

dataset = load_dataset("detection-datasets/fashionpedia", split = "train", cache_dir = "../data/raw/fashionpedia_cache")

example = dataset[0]

image = example["image"]
width = example["width"]
height = example["height"]
bbox = example["objects"]["bbox"][0]

processed = preprocess_image(image)
scaled_bbox = scale_bbox_xyxy(bbox, width, height)
cropped = crop_bbox_from_pil(image, bbox)
cropped_processed = preprocess_image(cropped)

print("original image size ", image.size)
print("processed image shape", processed.shape)
print("original bbox", bbox)
print("cropped image size", cropped.size)
print("cropped processed shape", cropped_processed.shape)