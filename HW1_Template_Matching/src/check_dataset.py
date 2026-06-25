from datasets import load_dataset

dataset = load_dataset("detection-datasets/fashionpedia", split="train", cache_dir="data/raw/fashionpedia_cache")

print(dataset)
print(dataset[0].keys())

#deneme
