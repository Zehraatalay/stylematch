import json 
from pathlib import Path

ROOT = Path.home() / "Datasets" / "polyvore-outfits"
SOURCE_DIR = ROOT / "disjoint"
OUTPUT_DIR = ROOT / "strict_disjoint"

OUTPUT_DIR.mkdir(exist_ok=True)

def load_json(path) : 
    with open(path, "r") as f :
        return json.load(f)

def save_json(data, path) : 
    with open(path, "w") as f : 
        json.dump(data, f)

def outfit_item_ids(outfit) : 
    return {str(item["item_id"]) for item in outfit["items"]}

def all_item_ids(outfits) : 
    ids = set()
    for outfit in outfits : 
        ids.update(outfit_item_ids(outfit))
    return ids 

train = load_json(SOURCE_DIR / "train.json")
valid = load_json(SOURCE_DIR / "valid.json")
test = load_json(SOURCE_DIR / "test.json")

print("ORIGINAL")
print("train outfits : ", len(train))
print("valid outfits : ", len(valid))
print("test outfits : ", len(test))

test_ids = all_item_ids(test)

strict_valid = []
removed_valid = 0 
for outfit in valid : 
    ids = outfit_item_ids(outfit)
    if ids & test_ids : 
        removed_valid += 1
        continue
    strict_valid.append(outfit)

valid_ids = all_item_ids(strict_valid)

forbidden_train_ids = test_ids | valid_ids
strict_train = []
removed_train = 0 
for outfit in train : 
    ids = outfit_item_ids(outfit)
    if ids & forbidden_train_ids : 
        removed_train += 1
        continue
    strict_train.append(outfit)


train_ids = all_item_ids(strict_train)
valid_ids = all_item_ids(strict_valid)
test_ids = all_item_ids(test)

print("\nREMOVED")
print("train removed : ", removed_train)
print("valid removed : ", removed_valid)
print("\nSTRICT SPLITS")
print("train outfits : ", len(strict_train))
print("valid outfits : ", len(strict_valid))
print("test outfits : ", len(test))
print("\nUNIQUE ITEMS")
print("train : ", len(train_ids))
print("valid : ", len(valid_ids))
print("test : ", len(test_ids))
print("\nOVERLAP")
print("train-valid : ", len(train_ids & valid_ids))
print("train-test : ", len(train_ids & test_ids))
print("valid-test : ", len(valid_ids & test_ids) )
assert len(train_ids & valid_ids) == 0
assert len(train_ids & test_ids) == 0
assert len(valid_ids & test_ids) == 0 

save_json(strict_train, OUTPUT_DIR / "train.json")
save_json(strict_valid, OUTPUT_DIR / "valid.json")
save_json(test, OUTPUT_DIR / "test.json")

print("\nSaved to : ")
print(OUTPUT_DIR)