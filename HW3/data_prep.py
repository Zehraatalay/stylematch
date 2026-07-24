import pickle
from collections import Counter

import cv2
import numpy as np
from datasets import load_dataset

IMG_SIZE = 128
N_CLASSES = 10
SEED = 42
TRAIN_MIN_IMAGES = 10_000
PER_CLASS_TRAIN = 1_000
PER_CLASS_VAL = 200
PER_CLASS_TEST = 200


def top_classes(ds, n=N_CLASSES):
    cnt = Counter()
    for cats in ds["objects"]["category"]:
        cnt.update(cats)
    return [c for c, _ in cnt.most_common(n)]


def index_images(ds, classes):
    classes_set = set(classes)
    img_classes = {}
    idx_by_class = {c: [] for c in classes}
    for i, cats in enumerate(ds["objects"]["category"]):
        present = classes_set.intersection(cats)
        if present:
            img_classes[i] = present
            for c in present:
                idx_by_class[c].append(i)
    return idx_by_class, img_classes


def sample_indices(
    idx_by_class,
    img_classes,
    classes,
    per_class_min,
    total_min=0,
    exclude=(),
    seed=SEED,
    initial_counts=None,
):
   
    rng = np.random.default_rng(seed)
    exclude = set(exclude)
    shuffled = {
        c: [int(i) for i in rng.permutation(idx_by_class[c]) if int(i) not in exclude]
        for c in classes
    }
    pos = {c: 0 for c in classes}
    selected = []
    selected_set = set()
    counts = Counter(initial_counts or {})

    def class_targets_met():
        return all(counts[c] >= per_class_min for c in classes)

    
    while not class_targets_met():
        progressed = False
        for c in classes:
            if counts[c] >= per_class_min:
                continue
            while pos[c] < len(shuffled[c]):
                cand = shuffled[c][pos[c]]
                pos[c] += 1
                if cand in selected_set:
                    continue
                selected_set.add(cand)
                selected.append(cand)
                for cc in img_classes.get(cand, ()):
                    counts[cc] += 1
                progressed = True
                break
        if not progressed:
            break

   
    if len(selected) < total_min:
        universe = [i for i in img_classes if i not in exclude and i not in selected_set]
        for cand in rng.permutation(universe):
            cand = int(cand)
            selected_set.add(cand)
            selected.append(cand)
            for cc in img_classes.get(cand, ()):
                counts[cc] += 1
            if len(selected) >= total_min:
                break

    return selected, counts


def resize_with_boxes(img, boxes):
    img = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    h, w = img.shape
    sx, sy = IMG_SIZE / w, IMG_SIZE / h
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    boxes = [(cls, [b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy]) for cls, b in boxes]
    return img, boxes


def point_in_box(x, y, box):
    x0, y0, x1, y1 = box
    return x0 <= x <= x1 and y0 <= y <= y1


def extract_labeled_features(img128, boxes, class_map):
    sift = cv2.SIFT_create()
    keypoints, desc = sift.detectAndCompute(img128, None)
    if desc is None:
        return np.empty((0, 128), dtype=np.float32), np.empty((0,), dtype=int)

    feats, labels = [], []
    for kp, descriptor in zip(keypoints, desc):
        x, y = kp.pt
        hits = [class_map[c] for c, box in boxes if point_in_box(x, y, box)]
        if len(hits) > 1:
            continue
        feats.append(descriptor)
        labels.append(hits[0] if hits else 0)

    if not feats:
        return np.empty((0, 128), dtype=np.float32), np.empty((0,), dtype=int)
    return np.asarray(feats, dtype=np.float32), np.asarray(labels, dtype=int)


def process_indices(ds, indices, classes, class_map, keep_images=False):
    subset = ds.select(indices)
    feature_parts, label_parts, raw = [], [], []

    for ex in subset:
        cats = ex["objects"]["category"]
        boxes = [(c, b) for c, b in zip(cats, ex["objects"]["bbox"]) if c in class_map]
        img128, boxes128 = resize_with_boxes(ex["image"], boxes)
        feats, labels = extract_labeled_features(img128, boxes128, class_map)
        if len(feats):
            feature_parts.append(feats)
            label_parts.append(labels)
        if keep_images:
            raw.append((img128, [(class_map[c], b) for c, b in boxes128]))

    X = np.vstack(feature_parts) if feature_parts else np.empty((0, 128), dtype=np.float32)
    y = np.concatenate(label_parts) if label_parts else np.empty((0,), dtype=int)
    return X, y, raw


def merge_feature_parts(parts):
    arrays = [a for a in parts if len(a)]
    return (
        np.vstack(arrays)
        if arrays
        else np.empty((0, 128), dtype=np.float32)
    )
def merge_label_parts(parts):
    arrays = [a for a in parts if len(a)]
    return np.concatenate(arrays) if arrays else np.empty((0,), dtype=int)

if __name__ == "__main__":
    train_ds = load_dataset("detection-datasets/fashionpedia", split = "train")
    official_test_ds = load_dataset("detection-datasets/fashionpedia", split="val")


    classes = top_classes(train_ds)
    class_map = { c: i+1 for i, c in enumerate(classes)}
    train_by_class, train_img_classes = index_images(train_ds, classes)
    train_idx, train_counts = sample_indices(
        train_by_class, train_img_classes, classes,
        per_class_min=PER_CLASS_TRAIN,
        total_min=TRAIN_MIN_IMAGES,
        seed=SEED,
    )


    val_idx,val_counts =sample_indices(
        train_by_class,train_img_classes,classes,
        per_class_min=PER_CLASS_VAL,
        exclude=train_idx,
        seed=SEED + 1,
    )


    test_by_class,test_img_classes = index_images(official_test_ds,classes)
    official_test_idx, official_test_counts = sample_indices(
        test_by_class, test_img_classes, classes,
        per_class_min=PER_CLASS_TEST,
        seed=SEED + 2,
    )


    used_train = set(train_idx) | set(val_idx)
    extra_test_idx, final_test_counts = sample_indices(
    train_by_class,
    train_img_classes,
    classes,
    per_class_min=PER_CLASS_TEST,
    exclude=used_train,
    seed=SEED + 3,
    initial_counts=official_test_counts,
)


    Xtr, ytr, _ = process_indices(train_ds, train_idx, classes, class_map, keep_images=False)
    Xval, yval, val_raw = process_indices(train_ds, val_idx, classes, class_map, keep_images=True)
    Xte_a, yte_a, test_raw_a = process_indices(
        official_test_ds, official_test_idx, classes, class_map, keep_images=True
    )

    Xte_b,yte_b,test_raw_b = process_indices(
        train_ds,extra_test_idx,classes,class_map,keep_images=True
    )

    Xte = merge_feature_parts([Xte_a,Xte_b])
    yte = merge_label_parts([yte_a, yte_b])
    test_raw = test_raw_a + test_raw_b

    np.savez("features.npz", Xtr = Xtr,ytr=ytr,Xval = Xval,yval=yval,Xte = Xte,yte=yte)
    with  open("raw_images.pkl","wb") as f:
        pickle.dump({"val": val_raw, "test": test_raw}, f)
    with open("split_info.pkl", "wb") as f:
        pickle.dump({
            "classes": classes,
            "class_map": class_map,
            "train_idx": train_idx,
            "val_idx": val_idx,
            "official_test_idx": official_test_idx,
           "extra_test_idx_from_train": extra_test_idx,
            "train_counts": dict(train_counts),
            "val_counts": dict(val_counts),
            "official_test_counts": dict(official_test_counts),
            "final_test_counts": dict(final_test_counts),
        }, f)
    print("classes:", class_map)
    print("train images:",len(train_idx), "per-class counts:", dict(train_counts))
    print("val images:" , len(val_idx),"per-class counts:", dict(val_counts))
    print("official test images:", len(official_test_idx),"per class counts:", dict(official_test_counts))


    print("extra test images from unused train:", len(extra_test_idx))
    print("final test images:", len(test_raw),"per-class-counts:", dict(final_test_counts))
    print("train feats", Xtr.shape,"val:", Xval.shape,"test:", Xte.shape)


    missing ={ class_map[c]: final_test_counts[c] for c in classes if final_test_counts[c] < PER_CLASS_TEST}
    if missing:
        print(missing)
    else:
        print("200 görüntü en az")

    

