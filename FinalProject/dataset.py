import json 
import random
from pathlib import Path 
import cv2 
import numpy as np 
import pandas as pd
import torch
from torch.utils.data import Dataset
from config import IMAGE_SIZE

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
SCRATCH_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
SCRATCH_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)

def load_pairs(path) : 
    with open(path, 'r', encoding='utf-8') as f :
        return json.load(f)

def decode_jpeg_bytes(image_bytes) : 
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None : 
        raise ValueError()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img 

def preprocess_rgb(img, mode='scratch', augment=False, image_size = IMAGE_SIZE) :
    img = cv2.resize(img, (image_size, image_size), interpolation = cv2.INTER_AREA)
    if augment and random.random() < 0.5 : 
        img = cv2.flip(img, 1)
    img = img.astype(np.float32) / 255.0
    if mode == 'transfer' : 
        img = (img - IMAGENET_MEAN) / IMAGE_STD
    elif mode == 'scratch' :
        img = (img - SCRATCH_MEAN) / SCRATCH_STD
    else:
        raise ValueError("mode must be 'scratch' or 'transfer'")

    img = np.transpose(img, (2, 0, 1)).copy()
    return torch.from_numpy(img)

class PolyvorePairDataset(Dataset) : 
    def __init__(self, parquet_path, pairs_path, mode='scratch', augment=False, image_size = IMAGE_SIZE, max_pairs = None):
        self.parquet_path = Path(parquet_path)
        self.pairs_path = Path(pairs_path)
        self.mode = mode
        self.augment = augment
        self.image_size = image_size
        if not self.parquet_path.exists() : 
            raise FileNotFoundError(self.parquet_path)
        if not self.pairs_path.exists() : 
            raise FileNotFoundError()

        self.df = pd.read_parquet(self.parquet_path, columns=['item_id', 'image'])
        self.item_to_index = {str(item_id) : idx for idx, item_id in enumerate(self.df['item_id'])}
        self.pairs = load_pairs(self.pairs_path)
        if max_pairs is not None : 
            self.pairs = self.pairs[:max_pairs]

        needed = set()
        for p in self.pairs : 
            needed.add(str(p['item_a']))
            needed.add(str(p['item_b']))
        missing = needed.difference(self.item_to_index)
        if missing : 
            example = list(missing)[:5]
            raise KeyError()

    def __len__(self) : 
        return len(self.pairs)

    def _load_item(self, item_id) : 
        row_idx = self.item_to_index[str(item_id)]
        image_obj = self.df.iloc[row_idx]['image']
        if not isinstance(image_obj, dict) or image_obj.get('bytes') is None : 
            raise ValueError()
        img = decode_jpeg_bytes(image_obj['bytes'])
        return preprocess_rgb(img, mode = self.mode, augment = self.augment, image_size = self.image_size,)

    def __getitem__(self, idx):
        p = self.pairs[idx]
        img_a = self._load_item(p['item_a'])
        img_b = self._load_item(p['item_b'])
        label = torch.tensor(float(p['label']), dtype=torch.float32)
        return img_a, img_b, label 