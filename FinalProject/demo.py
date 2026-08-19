import argparse
import torch
from config import OUTPUT_DIR, SPLITS
from dataset import PolyvorePairDataset, preprocess_rgb, decode_jpeg_bytes
from model_scratch import ScratchTwoTowerNet 
from model_transfer import TransferSiameseNet 
import pandas as pd 

def load_item_tensor(parquet_path, item_id, mode) : 
    df = pd.read_parquet(parquet_path, columns=['item_id', 'image'])
    row = df.loc[df['item_id'].astype(str) == str(item_id)]
    if row.empty : 
        raise KeyError()
    image_bytes = row.iloc[0]['image']['bytes']
    img = decode_jpeg_bytes(image_bytes)
    return preprocess_rgb(img, mode=mode, augment=False).unsqueeze(0)

def main() : 
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=['scratch', 'transfer'], required=True)
    p.add_argument('--split', choices=['train', 'valid', 'test'], default='test')
    p.add_argument('--item_a', required=True)
    p.add_argument('--item_b', required=True)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(OUTPUT_DIR / args.method / 'best.pt', map_location=device)

    if args.method == 'scratch' : 
        model = ScratchTwoTowerNet().to(device)
        mode = 'scratch'
    else : 
        model = TransferSiameseNet(pretrained = False).to(device)
        mode = 'transfer'
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    a = load_item_tensor(SPLITS[args.split]['parquet'], args.item_a, mode).to(device)
    b = load_item_tensor(SPLITS[args.split]['parquet'], args.item_b, mode).to(device)
    with torch.no_grad() : 
        prob = torch.sigmoid(model(a,b)).item()
    print(f'compatibility_probability={prob:.4f}')
    print('prediction=', 'UYUMLU' if prob >= 0.5 else 'UYUMSUZ')

if __name__=='__main__' : 
    main()