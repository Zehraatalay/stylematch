import torch
from torch.utils.data import DataLoader
from config import PAIR_DIR,SPLITS
from dataset import PolyvorePairDataset
from model_scratch import ScratchTwoTowerNet
from model_transfer import TransferSiameseNet

def test_dataset(mode):
    ds = PolyvorePairDataset(
        SPLITS['train']['parquet'],PAIR_DIR / 'train_pairs.json',
        mode = mode, augment=False,max_pairs=16
    )
    loader = DataLoader(ds,batch_size = 4,shuffle = False, num_workers = 0)
    a, b, y = next(iter(loader))
    print(f'[{mode}] A={tuple(a.shape)} B={tuple(b.shape)} y={tuple(y.shape)} labels={y.tolist()}')
    return a,b

def main():
    a,b = test_dataset('scratch')
    m1 = ScratchTwoTowerNet()
    print('[scratch forward]', tuple(m1(a,b).shape))

    a,b = test_dataset('transfer')
    m2 = TransferSiameseNet(pretrained = False)
    print('[transfer forward]', tuple(m2(a, b).shape))
    print('SMOKE TEST OK')
if __name__ == '__main__':
    main()
