import json 
from collections import Counter
import matplotlib.pyplot as plt 
import pandas as pd 
from config import DATA_ROOT, METADATA_PATH, OUTPUT_DIR, SPLITS
from data_prep import load_item_categories 

def load_json(path) : 
    with open(path, 'r', encoding='utf-8') as f : 
        return json.load(f)

def main() : 
    if not DATA_ROOT.exists() : 
        raise FileNotFoundError()
    categories = load_item_categories(METADATA_PATH)

    print('[root]', DATA_ROOT)
    print('[metadata items]', f'{len(categories):,}')

    all_split_ids = {}
    all_sizes = []
    for split in ('train', 'valid', 'test') : 
        outfits = load_json(SPLITS[split]['json'])
        df_ids = set(pd.read_parquet(SPLITS[split]['parquet'], columns=['item_id'])['item_id'].astype(str))
        json_ids = {str(item['item_id']) for o in outfits for item in o['items']}
        missing = json_ids - df_ids
        all_split_ids[split] = json_ids
        sizes = [len(o['items']) for o in outfits]
        all_sizes.extend(sizes)
        print(
            f'[{split}] outfits={len(outfits):,} parquet_items={len(df_ids):,} '
            f'json_items={len(json_ids):,} missing={len(missing):,} '
            f'avg_items/outfit={sum(sizes)/len(sizes):.2f}'
        )

    print('[overlap] train-valid:', len(all_split_ids['train'] & all_split_ids['valid']))
    print('[overlap] train-test :', len(all_split_ids['train'] & all_split_ids['test']))
    print('[overlap] valid-test :', len(all_split_ids['valid'] & all_split_ids['test']))

    cat_counts = Counter(categories[i] for i in all_split_ids['train'] if i in categories)
    print('[train top categories]')
    for cat, n in cat_counts.most_common(12) : 
        print(f'    {cat}: {n:,}')
    eda_dir = OUTPUT_DIR / 'eda'
    eda_dir.mkdir(parents=True, exist_ok=True)
    top = cat_counts.most_common(10)
    plt.figure(figsize=(8,4))
    plt.bar([x[0] for x in top], [x[1] for x in top])
    plt.xticks(rotation=45, ha='right')
    plt.title('Train split - top 10 categories')
    plt.tight_layout()
    plt.savefig(eda_dir / 'category_distribution.png', dpi=150)
    plt.close()

    plt.figure(figsize=(6,4))
    plt.hist(all_sizes, bins=range(1, max(all_sizes) + 2))
    plt.xlabel('Items per outfit')
    plt.ylabel('Outfit count')
    plt.title('Outfit size distribution')
    plt.tight_layout()
    plt.savefig(eda_dir / 'outfit_size_distribution.png', dpi=150)
    plt.close()
    print('[saved EDA]', eda_dir)
if __name__ == '__main__' : 
    main()