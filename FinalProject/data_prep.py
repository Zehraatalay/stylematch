import argparse
import json 
import random
from collections import defaultdict
from pathlib import Path
from config import DATA_ROOT,METADATA_PATH,PAIR_DIR,SEED,SPLITS

def load_json(path):
    with open(path,'r', encoding='utf-8') as f:
        return json.load(f)
def save_json(obj,path):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with open(path,'w', encoding='utf-8') as f:
        json.dump(obj,f,ensure_ascii = False)


def load_item_categories(metadata_path = METADATA_PATH):
    metadata = load_json(metadata_path)
    result = {}
    for item_id,rec in metadata.items():
        category = rec.get('semantic_category')
        if category is None:
            category = rec.get('category_id', rec.get('categoryid', 'unknown'))
        result[str(item_id)] = str(category)
    return result

def build_pairs(outfits,item_category,max_pos_per_outfit = 4,seed =SEED):
    rng = random.Random(seed)
    outfit_of_item = {}
    items_by_category = defaultdict(list)


    for outfiy in outfits:
        outfit_id = str(outfit['set_id'])
        for item in outfit['items']:
            iid = str(item['item_id'])
            outfit_of_item[iid] = outfit_id
            cat = item_category.get(iid,'unknown')
            items_by_category[cat].append(iid)
        pairs = []
        skipped_no_negative = 0
        for outfit in outfits:
            items = outfit['items']
            if len(items) < 2:
                continue

            candidates = []
            for i in range(len(items)):
                for j in range(i+1,len(items)):
                    a = str(items[i]['item_id'])
                    b = str(items[j]['item_id'])
                    cat_a = item_category.get(a,'unknown')
                    cat_b = item_category.get(b,'unknown')
                    if cat_a != cat_b:
                        candidates.append((a,b,cat_a,cat_b))
                rng.shuffle(candidates)
                for a,b,cat_a,cat_b in candidates[:max_pos_per_outfit]:
                    pairs.append({
                         'item_a': a, 'item_b': b, 'label': 1,
                                        'cat_a': cat_a, 'cat_b': cat_b,
                    })


                    valid_negatives = [
                        cand for cand in items_by_category[cat_b]
                        if cand != b and outfit_of_item.get(cand) != outfit_of_item[a]

                    ]
                    if not valid_negatives:
                        skipped_no_negative += 1
                        continue
                    neg_b = rng.choice(valid_negatives)
                    pairs.append(
                        {
                            'item_a': a, 'item_b': neg_b, 'label': 0,
                            'cat_a': cat_a,'cat_b': cat_b,
                    })
            rng.shuffle(pairs)
            return pairs,skipped_no_negative


        def main():
            parser = argparse.ArgumentParser()
            parser.add_argument('--max_pos_per_outfit', type = int,default=4)
            parser.add_argument('--seed', type= int, default=SEED)
            parser.add_argument('--overwrite', action='store_true')
            args = parser.parse_args()

            if not DATA_ROOT.exists():
                raise FileNotFoundError(
                    f'Dataset root not found: {DATA_ROOT}\nMount Google Drive first in Colab.'
                )

            item_category = load_item_categories()
            print(f'[info] metadata items: {len(item_category):,}')

            PAIR_DIR.mkdir(parents=True,exist_ok=True)
            for offset,split in enumerate(('train', 'valid', 'test')):
                out_path = PAIR_DIR / f'{split}_pairs.json'
                if out_path.exists() and not args.overwrite:
                    print(f'[skip] {out_path} already exists. Use --overwrite to rebuild.')
                    continue

                outfits = load_json(SPLITS[split]['json'])
                pairs,skipped = build_pairs(
                    outfits,
                    item_category,
                    max_pos_per_outfit= args.max_pos_per_outfit,
                    seed = args.seed + offset,
                )
                save_json(pairs,out_path)
                n_pos = sum(p['label'] == 1 for p in pairs)
                n_neg = len(pairs) - n_pos
                print(
                    f'[{split}] outfits={len(outfits):,} pairs={len(pairs):,} '
                    f'positive={n_pos:,} negative={n_neg:,} skipped_neg={skipped:,}'
                )
                print(f'        saved -> {out_path}')
        if __name__ == '__main__':
            main()





                    
