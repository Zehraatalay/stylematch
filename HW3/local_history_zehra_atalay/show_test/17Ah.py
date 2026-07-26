import cv2
import pickle
import matplotlib.pyplot as plt 
import argparse
from detect_eval import detect_image 

CLASS_NAMES = {0 : "no object", 1 : "sleeve", 2 : "shoe", 3 : "neckline", 4 : "pocket", 5 : "dress", 6 : "top", 7 : "pants", 8 : "collar", 9 : "zipper", 10 : "jacket"}


def _pt(x,y) : 
    return int(round(float(x))), int(round(float(y)))

def parse_args() : 
    p = argparse.ArgumentParser()
    p.add_argument("--iou", type=float, default=0.3)
    p.add_argument("--margin", type=float, default=0.08)
    p.add_argument("--min-object-prob", type=float, default=0.20)
    p.add_argument("--n", type=int, default=3)
    return p.parse_args()

def pick_and_show(m, test_raw, n_examples=3, iou_thresh=0.3, margin=0.08, min_object_prob=0.20) :
    print(f"selecting {n_examples} best and worst test examples")
    scores = []
    for i, (img128, gt_boxes) in enumerate(test_raw) : 
        if not gt_boxes : 
            continue
        preds, gts, _ = detect_image(img128, gt_boxes, m["kmeans_models"], m["K"], m["leaf_counts"], m["class_totals"], iou_thresh=iou_thresh, min_margin=margin, min_object_prob=min_object_prob,)
        positive = [j for j, g in enumerate(gts) if g != 0]
        if not positive : 
            continue
        correct = sum(preds[j] == gts[j] for j in positive)
        score = correct / len(positive)
        scores.append((i, score))
    if not scores :
        return
    
    scores.sort(key=lambda z: z[1], reverse=True)
    count = min(n_examples, len(scores))
    good = [i for i, _ in scores[:count]]
    bad = [i for i, _ in scores[-count:]]
    print(f"selected {len(good)} best and {len(bad)} worst examples")
    for tag, idx_list in (("good", good), ("bad", bad)) : 
        for j, i in enumerate(idx_list) : 
            img128, gt_boxes = test_raw[i]
            out_path = f"test_example_{tag}_{j}.png"
            draw(img128, gt_boxes, m["kmeans_models"], m["K"], m["leaf_counts"], m["class_totals"], out_path, iou_thresh=iou_thresh, margin=margin, min_object_prob=min_object_prob,)
            print(f"saved: {out_path}")

def draw(img128, gt_boxes, kmeans_models, K, leaf_counts, class_totals, out_path, iou_thresh=0.3, margin=0.08, min_object_prob=0.20,) :
    preds, gts, boxes_out = detect_image(img128, gt_boxes, kmeans_models, K, leaf_counts, class_totals, iou_thresh=iou_thresh, margin=margin, min_object_prob=min_object_prob,)
    vis = cv2.cvtColor(img128, cv2.COLOR_GRAY2BGR)
    colors = [(0,0,255), (0,255,0), (255,0,0), (0,255,255), (255,0,255), (255,255,0)]
    for cls, box, prob in boxes_out : 
        if prob < min_object_prob : 
            continue
        x0, y0, x1, y1 = box
        color = colors[int(cls) % len(colors)]
        cv2.rectangle(vis, _pt(x0, y0), _pt(x1, y1), color, 1)
        cv2.putText(vis, f"{    CLASS_NAMES.get(int(cls), str(cls))} {prob:.2f}", _pt(x0, max(float(y0) - 3.0, 8.0)), cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1,)
    for cls, box in gt_boxes : 
        x0, y0, x1, y1 = box
        cv2.rectangle(vis, _pt(x0, y0), _pt(x1, y1), (0, 255, 255), 1)
    plt.figure(figsize=(5,5))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=160)
    plt.close()
    n_correct = sum(1 for p, g in zip(preds, gts) if p == g and g != 0)
    n_gt_pos = sum(1 for g in gts if g!= 0)
    print(f"finished {out_path} ({n_correct}/{n_gt_pos} correct)")
    return n_correct, n_gt_pos


if __name__ == "__main__" : 
    args = parse_args()
    with open("ktree_model.pkl", "rb") as f : 
        model = pickle.load(f)
    with open("raw_images.pkl", "rb") as f : 
        test_raw = pickle.load(f)["test"]
    pick_and_show(model, test_raw, n_examples=args.n, iou_thresh=args.iou, margin=args.margin, min_object_prob=args.min_object_prob, )