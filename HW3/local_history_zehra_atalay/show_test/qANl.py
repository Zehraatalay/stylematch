import cv2
import pickle
import matplotlib.pyplot as plt 
import argparse
from detect_eval import detect_image 

CLASS_NAMES = {0 : "no object"}
PROB_THRESH = 0.20 

def _pt(x,y) : 
    return int(round(float(x))), int(round(float(y)))

def parse_args() : 
    p = argparse.ArgumentParser()
    p.add_argument("--iou", type=float, default=0.3)
    p.add_argument("--margin", type=float, default=0.008)
    p.add_argument("--min-object-prob", type=float, default=0.20)
    p.add_argument("--n", type=int, default=3)
    return p.parse_args()

def pick_and_show(m, test_raw, n_examples=3, iou_thresh=0.3, margin=0.08, min_object_prob=0.20) :
    scores = []
    for i, (img128, gt_boxes) in enumerate(test_raw) : 
        if not gt_boxes : 
            continue
        preds, gts, _ = detect_image(img128, gt_boxes, m["kmeans_models"], m["K"], m["leaf_counts"], m["class_totals"], iou_thresh=iou_thresh, min_margin=margin, min_object_prob=min_object_prob,)
        positive = [j for j, g in enumerate(gts) if g != 0]
        correct = sum(preds[j] == gts[j] for j in positive)
        score = correct / len(positive) if positive else 0.0 
        scores.append((i, score))
    scores.sort(key=lambda z: z[1], reverse=True)
    good = [i for i, _ in scores[:n_examples]]
    bad = [i for i, _ in scores[-n_examples:]]
    for tag, idx_list in (("good", good), "bad", bad)) : 
        for j, i in enumerate(idx_list) : 
            img128, gt_boxes = test_raw[i]
            out_path = f"test_example_{tag}_{j}.png"
            draw(img128, gt_boxes, m["kmeans_models"], m["K"], m["leaf_counts"], m["class_totals"], out_path, iou_thresh=iou_thresh, margin=margin, min_object_prob=min_object_prob,)

def draw(img128, gt_boxes, kmeans_models, K, leaf_counts, class_totals, out_path, iou_thresh=0.3, margin=0.08, min_object_prob=0.20,) :
    preds, gts, boxes_out = detect_image(img128, gt_boxes, kmeans_models, K, leaf_counts, class_totals, iou_thresh=iou_thresh, margin=margin, min_object_prob=min_object_prob,)
    vis = cv2.cvtColor(img128, cv2.COLOR_GRAY2BGR)
    colors = [(0,0,255), (0,255,0), (255,0,0), (0,255,255), (255,0,255), (255,255,0)]
    for cls, box, prob in boxes_out : 
        if prob < PROB_THRESH : 
            continue
        x0, y0, x1, y1 = box
        color = colors[int(cls) % len(colors)]
        cv2.rectangle(vis, _pt(x0, y0), _pt(x1, y1), color, 1)
        cv2