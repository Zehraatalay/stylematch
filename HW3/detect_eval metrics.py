import argparse
import pickle
import cv2
import numpy as np

from metrics import confusion_matrix, classification_report, print_report
from ktree import class_probs, choose_class


WINDOWS = [128,64,32,16]
IMG_SIZE = 128
DEFAULT_IOU_THRESH = 0.3
DEFAULT_NORM_POWER = 0.5
DEFAULT_MIN_MARGIN = 0.08
DEFAULT_MIN_OBJECT_PROB = 0.20


def iou(a,b):
    ax0,ay0,ax1,ay1 = map(float,a)
    bx0,by0,bx1,by1 = map(float,b)
    ix0,iy0 = max(ax0,bx0),max(ay0,by0)
    ix1,iy1 = min(ax1,bx1), min(ay1,by1)
    inter = max(0.0,ix1 -ix0) * max(0.0,iy1 - iy0)
    union = (ax1-ax0)*(ay1-ay0) + (bx1-bx0)*(by1-by0) - inter
    return inter / union if union > 0 else 0.0


def gt_label(box,gt_boxes,iou_thresh= DEFAULT_IOU_THRESH):
    best_iou,best_cls = 0.0,0
    for cls, gt_box in gt_boxes:
        score = iou(box,gt_box)
        if score > best_iou:
            best_iou,best_cls = score, int(cls)
    return best_cls if best_iou >= iou_thresh else 0


def sift_once(img128):
    sift = cv2.SIFT_create()
    keypoints,desc = sift.detectAndCompute(img128,None)
    if desc is None or not keypoints:
        return np.empty((0,2), dtype = np.float32),np.empty((0,128),dtype = np.float32)
    xy = np.array([kp.pt for kp in keypoints], dtype= np.float32)
    return xy,desc


def descriptors_in_window(keypoint_xy,desc,x,y,win):
    if desc is None or len(desc) == 0:
        return None
    x1,y1 = x + win, y + win
    mask = (
        (keypoint_xy[:,0] >= x)& (keypoint_xy[:,0] < x1)&
        (keypoint_xy[:,1] >= y) & (keypoint_xy[:,1] < y1)

    )

    selected = desc[mask]
    return selected if len(selected) else None



def detect_image(
    img128,
    gt_boxes,
    kmeans_models,
    K,
    leaf_counts,
    class_totals,
    num_classes=None,
    iou_thresh=DEFAULT_IOU_THRESH,
    norm_power=DEFAULT_NORM_POWER,
    min_margin=DEFAULT_MIN_MARGIN,
    min_object_prob=DEFAULT_MIN_OBJECT_PROB,
):
    keypoint_xy,image_desc = sift_once(img128)
    preds,gts,boxes_out = [], [], []


    for win in WINDOWS:
        stride = max(1,win // 2)
        for y in range(0,IMG_SIZE-win + 1, stride):
            for x in range(0,IMG_SIZE - win + 1,stride):
                desc = descriptors_in_window(keypoint_xy,image_desc,x,y,win)
                probs = class_probs(
                    desc,kmeans_models,K,leaf_counts,class_totals,
                    num_classes = num_classes, norm_power = norm_power,
                )
                pred = choose_class(
                    probs,
                    min_margin=min_margin,
                    min_object_prob=min_object_prob,

                )
                box = (int(x),int(y), int(x + win),int(y + win))
                preds.append(pred)
                gts.append(gt_label(box, gt_boxes, iou_thresh=iou_thresh))
                if pred != 0:
                    boxes_out.append((pred,box,float(probs[pred])))
    return preds,gts,boxes_out
def evaluate(
        images_with_boxes,
        kmeans_models,
        K,
        leaf_counts,
        class_totals,
          num_classes=None,
    iou_thresh=DEFAULT_IOU_THRESH,
    norm_power=DEFAULT_NORM_POWER,
    min_margin=DEFAULT_MIN_MARGIN,
    min_object_prob=DEFAULT_MIN_OBJECT_PROB,
):
    all_preds, all_gts = [], []
    for img128,gt_boxes in images_with_boxes:
        
        preds, gts, _ = detect_image(
            img128, gt_boxes, kmeans_models, K, leaf_counts, class_totals,
            num_classes=num_classes,
            iou_thresh=iou_thresh,
            norm_power=norm_power,
            min_margin=min_margin,
            min_object_prob=min_object_prob,
        )
        all_preds.extend(preds)
        all_gts.extend(gts)
    if num_classes is None:
        num_classes = len(class_totals) - 1
    cm = confusion_matrix(all_gts,all_preds,num_labels = num_classes + 1)
    rows,accuracy = classification_report(cm)
    return cm, rows,accuracy

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--iou", type=float,default=DEFAULT_IOU_THRESH,
                   help= "esik")
    p.add_argument("--norm-power",type=float,default=DEFAULT_NORM_POWER)
    p.add_argument("--margin", type=float,default=DEFAULT_MIN_MARGIN)
    p.add_argument(
    "--min-object-prob",
    type=float,
    default=DEFAULT_MIN_OBJECT_PROB,
)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    with open("ktree_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("raw_images.pkl", "rb") as f:
        raw = pickle.load(f)
    cm,rows,accuracy = evaluate(
        raw["test"],model["kmeans_models"],model["K"],
        model["leaf_counts"],model["class_totals"],
        iou_thresh=args.iou,
        norm_power=args.norm_power,
        min_margin= args.margin,
        min_object_prob= args.min_object_prob,

)
    print(
        f"IoU={args.iou:.2f}, "
        f"min_object_prob={args.min_object_prob:.2f}"
    )
    print(cm)
    print_report(rows,accuracy)


