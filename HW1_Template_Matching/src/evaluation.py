from pathlib import Path
import json


import cv2
 
from  sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from template_matching import detect_with_templates

def load_metadata(metadata_path):
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)
def intersection_area(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)

    return iw * ih

def box_ area(box)
   x1,y1,x2,y2 = box
   return max(0,x2-x1)*max(0,y2-y1)


def object_coverage(pred_bbox,gt_bbox):
   if pred_bbox is None or gt_bbox is None:
      return 0.0

    inter =intersection_area(pred_bbox,gt_bbox)
    gt_area = box_area(gt_bbox)

    if gt_area ==0:
       return 0.0


    return inter/gt_are
 
 

    
   








def evaluate_detection(image_dir,metadat_path,templates,threshold,area_threshold=0.5):
    metadata = load_metadata(metadata_path)

    y_true = []
    y_pred = []


    for item in metadata:
        image_path = Path(image_dir) / item["filename"]
        image = cv2.imread(str(image_path),cv2.IMREAD_GRAYSCALE)
       

        result = detect_with_templates(
            image = image,
            templates=templates,
            threshold= threshold
        )

        true_label = item["label"]
        gt_bbox = item["bbox"]


        if result["detected"]:
            if true_label == 1:
                coverage = object_coverage(result["bbox"],gt_bbox)
                predicted_label = 1 if coverage >= area_threshold else 0
            else:
                predicted_label = 0
        else:
            predicted_label = 0
        y_true.append(true_label)
        y_pred.append(predicted_label)
    metrics = {
        "threshold": threshold
        "area_threshold": area_threshold,
        "accuracy": accuracy_score()
        "precision": precission_score(y_true,y_pred,zero_division = 0),
        "f1" : f1_score(y_true,y_pred,zero_diviison = 0),
        "confusion_matrix": confusion_matrix(y_true,y_pred).tolist()

    }

    return metrics


        
