from __future__ import annotations
import csv 
import json 
from pathlib import Path 
import cv2 
import matplotlib.pyplot as plt
import numpy as np 
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from stylematch.data import get_category_names, image_to_gray, load_fashionpedia, normalize_bbox
from stylematch.hw2_features import IMAGE_SIZE, WINDOW_COORDS, resize_gray_image
from stylematch.image_io import write_image 

def save_json(path : Path, payload : dict | list) -> None : 
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

def load_window_rows(prepared_dir : Path, split_name : str) -> list[dict] : 
    int_fields = ["label_id", "window_index", "row_index", "image_id"]
    json_fields = ["window_box", "matched_category_ids", "matched_ious"]
    rows = []
    csv_path = prepared_dir / "manifests" / f"{split_name}_windows.csv"
    with csv_path.open("r", encoding="utf-8", newline="") as handle : 
        for row in csv.DictReader(handle) : 
            for field in int_fields : 
                row[field] = int(row[field])

            for field in json_fields : 
                row[field] = json.loads(row[field])
            row["is_ambiguous"] = bool(int(row["is_ambiguous"]))
            rows.append(row)
    return rows

def evaluate_predictions(true_labels : np.ndarray, pred_labels : np.ndarray, label_order : list[str]) -> dict : 
    labels = list(range(len(label_order)))
    cm = confusion_matrix(true_labels, pred_labels, labels=labels)

    report = classification_report(
        true_labels, pred_labels, labels=labels, target_names=label_order, output_dict=True, zero_division=0,)
    
    total = int(cm.sum())
    per_class = {}
    for label_id, label_name in enumerate(label_order) : 
        tp = int(cm[label_id, label_id])
        fp = int(cm[:, label_id].sum()-tp)
        fn = int(cm[label_id, :].sum()-tp)
        tn = total-tp-fp-fn

        per_class[label_name] = {
            "precision": float(report.get(label_name, {}).get("precision", 0.0)),
            "recall": float(report.get(label_name, {}).get("recall", 0.0)),
            "f1": float(report.get(label_name, {}).get("f1-score", 0.0)),
            "support": int(report.get(label_name, {}).get("support", 0)),

            "specificity" : 0.0 if tn + fp == 0 else float(tn / (tn + fp)), 
        }
    return {
        "accuracy" : float(accuracy_score(true_labels, pred_labels)),
        "macro_f1" : float(f1_score(true_labels, pred_labels, average="macro", zero_division=0)),
        "weighted_f1" : float(f1_score(true_labels, pred_labels, average="weighted", zero_division=0)),
        "classification_report" : report,
        "confusion_matrix" : cm.tolist(),
        "labels" : label_order,
        "per_class" : per_class,
    }

def create_demo_images(
        prepared_dir : Path, prediction_rows : list[dict], label_order : list[str], output_dir : Path,
) -> None : 
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_dict = load_fashionpedia()
    category_names = get_category_names(dataset_dict)
    target_names = set(label_order[:-1])

    for image_key, rows in choose_demo_images(prediction_rows) : 
        reference = rows[0]
        source_row = dataset_dict[reference["source_split"]][reference["row_index"]]
        gray = resize_gray_image(image_to_gray(source_row["image"]), IMAGE_SIZE)
        canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        height = int(source_row["height"])
        width =int(source_row["width"])
        for category_id, bbox in zip(source_row["objects"]["category"], source_row["objects"]["bbox"]) :
            category_name =category_names.get(int(category_id), f"category_{int(category_id)}")
            if category_name in target_names :
                draw_label(canvas, scale_bbox(bbox, width, height), f"GT {category_name}", (0,255,0))

        for row in rows :
            if row["pred_label"] != "background" : 
                draw_label(canvas, json.loads(row["window_box"]), f"PR {row['pred_label']}", (0,0,255))

        for x1,y1,x2,y2 in WINDOW_COORDS : 
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (255,255,0), 1)

        out_path = output_dir / f"demo_{image_key.replace(':', '_')}.png"
        write_image(out_path, canvas)

def choose_demo_images(prediction_rows : list[dict], max_images : int = 4) -> list[tuple[str, list[dict]]] : 
    grouped = {}
    for row in prediction_rows : 
        grouped.setdefault(row["image_key"], []).append(row)
    chosen = []
    for image_key, rows in grouped.items() : 
        if any(row["pred_label"] != "background" for row in rows) : 
            chosen.append((image_key, rows))
            if len(chosen) == max_images : 
               break
    return chosen

def scale_bbox(bbox, width:int, height:int) -> list[int] : 
    norm_bbox = normalize_bbox(bbox, width, height)
    scaled = [int(round(norm_bbox[0]*IMAGE_SIZE / width)), int(round(norm_bbox[1]*IMAGE_SIZE / height)), int(round(norm_bbox[2] * IMAGE_SIZE / width)), int(round(norm_bbox[3] * IMAGE_SIZE / height)),]

    scaled[0] = max(0, min(scaled[0], IMAGE_SIZE - 1))
    scaled[1] = max(0, min(scaled[1], IMAGE_SIZE-1))
    scaled[2] = max(scaled[0] +1, min(scaled[2], IMAGE_SIZE))
    scaled[3] = max(scaled[1] +1, min(scaled[3], IMAGE_SIZE))
    return scaled 

def draw_label(image : np.ndarray, box : list[int], text : str, color : tuple[int, int, int]) -> None : 
    x1,y1,x2,y2 = box 
    cv2.rectangle(image, (x1,y1), (x2,y2), color, 2)
    cv2.putText(image, text, (x1+4, min(y2-6, y1+18)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,)


def write_predictions_csv(rows : list[dict], output_path: Path) -> None : 
    fieldnames = ["sample_id", "image_key", "source_split", "row_index", "image_id", "window_index", "window_box", "true_label_id", "true_label", "pred_label_id", "pred_label", "is_ambiguous", "matched_category_ids", "matched_ious",]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle : 
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_confusion_matrix(cm : list[list[int]], labels : list[str], output_path : Path, title : str) -> None : 
    cm_array = np.asarray(cm)
    fig, ax = plt.subplots(figsize = (10,8))
    image = ax.imshow(cm_array, cmap = "Blues")
    ax.set(xticks = np.arange(len(labels)), yticks=np.arange(len(labels)), xticklabels = labels, yticklabels = labels, xlabel = "Predicted", ylabel = "True", title = title,)

    plt.setp(ax.get_xticklabels(), rotation = 30, ha = "right")

    for row in range(cm_array.shape[0]) : 
        for col in range(cm_array.shape[1]) : 
            ax.text(col, row, str(cm_array[row, col]), ha = "center", va = "center")

    fig.colorbar(image, ax = ax)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi = 180, bbox_inches = "tight")
    plt.close(fig)


def run_test_evaluation(prepared_dir : Path, outputs_dir : Path, label_order : list[str], model, scaler_mean : np.ndarray, scaler_std : np.ndarray, test_x : np.ndarray, test_y : np.ndarray, rows : list[dict], device : torch.device, ) -> dict : 
    test_scaled = ((test_x -scaler_mean) / scaler_std).astype(np.float32)

    model.eval()
    with torch.no_grad() : 
        logits = model(torch.from_numpy(test_scaled).to(device))
        pred_all = torch.argmax(logits, dim=1).cpu().numpy()

    valid_mask = test_y >= 0 
    metrics = evaluate_predictions(test_y[valid_mask], pred_all[valid_mask], label_order)
    metrics["ambiguous_window_count"] = int(np.sum(test_y == -2))

    result_rows = [
        {
            "sample_id" : row["sample_id"], "image_key" : row["image_key"], "source_split" : row["source_split"], "row_index" : row["row_index"], "image_id" : row["image_id"], "window_index" : row["window_index"], "window_box" : json.dumps(row["window_box"]), "true_label_id" : row["label_id"], "true_label" : row["label_name"], "pred_label_id" : int(pred_label_id), "pred_label": label_order[int(pred_label_id)], "is_ambiguous" : int(row["is_ambiguous"]), "matched_category_ids" : json.dumps(row["matched_category_ids"]), "matched_ious" : json.dumps(row["matched_ious"]),
        }
        for row, pred_label_id in zip(rows, pred_all)
    ]
    test_dir = outputs_dir / "test"
    save_json(test_dir / "metrics.json", metrics)
    write_predictions_csv(result_rows, test_dir / "predictions.csv")
    plot_confusion_matrix(metrics["confusion_matrix"], label_order, test_dir / "confusion_matrix.png", "HW2 Test Confusion Matrix", )
    create_demo_images(prepared_dir, result_rows, label_order, test_dir)
    print(json.dumps(metrics, indent=2, ensure_ascii=False, ))
    return {"metrics" : metrics, "rows" : result_rows}

