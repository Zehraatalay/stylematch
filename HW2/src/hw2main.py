from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import csv

from hw2_data import prepare_hw2_fashionpedia_data
from hw2_evaluation import load_window_rows, run_test_evaluation
from hw2_model import load_feature_split, load_trained_model, run_validation_search


def add_common_arguments(parser:argparse.ArgumentParser) -> None:
    parser.add_argument("--project-root" , type= Path,default=Path.cwd())
    parser.add_argument("--prepared-dir", type=Path, default=None)
    parser.add_argument("--outputs-dir", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed" , type=int,default=42)

def resolve_dirs(args: argparse.Namespace) -> tuple[Path,Path]:
    root = args.project_root.resolve()
    prepared = (args.prepared_dir or (root / "outputs" / "hw2")).resolve()
    outputs = (args.outputs_dir or (root / "outputs" / "hw2")).resolve()
    prepared.mkdir(parents=True, exist_ok=True)
    outputs.mkdir(parents=True, exist_ok=True)
    return prepared, outputs



def load_metadata(prepared_dir: Path) -> dict:
    path = prepared_dir / "manifests" / "metadata.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    

    train_image_path = prepared_dir / "manifests" / "train_images.json"
    val_image_path = prepared_dir / "manifests" / "validation_images.json"

    if not train_image_path.exists() or not val_image_path.exists():
        raise FileNotFoundError(f"HW2 metadata bulunamadı: {path}")
    

    label_map: dict[int,str] = {}

    for image_path in (train_image_path,val_image_path):
        items = json.loads(image_path.read_text(encoding= "utf-8"))

    
    label_map: dict[int,str] = {}
    for image_path in (train_image_path,val_image_path):
        items = json.loads(image_path.read_text(encoding="utf-8"))
        for item in items:
            for obj in item.get("scaled_objects",[]):
                label_map[int(obj["label_id"])] = obj["label_name"]


        
    if not label_map:
            raise FileNotFoundError(f"HW2 metadata türetilemedi: {path}")
        

    max_label_id = max(label_map)
    ordered = [label_map[index] for index in range(max_label_id+1)]
    ordered.append("background")
    metadata = {"label_order": ordered}
    return metadata 
    


def run_prepare(args: argparse.Namespace) -> None:
        prepared_dir, _ = resolve_dirs(args)
        prepare_hw2_fashionpedia_data(
        output_dir = prepared_dir,
        num_classes = args.num_classes,
        train_target_per_class = args.train_target_per_class,
        val_target_per_class = args.validation_target_per_class,
        test_target_per_class = args.test_target_per_class,
        image_size = args.image_size,
        seed = args.seed,
        iou_threshold = args.iou_threshold,
                            )
def run_validation(args: argparse.Namespace) -> dict:
    prepared_dir,outputs_dir = resolve_dirs(args)
    metadata = load_metadata(prepared_dir)
    config_names = None
    if args.config_names:
        config_names = [item.strip() for item in args.config_names.split(",") if item.strip()]
    return run_validation_search(
        prepared_dir = prepared_dir,
        outputs_dir = outputs_dir,
        label_order = metadata["label_order"],
        seed = args.seed,
        config_names = config_names,
        max_epochs_override = args.max_epochs,
    )



def run_test(args: argparse.Namespace) -> dict:
    prepared_dir,outputs_dir = resolve_dirs(args)
    metadata = load_metadata(prepared_dir)
    best_path = outputs_dir/ "validation" / "best_config.json"
    if not best_path.exists():
        raise FileNotFoundError(f"HW2 validation sonucu bulunamadı: {best_path}")
    best = json.loads(best_path.read_text(encoding="utf-8"))




    test_x,test_y, _ = load_feature_split(prepared_dir, "test")
    rows = load_window_rows(prepared_dir,"test")
    mean = np.load(outputs_dir / "validation" / "scaler_mean.npy")
    std = np.load(outputs_dir / "validation" / "scaler_std.npy")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, _  = load_trained_model(
        model_path = Path(best["model_path"]),
         label_order= metadata["label_order"],
         input_dim = int(test_x.shape[1]),
        device= device,
)

    result = run_test_evaluation(
        prepared_dir=prepared_dir,
        outputs_dir=outputs_dir,
        label_order=metadata["label_order"],
        model=model,
        scaler_mean=mean,
        scaler_std=std,
        test_x=test_x,
        test_y=test_y,
        rows=rows,
        device=device,
    )


    save_path = outputs_dir / "test" / "best_config_applied.json"
    save_path.write_text(json.dumps(best, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def run_all(args: argparse.Namespace) -> None:
    run_prepare(args)
    run_validation(args)
    run_test(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description = "hw2pipeline")
    subparsers = parser.add_subparsers(dest = "command",required = True)


    prepare = subparsers.add_parser("prepare-data")
    add_common_arguments(prepare)
    prepare.add_argument("--num-classes",type = int, default= 10)
    prepare.add_argument("--train-target-per-class", type=int, default=600)
    prepare.add_argument("--validation-target-per-class", type=int, default=200)
    prepare.add_argument("--test-target-per-class",type = int,default=200)
    prepare.add_argument("--iou-threshold", type=float, default=0.2)
    prepare.set_defaults(func=run_prepare)



    validation = subparsers.add_parser("run-validation")
    add_common_arguments(validation)
    validation.add_argument("--config-names", type=str, default=None)
    validation.add_argument("--max-epochs", type=int, default=None)
    validation.set_defaults(func = run_validation)



    test = subparsers.add_parser("run-test")
    add_common_arguments(test)
    test.set_defaults(func = run_test)




    run_all_parser = subparsers.add_parser("run-experiments")





    add_common_arguments(run_all_parser)
    run_all_parser.add_argument("--num-classes", type=int, default=10)
    run_all_parser.add_argument("--train-target-per-class", type=int, default=600)
    run_all_parser.add_argument("--validation-target-per-class", type=int, default=200)
    run_all_parser.add_argument("--test-target-per-class", type=int, default=200)
    run_all_parser.add_argument("--iou-threshold", type=float, default=0.2)
    run_all_parser.add_argument("--config-names", type=str, default=None)
    run_all_parser.add_argument("--max-epochs", type=int, default=None)
    run_all_parser.set_defaults(func=run_all)

    return parser



def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
            
