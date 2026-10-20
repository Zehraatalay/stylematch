from template_matching import load_templates
from evaluation import evaluate_detection

templates = load_templates("data/processed/train/templates")

thresholds = [0.50, 0.60, 0.70, 0.80]
area_thresholds = [0.3, 0.5]
stride_ratios = [0.10, 0.25, 0.50]

best_result = None


for threshold in thresholds:
    for area_threshold in area_thresholds:
        for stride_ratio in stride_ratios:
            metrics = evaluate_detection(
                image_dir="data/processed/val/images",
                metadata_path="data/processed/val/metadata.json",
                templates=templates,
                threshold=threshold,
                area_threshold=area_threshold,
                stride_ratio=stride_ratio
            )
            print(metrics)

            if best_result is None or metrics["f1"] > best_result["f1"]:
                best_result = metrics

print("\nBest validation result:")
print(best_result)