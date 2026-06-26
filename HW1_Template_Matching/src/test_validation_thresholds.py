from template_matching import load_templates
from evaluation import evaluate_detection

templates = load_templates("data/processed/train/templates")

thresholds = [0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85]
area_thresholds =[0.3,0.5,0.7]


best_result = None


for threshold in thresholds:
    for area_threshold in area_thresholds:
        metrics = evaluate_detection(
            image_dir = "data/processed/val/images",
            metadata_path= "data/processed/val/images",
            templates = templates,
            threshold = threshold,
            area_threshold = area_threshold
        )
        print(metrics)

        if best_result is None or metrics["f1"] > best_result["f1"]:
                                    best_result = metrics
print("\nBest validation result:")
print(best_result)
                                                              