from template_matching import load_templates
from evaluation import evaluate_detection

BEST_THRESHOLD = 0.60
BEST_AREA_THRESHOLD = 0.30

templates = load_templates(
    "data/processed/train/templates"
)

metrics = evaluate_detection(
    image_dir = "data/processed/test/images",
    metadata_path = "data/processed/test/metadata.json",
    templates = BEST_THRESHOLD,
    area_threshold = BEST_AREA_THRESHOLD
)


print("\nFINAL TEST RESULTS\n")
print(metrics)