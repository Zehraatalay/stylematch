from template_matching import load_templates
from evaluation import evaluate_detection

BEST_THRESHOLD = 0.50
BEST_AREA_THRESHOLD = 0.30
BEST_STRIDE_RATIO = 0.50

templates = load_templates(
    "data/processed/train/templates"
)

metrics = evaluate_detection(
    image_dir = "data/processed/test/images",
    metadata_path = "data/processed/test/metadata.json",
    templates = templates,
    threshold=BEST_THRESHOLD,
    area_threshold = BEST_AREA_THRESHOLD,
    stride_ratio=BEST_STRIDE_RATIO
)


print("\nfinal-test-results\n")
print(metrics)