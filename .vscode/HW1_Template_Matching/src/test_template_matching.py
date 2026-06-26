import cv2

from template_matching import load_templates,detect_with_templates

TEMPLATE_DIR = "data/processed/train/templates"
TEST_IMAGE_PATH = "data/processed/val/images/pos_0000.png"

templates = load_templates(TEMPLATE_DIR)

image = cv2.imread(TEST_IMAGE_PATH,cv2.IMREAD_GRAYSCALE)


result = detect_with_templates(
    image=image,
    templates=templates,
    threshold=0.5
)

print("Template count:",len(templates))
print("Detection resukt:")
print(result)