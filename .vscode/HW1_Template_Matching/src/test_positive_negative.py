import cv2

from template_matching import load_templates,detect_with_templates


templates = load_templates(
    "data/processed/train/templates"
)

positive = cv2.imread(
    "data/processed/val/images/pos_0000.png",
    cv2.IMREAD_GRAYSCALE
)


negative = cv2.imread(
    "data/processed/val/images/neg_0000.png",
    cv2.IMREAD_GRAYSCALE
)

pos_result = detect_with_templates(
    positive,
    templates,
    threshold=0.5

)

neg_result = detect_with_templates(
    negative,
    templates,
    threshold = 0.5
)

neg_result = detect_with_templates(
    negative,
    templates,
    threshold=0.5
)

print("POSITIVE")
print(pos_result)

print()

print("NEGATIVE")
print(neg_result)
      