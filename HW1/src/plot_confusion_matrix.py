
#confusion matrix kodu 
import matplotlib.pyplot as plt
import numpy as np

cm = np.array([
    [37, 563],
    [172, 28]
])

labels = ["Negative", "Positive"]

plt.figure(figsize=(5,5))
plt.imshow(cm, cmap="Blues")

plt.title("Confusion Matrix")
plt.colorbar()

plt.xticks([0,1], labels)
plt.yticks([0,1], labels)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
            color="black",
            fontsize=12
        )

plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()