# StyleMatch — Computer Vision & Fashion AI

A collection of computer vision and deep learning projects exploring image matching, visual feature extraction, clustering, and fashion compatibility prediction.

Developed as part of the **Computer Vision and Deep Learning coursework at TOBB University of Economics and Technology**.

The repository covers both classical computer vision techniques and CNN-based approaches, progressing from template matching to deep learning models for fashion compatibility.

## Projects

### 01 — Template Matching

**Directory:** [`HW1`](HW1)

Explores traditional template matching techniques for detecting fashion items in images.

**Key concepts:**
- Template matching and similarity scoring
- Detection thresholds and parameter analysis
- Image preprocessing
- Evaluation of detection performance

**Technologies:** Python, OpenCV, NumPy

[Explore HW1 →](HW1)

---

### 02 — HOG Feature Extraction & Classification

**Directory:** [`HW2`](HW2)

Investigates sliding-window image classification using Histogram of Oriented Gradients (HOG) features.

Images are divided into candidate windows, and visual descriptors are extracted to distinguish target fashion categories.

**Key concepts:**
- Sliding-window detection
- Histogram of Oriented Gradients (HOG)
- Feature-based classification
- Intersection over Union (IoU)
- Classification performance evaluation

**Technologies:** Python, OpenCV, NumPy, scikit-learn

[Explore HW2 →](HW2)

---

### 03 — SIFT & Hierarchical K-Means

**Directory:** [`HW3`](HW3)

Explores local feature descriptors and hierarchical clustering for visual representation and image analysis.

**Key concepts:**
- Scale-Invariant Feature Transform (SIFT)
- Local feature extraction
- Hierarchical K-Means / K-tree
- Feature representation
- Hyperparameter experimentation and evaluation

**Technologies:** Python, OpenCV, NumPy, scikit-learn

[Explore HW3 →](HW3)

---

### 04 — Fashion Item Compatibility Prediction

**Directory:** [`FinalProject`](FinalProject)

Compares two deep learning approaches for predicting whether two fashion items are visually compatible using the Polyvore Outfits dataset.

**Approach 1 — Scratch Early-Fusion CNN**

Two RGB images are concatenated into a six-channel tensor and processed using a CNN trained without pretrained weights.

The architecture includes multi-scale convolutions, residual blocks, and squeeze-and-excitation attention.

**Approach 2 — Siamese ResNet-18**

Uses a shared ImageNet-pretrained ResNet-18 encoder to extract embeddings from both images.

The embeddings are combined using symmetric pairwise operations and passed through a compatibility classifier.

**Key concepts:**
- CNN architecture design
- Transfer learning and fine-tuning
- Siamese networks
- Fashion compatibility prediction
- Data leakage prevention through item-disjoint splits
- Hyperparameter tuning and early stopping
- ROC-AUC-based model selection

**Technologies:** Python, PyTorch, Torchvision, OpenCV, pandas, scikit-learn

#### Reported experimental results

| Model | Test Accuracy | Test F1 | Test ROC-AUC |
| --- | ---: | ---: | ---: |
| Scratch Early-Fusion CNN | **0.6202** | 0.5951 | **0.6651** |
| Siamese ResNet-18 | 0.5965 | **0.6075** | 0.6346 |

The scratch CNN achieved higher ROC-AUC and accuracy, while the Siamese model achieved a slightly higher F1-score.

Results are taken from the original academic experiments. See the project report for the complete methodology and evaluation.

[Explore FinalProject →](FinalProject)

---

## Technical Skills Demonstrated

| Area | Techniques |
| --- | --- |
| Image Processing | OpenCV, preprocessing, template matching |
| Feature Engineering | HOG, SIFT, sliding windows |
| Machine Learning | Classification, clustering, hyperparameter tuning |
| Deep Learning | CNNs, residual connections, attention mechanisms |
| Transfer Learning | ResNet-18, frozen backbones, fine-tuning |
| Evaluation | Accuracy, precision, recall, F1, ROC-AUC |
| Experimental Design | Data splitting, leakage prevention, model comparison |

## Repository Structure

```text
stylematch/
├── HW1/
│   ├── README.md
│   ├── requirements.txt
│   └── ...
├── HW2/
│   ├── README.md
│   ├── requirements.txt
│   └── ...
├── HW3/
│   ├── README.md
│   ├── requirements.txt
│   └── ...
├── FinalProject/
│   ├── README.md
│   ├── requirements.txt
│   ├── model_scratch.py
│   ├── model_transfer.py
│   └── ...
└── README.md
```

Each project directory contains its own documentation, implementation details, and available experiment reports.

## Getting Started

Clone the repository:

```bash
git clone https://github.com/Zehraatalay/stylematch.git
cd stylematch
```

Navigate to the project you want to explore and follow its README for setup and execution instructions.

For example:

```bash
cd FinalProject
python -m pip install -r requirements.txt
```

The original datasets and trained model checkpoints are not included in the repository.

## Academic Context

This repository contains coursework and experimental implementations developed during undergraduate Computer Engineering studies at **TOBB University of Economics and Technology**.

The projects focus on understanding computer vision algorithms, implementing machine learning pipelines, and evaluating different approaches to visual recognition and compatibility prediction.