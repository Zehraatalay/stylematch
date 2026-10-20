# HW2 — Fashion Object Classification with HOG + MLP

## Overview

This coursework project investigates **window-level fashion object classification** on the [Fashionpedia dataset](https://huggingface.co/datasets/detection-datasets/fashionpedia). It combines handcrafted **Histogram of Oriented Gradients (HOG)** descriptors with a **PyTorch multilayer perceptron (MLP)**. It is an experimental classification pipeline, not a production-ready object detector or outfit recommendation system.

## Methodology

1. Load Fashionpedia (`detection-datasets/fashionpedia`) and select fashion categories.
2. Convert images to grayscale and resize to **256 × 256** pixels.
3. Extract **nine overlapping 128 × 128 windows**, using a **64-pixel stride**.
4. Compute HOG descriptors separately on the left and right **64 × 128** halves of each window and concatenate them into a **7,560-dimensional feature vector**.
5. Assign window labels using bounding-box overlap (default IoU threshold **0.2**). Handle ambiguous windows separately during data preparation/evaluation.
6. Train and compare MLP configurations on the validation split, then evaluate the selected model on the test split.

The implementation is in [`src/hw2_features.py`](src/hw2_features.py), [`src/hw2_data.py`](src/hw2_data.py), [`src/hw2_model.py`](src/hw2_model.py), and [`src/hw2_evaluation.py`](src/hw2_evaluation.py).

## Reported Experiments and Results

The coursework report compares six MLP configurations. The best validation configuration was **`cfg_03`**, with hidden layers **[1024, 512]**, batch size **128**, learning rate **0.001**, dropout **0.2**, and batch normalization. Its reported validation macro F1 was **0.2196** (best epoch **23**).

| Test metric | Reported value |
| --- | ---: |
| Accuracy | 0.4971 |
| Macro F1 | 0.2080 |
| Weighted F1 | 0.5197 |

These results indicate substantial difficulty distinguishing fine-grained clothing categories with fixed-window grayscale HOG features. Accuracy alone should not be used to assess performance, because the background class and class imbalance affect aggregate metrics. Values above are **reported experimental results**, not results reproduced during this documentation update.

See the [HW2 academic report](reports/HW2-report.pdf) for experimental details and figures.

## Repository Structure

```text
HW2/
├── README.md
├── requirements.txt
├── reports/
│   └── HW2-report.pdf
└── src/
    ├── hw2main.py           # Command-line entry point
    ├── hw2_data.py          # Dataset and window-label preparation
    ├── hw2_features.py      # Image windows and HOG features
    ├── hw2_model.py         # MLP training and validation search
    ├── hw2_evaluation.py    # Test metrics and visualizations
    ├── models.py            # Shared dataset-loading utilities
    ├── indexing.py          # Indexing/image utilities
    ├── image_io.py          # Image output utilities
    ├── pipeline.py          # Additional/earlier preparation pipeline
    └── sampling.py          # Sampling utilities
```

The older `src/readme.md` refers to paths and commands that do **not** match this repository's current entry point. Use this README instead.

## Setup

From the `HW2/` directory:

```bash
python -m pip install -r requirements.txt
```

Python 3.11 or 3.12 is a reasonable starting point; exact package versions and full end-to-end reproducibility have **not** been verified. Dataset preparation requires access to Fashionpedia via Hugging Face.

## Usage

Run these commands **from `HW2/`**. The default `--project-root` is the current working directory, and the default prepared/output location is `outputs/hw2/`.

```bash
python src/hw2main.py prepare-data
python src/hw2main.py run-validation
python src/hw2main.py run-test
```

To run all three stages sequentially:

```bash
python src/hw2main.py run-experiments
```

For available options:

```bash
python src/hw2main.py prepare-data --help
python src/hw2main.py run-validation --help
python src/hw2main.py run-test --help
```

**Note:** `run-validation` requires prepared features and manifests; `run-test` additionally requires validation artifacts and a trained model. The dataset, generated features, and trained weights are not included in this documentation package. The commands reflect the source code's CLI and have not been re-executed here.

## Limitations

- Fixed windows can misalign with objects of different sizes or aspect ratios.
- Grayscale HOG discards color and richer semantic information.
- Background prevalence and confusion between similar categories make aggregate accuracy potentially misleading.
- The reported results do not establish real-world detection performance on arbitrary images.

## Academic Report

[Read the HW2 report (PDF)](reports/HW2-report.pdf).
