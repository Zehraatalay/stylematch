# StyleMatch — Fashion Item Compatibility Prediction

A computer vision project comparing **two CNN-based approaches** to predict whether two fashion items are compatible using the [Polyvore Outfits dataset](https://huggingface.co/datasets/mvasil/polyvore-outfits). Given two product images, the models output a single compatibility logit for binary classification.

> **Implementation status:** Both model architectures are implemented in the repository. The scratch early-fusion CNN has passed a syntax check and a forward-pass test using synthetic image pairs. The experimental results below are taken from the original academic report and have not been independently reproduced from the current repository snapshot.

## Approaches

### 1. Scratch early-fusion CNN

The scratch model concatenates two RGB images into a **6-channel tensor** before feature extraction. Its architecture combines a parallel 3×3/5×5 convolutional stem, residual blocks with squeeze-and-excitation channel attention, pooling, and a fully connected compatibility classifier.

The model is trained entirely from scratch, without pretrained weights.

**Architecture:**

```text
Image A (3 × 224 × 224) ──┐
                          ├── Channel Concatenation (6 × 224 × 224)
Image B (3 × 224 × 224) ──┘
                                      │
                                      ▼
                           Multi-Scale Convolution
                              (3×3 and 5×5)
                                      │
                                      ▼
                                  MaxPool
                                      │
                                      ▼
                              Residual SE Block
                                      │
                                      ▼
                                  MaxPool
                                      │
                                      ▼
                              Residual SE Block
                                      │
                                      ▼
                                  MaxPool
                                      │
                                      ▼
                              Residual SE Block
                                      │
                                      ▼
                           Global Average Pooling
                                      │
                                      ▼
                            Fully Connected Layers
                                      │
                                      ▼
                              Compatibility Logit
```

The implementation is available in [`model_scratch.py`](model_scratch.py).

### 2. ResNet-18 Siamese transfer learning

Two images pass through a **shared ImageNet-pretrained ResNet-18** encoder, followed by a projection head. Embeddings are combined with symmetric pairwise features:

```text
Image A -> shared ResNet-18 -> projection -> ea --+
                                              |--> [|ea-eb|, ea*eb, ea+eb] -> MLP -> logit
Image B -> shared ResNet-18 -> projection -> eb --+
```

Training has two stages:

1. Train the projection and comparison heads while keeping the backbone frozen.
2. Fine-tune selected ResNet layers using a lower backbone learning rate.

The implementation is available in [`model_transfer.py`](model_transfer.py).

## Dataset and leakage prevention

The project starts from Polyvore Outfits' disjoint split and builds stricter **item-disjoint** train/validation/test partitions to prevent the same product from appearing across splits.

Positive pairs contain items from the same outfit and different categories.

Negative pairs keep the first item fixed and substitute a same-category item from a different outfit. This strategy is referred to as **category-matched negative sampling**, rather than online hard-negative mining.

Each split contains equal numbers of positive and negative pairs.

| Split | Outfits | Unique items | Pairs | Positive | Negative |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 11,379 | 51,042 | 86,608 | 43,304 | 43,304 |
| Validation | 2,954 | 14,457 | 22,598 | 11,299 | 11,299 |
| Test | 15,145 | 70,035 | 111,462 | 55,731 | 55,731 |

These counts and zero cross-split item overlap are reported in the project report.

### Image preprocessing

- Input resolution: **224 × 224**
- Scratch model: normalization to `[-1, 1]`
- Transfer model: ImageNet mean/std normalization
- Training augmentation: random horizontal flipping
- Validation and test: deterministic preprocessing without random augmentation

## Training and evaluation

- **Loss:** `BCEWithLogitsLoss`
- **Optimizers:** Adam and AdamW
- **Model selection:** validation ROC-AUC
- **Training controls:** checkpointing and early stopping
- **Evaluation metrics:** loss, accuracy, precision, recall, F1-score, and ROC-AUC
- **Classification threshold:** 0.5
- **Tuning subset:** 20,000 training pairs and 5,000 validation pairs
- **Final experiments:** all available training and validation pairs

### Reported final test results

| Model | Best validation ROC-AUC | Test loss | Accuracy | Precision | Recall | F1 | Test ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Scratch early-fusion CNN | 0.6844 | 0.6471 | **0.6202** | **0.6372** | 0.5583 | 0.5951 | **0.6651** |
| ResNet-18 Siamese | 0.6459 | 0.6646 | 0.5965 | 0.5913 | **0.6246** | **0.6075** | 0.6346 |

**Interpretation:** The scratch approach achieved higher test ROC-AUC, accuracy, and precision, while the Siamese approach achieved higher recall and F1.

The comparison changes **both the feature extractor and the fusion strategy**, so the performance difference cannot be attributed solely to pretraining.

### Selected final configurations

| Setting | Scratch | Transfer |
| --- | --- | --- |
| Optimizer | AdamW | AdamW |
| Embedding dimension | 128 | 256 |
| Dropout | 0.4 | 0.4 |
| Weight decay | 1e-4 | 1e-4 |
| Batch size | 64 | 64 |
| Learning rate | 3e-4 | Head: 3e-4 / 1e-4; backbone: 1e-5 |
| Training | Up to 8 epochs | 2 head epochs + up to 6 fine-tuning epochs |
| Early stopping patience | 3 | 3 |
| Backbone fine-tuning | Not applicable | ResNet layer4 |

## Project layout

| File | Purpose |
| --- | --- |
| `config.py` | Dataset, split, seed, and output paths |
| `make_strict_disjoint.py` | Generate item-disjoint partitions |
| `data_prep.py` | Construct balanced compatibility pairs |
| `dataset.py` | Load parquet image bytes, preprocess, and augment pairs |
| `model_scratch.py` | Scratch early-fusion CNN with multi-scale convolutions, residual SE blocks, and compatibility classifier |
| `model_transfer.py` | Shared-backbone Siamese ResNet-18 model |
| `train_scratch.py`, `train_transfer.py` | Training entry points |
| `train_utils.py` | Evaluation, checkpoints, and early stopping |
| `tune.py` | Configured hyperparameter experiments |
| `inspect_dataset.py`, `smoke_test.py` | Dataset checks and pipeline smoke tests |
| `demo.py` | Inference demonstration |

## Setup and running

The code was developed with **Kaggle-style paths**.

Set `POLYVORE_ROOT` and `STYLEMATCH_OUTPUT_DIR` to match your environment.

The dataset and trained weights are **not included** in this repository.

### Installation

```bash
cd FinalProject

python -m pip install -r requirements.txt

export POLYVORE_ROOT=/path/to/polyvore-outfits
export STYLEMATCH_OUTPUT_DIR=/path/to/outputs
```

### Data preparation

```bash
python make_strict_disjoint.py
python data_prep.py
python inspect_dataset.py
```

### Pipeline validation

```bash
python smoke_test.py
```

### Hyperparameter tuning

```bash
python tune.py
```

### Final training

**Scratch early-fusion CNN:**

```bash
python train_scratch.py --final --dropout 0.4 --patience 3
```

**ResNet-18 Siamese:**

```bash
python train_transfer.py --final --embed_dim 256 --dropout 0.4 --patience 3
```

**Before running:** Confirm the dataset layout and the CLI options for each script. The scratch model has passed a synthetic forward-pass test, but the complete training and evaluation pipelines have not been revalidated in the current environment. The transfer script may download pretrained ResNet-18 weights on first use.

## Model validation

The scratch early-fusion CNN has been checked using Python compilation and a synthetic forward-pass test.

**Test configuration:**

```python
model = ScratchEarlyFusionNet(
    embed_dim=128,
    dropout=0.4,
    base_channels=24
).eval()

a = torch.randn(2, 3, 224, 224)
b = torch.randn(2, 3, 224, 224)

with torch.no_grad():
    output = model(a, b)

assert output.shape == (2,)
```

**Observed output:**

```text
Output shape: torch.Size([2])
Scratch model forward pass: PASSED
```

This confirms that the model accepts image pairs and produces one compatibility logit per pair. It does not independently verify the reported training results.

## Limitations and next steps

- Only two product images are considered; outfit context, categories, and text metadata are not modeled directly.
- Compatibility labels are derived from outfit co-occurrence and category-matched substitutions, not explicit human preference ratings.
- The reported experimental metrics have not been reproduced from the current repository snapshot.
- The complete training and evaluation pipelines have not been revalidated after restoring the scratch model implementation.
- Future directions include hard-negative mining, category-aware modeling, validation-based threshold tuning, broader backbone comparisons, and controlled ablations separating backbone effects from fusion strategy.

## Report

See [`reports/FinalProject-report.pdf`](reports/FinalProject-report.pdf) for the full academic methodology, tuning configurations, architecture diagrams, and experimental results.