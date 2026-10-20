# StyleMatch — Fashion Item Compatibility Prediction

A computer vision project comparing **two CNN-based approaches** to predict whether two fashion items are compatible using the [Polyvore Outfits dataset](https://huggingface.co/datasets/mvasil/polyvore-outfits). Given two product images, the models output a single compatibility logit for binary classification.

> **Implementation status:** The academic report documents completed experiments, but the current repository snapshot of `model_scratch.py` is incomplete: it contains `SqueezeExcite`, `ResidualSEBlock`, and `MultiScaleStem`, but not the full scratch-model class referenced by the training pipeline. Consequently, the scratch training commands below cannot be assumed to run successfully from this checkout. Reported metrics are historical experimental results, **not reproduced from the current snapshot**.

## Approaches

### 1. Scratch early-fusion CNN

The reported scratch model concatenates the two RGB images into a **6-channel tensor** before feature extraction. Its design combines a parallel 3×3/5×5 convolutional stem, residual blocks with squeeze-and-excitation channel attention, pooling, and a fully connected compatibility classifier. It is trained without pretrained weights.

**Repository caveat:** `model_scratch.py` currently ends after the `MultiScaleStem` definition. The complete classifier implementation described in the report is not present in this version.

### 2. ResNet-18 Siamese transfer learning

Two images pass through a **shared ImageNet-pretrained ResNet-18** encoder, followed by a projection head. Embeddings are combined with symmetric pairwise features:

```text
Image A -> shared ResNet-18 -> projection -> ea --+
                                              |--> [|ea-eb|, ea*eb, ea+eb] -> MLP -> logit
Image B -> shared ResNet-18 -> projection -> eb --+
```

Training has two stages: first train the projection/comparison heads with the backbone frozen; then fine-tune selected ResNet layers with a lower backbone learning rate.

## Dataset and leakage prevention

The project starts from Polyvore Outfits' disjoint split and builds stricter **item-disjoint** train/validation/test partitions to prevent the same product from appearing across splits. Positive pairs contain items from the same outfit and different categories. Negative pairs keep the first item fixed and substitute a same-category item from a different outfit (**category-matched negative sampling**, not online hard-negative mining). Each split has equal numbers of positive and negative pairs.

| Split | Outfits | Unique items | Pairs | Positive | Negative |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 11,379 | 51,042 | 86,608 | 43,304 | 43,304 |
| Validation | 2,954 | 14,457 | 22,598 | 11,299 | 11,299 |
| Test | 15,145 | 70,035 | 111,462 | 55,731 | 55,731 |

These counts and zero cross-split item overlap are reported in the project report. Input images are resized to **224×224**. The scratch pipeline uses normalization to [-1, 1]; transfer learning uses ImageNet mean/std normalization. Horizontal flipping is used for training augmentation.

## Training and evaluation

- **Loss:** `BCEWithLogitsLoss`; probabilities are obtained using sigmoid for evaluation.
- **Optimizers:** Adam and AdamW.
- **Model selection:** validation ROC-AUC, with checkpointing and early stopping.
- **Metrics:** loss, accuracy, precision, recall, F1, and ROC-AUC; classification threshold is 0.5.
- **Tuning subset:** 20,000 training pairs and 5,000 validation pairs; final experiments used all available pairs.

### Reported final test results

| Model | Best validation ROC-AUC | Test loss | Accuracy | Precision | Recall | F1 | Test ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Scratch early-fusion CNN | 0.6844 | 0.6471 | **0.6202** | **0.6372** | 0.5583 | 0.5951 | **0.6651** |
| ResNet-18 Siamese | 0.6459 | 0.6646 | 0.5965 | 0.5913 | **0.6246** | **0.6075** | 0.6346 |

**Interpretation:** The scratch approach achieved higher test ROC-AUC, accuracy, and precision, while the Siamese approach achieved higher recall and F1. The comparison changes **both feature extractor and fusion strategy**, so the performance gap cannot be attributed solely to pretraining.

### Selected final configurations (from report)

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
| `dataset.py` | Load parquet image bytes, preprocess and augment pairs |
| `model_scratch.py` | Partial scratch-model building blocks (see caveat above) |
| `model_transfer.py` | Shared-backbone Siamese ResNet-18 model |
| `train_scratch.py`, `train_transfer.py` | Training entry points |
| `train_utils.py` | Evaluation, checkpoints, and early stopping |
| `tune.py` | Configured hyperparameter experiments |
| `inspect_dataset.py`, `smoke_test.py` | Dataset checks and pipeline smoke tests |
| `demo.py` | Inference demonstration |

## Setup and running

The code was developed with **Kaggle-style paths**. Set `POLYVORE_ROOT` and `STYLEMATCH_OUTPUT_DIR` to match your environment. The dataset and trained weights are **not included** in this repository.

```bash
cd FinalProject
python -m pip install -r requirements.txt
export POLYVORE_ROOT=/path/to/polyvore-outfits
export STYLEMATCH_OUTPUT_DIR=/path/to/outputs
```

The documented processing sequence is:

```bash
python make_strict_disjoint.py
python data_prep.py
python inspect_dataset.py
python smoke_test.py
python tune.py
python train_scratch.py --final --dropout 0.4 --patience 3
python train_transfer.py --final --embed_dim 256 --dropout 0.4 --patience 3
```

**Before running:** confirm the dataset layout and the CLI options for each script. Restore the missing scratch classifier implementation before running `train_scratch.py`, `smoke_test.py`, or tuning jobs that import it. The commands above describe the intended workflow, not a verified end-to-end run on this checkout. The transfer script may download pretrained ResNet-18 weights on first use.

## Limitations and next steps

- Only two product images are considered; outfit context, categories, and text metadata are not modeled directly.
- Compatibility labels are derived from outfit co-occurrence and category-matched substitutions, not explicit human preference ratings.
- The scratch model definition needs to be restored before the repository is fully reproducible.
- Future directions: hard-negative mining, category-aware modeling, validation-based threshold tuning, broader backbone comparisons, and controlled ablations that separate backbone effects from fusion strategy.

## Report

See [`reports/FinalProject-report.pdf`](reports/FinalProject-report.pdf) for the full academic methodology, tuning configurations, architecture diagrams, and experimental results.
