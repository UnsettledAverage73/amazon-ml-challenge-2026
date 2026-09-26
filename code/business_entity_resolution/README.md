# Business Entity Resolution Pipeline

## Overview
This repository provides an end-to-end Machine Learning pipeline for the Amazon ML Challenge 2026: Business Entity Resolution.
Given business entity records across 3 independent data sources with noisy and inconsistent fields, the pipeline resolves which records across sources refer to the same real-world entity.

## Structure
```
code/business_entity_resolution/
├── src/
│   ├── utils.py       # Entity normalization, regex, cleaning
│   ├── blocking.py    # Memory-efficient inverted index blocking
│   ├── features.py    # Pairwise similarity feature extraction
│   ├── train.py       # Model training with LightGBM & F_0.5 optimization
│   └── inference.py   # Test set inference generating matching & candidate TSVs
├── requirements.txt   # Pinned dependencies
└── README.md          # Execution instructions
```

## Setup & Dependencies
```bash
pip install -r requirements.txt
```

## Running the Pipeline

### 1. Training
```bash
python3 src/train.py
```
This trains the LightGBM classifier on candidate pairs generated from the training dataset and saves the model to `models/lgb_model.pkl`.

### 2. Inference & Submission Generation
```bash
python3 src/inference.py
```
This runs inference over the test dataset and outputs:
- `output/matching_results.tsv` (final entity predictions)
- `output/candidate_pairs.tsv` (blocking candidate pool)

### 3. Submission Validation
```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
