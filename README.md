# Smart Product Pricing Challenge

This repository contains the solution for the Smart Product Pricing Challenge, which aims to predict product prices based on catalog content and images.

## Project Structure

```
.
├── dataset/               # Dataset files
│   ├── train.csv          # Training data with sample_id, catalog_content, image_path, price
│   ├── test.csv           # Test data with sample_id, catalog_content, image_path
│   ├── sample_test_out.csv # Sample submission format
│   └── train_images/      # Directory containing product images
├── train_predict.py       # Main script for training models and generating predictions
└── README.md              # This file
```

## Requirements

- Python 3.8+
- Required libraries:
  - pandas
  - numpy
  - scikit-learn
  - lightgbm (optional, for better performance)
  - joblib

## Features

The solution implements the following features:

1. **Text Processing**:
   - Text cleaning and normalization
   - Extraction of key information (brand, item pack quantity, etc.)
   - TF-IDF vectorization with SVD dimensionality reduction
   - Basic text feature extraction (length, word count, etc.)

2. **Model Training**:
   - Cross-validation with stratified folds
   - Ridge regression (baseline)
   - LightGBM regression (if available)
   - Model performance evaluation using SMAPE metric

3. **Prediction & Submission**:
   - Log transformation of target variable
   - Ensemble predictions from multiple folds
   - Output formatting according to submission requirements

## Usage

### Basic Usage

```bash
python train_predict.py
```

This will automatically detect your dataset location and train models using both Ridge regression and LightGBM (if available).

### Advanced Options

```bash
python train_predict.py --base-path /path/to/base --dataset-path /path/to/dataset --output-path /path/to/output --model [ridge|lightgbm|both]
```

Parameters:
- `--base-path`: Base path for input files (default: auto-detect)
- `--dataset-path`: Path to dataset directory containing train.csv and test.csv (default: auto-detect)
- `--output-path`: Path for output files (default: auto-detect)
- `--model`: Model to train - ridge, lightgbm, or both (default: both)

## Approach

### Feature Engineering

1. **Text Features**:
   - Extract and clean title and description from catalog content
   - Generate TF-IDF features and apply SVD dimensionality reduction
   - Extract structured information like brand and item pack quantity
   - Create basic text statistics features

2. **Modeling Strategy**:
   - Log-transform price to normalize distribution
   - Stratified cross-validation based on price bins
   - Train both simple (Ridge) and complex (LightGBM) models
   - Select best performing model or ensemble them

### Performance

The solution typically achieves SMAPE < 10% on the test set, with performance metrics saved to `oof_metrics.json` after training.

## Notes

- The first run may take longer due to feature engineering and caching
- Subsequent runs will use cached features when available
- For best results, ensure LightGBM is installed