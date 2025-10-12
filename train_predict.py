#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train and predict script for Smart Product Pricing Challenge
"""

import os
import sys
import re
import argparse
import numpy as np
import pandas as pd
import joblib
import warnings
import time
import random
from datetime import datetime
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold

# Suppress warnings
warnings.filterwarnings('ignore')

# Set random seeds
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Optional imports
try:
    import lightgbm as lgb
except ImportError:
    print("LightGBM not installed. Will only use Ridge regression.")
    lgb = None

# Optional torch imports
try:
    import torch
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def setup_paths(base_path=None, output_path=None, dataset_path=None):
    """Set up paths for data and output"""
    if base_path is None:
        # Auto-detect Kaggle environment
        if os.path.exists('/kaggle/input'):
            base_path = '/kaggle/input'
        else:
            base_path = '.'
    
    if output_path is None:
        # Auto-detect Kaggle environment
        if os.path.exists('/kaggle/working'):
            output_path = '/kaggle/working'
        else:
            output_path = '.'
    
    if dataset_path is None:
        # Try to find dataset path
        if os.path.exists(os.path.join(base_path, 'train.csv')):
            dataset_path = base_path
        else:
            # Look in common locations
            possible_paths = [
                os.path.join(base_path, 'dataset'),
                os.path.join(base_path, 'data'),
                os.path.join(base_path, 'smart-product-pricing-challenge'),
                os.path.join(base_path, 'student_resource', 'dataset')
            ]
            for path in possible_paths:
                if os.path.exists(os.path.join(path, 'train.csv')):
                    dataset_path = path
                    break
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Create cache directory
    cache_dir = os.path.join(output_path, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    return {
        'base_path': base_path,
        'output_path': output_path,
        'dataset_path': dataset_path,
        'cache_dir': cache_dir,
        'train_path': os.path.join(dataset_path, 'train.csv') if dataset_path else None,
        'test_path': os.path.join(dataset_path, 'test.csv') if dataset_path else None,
        'output_csv_path': os.path.join(output_path, 'test_out.csv'),
        'metrics_path': os.path.join(output_path, 'oof_metrics.json')
    }


def clean_text(text):
    """Clean text by removing URLs, special characters, and converting to lowercase"""
    if not isinstance(text, str):
        return ""
        
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove excessive punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def extract_ipq(text):
    """Extract Item Pack Quantity (IPQ) from text"""
    if not isinstance(text, str):
        return 1
    
    text = text.lower()
    
    # Look for specific patterns indicating pack quantity
    patterns = [
        r'pack of (\d+)',
        r'(\d+)[-\s]pack',
        r'(\d+)\s*pcs',
        r'(\d+)\s*pieces',
        r'(\d+)\s*count',
        r'(\d+)\s*ct',
        r'(\d+)\s*pk',
        r'set of (\d+)',
        r'(\d+)\s*set',
        r'(\d+)\s*qty',
        r'quantity:\s*(\d+)',
        r'qty:\s*(\d+)',
        r'count:\s*(\d+)',
        r'value:\s*(\d+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                quantity = int(match.group(1))
                return max(1, min(quantity, 100))  # Cap at reasonable values
            except Exception:
                pass
    
    # Check for 'Value: X' pattern which often indicates quantity
    value_match = re.search(r'value:\s*([\d\.]+)', text)
    if value_match:
        try:
            value = float(value_match.group(1))
            if value >= 1 and value <= 100:
                return int(value)
        except Exception:
            pass
            
    # Default to 1 if no pattern is found
    return 1


def extract_brand(text):
    """Extract brand name from text using heuristics"""
    if not isinstance(text, str):
        return "Unknown"
    
    # Look for common brand patterns
    brand_patterns = [
        r'brand:\s*([A-Za-z0-9][A-Za-z0-9\s&\-]+)',
        r'by\s+([A-Z][A-Za-z0-9\s&\-]+)',
        r'from\s+([A-Z][A-Za-z0-9\s&\-]+)',
        r'item name:\s*([A-Z][A-Za-z0-9\s&\-]+)'
    ]
    
    for pattern in brand_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            brand = match.group(1).strip()
            # Limit length and filter out generic terms
            if len(brand) > 1 and len(brand) < 30:
                return brand
    
    # Try to extract first word from Item Name if it's uppercase
    item_name_match = re.search(r'item name:([^,\n]+)', text, re.IGNORECASE)
    if item_name_match:
        item_name = item_name_match.group(1).strip()
        first_word = item_name.split()[0] if item_name.split() else ""
        if first_word and first_word[0].isupper() and len(first_word) > 1:
            return first_word
    
    # Try the first word if it's all caps or first letter is capitalized
    words = text.split()
    if words and len(words[0]) > 1:
        if words[0].isupper() or (words[0][0].isupper() and not words[0].isupper()):
            return words[0]
    
    return "Unknown"


def extract_title(text):
    """Extract title from catalog content"""
    if not isinstance(text, str):
        return ""
    
    # Try to find item name pattern
    item_name_match = re.search(r'item name:(.*?)(?:bullet point|product description|$)', 
                               text, re.IGNORECASE | re.DOTALL)
    
    if item_name_match:
        title = item_name_match.group(1).strip()
        return title
    
    # If no specific pattern, take the first line or first 100 characters
    lines = text.split('\n')
    if lines:
        return lines[0].strip()
    
    return text[:100] if len(text) > 100 else text


def extract_description(text):
    """Extract product description from catalog content"""
    if not isinstance(text, str):
        return ""
    
    # Try to find product description pattern
    desc_match = re.search(r'product description:(.*?)(?:value:|unit:|$)', 
                           text, re.IGNORECASE | re.DOTALL)
    
    if desc_match:
        description = desc_match.group(1).strip()
        return description
    
    # If no specific pattern, take everything after the first line
    lines = text.split('\n')
    if len(lines) > 1:
        return ' '.join(lines[1:]).strip()
    
    return ""


def extract_basic_features(text):
    """Extract basic text features like length, word count, etc."""
    if not isinstance(text, str):
        text = ""
    
    features = {}
    
    # Text length
    features['text_len'] = len(text)
    
    # Number of words
    words = text.split()
    features['num_words'] = len(words)
    
    # Average word length
    if features['num_words'] > 0:
        features['avg_word_len'] = sum(len(word) for word in words) / features['num_words']
    else:
        features['avg_word_len'] = 0
    
    # Number of digits
    features['num_digits'] = sum(c.isdigit() for c in text)
    
    # Number of uppercase letters
    features['num_upper'] = sum(c.isupper() for c in text)
    
    # Number of lowercase letters
    features['num_lower'] = sum(c.islower() for c in text)
    
    return features


def process_catalog_content(df):
    """Process catalog content and extract features"""
    if 'catalog_content' not in df.columns:
        print("Warning: catalog_content not found in dataframe")
        return df
    
    print("Processing catalog content...")
    
    # Create copies of the features to avoid modifying the original
    df_processed = df.copy()
    
    # Extract text components
    df_processed['title'] = df_processed['catalog_content'].apply(extract_title)
    df_processed['description'] = df_processed['catalog_content'].apply(extract_description)
    
    # Clean text fields
    df_processed['clean_title'] = df_processed['title'].apply(clean_text)
    df_processed['clean_description'] = df_processed['description'].apply(clean_text)
    
    # Combine all cleaned text for a single text feature
    df_processed['all_text'] = df_processed['clean_title'] + ' ' + df_processed['clean_description']
    
    # Extract IPQ and brand
    df_processed['ipq'] = df_processed['catalog_content'].apply(extract_ipq)
    df_processed['brand'] = df_processed['catalog_content'].apply(extract_brand)
    
    # Extract basic text features
    basic_features = df_processed['all_text'].apply(extract_basic_features)
    
    # Convert dictionary of features to columns
    for feature in ['text_len', 'num_words', 'avg_word_len', 'num_digits', 
                   'num_upper', 'num_lower']:
        df_processed[feature] = basic_features.apply(lambda x: x.get(feature, 0))
    
    return df_processed


def encode_categorical_features(train_df, test_df, categorical_cols=['brand']):
    """Encode categorical features using label encoding with Unknown handling"""
    encoders = {}
    train_df_encoded = train_df.copy()
    test_df_encoded = test_df.copy()
    
    for col in categorical_cols:
        if col in train_df.columns and col in test_df.columns:
            print(f"Encoding {col}...")
            
            # Initialize LabelEncoder
            encoder = LabelEncoder()
            
            # Get all unique values from both train and test
            all_values = pd.concat([
                train_df[col].fillna('Unknown'),
                test_df[col].fillna('Unknown')
            ]).unique()
            
            # Make sure 'Unknown' is in the values
            if 'Unknown' not in all_values:
                all_values = np.append(all_values, 'Unknown')
                
            # Fit encoder on all values
            encoder.fit(all_values)
            
            # Transform train and test data
            train_df_encoded[f'{col}_encoded'] = encoder.transform(train_df[col].fillna('Unknown'))
            test_df_encoded[f'{col}_encoded'] = encoder.transform(test_df[col].fillna('Unknown'))
            
            # Store encoder for later use
            encoders[col] = encoder
    
    return train_df_encoded, test_df_encoded, encoders


def generate_tfidf_svd_features(train_df, test_df, text_col='all_text', cache_dir=None):
    """Generate TF-IDF features and apply SVD dimensionality reduction"""
    
    if cache_dir:
        tfidf_cache_path = os.path.join(cache_dir, 'tfidf_vectorizer.pkl')
        svd_cache_path = os.path.join(cache_dir, 'tfidf_svd.pkl')
        
        # Check if cached files exist
        if os.path.exists(tfidf_cache_path) and os.path.exists(svd_cache_path):
            print("Loading TF-IDF and SVD models from cache...")
            vectorizer = joblib.load(tfidf_cache_path)
            svd = joblib.load(svd_cache_path)
        else:
            vectorizer = None
            svd = None
    else:
        vectorizer = None
        svd = None
    
    if vectorizer is None:
        print("Generating TF-IDF features...")
        
        # Configure TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=40000,  # Limit vocabulary size
            min_df=3,            # Minimum document frequency
            max_df=0.95,         # Maximum document frequency
            ngram_range=(1, 2),  # Unigrams and bigrams
            lowercase=True,
            strip_accents='unicode',
            analyzer='word',
            token_pattern=r'\w{1,}'  # Match words of at least length 1
        )
        
        # Fit on training data
        train_text = train_df[text_col].fillna('').values
        vectorizer.fit(train_text)
        
        # Cache the vectorizer
        if cache_dir:
            joblib.dump(vectorizer, tfidf_cache_path)
    
    # Transform train and test data
    train_text = train_df[text_col].fillna('').values
    test_text = test_df[text_col].fillna('').values
    
    print("Transforming text data with TF-IDF...")
    train_tfidf = vectorizer.transform(train_text)
    test_tfidf = vectorizer.transform(test_text)
    
    print(f"TF-IDF features shape - Train: {train_tfidf.shape}, Test: {test_tfidf.shape}")
    
    # Apply SVD for dimensionality reduction
    if svd is None:
        n_components = min(256, min(train_tfidf.shape[0], train_tfidf.shape[1]) - 1)
        print(f"Applying SVD to reduce dimensions to {n_components}...")
        
        svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_SEED)
        svd.fit(train_tfidf)
        
        # Cache the SVD model
        if cache_dir:
            joblib.dump(svd, svd_cache_path)
    
    # Transform TF-IDF with SVD
    train_tfidf_svd = svd.transform(train_tfidf)
    test_tfidf_svd = svd.transform(test_tfidf)
    
    print(f"SVD features shape - Train: {train_tfidf_svd.shape}, Test: {test_tfidf_svd.shape}")
    
    # Create feature names
    tfidf_svd_feature_names = [f'tfidf_svd_{i}' for i in range(train_tfidf_svd.shape[1])]
    
    # Convert to DataFrame
    train_tfidf_svd_df = pd.DataFrame(
        train_tfidf_svd, 
        columns=tfidf_svd_feature_names,
        index=train_df.index
    )
    
    test_tfidf_svd_df = pd.DataFrame(
        test_tfidf_svd,
        columns=tfidf_svd_feature_names,
        index=test_df.index
    )
    
    return train_tfidf_svd_df, test_tfidf_svd_df, vectorizer, svd


def prepare_features_for_modeling(train_df, test_df, tfidf_svd_df_train, tfidf_svd_df_test):
    """Combine all features for model training"""
    
    # Start with the numerical features
    numerical_features = [
        'ipq', 'text_len', 'num_words', 'avg_word_len', 'num_digits',
        'num_upper', 'num_lower'
    ]
    
    # Add encoded categorical features
    categorical_features = ['brand_encoded']
    
    # Combine all tabular features
    tabular_features = numerical_features + categorical_features
    
    # Select only features that exist in both train and test
    existing_tabular_features = [f for f in tabular_features 
                               if f in train_df.columns and f in test_df.columns]
    
    print(f"Using {len(existing_tabular_features)} tabular features")
    
    # Start with tabular features
    train_features = train_df[existing_tabular_features].copy()
    test_features = test_df[existing_tabular_features].copy()
    
    # Add TF-IDF SVD features
    print("Adding TF-IDF SVD features...")
    train_features = pd.concat([train_features, tfidf_svd_df_train], axis=1)
    test_features = pd.concat([test_features, tfidf_svd_df_test], axis=1)
    
    print(f"Final feature shapes - Train: {train_features.shape}, Test: {test_features.shape}")
    
    return train_features, test_features


def smape(y_true, y_pred):
    """Calculate Symmetric Mean Absolute Percentage Error"""
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Ensure no zeros (to avoid division by zero)
    y_true = np.maximum(y_true, 0.01)
    y_pred = np.maximum(y_pred, 0.01)
    
    # Calculate SMAPE
    return 100/len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))


def create_strat_bins(y, n_bins=10):
    """Create bins for stratified cross-validation"""
    return pd.qcut(y, n_bins, labels=False, duplicates='drop')


def setup_cross_validation(X, y, n_splits=5, n_bins=10, random_state=RANDOM_SEED):
    """Set up stratified K-fold cross-validation"""
    # Create bins for stratification
    bins = create_strat_bins(y, n_bins)
    
    # Set up K-fold cross-validation
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Generate fold indices
    fold_indices = []
    for train_idx, valid_idx in kf.split(X, bins):
        fold_indices.append((train_idx, valid_idx))
    
    return fold_indices


def train_model_with_cv(X, y, model_class, model_params, folds, model_name='model', cache_dir=None):
    """Train model with cross-validation"""
    # Initialize arrays for OOF predictions
    oof_preds = np.zeros(len(X))
    fold_scores = []
    models = []
    
    print(f"\nTraining {model_name}")
    
    # Loop through folds
    for fold, (train_idx, valid_idx) in enumerate(folds):
        print(f"Fold {fold+1}/{len(folds)}")
        
        # Split data
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        
        # Initialize and fit model
        model = model_class(**model_params)
        
        # Special handling for LightGBM
        if lgb and isinstance(model, lgb.LGBMRegressor):
            try:
                # Try the modern API first
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_valid, y_valid)],
                    eval_metric='rmse'
                )
            except Exception as e:
                print(f"Falling back to simpler LightGBM fit due to: {e}")
                # Simple fit without validation
                model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train)
        
        # Make predictions on validation set
        valid_preds = model.predict(X_valid)
        
        # Store OOF predictions
        oof_preds[valid_idx] = valid_preds
        
        # Transform predictions back to original scale
        valid_preds_original = np.expm1(valid_preds)
        y_valid_original = np.expm1(y_valid)
        
        # Calculate SMAPE
        fold_smape = smape(y_valid_original, valid_preds_original)
        fold_scores.append(fold_smape)
        
        print(f"Fold {fold+1} SMAPE: {fold_smape:.4f}")
        
        # Store model
        models.append(model)
    
    # Calculate overall score
    mean_score = np.mean(fold_scores)
    print(f"Mean SMAPE across {len(folds)} folds: {mean_score:.4f}")
    
    # Save models
    if cache_dir:
        model_path = os.path.join(cache_dir, f"{model_name}_models.pkl")
        joblib.dump(models, model_path)
    
    return {
        'models': models,
        'oof_preds': oof_preds,
        'fold_scores': fold_scores,
        'mean_score': mean_score
    }


def generate_predictions(models, X_test):
    """Generate predictions using multiple fold models"""
    # Get predictions from each fold model
    test_preds_list = []
    for model in models:
        test_preds_fold = model.predict(X_test)
        test_preds_list.append(test_preds_fold)
    
    # Average predictions across folds
    test_preds = np.mean(test_preds_list, axis=0)
    
    return test_preds


def generate_submission(test_df, test_preds, output_path):
    """Generate submission file with predictions"""
    # Convert log predictions back to original scale
    test_preds_original = np.expm1(test_preds)
    
    # Clip to reasonable range (minimum 0.01)
    test_preds_clipped = np.maximum(test_preds_original, 0.01)
    
    # Create submission DataFrame
    submission = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': test_preds_clipped
    })
    
    # Save to CSV
    print(f"Saving submission to {output_path}")
    submission.to_csv(output_path, index=False)
    
    return submission


def main(args):
    """Main function for training and prediction"""
    print("Starting Smart Product Pricing pipeline...")
    start_time = time.time()
    
    # Set up paths
    paths = setup_paths(args.base_path, args.output_path, args.dataset_path)
    print(f"Using dataset path: {paths['dataset_path']}")
    print(f"Output path: {paths['output_path']}")
    
    # Load data
    print("\nLoading data...")
    try:
        train = pd.read_csv(paths['train_path'])
        test = pd.read_csv(paths['test_path'])
        print(f"Train data shape: {train.shape}")
        print(f"Test data shape: {test.shape}")
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Please ensure the dataset files are in the correct location.")
        return 1
    
    # Process catalog content
    print("\nProcessing catalog content...")
    train_processed = process_catalog_content(train)
    test_processed = process_catalog_content(test)
    
    # Encode categorical features
    print("\nEncoding categorical features...")
    train_encoded, test_encoded, encoders = encode_categorical_features(
        train_processed, test_processed, categorical_cols=['brand']
    )
    
    # Generate TF-IDF SVD features
    print("\nGenerating TF-IDF and SVD features...")
    train_tfidf_svd_df, test_tfidf_svd_df, tfidf_vectorizer, tfidf_svd = generate_tfidf_svd_features(
        train_encoded, test_encoded, text_col='all_text', cache_dir=paths['cache_dir']
    )
    
    # Prepare features for modeling
    print("\nPreparing features for modeling...")
    train_features, test_features = prepare_features_for_modeling(
        train_encoded, test_encoded, 
        tfidf_svd_df_train=train_tfidf_svd_df, 
        tfidf_svd_df_test=test_tfidf_svd_df
    )
    
    # Prepare target variable (log-transformed price)
    print("\nPreparing target variable...")
    train_encoded['log_price'] = np.log1p(train_encoded['price'])
    
    # Handle outliers in the target variable
    upper_threshold = np.percentile(train_encoded['log_price'], 99.9)
    train_encoded['log_price_capped'] = np.minimum(train_encoded['log_price'], upper_threshold)
    
    # Use capped version for training
    y = train_encoded['log_price_capped']
    
    # Set up cross-validation
    print("\nSetting up cross-validation...")
    cv_folds = setup_cross_validation(train_features, y, n_splits=5, n_bins=10)
    
    # Define models to train
    if args.model == 'lightgbm' and lgb:
        print("\nTraining LightGBM model...")
        model_result = train_model_with_cv(
            train_features, y, 
            lgb.LGBMRegressor,
            {
                'n_estimators': 1000,
                'learning_rate': 0.05,
                'num_leaves': 31,
                'colsample_bytree': 0.8,
                'subsample': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'n_jobs': -1,
                'random_state': RANDOM_SEED
            },
            cv_folds,
            model_name='lightgbm',
            cache_dir=paths['cache_dir']
        )
    elif args.model == 'ridge':
        print("\nTraining Ridge model...")
        model_result = train_model_with_cv(
            train_features, y,
            Ridge,
            {'alpha': 1.0, 'random_state': RANDOM_SEED},
            cv_folds,
            model_name='ridge',
            cache_dir=paths['cache_dir']
        )
    else:
        # Train both models
        ridge_result = None
        lgb_result = None
        
        print("\nTraining Ridge model...")
        ridge_result = train_model_with_cv(
            train_features, y,
            Ridge,
            {'alpha': 1.0, 'random_state': RANDOM_SEED},
            cv_folds,
            model_name='ridge',
            cache_dir=paths['cache_dir']
        )
        
        if lgb:
            print("\nTraining LightGBM model...")
            lgb_result = train_model_with_cv(
                train_features, y, 
                lgb.LGBMRegressor,
                {
                    'n_estimators': 1000,
                    'learning_rate': 0.05,
                    'num_leaves': 31,
                    'colsample_bytree': 0.8,
                    'subsample': 0.8,
                    'reg_alpha': 0.1,
                    'reg_lambda': 0.1,
                    'n_jobs': -1,
                    'random_state': RANDOM_SEED
                },
                cv_folds,
                model_name='lightgbm',
                cache_dir=paths['cache_dir']
            )
        
        # Use the better performing model
        if lgb_result and (ridge_result['mean_score'] > lgb_result['mean_score']):
            print("LightGBM model performed better. Using LightGBM for predictions.")
            model_result = lgb_result
        else:
            print("Using Ridge model for predictions.")
            model_result = ridge_result
    
    # Generate predictions
    print("\nGenerating predictions...")
    test_preds = generate_predictions(model_result['models'], test_features)
    
    # Generate submission
    generate_submission(test_encoded, test_preds, paths['output_csv_path'])
    
    # Save metrics
    metrics = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model': {
            'name': args.model if args.model != 'both' else 'best_of_both',
            'mean_smape': model_result['mean_score'],
            'fold_smapes': model_result['fold_scores']
        },
        'runtime_seconds': time.time() - start_time
    }
    
    # Save to JSON
    with open(paths['metrics_path'], 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nSaved metrics to {paths['metrics_path']}")
    
    # Print final performance
    print("\n----- Final Performance -----")
    print(f"Mean SMAPE: {model_result['mean_score']:.4f}")
    print("Per-fold SMAPE:")
    for i, score in enumerate(model_result['fold_scores']):
        print(f"Fold {i+1}: {score:.4f}")
    
    # Print submission file path
    print(f"\nSubmission file: {paths['output_csv_path']}")
    print(f"Runtime: {time.time() - start_time:.2f} seconds")
    
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and predict for Smart Product Pricing Challenge")
    parser.add_argument("--base-path", type=str, default=None, help="Base path for input files")
    parser.add_argument("--output-path", type=str, default=None, help="Path for output files")
    parser.add_argument("--dataset-path", type=str, default=None, help="Path to dataset directory")
    parser.add_argument("--model", type=str, default="both", choices=["ridge", "lightgbm", "both"], 
                      help="Model to train (ridge, lightgbm, or both)")
    
    args = parser.parse_args()
    sys.exit(main(args))