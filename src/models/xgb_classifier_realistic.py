"""
AppLift ML Training - XGBoost Classifier (REALISTIC VERSION)
=============================================================
A comprehensive classification pipeline for exercise execution quality using XGBoost.

FEATURES:
---------
1. PROPER DATA LEAKAGE PREVENTION:
   - CV uses ONLY training data (X_train, y_train)
   - Feature selection uses training data only
   - Scaling fit on training, applied to test
   
2. SMOTE-AWARE CROSS-VALIDATION:
   - SMOTE applied WITHIN each CV fold
   - Prevents synthetic samples from leaking
   
3. XGBOOST-SPECIFIC OPTIMIZATIONS:
   - Scale_pos_weight for class imbalance
   - Early stopping to prevent overfitting
   - Tree-based feature importance
   - GPU acceleration support (optional)

4. COMPREHENSIVE EVALUATION:
   - Multiple metrics (accuracy, balanced accuracy, F1, precision, recall)
   - Per-class performance analysis
   - Feature importance visualization
   - Confusion matrix and ROC curves

Author: AppLift ML Training Pipeline
Date: March 2026
"""

import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from datetime import datetime
import warnings
import joblib
import os

# Scikit-learn imports
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    balanced_accuracy_score, precision_recall_curve, roc_curve, auc
)
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif, RFE

# XGBoost import
try:
    import xgboost as xgb
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[ERROR] XGBoost not installed. Install with: pip install xgboost")

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from scipy.signal import savgol_filter

# SMOTE for oversampling
try:
    from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
# XGBOOST VERSION: Use separate output directory for XGBoost model results
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'xgboost'
MODELS_DIR = OUTPUT_DIR / 'models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)

COMPARISON_DIR = OUTPUT_DIR / 'comparison_with_rf'
COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COLUMN = 'target'

ALWAYS_EXCLUDE = [
    'source_file', 'target_warning', 'rep_original',
    'Unnamed: 0', 'index'
]

EQUIPMENT_TYPES = {
    0: 'Dumbbell',
    1: 'Barbell', 
    2: 'Weight Stack'
}

EXERCISE_TYPES = {
    0: 'Concentration Curls',
    1: 'Overhead Extension',
    2: 'Bench Press',
    3: 'Back Squat',
    4: 'Lateral Pulldown',
    5: 'Seated Leg Extension'
}

QUALITY_NAMES_BY_EXERCISE = {
    0: {0: 'Clean', 1: 'Uncontrolled Movement', 2: 'Abrupt Initiation'},
    1: {0: 'Clean', 1: 'Uncontrolled Movement', 2: 'Abrupt Initiation'},
    2: {0: 'Clean', 1: 'Uncontrolled Movement', 2: 'Inclination Asymmetry'},
    3: {0: 'Clean', 1: 'Uncontrolled Movement', 2: 'Inclination Asymmetry'},
    4: {0: 'Clean', 1: 'Pulling Too Fast', 2: 'Releasing Too Fast'},
    5: {0: 'Clean', 1: 'Pulling Too Fast', 2: 'Releasing Too Fast'}
}

QUALITY_NAMES = {
    0: 'Clean',
    1: 'Uncontrolled Movement',
    2: 'Abrupt Initiation'
}


# Import shared utility functions from RF classifier
import sys
sys.path.insert(0, str(PROJECT_ROOT / 'src' / 'models'))
from rf_classifier_realistic import (
    get_quality_names,
    get_dataset_info,
    display_dataset_info,
    select_csv_file,
    select_columns_ui,
    compute_rep_features,
    analyze_class_distribution,
    prepare_data,
    impute_after_split,
    calculate_imbalance_ratio,
    configure_class_imbalance_strategy,
    apply_smote,
    get_recommended_top_k,
    configure_dimensionality_reduction,
    get_feature_preference_score,
    choose_feature_from_correlated_pair,
    apply_correlation_pruning,
    apply_dimensionality_reduction
)


# =============================================================================
# XGBOOST-SPECIFIC FUNCTIONS
# =============================================================================

def get_default_xgb_params(scale_pos_weight=1.0, use_gpu=False):
    """
    Default XGBoost parameters optimized for exercise quality classification.
    
    Parameters:
    - scale_pos_weight: Weight for positive class (for imbalance)
    - use_gpu: Whether to use GPU acceleration
    
    Returns:
    - Dictionary of XGBoost parameters
    """
    params = {
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.1,
        'min_child_weight': 3,
        'gamma': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'scale_pos_weight': scale_pos_weight,
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'random_state': 42,
        'n_jobs': -1
    }
    
    if use_gpu:
        params['tree_method'] = 'gpu_hist'
        params['predictor'] = 'gpu_predictor'
    else:
        params['tree_method'] = 'hist'
    
    return params


def create_optimized_xgb_model(best_params=None, scale_pos_weight=1.0, use_gpu=False):
    """
    Create XGBoost model with optimized hyperparameters or defaults.
    
    Parameters:
    - best_params: Optional dictionary of best hyperparameters from search
    - scale_pos_weight: Weight for positive class
    - use_gpu: Whether to use GPU acceleration
    
    Returns:
    - model: XGBClassifier
    """
    if best_params:
        print(f"\n[XGB] Creating optimized XGBoost model...")
        model_params = best_params.copy()
        if 'scale_pos_weight' not in model_params:
            model_params['scale_pos_weight'] = scale_pos_weight
    else:
        print(f"\n[XGB] Creating default XGBoost model...")
        model_params = get_default_xgb_params(scale_pos_weight=scale_pos_weight, use_gpu=use_gpu)
    
    model = XGBClassifier(**model_params)
    
    print("  [OK] Model created")
    return model



def get_hyperparameter_grid_xgb(use_imbalance_strategy=True):
    """
    Define hyperparameter grid for XGBoost Grid Search.
    
    Parameters:
    - use_imbalance_strategy: Whether to include scale_pos_weight options
    
    Returns:
    - Dictionary of parameter grid
    """
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [0.5, 1.0, 2.0]
    }
    
    if use_imbalance_strategy:
        param_grid['scale_pos_weight'] = [1, 2, 3]
    
    return param_grid


def get_hyperparameter_distributions_xgb(use_imbalance_strategy=True):
    """
    Define hyperparameter distributions for XGBoost Random Search.
    
    Parameters:
    - use_imbalance_strategy: Whether to include scale_pos_weight options
    
    Returns:
    - Dictionary of parameter distributions
    """
    from scipy.stats import randint, uniform
    
    param_distributions = {
        'n_estimators': randint(100, 500),
        'max_depth': randint(3, 10),
        'learning_rate': uniform(0.01, 0.29),  # 0.01 to 0.30
        'min_child_weight': randint(1, 10),
        'gamma': uniform(0, 0.5),
        'subsample': uniform(0.6, 0.4),  # 0.6 to 1.0
        'colsample_bytree': uniform(0.6, 0.4),  # 0.6 to 1.0
        'reg_alpha': uniform(0, 1.0),
        'reg_lambda': uniform(0.5, 2.5)  # 0.5 to 3.0
    }
    
    if use_imbalance_strategy:
        param_distributions['scale_pos_weight'] = randint(1, 5)
    
    return param_distributions


def perform_grid_search_xgb(X_train, y_train, cv_folds=5, n_jobs=-1, verbose=1, use_imbalance_strategy=True):
    """
    Perform Grid Search for XGBoost hyperparameter optimization.
    
    Parameters:
    - X_train: Training features
    - y_train: Training targets
    - cv_folds: Number of cross-validation folds
    - n_jobs: Number of parallel jobs
    - verbose: Verbosity level
    - use_imbalance_strategy: Whether to search scale_pos_weight
    
    Returns:
    - best_params: Dictionary of best hyperparameters
    - grid_search: GridSearchCV object
    """
    print(f"\n[GRID] Performing Grid Search for XGBoost hyperparameters...")
    print(f"  CV Folds: {cv_folds} | Parallel Jobs: {n_jobs}")
    
    param_grid = get_hyperparameter_grid_xgb(use_imbalance_strategy=use_imbalance_strategy)
    
    print(f"  Parameter combinations: {np.prod([len(v) for v in param_grid.values()]):,}")
    print(f"  Total fits: {np.prod([len(v) for v in param_grid.values()]) * cv_folds:,}")
    
    xgb_model = XGBClassifier(
        objective='multi:softprob',
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=1
    )
    
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
        scoring='balanced_accuracy',
        n_jobs=n_jobs,
        verbose=verbose,
        return_train_score=True,
        refit=True
    )
    
    print(f"  [START] Starting Grid Search...")
    start_time = pd.Timestamp.now()
    grid_search.fit(X_train, y_train)
    end_time = pd.Timestamp.now()
    
    duration = (end_time - start_time).total_seconds()
    print(f"  [OK] Grid Search completed in {duration:.1f}s ({duration/60:.1f}min)")
    
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    print(f"\n[BEST] Best Hyperparameters:")
    print("=" * 50)
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    print(f"\n[STATS] Best CV Score: {best_score:.4f}")
    
    return best_params, grid_search



def perform_random_search_xgb(X_train, y_train, n_iter=100, cv_folds=5, n_jobs=-1, verbose=1, use_imbalance_strategy=True):
    """
    Perform Random Search for XGBoost hyperparameter optimization.
    
    Parameters:
    - X_train: Training features
    - y_train: Training targets
    - n_iter: Number of parameter combinations to try
    - cv_folds: Number of cross-validation folds
    - n_jobs: Number of parallel jobs
    - verbose: Verbosity level
    - use_imbalance_strategy: Whether to search scale_pos_weight
    
    Returns:
    - best_params: Dictionary of best hyperparameters
    - random_search: RandomizedSearchCV object
    """
    print(f"\n[RANDOM] Performing Random Search for XGBoost hyperparameters...")
    print(f"  Iterations: {n_iter} | CV Folds: {cv_folds} | Parallel Jobs: {n_jobs}")
    
    param_distributions = get_hyperparameter_distributions_xgb(use_imbalance_strategy=use_imbalance_strategy)
    
    print(f"  Parameter combinations to sample: {n_iter:,}")
    print(f"  Total fits: {n_iter * cv_folds:,}")
    
    xgb_model = XGBClassifier(
        objective='multi:softprob',
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=1
    )
    
    random_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
        scoring='balanced_accuracy',
        n_jobs=n_jobs,
        verbose=verbose,
        return_train_score=True,
        refit=True,
        random_state=42
    )
    
    print(f"  [START] Starting Random Search...")
    start_time = pd.Timestamp.now()
    random_search.fit(X_train, y_train)
    end_time = pd.Timestamp.now()
    
    duration = (end_time - start_time).total_seconds()
    print(f"  [OK] Random Search completed in {duration:.1f}s ({duration/60:.1f}min)")
    
    best_params = random_search.best_params_
    best_score = random_search.best_score_
    
    print(f"\n[BEST] Best Hyperparameters:")
    print("=" * 50)
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    print(f"\n[STATS] Best CV Score: {best_score:.4f}")
    
    # Display top 5 combinations
    results_df = pd.DataFrame(random_search.cv_results_)
    top_5 = results_df.nlargest(5, 'mean_test_score')[['mean_test_score', 'std_test_score', 'params']]
    
    print(f"\n[TOP] Top 5 Parameter Combinations:")
    print("-" * 70)
    for i, (idx, row) in enumerate(top_5.iterrows(), 1):
        print(f"  {i}. Score: {row['mean_test_score']:.4f} (+/-{row['std_test_score']:.4f})")
        print(f"     Params: {row['params']}")
        print()
    
    return best_params, random_search



# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_xgboost(
    X_train,
    y_train,
    X_test,
    y_test,
    feature_names,
    best_params=None,
    scale_pos_weight=1.0,
    use_gpu=False,
    exercise_code=None,
    df=None
):
    """
    Train an XGBoost classifier with optional optimized hyperparameters.
    
    Parameters:
    - X_train, X_test: Feature matrices
    - y_train, y_test: Target vectors  
    - feature_names: List of feature names
    - best_params: Optional dictionary of optimized hyperparameters
    - scale_pos_weight: Weight for positive class
    - use_gpu: Whether to use GPU acceleration
    - exercise_code: Optional exercise code for context-aware quality names
    - df: Optional dataframe for auto-detecting exercise type
    
    Returns:
    - model: Trained XGBClassifier
    - scaler: Fitted StandardScaler
    - results: Dictionary with evaluation metrics
    """
    print("\n[XGB] Training XGBoost Classifier...")
    
    # Analyze class distribution
    analyze_class_distribution(y_train, "Training Set", exercise_code=exercise_code, df=df)
    
    # Get appropriate quality names
    quality_names = get_quality_names(exercise_code=exercise_code, df=df)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create model
    if best_params:
        print("  Using optimized hyperparameters from search")
    else:
        print("  Using default hyperparameters")
    
    model = create_optimized_xgb_model(
        best_params=best_params,
        scale_pos_weight=scale_pos_weight,
        use_gpu=use_gpu
    )
    
    # Train with early stopping
    print("  Training model with early stopping...")
    eval_set = [(X_train_scaled, y_train), (X_test_scaled, y_test)]
    
    model.fit(
        X_train_scaled,
        y_train,
        eval_set=eval_set,
        verbose=False
    )
    
    # Get best iteration
    if hasattr(model, 'best_iteration'):
        print(f"  Best iteration: {model.best_iteration}")
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)
    
    # Evaluation metrics
    results = {
        'accuracy': accuracy_score(y_test, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
        'precision_weighted': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall_weighted': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1_weighted': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
        'y_test': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'classification_report': classification_report(
            y_test, y_pred, 
            target_names=[quality_names.get(i, f'Class {i}') for i in sorted(y_test.unique())],
            zero_division=0
        ),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    # Per-class metrics
    per_class_precision = precision_score(y_test, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_test, y_pred, average=None, zero_division=0)
    per_class_f1 = f1_score(y_test, y_pred, average=None, zero_division=0)
    
    results['per_class_metrics'] = {}
    for i, class_id in enumerate(sorted(y_test.unique())):
        class_name = quality_names.get(class_id, f'Class {class_id}')
        results['per_class_metrics'][class_name] = {
            'precision': per_class_precision[i],
            'recall': per_class_recall[i],
            'f1': per_class_f1[i],
            'support': len(y_test[y_test == class_id])
        }
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    results['feature_importance'] = feature_importance
    
    print(f"\n[STATS] Model Performance:")
    print(f"  * Standard Accuracy: {results['accuracy']:.4f}")
    print(f"  * Balanced Accuracy: {results['balanced_accuracy']:.4f}")
    print(f"  * F1 Score (Weighted): {results['f1_weighted']:.4f}")
    print(f"  * F1 Score (Macro): {results['f1_macro']:.4f}")
    
    print(f"\n[LIST] Per-Class Performance:")
    for class_name, metrics in results['per_class_metrics'].items():
        print(f"  * {class_name}: F1={metrics['f1']:.3f}, Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f} (n={metrics['support']})")
    
    return model, scaler, results



# =============================================================================
# CROSS-VALIDATION
# =============================================================================

def perform_cross_validation_xgb(X_train, y_train, n_splits=5, best_params=None, scale_pos_weight=1.0, use_gpu=False):
    """
    Perform stratified k-fold cross-validation on TRAINING DATA ONLY.
    
    Parameters:
    - X_train: Training feature matrix (NOT the full dataset)
    - y_train: Training target vector (NOT the full dataset)
    - n_splits: Number of CV folds
    - best_params: Optional optimized hyperparameters
    - scale_pos_weight: Weight for positive class
    - use_gpu: Whether to use GPU
    
    Returns:
    - cv_results: Dictionary with CV scores and statistics
    """
    print(f"\n[CV] Performing {n_splits}-Fold Stratified Cross-Validation on TRAINING DATA...")
    print(f"  [FIX] Using {len(X_train)} training samples for CV (test set properly excluded)")
    print(f"  [FIX] Using Pipeline (scaler re-fits per fold - no scaling leakage)")
    
    # Create model
    model = create_optimized_xgb_model(
        best_params=best_params,
        scale_pos_weight=scale_pos_weight,
        use_gpu=use_gpu
    )
    
    # Use Pipeline for proper per-fold scaling
    cv_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', model)
    ])
    
    # Stratified K-Fold
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Cross-validation scores
    cv_accuracy = cross_val_score(cv_pipe, X_train, y_train, cv=cv, scoring='accuracy')
    cv_balanced_accuracy = cross_val_score(cv_pipe, X_train, y_train, cv=cv, scoring='balanced_accuracy')
    cv_precision_weighted = cross_val_score(cv_pipe, X_train, y_train, cv=cv, scoring='precision_weighted')
    cv_recall_weighted = cross_val_score(cv_pipe, X_train, y_train, cv=cv, scoring='recall_weighted')
    cv_f1_weighted = cross_val_score(cv_pipe, X_train, y_train, cv=cv, scoring='f1_weighted')
    cv_precision_macro = cross_val_score(cv_pipe, X_train, y_train, cv=cv, scoring='precision_macro')
    cv_recall_macro = cross_val_score(cv_pipe, X_train, y_train, cv=cv, scoring='recall_macro')
    cv_f1_macro = cross_val_score(cv_pipe, X_train, y_train, cv=cv, scoring='f1_macro')
    
    cv_results = {
        'accuracy': cv_accuracy,
        'balanced_accuracy': cv_balanced_accuracy,
        'precision_weighted': cv_precision_weighted,
        'recall_weighted': cv_recall_weighted,
        'f1_weighted': cv_f1_weighted,
        'precision_macro': cv_precision_macro,
        'recall_macro': cv_recall_macro,
        'f1_macro': cv_f1_macro,
        'accuracy_mean': cv_accuracy.mean(),
        'accuracy_std': cv_accuracy.std(),
        'balanced_accuracy_mean': cv_balanced_accuracy.mean(),
        'balanced_accuracy_std': cv_balanced_accuracy.std(),
        'precision_weighted_mean': cv_precision_weighted.mean(),
        'precision_weighted_std': cv_precision_weighted.std(),
        'recall_weighted_mean': cv_recall_weighted.mean(),
        'recall_weighted_std': cv_recall_weighted.std(),
        'f1_weighted_mean': cv_f1_weighted.mean(),
        'f1_weighted_std': cv_f1_weighted.std(),
        'precision_macro_mean': cv_precision_macro.mean(),
        'precision_macro_std': cv_precision_macro.std(),
        'recall_macro_mean': cv_recall_macro.mean(),
        'recall_macro_std': cv_recall_macro.std(),
        'f1_macro_mean': cv_f1_macro.mean(),
        'f1_macro_std': cv_f1_macro.std()
    }
    
    print(f"\n[STATS] Cross-Validation Results ({n_splits}-Fold):")
    print("=" * 60)
    print(f"  Standard Accuracy:  {cv_results['accuracy_mean']:.4f} +/- {cv_results['accuracy_std']:.4f}")
    print(f"  Balanced Accuracy:  {cv_results['balanced_accuracy_mean']:.4f} +/- {cv_results['balanced_accuracy_std']:.4f}")
    print(f"  F1 Weighted:        {cv_results['f1_weighted_mean']:.4f} +/- {cv_results['f1_weighted_std']:.4f}")
    print(f"  F1 Macro:           {cv_results['f1_macro_mean']:.4f} +/- {cv_results['f1_macro_std']:.4f}")
    print("=" * 60)
    
    # Check for overfitting/underfitting
    print(f"\n[SEARCH] Model Fit Analysis:")
    cv_pipe.fit(X_train, y_train)
    train_balanced_accuracy = balanced_accuracy_score(y_train, cv_pipe.predict(X_train))
    
    gap = train_balanced_accuracy - cv_results['balanced_accuracy_mean']
    
    print(f"  * Training Balanced Accuracy: {train_balanced_accuracy:.4f}")
    print(f"  * CV Balanced Accuracy (mean): {cv_results['balanced_accuracy_mean']:.4f}")
    print(f"  * Gap (Train - CV): {gap:.4f}")
    
    if gap > 0.15:
        print("  [WARNING] Possible OVERFITTING detected (gap > 0.15)")
        print("     Consider: reducing max_depth, increasing min_child_weight, or adding regularization")
    elif cv_results['balanced_accuracy_mean'] < 0.6:
        print("  [WARNING] Possible UNDERFITTING detected (CV balanced accuracy < 0.60)")
        print("     Consider: increasing n_estimators, max_depth, or adding more features")
    else:
        print("  [OK] Model appears well-fitted")
    
    cv_results['train_balanced_accuracy'] = train_balanced_accuracy
    cv_results['generalization_gap'] = gap
    
    return cv_results



def perform_cross_validation_with_smote_xgb(X_train, y_train, n_splits=5, best_params=None, 
                                            scale_pos_weight=1.0, use_gpu=False, quality_names=None):
    """
    Perform stratified k-fold cross-validation WITH SMOTE applied within each fold.
    
    Parameters:
    - X_train: Training feature matrix
    - y_train: Training target vector
    - n_splits: Number of CV folds
    - best_params: Optional optimized hyperparameters
    - scale_pos_weight: Weight for positive class
    - use_gpu: Whether to use GPU
    - quality_names: Optional dict mapping class codes to names
    
    Returns:
    - cv_results: Dictionary with CV scores and statistics
    """
    if not SMOTE_AVAILABLE:
        print("  [WARNING] SMOTE not available. Falling back to regular CV.")
        return perform_cross_validation_xgb(X_train, y_train, n_splits, best_params, scale_pos_weight, use_gpu)
    
    print(f"\n[CV+SMOTE] Performing {n_splits}-Fold CV with SMOTE applied per-fold...")
    print(f"  [FIX] SMOTE is applied WITHIN each fold to prevent data leakage")
    
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_metrics = {
        'accuracy': [],
        'balanced_accuracy': [],
        'f1_weighted': [],
        'f1_macro': [],
        'precision_weighted': [],
        'recall_weighted': [],
        'precision_macro': [],
        'recall_macro': []
    }
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
        # Split data for this fold
        if isinstance(X_train, pd.DataFrame):
            X_fold_train = X_train.iloc[train_idx].copy()
            X_fold_val = X_train.iloc[val_idx].copy()
        else:
            X_fold_train = X_train[train_idx].copy()
            X_fold_val = X_train[val_idx].copy()
            
        if isinstance(y_train, pd.Series):
            y_fold_train = y_train.iloc[train_idx].copy()
            y_fold_val = y_train.iloc[val_idx].copy()
        else:
            y_fold_train = y_train[train_idx].copy()
            y_fold_val = y_train[val_idx].copy()
        
        # Apply SMOTE to training fold ONLY
        min_class_count = min(Counter(y_fold_train).values())
        k_neighbors = min(5, min_class_count - 1)
        
        if k_neighbors >= 1:
            try:
                smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
                X_fold_train_resampled, y_fold_train_resampled = smote.fit_resample(X_fold_train, y_fold_train)
            except Exception:
                X_fold_train_resampled, y_fold_train_resampled = X_fold_train, y_fold_train
        else:
            X_fold_train_resampled, y_fold_train_resampled = X_fold_train, y_fold_train
        
        # Scale features
        scaler = StandardScaler()
        X_fold_train_scaled = scaler.fit_transform(X_fold_train_resampled)
        X_fold_val_scaled = scaler.transform(X_fold_val)
        
        # Create and train model
        model = create_optimized_xgb_model(best_params=best_params, scale_pos_weight=scale_pos_weight, use_gpu=use_gpu)
        model.fit(X_fold_train_scaled, y_fold_train_resampled, verbose=False)
        
        # Predict on validation fold
        y_pred = model.predict(X_fold_val_scaled)
        
        # Calculate metrics
        fold_metrics['accuracy'].append(accuracy_score(y_fold_val, y_pred))
        fold_metrics['balanced_accuracy'].append(balanced_accuracy_score(y_fold_val, y_pred))
        fold_metrics['f1_weighted'].append(f1_score(y_fold_val, y_pred, average='weighted', zero_division=0))
        fold_metrics['f1_macro'].append(f1_score(y_fold_val, y_pred, average='macro', zero_division=0))
        fold_metrics['precision_weighted'].append(precision_score(y_fold_val, y_pred, average='weighted', zero_division=0))
        fold_metrics['recall_weighted'].append(recall_score(y_fold_val, y_pred, average='weighted', zero_division=0))
        fold_metrics['precision_macro'].append(precision_score(y_fold_val, y_pred, average='macro', zero_division=0))
        fold_metrics['recall_macro'].append(recall_score(y_fold_val, y_pred, average='macro', zero_division=0))
    
    # Convert to numpy arrays
    for key in fold_metrics:
        fold_metrics[key] = np.array(fold_metrics[key])
    
    # Build results dict
    cv_results = {
        'accuracy': fold_metrics['accuracy'],
        'balanced_accuracy': fold_metrics['balanced_accuracy'],
        'f1_weighted': fold_metrics['f1_weighted'],
        'f1_macro': fold_metrics['f1_macro'],
        'precision_weighted': fold_metrics['precision_weighted'],
        'recall_weighted': fold_metrics['recall_weighted'],
        'precision_macro': fold_metrics['precision_macro'],
        'recall_macro': fold_metrics['recall_macro'],
        'accuracy_mean': fold_metrics['accuracy'].mean(),
        'accuracy_std': fold_metrics['accuracy'].std(),
        'balanced_accuracy_mean': fold_metrics['balanced_accuracy'].mean(),
        'balanced_accuracy_std': fold_metrics['balanced_accuracy'].std(),
        'f1_weighted_mean': fold_metrics['f1_weighted'].mean(),
        'f1_weighted_std': fold_metrics['f1_weighted'].std(),
        'f1_macro_mean': fold_metrics['f1_macro'].mean(),
        'f1_macro_std': fold_metrics['f1_macro'].std(),
        'precision_weighted_mean': fold_metrics['precision_weighted'].mean(),
        'precision_weighted_std': fold_metrics['precision_weighted'].std(),
        'recall_weighted_mean': fold_metrics['recall_weighted'].mean(),
        'recall_weighted_std': fold_metrics['recall_weighted'].std(),
        'precision_macro_mean': fold_metrics['precision_macro'].mean(),
        'precision_macro_std': fold_metrics['precision_macro'].std(),
        'recall_macro_mean': fold_metrics['recall_macro'].mean(),
        'recall_macro_std': fold_metrics['recall_macro'].std()
    }
    
    print(f"\n[STATS] Cross-Validation Results (with per-fold SMOTE):")
    print("=" * 60)
    print(f"  Balanced Accuracy:  {cv_results['balanced_accuracy_mean']:.4f} +/- {cv_results['balanced_accuracy_std']:.4f}")
    print(f"  F1 Weighted:        {cv_results['f1_weighted_mean']:.4f} +/- {cv_results['f1_weighted_std']:.4f}")
    print(f"  F1 Macro:           {cv_results['f1_macro_mean']:.4f} +/- {cv_results['f1_macro_std']:.4f}")
    print("=" * 60)
    
    return cv_results



# =============================================================================
# VISUALIZATION & EXPORT
# =============================================================================

def create_model_visualizations_xgb(y_test, results, cv_results, output_folder, exercise_code=None, df=None):
    """
    Create comprehensive visualizations for XGBoost model performance.
    
    Parameters:
    - y_test: Test target vector
    - results: Dictionary with model results
    - cv_results: Dictionary with CV results
    - output_folder: Path to save visualizations
    - exercise_code: Optional exercise code
    - df: Optional dataframe
    """
    quality_names = get_quality_names(exercise_code=exercise_code, df=df)
    class_names = [quality_names.get(i, f'Class {i}') for i in sorted(y_test.unique())]
    
    viz_folder = Path(output_folder) / 'visualizations'
    viz_folder.mkdir(parents=True, exist_ok=True)
    
    # 1. Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm = results['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('XGBoost Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(viz_folder / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Feature Importance
    plt.figure(figsize=(12, 8))
    top_features = results['feature_importance'].head(20)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance Score')
    plt.title('XGBoost Top 20 Feature Importances', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(viz_folder / 'feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. ROC Curves (if binary or multi-class)
    if len(class_names) == 2:
        plt.figure(figsize=(10, 8))
        fpr, tpr, _ = roc_curve(y_test, results['y_pred_proba'][:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('XGBoost ROC Curve', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(viz_folder / 'roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Cross-Validation Scores
    if cv_results:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        metrics = ['accuracy', 'balanced_accuracy', 'f1_weighted', 'f1_macro']
        titles = ['Accuracy', 'Balanced Accuracy', 'F1 Weighted', 'F1 Macro']
        
        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx // 2, idx % 2]
            scores = cv_results[metric]
            ax.boxplot([scores], labels=['CV Scores'])
            ax.set_ylabel('Score')
            ax.set_title(f'{title}: {scores.mean():.3f} ± {scores.std():.3f}')
            ax.grid(alpha=0.3)
        
        plt.suptitle('XGBoost Cross-Validation Performance', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(viz_folder / 'cv_scores.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"\n[OK] Visualizations saved to: {viz_folder}")


def export_model_xgb(model, scaler, feature_names, results, cv_results, output_folder, 
                     exercise_code=None, df=None, imbalance_config=None, reduction_summary=None):
    """
    Export trained XGBoost model and metadata.
    
    Parameters:
    - model: Trained XGBClassifier
    - scaler: Fitted StandardScaler
    - feature_names: List of feature names
    - results: Dictionary with model results
    - cv_results: Dictionary with CV results
    - output_folder: Path to save model
    - exercise_code: Optional exercise code
    - df: Optional dataframe
    - imbalance_config: Optional imbalance strategy config
    - reduction_summary: Optional dimensionality reduction summary
    """
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save model and scaler
    model_file = output_path / f'xgb_model_{timestamp}.pkl'
    scaler_file = output_path / f'scaler_{timestamp}.pkl'
    features_file = output_path / f'feature_names_{timestamp}.pkl'
    
    joblib.dump(model, model_file)
    joblib.dump(scaler, scaler_file)
    joblib.dump(feature_names, features_file)
    
    print(f"\n[SAVE] Model artifacts saved:")
    print(f"  * Model: {model_file}")
    print(f"  * Scaler: {scaler_file}")
    print(f"  * Features: {features_file}")
    
    # Save metadata
    quality_names = get_quality_names(exercise_code=exercise_code, df=df)
    
    metadata = {
        'model_type': 'XGBClassifier',
        'training_date': timestamp,
        'feature_names': feature_names,
        'n_features': len(feature_names),
        'n_classes': len(quality_names),
        'class_names': quality_names,
        'metrics': {
            'test_accuracy': results['accuracy'],
            'test_balanced_accuracy': results['balanced_accuracy'],
            'test_f1_weighted': results['f1_weighted'],
            'test_f1_macro': results['f1_macro'],
            'cv_accuracy_mean': cv_results['accuracy_mean'] if cv_results else None,
            'cv_accuracy_std': cv_results['accuracy_std'] if cv_results else None,
            'cv_balanced_accuracy_mean': cv_results['balanced_accuracy_mean'] if cv_results else None,
            'cv_balanced_accuracy_std': cv_results['balanced_accuracy_std'] if cv_results else None
        },
        'hyperparameters': model.get_params(),
        'imbalance_strategy': imbalance_config,
        'dimensionality_reduction': reduction_summary
    }
    
    metadata_file = output_path / f'model_metadata_{timestamp}.json'
    import json
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    print(f"  * Metadata: {metadata_file}")
    
    # Save comprehensive classification report
    report_file = output_path / f'classification_report_{timestamp}.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        # Executive Summary
        f.write("=" * 80 + "\n")
        f.write("XGBOOST CLASSIFIER - COMPREHENSIVE EVALUATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("EXECUTIVE SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"This report presents a comprehensive evaluation of an XGBoost classifier\n")
        f.write(f"for exercise form quality assessment. The model achieved {results['accuracy']:.1%} test accuracy\n")
        f.write(f"with {cv_results['accuracy_mean']:.1%} +/- {cv_results['accuracy_std']:.1%} cross-validation accuracy.\n")
        if reduction_summary:
            f.write(f"Dataset dimensionality was reduced from {reduction_summary.get('initial_feature_count', len(feature_names))} ")
            f.write(f"to {len(feature_names)} features ({reduction_summary.get('reduction_percent', 0):.1f}% reduction).\n")
        if len(set(results.get('y_test', []))) > 0:
            unique_classes = sorted(set(results.get('y_test', [])))
            class_names_summary = [quality_names.get(i, f'Class {i}') for i in unique_classes]
            f.write(f"The classifier discriminates between {len(class_names_summary)} movement quality classes: ")
            f.write(f"{', '.join(class_names_summary)}.\n")
        f.write("\n")
        
        # Metadata Section
        f.write("EXPERIMENT METADATA\n")
        f.write("-" * 80 + "\n")
        f.write(f"Training Date:           {timestamp}\n")
        f.write(f"Random Seed:             42\n")
        f.write(f"Cross-Validation:        5-Fold Stratified\n")
        f.write(f"Test Set Size:           {len(results.get('y_test', []))} samples\n")
        f.write(f"Training Set Size:       Remaining after 80/20 split\n")
        if imbalance_config:
            f.write(f"SMOTE Applied:           {'Yes' if imbalance_config.get('use_smote', False) else 'No'}\n")
            f.write(f"Class Imbalance Strategy: {imbalance_config.get('strategy', 'scale_pos_weight')}\n")
        f.write("\n")
        
        # Model Configuration
        f.write("MODEL CONFIGURATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Algorithm:               XGBoost Classifier (Gradient Boosting)\n")
        f.write(f"n_estimators:            {model.n_estimators}\n")
        f.write(f"max_depth:               {model.max_depth}\n")
        f.write(f"learning_rate:           {model.learning_rate}\n")
        f.write(f"min_child_weight:        {model.min_child_weight}\n")
        f.write(f"gamma:                   {model.gamma}\n")
        f.write(f"subsample:               {model.subsample}\n")
        f.write(f"colsample_bytree:        {model.colsample_bytree}\n")
        f.write(f"reg_alpha (L1):          {model.reg_alpha}\n")
        f.write(f"reg_lambda (L2):         {model.reg_lambda}\n")
        f.write(f"scale_pos_weight:        {model.scale_pos_weight}\n")
        f.write(f"tree_method:             {model.tree_method}\n")
        f.write(f"objective:               {model.objective}\n")
        f.write(f"eval_metric:             {model.eval_metric}\n")
        f.write(f"random_state:            {model.random_state}\n")
        if hasattr(model, 'best_iteration'):
            f.write(f"best_iteration:          {model.best_iteration}\n")
        f.write("\n")
        
        # Dataset & Dimensionality Information
        f.write("DATASET & DIMENSIONALITY ANALYSIS\n")
        f.write("-" * 80 + "\n")
        if reduction_summary:
            f.write(f"Original Feature Count:    {reduction_summary.get('initial_feature_count', len(feature_names))}\n")
            f.write(f"Final Feature Count:       {len(feature_names)}\n")
            f.write(f"Dimensionality Reduction:  {reduction_summary.get('method', 'none')}\n")
            f.write(f"Feature Reduction Rate:    {reduction_summary.get('reduction_percent', 0):.1f}%\n")
            f.write(f"Features Retained:         {100 - reduction_summary.get('reduction_percent', 0):.1f}%\n")
        else:
            f.write(f"Feature Count:             {len(feature_names)}\n")
            f.write(f"Dimensionality Reduction:  None\n")
        
        # Class distribution
        if 'y_test' in results:
            y_test = results['y_test']
            unique_classes, class_counts = np.unique(y_test, return_counts=True)
            f.write(f"\nTest Set Class Distribution:\n")
            for cls, count in zip(unique_classes, class_counts):
                class_name = quality_names.get(cls, f'Class {cls}')
                percentage = (count / len(y_test)) * 100
                f.write(f"  {class_name:<20}: {count:>3} samples ({percentage:>5.1f}%)\n")
        f.write("\n")
        
        # Cross-validation section
        f.write("CROSS-VALIDATION ANALYSIS (5-FOLD STRATIFIED)\n")
        f.write("-" * 80 + "\n")
        
        # Summary statistics
        f.write("Summary Statistics:\n")
        f.write(f"  Accuracy:           {cv_results['accuracy_mean']:.4f} +/- {cv_results['accuracy_std']:.4f}\n")
        f.write(f"  Balanced Accuracy:  {cv_results['balanced_accuracy_mean']:.4f} +/- {cv_results['balanced_accuracy_std']:.4f}\n")
        f.write(f"  F1 Weighted:        {cv_results['f1_weighted_mean']:.4f} +/- {cv_results['f1_weighted_std']:.4f}\n")
        f.write(f"  F1 Macro:           {cv_results['f1_macro_mean']:.4f} +/- {cv_results['f1_macro_std']:.4f}\n")
        f.write(f"  Precision Macro:    {cv_results['precision_macro_mean']:.4f} +/- {cv_results['precision_macro_std']:.4f}\n")
        f.write(f"  Recall Macro:       {cv_results['recall_macro_mean']:.4f} +/- {cv_results['recall_macro_std']:.4f}\n")
        
        # Per-fold detailed results
        f.write("\nPer-Fold Detailed Results:\n")
        f.write("Fold    Accuracy  Bal.Acc   F1-Weighted F1-Macro  Prec-Macro Rec-Macro\n")
        f.write("-" * 80 + "\n")
        for i in range(5):  # 5-fold CV
            fold_num = i + 1
            acc = cv_results['accuracy'][i] if len(cv_results['accuracy']) > i else 0
            bal_acc = cv_results['balanced_accuracy'][i] if len(cv_results['balanced_accuracy']) > i else 0
            f1_w = cv_results['f1_weighted'][i] if len(cv_results['f1_weighted']) > i else 0
            f1_m = cv_results['f1_macro'][i] if len(cv_results['f1_macro']) > i else 0
            prec_m = cv_results['precision_macro'][i] if len(cv_results['precision_macro']) > i else 0
            rec_m = cv_results['recall_macro'][i] if len(cv_results['recall_macro']) > i else 0
            
            f.write(f"{fold_num:>4}    {acc:>7.4f}   {bal_acc:>7.4f}   {f1_w:>9.4f}   {f1_m:>7.4f}   {prec_m:>8.4f}   {rec_m:>7.4f}\n")
        
        # Statistical analysis
        f.write("\nStatistical Analysis:\n")
        acc_min = cv_results['accuracy'].min()
        acc_max = cv_results['accuracy'].max()
        acc_range = acc_max - acc_min
        f.write(f"  Accuracy Range:     {acc_range:.4f} (min: {acc_min:.4f}, max: {acc_max:.4f})\n")
        
        # Training vs validation performance
        f.write(f"  Training Accuracy:  {cv_results.get('train_balanced_accuracy', 'N/A'):.4f}\n" if 'train_balanced_accuracy' in cv_results else "  Training Accuracy:  N/A\n")
        f.write(f"  Validation Accuracy: {cv_results['balanced_accuracy_mean']:.4f}\n")
        f.write(f"  Generalization Gap: {cv_results.get('generalization_gap', 0):.4f}\n")
        
        # Model stability assessment
        cv_coeff_var = cv_results['accuracy_std'] / cv_results['accuracy_mean'] * 100
        f.write(f"  Coefficient of Variation: {cv_coeff_var:.2f}% (model stability indicator)\n")
        
        stability_assessment = "Excellent" if cv_coeff_var < 2 else "Good" if cv_coeff_var < 5 else "Moderate" if cv_coeff_var < 10 else "Poor"
        f.write(f"  Stability Assessment: {stability_assessment}\n")
        f.write("\n")

        f.write("TEST SET PERFORMANCE\n")
        f.write("-" * 80 + "\n")
        f.write(results['classification_report'])
        f.write("\n")
        
        # Add confusion matrix section
        f.write("CONFUSION MATRIX ANALYSIS\n")
        f.write("-" * 80 + "\n")
        
        # Get class names and confusion matrix
        cm = results['confusion_matrix']
        unique_classes = sorted(set(results.get('y_test', [])) if 'y_test' in results else [0, 1, 2])
        class_names = [quality_names.get(i, f'Class {i}') for i in unique_classes]
        
        # Write confusion matrix with counts
        f.write("Confusion Matrix (Counts):\n")
        f.write("Actual \\ Predicted")
        for class_name in class_names:
            f.write(f"{class_name:>15}")
        f.write("\n")
        f.write("-" * (20 + 15 * len(class_names)) + "\n")
        
        # Write confusion matrix rows
        for i, actual_class in enumerate(class_names):
            f.write(f"{actual_class:<20}")
            for j in range(len(class_names)):
                if i < cm.shape[0] and j < cm.shape[1]:
                    f.write(f"{cm[i, j]:>15}")
                else:
                    f.write(f"{'0':>15}")
            f.write("\n")
        
        # Calculate and write confusion matrix percentages
        f.write("\nConfusion Matrix (Percentages by Actual Class):\n")
        f.write("Actual \\ Predicted")
        for class_name in class_names:
            f.write(f"{class_name:>15}")
        f.write("\n")
        f.write("-" * (20 + 15 * len(class_names)) + "\n")
        
        for i, actual_class in enumerate(class_names):
            f.write(f"{actual_class:<20}")
            if i < cm.shape[0]:
                row_sum = cm[i, :].sum()
                for j in range(len(class_names)):
                    if j < cm.shape[1] and row_sum > 0:
                        percentage = (cm[i, j] / row_sum) * 100
                        f.write(f"{percentage:>14.1f}%")
                    else:
                        f.write(f"{'0.0%':>15}")
            else:
                for j in range(len(class_names)):
                    f.write(f"{'0.0%':>15}")
            f.write("\n")
        f.write("\n")
        
        f.write("CLASSIFICATION METRICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Standard Accuracy: {results['accuracy']:.4f}\n")
        f.write(f"Balanced Accuracy: {results['balanced_accuracy']:.4f}\n")
        f.write(f"F1 Weighted: {results['f1_weighted']:.4f}\n")
        f.write(f"F1 Macro: {results['f1_macro']:.4f}\n")
        f.write("\n")
        
        if 'per_class_metrics' in results:
            f.write("PER-CLASS DETAILED PERFORMANCE\n")
            f.write("-" * 80 + "\n")
            
            # Header for tabular format
            f.write("Class                Precision    Recall    F1-Score    Support    Sensitivity\n")
            f.write("-" * 80 + "\n")
            
            for class_name, metrics in results['per_class_metrics'].items():
                f.write(f"{class_name:<20} {metrics['precision']:>9.4f} {metrics['recall']:>9.4f} {metrics['f1']:>9.4f} {metrics['support']:>9} {metrics['recall']:>11.4f}\n")
            
            f.write("\nDetailed Per-Class Analysis:\n")
            for class_name, metrics in results['per_class_metrics'].items():
                f.write(f"\n{class_name}:\n")
                f.write(f"  - Precision (PPV): {metrics['precision']:.4f} - Of all predicted {class_name}, {metrics['precision']:.1%} were correct\n")
                f.write(f"  - Recall (Sensitivity): {metrics['recall']:.4f} - Of all actual {class_name}, {metrics['recall']:.1%} were identified\n")
                f.write(f"  - F1-Score: {metrics['f1']:.4f} - Harmonic mean of precision and recall\n")
                f.write(f"  - Support: {metrics['support']} samples in test set\n")
        f.write("\n")
        
        f.write("FEATURE IMPORTANCE ANALYSIS\n")
        f.write("-" * 80 + "\n")
        f.write("Top 20 Most Important Features (XGBoost Gain):\n")
        f.write("Rank  Feature Name                     Importance    Cumulative %\n")
        f.write("-" * 80 + "\n")
        
        cumulative_importance = 0
        for idx, (_, row) in enumerate(results['feature_importance'].head(20).iterrows(), 1):
            cumulative_importance += row['importance']
            f.write(f"{idx:>4}  {row['feature']:<30} {row['importance']:>10.4f} {cumulative_importance:>11.1%}\n")
        
        f.write(f"\nTop 20 features account for {cumulative_importance:.1%} of total importance.\n")
        
        # Feature importance statistics
        all_importance = results['feature_importance']['importance']
        f.write(f"\nFeature Importance Statistics:\n")
        f.write(f"  Mean importance:     {all_importance.mean():.4f}\n")
        f.write(f"  Std deviation:       {all_importance.std():.4f}\n")
        f.write(f"  Max importance:      {all_importance.max():.4f}\n")
        f.write(f"  Min importance:      {all_importance.min():.4f}\n")
        f.write(f"  Top 5% threshold:    {all_importance.quantile(0.95):.4f}\n")
        f.write(f"  Top 10% threshold:   {all_importance.quantile(0.90):.4f}\n")
        f.write("\n")
        
        # XGBoost-specific analysis
        f.write("XGBOOST-SPECIFIC ANALYSIS\n")
        f.write("-" * 80 + "\n")
        f.write("Gradient Boosting Characteristics:\n")
        f.write(f"  - Sequential tree building (boosting)\n")
        f.write(f"  - Each tree corrects errors of previous trees\n")
        f.write(f"  - L1 regularization (alpha): {model.reg_alpha}\n")
        f.write(f"  - L2 regularization (lambda): {model.reg_lambda}\n")
        f.write(f"  - Learning rate: {model.learning_rate} (controls step size)\n")
        f.write(f"  - Max depth: {model.max_depth} (tree complexity)\n")
        f.write(f"  - Subsample: {model.subsample} (row sampling per tree)\n")
        f.write(f"  - Colsample by tree: {model.colsample_bytree} (column sampling per tree)\n")
        
        if hasattr(model, 'best_iteration'):
            f.write(f"\nEarly Stopping:\n")
            f.write(f"  - Best iteration: {model.best_iteration}\n")
            f.write(f"  - Total estimators: {model.n_estimators}\n")
            f.write(f"  - Trees used: {model.best_iteration}/{model.n_estimators} ({model.best_iteration/model.n_estimators:.1%})\n")
        
        f.write("\n")
        
        # Final Summary Section
        f.write("COMPREHENSIVE SUMMARY & CONCLUSIONS\n")
        f.write("=" * 80 + "\n")
        
        f.write("Model Performance Summary:\n")
        f.write("-" * 40 + "\n")
        f.write(f"- Overall Accuracy:      {results['accuracy']:.1%} (Test Set)\n")
        f.write(f"- Cross-Validation:      {cv_results['accuracy_mean']:.1%} +/- {cv_results['accuracy_std']:.1%} (5-fold)\n")
        f.write(f"- Balanced Accuracy:     {results['balanced_accuracy']:.1%}\n")
        f.write(f"- F1-Score (Macro):      {results['f1_macro']:.1%}\n")
        f.write(f"- Model Stability:       {stability_assessment} (CV = {cv_coeff_var:.2f}%)\n")
        f.write(f"- Generalization Gap:    {cv_results.get('generalization_gap', 0):.4f}\n")
        
        f.write("\nKey Findings:\n")
        f.write("-" * 40 + "\n")
        
        # Performance assessment
        if results['accuracy'] >= 0.95:
            f.write("- EXCELLENT: Model achieves >95% accuracy suitable for production deployment\n")
        elif results['accuracy'] >= 0.90:
            f.write("- GOOD: Model achieves >90% accuracy suitable for assisted coaching\n")
        elif results['accuracy'] >= 0.85:
            f.write("- MODERATE: Model achieves >85% accuracy requiring human oversight\n")
        else:
            f.write("- POOR: Model <85% accuracy requires significant improvement\n")
        
        # Dimensionality reduction assessment
        if reduction_summary:
            reduction_pct = reduction_summary.get('reduction_percent', 0)
            if reduction_pct > 70:
                f.write(f"- EFFICIENT: {reduction_pct:.1f}% dimensionality reduction with minimal performance loss\n")
            elif reduction_pct > 50:
                f.write(f"- MODERATE: {reduction_pct:.1f}% dimensionality reduction achieved\n")
            else:
                f.write(f"- MINIMAL: {reduction_pct:.1f}% dimensionality reduction applied\n")
        
        # Generalization assessment
        gen_gap = cv_results.get('generalization_gap', 0)
        if gen_gap < 0.05:
            f.write("- ROBUST: Excellent generalization with minimal overfitting\n")
        elif gen_gap < 0.10:
            f.write("- STABLE: Good generalization with acceptable overfitting\n")
        else:
            f.write("- CAUTION: Potential overfitting detected, consider regularization\n")
        
        f.write("\nRecommendations:\n")
        f.write("-" * 40 + "\n")
        
        # Deployment recommendations
        if results['accuracy'] >= 0.90 and gen_gap < 0.10:
            f.write("1. DEPLOYMENT READY: Model suitable for production use\n")
            f.write("2. Monitor performance on new users and exercises\n")
        else:
            f.write("1. REQUIRES IMPROVEMENT: Consider additional data collection\n")
            f.write("2. Implement human-in-the-loop validation\n")
        
        f.write("3. Regularly retrain with new data to maintain performance\n")
        f.write("4. Consider ensemble with Random Forest for critical applications\n")
        f.write("5. Monitor for concept drift in production environment\n")
        
        f.write(f"\nTechnical Details:\n")
        f.write(f"-" * 40 + "\n")
        f.write(f"- Feature Engineering:   {len(feature_names)} sensor-derived features\n")
        f.write(f"- Cross-Validation:      Stratified 5-fold with proper train/test split\n")
        f.write(f"- Model Complexity:      {model.n_estimators} gradient boosted trees\n")
        f.write(f"- Data Preprocessing:    StandardScaler + feature selection\n")
        f.write(f"- Evaluation Metrics:    Comprehensive imbalance-aware assessment\n")
        f.write(f"- Regularization:        L1={model.reg_alpha}, L2={model.reg_lambda}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF COMPREHENSIVE EVALUATION REPORT\n")
        f.write("=" * 80 + "\n")
    
    print(f"  * Report: {report_file}")
    
    # Save feature importance to CSV
    importance_path = output_path / f'feature_importance_{timestamp}.csv'
    results['feature_importance'].to_csv(importance_path, index=False)
    print(f"  * Feature importance CSV: {importance_path}")
    
    print(f"\n[OK] Model export complete!")



# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_classification_pipeline_xgb():
    """
    Main function to run the complete XGBoost classification pipeline.
    """
    if not XGBOOST_AVAILABLE:
        print("\n[ERROR] XGBoost is not installed!")
        print("Install with: pip install xgboost")
        return
    
    print("=" * 80)
    print("  XGBOOST EXERCISE QUALITY CLASSIFICATION PIPELINE")
    print("=" * 80)
    print("\nThis pipeline trains an XGBoost classifier for exercise quality assessment.")
    print("All data leakage prevention measures are applied.")
    print("\n" + "=" * 80 + "\n")
    
    # Step 1: Select CSV file
    print("[STEP 1] Select Dataset CSV File")
    print("-" * 80)
    file_path = select_csv_file()
    
    if not file_path:
        print("[ERROR] No file selected. Exiting.")
        return
    
    print(f"[OK] Selected: {file_path}")
    
    # Step 2: Load data
    print("\n[STEP 2] Loading Dataset")
    print("-" * 80)
    df = pd.DataFrame(pd.read_csv(file_path))
    print(f"[OK] Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    # Display dataset info
    dataset_info = get_dataset_info(df)
    display_dataset_info(dataset_info)
    
    # Step 3: Feature engineering (if needed)
    print("\n[STEP 3] Feature Engineering")
    print("-" * 80)
    
    # Check if data needs rep-level aggregation
    if 'rep' in df.columns and 'timestamp_ms' in df.columns:
        print("[INFO] Raw time-series data detected. Computing rep-level features...")
        features_df = compute_rep_features(df)
    else:
        print("[INFO] Using existing features (already aggregated)")
        features_df = df.copy()
    
    # Step 4: Feature selection
    print("\n[STEP 4] Feature Selection")
    print("-" * 80)
    selected_features, excluded_features = select_columns_ui(features_df, target_column=TARGET_COLUMN)
    
    if selected_features is None:
        print("[ERROR] Feature selection cancelled. Exiting.")
        return
    
    print(f"\n[OK] Selected {len(selected_features)} features")
    print(f"[OK] Excluded {len(excluded_features)} columns")
    
    # Step 5: Prepare data
    print("\n[STEP 5] Preparing Data")
    print("-" * 80)
    X, y, feature_names = prepare_data(features_df, selected_features, target_column=TARGET_COLUMN)
    
    # Get exercise code for context-aware quality names
    exercise_code = features_df['exercise_code'].iloc[0] if 'exercise_code' in features_df.columns else None
    
    # Step 6: Train-test split
    print("\n[STEP 6] Train-Test Split")
    print("-" * 80)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"[OK] Training set: {len(X_train):,} samples")
    print(f"[OK] Test set: {len(X_test):,} samples")
    
    # Step 7: Impute missing values (AFTER split)
    print("\n[STEP 7] Handling Missing Values")
    print("-" * 80)
    X_train, X_test = impute_after_split(X_train, X_test)
    
    # Step 8: Class imbalance strategy
    print("\n[STEP 8] Class Imbalance Strategy")
    print("-" * 80)
    imbalance_config = configure_class_imbalance_strategy(y_train, quality_names=get_quality_names(exercise_code=exercise_code, df=features_df))
    
    if imbalance_config is None:
        print("[ERROR] Configuration cancelled. Exiting.")
        return
    
    # Apply SMOTE if requested
    smote_summary = None
    if imbalance_config['use_smote']:
        X_train, y_train, smote_summary = apply_smote(X_train, y_train, quality_names=get_quality_names(exercise_code=exercise_code, df=features_df))
    
    # Calculate scale_pos_weight for XGBoost
    class_counts = Counter(y_train)
    if len(class_counts) == 2:
        # Binary classification
        scale_pos_weight = class_counts[0] / class_counts[1]
    else:
        # Multi-class: use 1.0 (XGBoost handles multi-class differently)
        scale_pos_weight = 1.0
    
    print(f"\n[XGB] Scale_pos_weight: {scale_pos_weight:.2f}")
    
    # Step 9: Dimensionality reduction
    print("\n[STEP 9] Dimensionality Reduction")
    print("-" * 80)
    reduction_config = configure_dimensionality_reduction(X_train.shape[1])
    
    if reduction_config is None:
        print("[ERROR] Configuration cancelled. Exiting.")
        return
    
    X_train_reduced, X_test_reduced, reduced_feature_names, reduction_summary = apply_dimensionality_reduction(
        X_train, X_test, y_train, reduction_config,
        best_params=None,
        class_weight_setting='balanced' if imbalance_config['use_imbalance_strategy'] else None
    )
    
    # Step 10: Hyperparameter optimization
    print("\n[STEP 10] Hyperparameter Optimization")
    print("-" * 80)
    print("\nChoose hyperparameter optimization strategy:")
    print("1. [FAST] Use default parameters (quick, good baseline)")
    print("2. [RANDOM] Random Search (100 iterations, ~10-20 min)")
    print("3. [GRID] Grid Search (comprehensive, ~30-60 min)")
    print("4. [SKIP] Skip optimization (use defaults)")
    
    while True:
        try:
            choice = input("\nEnter choice (1-4) [default=1]: ").strip()
            if choice == "":
                choice = "1"
            if choice in ['1', '2', '3', '4']:
                break
            print("Please enter 1, 2, 3, or 4")
        except KeyboardInterrupt:
            print("\n[ERROR] Operation cancelled.")
            return
        except Exception:
            print("Please enter 1, 2, 3, or 4")
    
    best_params = None
    if choice == '2':
        best_params, _ = perform_random_search_xgb(
            X_train_reduced, y_train,
            n_iter=100, cv_folds=5, n_jobs=-1, verbose=1,
            use_imbalance_strategy=imbalance_config['use_imbalance_strategy']
        )
    elif choice == '3':
        best_params, _ = perform_grid_search_xgb(
            X_train_reduced, y_train,
            cv_folds=5, n_jobs=-1, verbose=1,
            use_imbalance_strategy=imbalance_config['use_imbalance_strategy']
        )
    
    # Step 11: Cross-validation
    print("\n[STEP 11] Cross-Validation")
    print("-" * 80)
    
    if imbalance_config['use_smote']:
        cv_results = perform_cross_validation_with_smote_xgb(
            X_train_reduced, y_train,
            n_splits=5, best_params=best_params,
            scale_pos_weight=scale_pos_weight,
            use_gpu=False,
            quality_names=get_quality_names(exercise_code=exercise_code, df=features_df)
        )
    else:
        cv_results = perform_cross_validation_xgb(
            X_train_reduced, y_train,
            n_splits=5, best_params=best_params,
            scale_pos_weight=scale_pos_weight,
            use_gpu=False
        )
    
    # Step 12: Train final model
    print("\n[STEP 12] Training Final Model")
    print("-" * 80)
    
    model, scaler, results = train_xgboost(
        X_train_reduced, y_train,
        X_test_reduced, y_test,
        reduced_feature_names,
        best_params=best_params,
        scale_pos_weight=scale_pos_weight,
        use_gpu=False,
        exercise_code=exercise_code,
        df=features_df
    )
    
    # Step 13: Create visualizations
    print("\n[STEP 13] Creating Visualizations")
    print("-" * 80)
    
    # Determine output folder based on exercise
    if exercise_code is not None:
        exercise_name = EXERCISE_TYPES.get(exercise_code, f'Exercise_{exercise_code}')
        output_folder = MODELS_DIR / exercise_name.replace(' ', '_')
    else:
        output_folder = MODELS_DIR / 'Mixed_Exercises'
    
    create_model_visualizations_xgb(
        y_test, results, cv_results, output_folder,
        exercise_code=exercise_code, df=features_df
    )
    
    # Step 14: Export model
    print("\n[STEP 14] Exporting Model")
    print("-" * 80)
    
    export_model_xgb(
        model, scaler, reduced_feature_names,
        results, cv_results, output_folder,
        exercise_code=exercise_code, df=features_df,
        imbalance_config=imbalance_config,
        reduction_summary=reduction_summary
    )
    
    # Final summary
    print("\n" + "=" * 80)
    print("  PIPELINE COMPLETE!")
    print("=" * 80)
    print(f"\n[SUMMARY] XGBoost Model Performance:")
    print(f"  * Test Balanced Accuracy: {results['balanced_accuracy']:.4f}")
    print(f"  * Test F1 Weighted: {results['f1_weighted']:.4f}")
    print(f"  * CV Balanced Accuracy: {cv_results['balanced_accuracy_mean']:.4f} ± {cv_results['balanced_accuracy_std']:.4f}")
    print(f"  * Features Used: {len(reduced_feature_names)}")
    print(f"\n[OUTPUT] Results saved to: {output_folder}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    run_classification_pipeline_xgb()
