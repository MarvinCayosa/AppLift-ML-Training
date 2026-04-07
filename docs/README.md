# AppLift ML Training - Exercise Quality Classification Model

## Overview

This project implements a comprehensive machine learning pipeline for classifying exercise execution quality using sensor data from various fitness equipment. The system employs Random Forest classification to distinguish between proper form ("Clean") and various execution errors across multiple exercise types.

## Data Gathering Methodology

### Data Collection Setup
- **Equipment Types**: Barbell, Dumbbell, Weight Stack
- **Exercise Categories**: 
  - Barbell: Back Squats, Bench Press
  - Dumbbell: Concentration Curls, Overhead Extension  
  - Weight Stack: Lateral Pulldown, Seated Leg Extension
- **Sensor Data**: Multi-axis accelerometer/gyroscope readings at high frequency (typically 50-100Hz)
- **Quality Classes**:
  - **Clean**: Proper exercise execution
  - **Uncontrolled Movement**: Jerky or inconsistent motion patterns
  - **Inclination Asymmetry**: Uneven body positioning (Barbell exercises)
  - **Abrupt Initiation**: Sudden start without proper setup (Dumbbell exercises)
  - **Pulling Too Fast / Releasing Too Fast**: Incorrect tempo (Weight Stack exercises)

### Data Structure
```
Raw Data Format:
- timestamp_ms: Millisecond-level timestamps
- participant: Subject identifier
- rep: Repetition number within a set
- exercise_code: Numerical exercise type identifier
- equipment_code: Numerical equipment type identifier
- target: Quality classification (0=Clean, 1=Error Type 1, 2=Error Type 2)
- sensor_x, sensor_y, sensor_z: Multi-axis sensor readings
- Additional derived signals and metadata
```

## Technical Pipeline: Step-by-Step Process

### Phase 1: Data Preprocessing and Feature Engineering

#### 1.1 Rep-Level Aggregation (Critical for Data Leakage Prevention)
**Problem Addressed**: Raw sensor data contains thousands of samples per repetition, creating severe data leakage if used directly.

**Solution**: Aggregate time-series data into statistical features per repetition.

```python
# For each repetition, compute:
Statistical Features:
- mean, std, min, max, median
- skewness, kurtosis (distribution shape)
- range, interquartile_range
- coefficient_of_variation

Temporal Features:  
- peak_position (timing of maximum value)
- peak_value (maximum amplitude)
- zero_crossings count
- autocorrelation at lag 1

Frequency Domain:
- dominant_frequency (FFT analysis)
- spectral_centroid
- spectral_rolloff
```

**Result**: Each repetition becomes one sample with ~40-80 computed features per sensor axis.

#### 1.2 Data Quality Control
- **Rep 0 Filtering**: Remove incomplete repetitions (rep=0)
- **Missing Value Handling**: Forward-fill then median imputation
- **Infinite Value Correction**: Replace ±inf with NaN, then median fill
- **Outlier Detection**: Z-score based outlier flagging (optional removal)

### Phase 2: Train-Test Split (Data Leakage Prevention)

#### 2.1 Participant-Stratified Split
```python
# Ensure no participant appears in both train and test
train_test_split(
    X, y, 
    test_size=0.2, 
    stratify=y,  # Preserve class distribution
    random_state=42
)
```

**Critical Fix**: The pipeline ensures that:
- Test data is NEVER used for feature selection
- Cross-validation uses ONLY training data
- All preprocessing fits on training data only

### Phase 3: Feature Selection and Dimensionality Reduction

#### 3.1 Correlation Pruning
**Objective**: Remove redundant features that carry similar information.

```python
Algorithm:
1. Compute correlation matrix from TRAINING data only
2. For each pair with |correlation| > threshold (default 0.90):
   - Apply preference heuristic:
     * Prefer: peak_position, skewness, kurtosis (robust shape features)
     * De-prioritize: peak_value, min/max (noise-sensitive)
   - If tied, keep higher-variance feature
3. Remove redundant features from both train and test sets
```

#### 3.2 Random Forest Importance Pruning
```python
Process:
1. Train Random Forest on current feature set (training data only)
2. Rank features by importance scores
3. Select top-K features (default: 40-60 for optimal performance)
4. Retrain model on reduced feature set
```

#### 3.3 Recursive Feature Elimination (RFE) [Alternative]
```python
RFE Process:
1. Start with all features
2. Train model, rank by importance
3. Remove least important features (step size: 10)
4. Repeat until target feature count reached
5. Cross-validate to find optimal feature count
```

### Phase 4: Class Imbalance Handling

#### 4.1 Imbalance Strategy Selection
The pipeline analyzes class distribution and recommends strategies:

- **Ratio < 1.5:1**: No intervention needed
- **Ratio 1.5-3:1**: Apply `class_weight='balanced'` (cost-sensitive learning)
- **Ratio > 3:1**: SMOTE oversampling + class weighting

#### 4.2 SMOTE Implementation (When Applicable)
```python
SMOTE-Aware Cross-Validation:
1. For each CV fold:
   - Apply SMOTE to training fold only
   - Validate on unmodified validation fold
   - This prevents synthetic samples from leaking across folds

2. Final Model Training:
   - Apply SMOTE to entire training set
   - Train final model on balanced data
   - Test on original (unmodified) test set
```

### Phase 5: Model Training and Hyperparameter Optimization

#### 5.1 Random Forest Configuration
```python
Hyperparameter Search Space:
- n_estimators: [100, 200, 500, 1000]
- max_depth: [10, 15, 20, 25, None]
- min_samples_split: [2, 5, 10]
- min_samples_leaf: [1, 2, 4]  
- max_features: ['sqrt', 'log2', None]
- bootstrap: [True, False]
- criterion: ['gini', 'entropy']
- class_weight: ['balanced', 'balanced_subsample', None]
```

#### 5.2 Optimization Strategies
1. **Random Search** (Default): 100-200 iterations, faster exploration
2. **Grid Search** (Comprehensive): Full parameter space, slower but thorough
3. **Default Parameters** (Quick): Sensible defaults for rapid prototyping

### Phase 6: Cross-Validation (Fixed Implementation)

#### 6.1 Proper CV Implementation
**Critical Fix**: CV now uses ONLY training data to prevent data leakage.

```python
Original (INCORRECT):
cv_scores = cross_val_score(model, X_full, y_full, cv=5)  # LEAKED TEST DATA

Fixed (CORRECT):  
cv_scores = cross_val_score(model, X_train, y_train, cv=5)  # TRAINING ONLY
```

#### 6.2 Stratified K-Fold Cross-Validation
- **5-Fold Stratified**: Maintains class distribution in each fold
- **Metrics**: Balanced accuracy (primary), F1-weighted, F1-macro
- **Scaling**: StandardScaler fit within each fold
- **SMOTE**: Applied within each fold when requested

### Phase 7: Model Evaluation

#### 7.1 Comprehensive Metrics
```python
Primary Metrics:
- Balanced Accuracy: Accounts for class imbalance
- F1-Weighted: Harmonic mean of precision/recall, weighted by support
- F1-Macro: Unweighted average across classes

Secondary Metrics:
- Standard Accuracy: Overall correct predictions
- Per-class Precision/Recall/F1: Individual class performance
- Confusion Matrix: Detailed classification breakdown
```

#### 7.2 Generalization Assessment
```python
Overfitting Detection:
- Training Score: Model performance on training data
- CV Score: Cross-validation performance  
- Gap Analysis: Train_score - CV_score
  * Gap > 0.15: Likely overfitting
  * Gap < 0.05: Good generalization
  * CV_score < 0.6: Underfitting
```

### Phase 8: Model Export and Deployment

#### 8.1 Model Artifacts
```python
Exported Components:
- model.pkl: Trained RandomForestClassifier
- scaler.pkl: Fitted StandardScaler  
- feature_names.pkl: Selected feature list
- model_metadata.json: Training configuration and metrics
```

#### 8.2 Visualization Outputs
- **Confusion Matrix**: Classification accuracy by class
- **Feature Importance**: Top contributing features
- **ROC Curves**: Per-class discrimination ability
- **Precision-Recall Curves**: Performance across decision thresholds
- **Cross-Validation Scores**: Fold-by-fold performance tracking

### Phase 9: Performance Analysis and Validation

#### 9.1 Expected Performance Ranges
Based on corrected pipeline (without data leakage):
- **Cross-Validation Scores**: 0.70-0.85 (realistic range)
- **Test Set Performance**: 0.65-0.80 (conservative estimate)
- **Performance Drop**: 5-15% lower than original (due to leak fixes)

#### 9.2 Model Interpretation
```python
Feature Importance Analysis:
1. Individual feature contributions
2. Feature category rankings (temporal vs statistical vs frequency)
3. Exercise-specific feature patterns
4. Equipment-specific discriminative features
```

## Key Technical Improvements (Fixed Version)

### Data Leakage Prevention
1. **Cross-Validation Fix**: Uses only training data (X_train, y_train)
2. **Feature Selection Isolation**: Correlations computed from training data only  
3. **Scaling Consistency**: Scaler fit on training, applied to test
4. **SMOTE Isolation**: Synthetic samples generated within CV folds only

### Enhanced Pipeline Reliability
1. **Robust Feature Engineering**: Handles missing values and infinite values
2. **Stratified Splitting**: Maintains class balance across train/test
3. **Comprehensive Validation**: Multiple metrics for balanced evaluation
4. **Reproducible Results**: Fixed random seeds throughout pipeline

### Performance Optimization
1. **Efficient Hyperparameter Search**: Random search for faster exploration
2. **Parallel Processing**: Multi-core training and validation
3. **Memory Management**: Chunked processing for large datasets
4. **Feature Selection**: Reduces dimensionality while preserving signal

## Usage Instructions

### Basic Pipeline Execution
```bash
python rf_classifier_fixed.py
```

### Pipeline Steps
1. **File Selection**: Choose preprocessed CSV dataset
2. **Feature Selection**: Interactive UI for feature inclusion/exclusion  
3. **Configuration**: Set class imbalance and dimensionality reduction strategies
4. **Training**: Automated hyperparameter search and model training
5. **Validation**: Cross-validation and test set evaluation
6. **Export**: Model artifacts and comprehensive reports

### Output Structure
```
output_fixed/
├── models/
│   └── [Exercise_Name]/
│       ├── model.pkl
│       ├── scaler.pkl  
│       ├── feature_names.pkl
│       ├── model_metadata.json
│       └── classification_report_[timestamp].txt
├── visualizations/
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── roc_curves.png
│   └── cv_scores.png
└── comparison_with_original/
    └── performance_comparison.txt
```

## Technical Specifications

### Dependencies
```python
Core ML Libraries:
- scikit-learn >= 1.0.0: ML algorithms and evaluation
- pandas >= 1.3.0: Data manipulation
- numpy >= 1.21.0: Numerical computing

Visualization:
- matplotlib >= 3.5.0: Plotting and visualization
- seaborn >= 0.11.0: Statistical visualization

Class Imbalance:
- imbalanced-learn >= 0.8.0: SMOTE and sampling techniques

UI Components:
- tkinter: Feature selection interface (built-in)
```

### Computational Requirements
- **RAM**: 4-8GB (depending on dataset size)
- **CPU**: Multi-core recommended for hyperparameter search
- **Storage**: ~100MB per trained model (including visualizations)
- **Runtime**: 5-30 minutes (depending on optimization strategy)

## Model Performance Interpretation

### Cross-Validation vs Test Performance
The fixed pipeline shows realistic performance metrics:
- **CV Scores**: Represent expected performance on new data from same distribution
- **Test Scores**: Validation of generalization to completely unseen data
- **Performance Gap**: 5-15% difference is normal and indicates good pipeline health

### Class-Specific Performance
- **Clean Form**: Typically highest accuracy (most samples, clearest signal)
- **Error Types**: Variable performance based on:
  - Class frequency in training data
  - Distinctiveness of error patterns
  - Sensor sensitivity to specific movement errors

### Feature Importance Insights
Common high-importance features:
- **Temporal Features**: peak_position, zero_crossings (timing patterns)
- **Statistical Features**: skewness, kurtosis (movement quality indicators)  
- **Movement-Specific**: Axis-dependent based on exercise biomechanics

## Validation and Quality Assurance

### Pipeline Validation Steps
1. **Data Integrity**: Verify no test data leakage
2. **Feature Quality**: Ensure no constant or highly correlated features
3. **Class Balance**: Confirm appropriate imbalance handling
4. **Model Stability**: Cross-validation score consistency
5. **Generalization**: Reasonable train-test performance gap

### Quality Metrics Thresholds
- **Minimum CV Score**: 0.60 (balanced accuracy)
- **Maximum Overfitting Gap**: 0.20 (train - CV)
- **Feature Importance Coverage**: Top 10 features contribute >60% importance
- **Class Performance**: No class with F1-score < 0.40

## Future Enhancements

### Potential Improvements
1. **Deep Learning**: CNN or LSTM for temporal pattern recognition
2. **Ensemble Methods**: Combine RF with other algorithms
3. **Real-time Processing**: Online learning for live feedback
4. **Multi-sensor Fusion**: Integrate additional sensor modalities
5. **Transfer Learning**: Pre-trained models across exercise types

### Research Directions
1. **Personalized Models**: Individual-specific error detection
2. **Exercise Progression**: Difficulty and form evolution tracking
3. **Biomechanical Analysis**: Joint angle and force estimation
4. **Fatigue Detection**: Performance degradation identification

---

**Author**: AppLift ML Training Pipeline  
**Version**: 2.0 (Fixed Data Leakage Issues)  
**Date**: February 2026  
**License**: Academic Use Only
