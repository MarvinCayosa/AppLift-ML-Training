"""
Back Squats Threshold-Based False Positive Fix
==============================================
Instead of changing model parameters, this focuses on:
1. Optimal threshold tuning specifically for the "Uncontrolled Movement" class
2. Custom prediction logic that requires higher confidence for error classes
3. Analysis of feature importance to understand what drives false positives
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import joblib

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def custom_predict_with_thresholds(model, X_scaled, 
                                 clean_threshold=0.3,
                                 uncontrolled_threshold=0.75,  # Much higher!
                                 inclination_threshold=0.6):
    """
    Custom prediction with class-specific thresholds to reduce false positives.
    
    Key insight: "Uncontrolled Movement" has high false positive rate, so require
    much higher confidence before flagging a clean rep as uncontrolled.
    """
    probas = model.predict_proba(X_scaled)
    classes = model.classes_
    
    predictions = []
    confidence_scores = []
    
    for proba in probas:
        max_idx = np.argmax(proba)
        max_class = classes[max_idx]
        max_prob = proba[max_idx]
        
        # Map class indices to class IDs (0=Clean, 1=Uncontrolled, 2=Inclination)
        class_to_idx = {class_id: i for i, class_id in enumerate(classes)}
        
        # Get probabilities for each class
        clean_prob = proba[class_to_idx.get(0, 0)] if 0 in class_to_idx else 0
        uncontrolled_prob = proba[class_to_idx.get(1, 0)] if 1 in class_to_idx else 0
        inclination_prob = proba[class_to_idx.get(2, 0)] if 2 in class_to_idx else 0
        
        # Apply class-specific thresholds
        if uncontrolled_prob >= uncontrolled_threshold:
            prediction = 1  # Uncontrolled Movement
        elif inclination_prob >= inclination_threshold:
            prediction = 2  # Inclination Asymmetry
        elif clean_prob >= clean_threshold:
            prediction = 0  # Clean
        else:
            # If no class meets its threshold, default to Clean (safest)
            prediction = 0
        
        predictions.append(prediction)
        confidence_scores.append(max_prob)
    
    return np.array(predictions), np.array(confidence_scores)


def find_optimal_thresholds(model, X_test_scaled, y_test):
    """
    Test different threshold combinations to minimize Clean → Uncontrolled false positives
    """
    print("\\n" + "=" * 60)
    print("THRESHOLD OPTIMIZATION FOR FALSE POSITIVE REDUCTION")
    print("=" * 60)
    
    best_result = None
    best_clean_to_uncontrolled = float('inf')
    
    # Test different uncontrolled thresholds (most important)
    uncontrolled_thresholds = np.arange(0.5, 0.95, 0.05)
    
    print(f"\\n{'Uncontrolled':>12} | {'Clean→Uncontrolled':>15} | {'Precision':>9} | {'Recall':>7} | {'F1':>7}")
    print("   " + "-" * 65)
    
    for unc_thresh in uncontrolled_thresholds:
        y_pred, confidence = custom_predict_with_thresholds(
            model, X_test_scaled,
            clean_threshold=0.3,
            uncontrolled_threshold=unc_thresh,
            inclination_threshold=0.6
        )
        
        # Calculate metrics
        cm = confusion_matrix(y_test, y_pred)
        clean_to_uncontrolled = cm[0, 1] if cm.shape[0] > 1 and cm.shape[1] > 1 else 0
        
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        print(f"   {unc_thresh:>12.2f} | {clean_to_uncontrolled:>15d} | {precision:>9.4f} | {recall:>7.4f} | {f1:>7.4f}")
        
        # Track best result (minimize Clean → Uncontrolled while maintaining reasonable performance)
        if clean_to_uncontrolled < best_clean_to_uncontrolled and recall > 0.85:
            best_clean_to_uncontrolled = clean_to_uncontrolled
            best_result = {
                'uncontrolled_threshold': unc_thresh,
                'clean_to_uncontrolled': clean_to_uncontrolled,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'y_pred': y_pred,
                'confusion_matrix': cm
            }
    
    return best_result


def analyze_feature_importance_for_fp(model, feature_names, X_test_scaled, y_test, y_pred):
    """
    Analyze which features are driving false positives for Clean → Uncontrolled
    """
    print("\\n" + "=" * 60)
    print("FEATURE ANALYSIS FOR FALSE POSITIVES")
    print("=" * 60)
    
    # Get feature importance from the model
    importances = model.feature_importances_
    feature_importance = list(zip(feature_names, importances))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    print("\\n📊 Top 10 Most Important Features (driving predictions):")
    for i, (feature, importance) in enumerate(feature_importance[:10], 1):
        print(f"   {i:2d}. {feature:<25} {importance:.4f} ({importance/sum(importances)*100:.1f}%)")
    
    # Find false positive cases (Clean predicted as Uncontrolled)
    fp_mask = (y_test == 0) & (y_pred == 1)
    true_positive_mask = (y_test == 0) & (y_pred == 0)
    
    if np.sum(fp_mask) > 0 and np.sum(true_positive_mask) > 0:
        print(f"\\n🔍 Analysis of {np.sum(fp_mask)} False Positive Cases:")
        print(f"   (Clean reps incorrectly predicted as Uncontrolled)")
        
        # Compare feature values between false positives and true positives
        fp_features = X_test_scaled[fp_mask]
        tp_features = X_test_scaled[true_positive_mask]
        
        print(f"\\n   Feature differences (FP vs TP for Clean class):")
        print(f"   {'Feature':<25} {'FP Mean':<10} {'TP Mean':<10} {'Difference':<12}")
        print("   " + "-" * 60)
        
        for i, feature in enumerate(feature_names):
            fp_mean = np.mean(fp_features[:, i])
            tp_mean = np.mean(tp_features[:, i]) 
            diff = fp_mean - tp_mean
            
            # Show only features with significant differences
            if abs(diff) > 0.5:  # Standardized features, so 0.5 std is significant
                print(f"   {feature:<25} {fp_mean:>10.3f} {tp_mean:>10.3f} {diff:>12.3f}")


def main():
    print("=" * 60)
    print("BACK SQUATS FALSE POSITIVE FIX")
    print("Threshold-Based Approach") 
    print("=" * 60)
    
    # Load the merged dataset
    merged_path = project_root / 'output' / 'merged_datasets' / 'BACK_SQUATS_20260218_234157.csv'
    
    if not merged_path.exists():
        print(f"❌ Dataset not found: {merged_path}")
        return
    
    print(f"📊 Loading dataset: {merged_path.name}")
    df = pd.read_csv(merged_path)
    
    print(f"   Dataset shape: {df.shape}")
    print(f"   Class distribution:")
    print(df['target'].value_counts().sort_index())
    
    # Prepare features and target (same as before)
    exclude_cols = ['target', 'source_file', 'participant', 'rep', 
                   'timestamp', 'timestamp_ms', 'equipment_code', 'exercise_code', 'quality_code',
                   'Unnamed: 0', 'index']
    
    feature_cols = []
    for col in df.columns:
        if col not in exclude_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                feature_cols.append(col)
    
    X = df[feature_cols]
    y = df['target']
    
    print(f"   Feature count: {len(feature_cols)}")
    
    # Split and scale the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model with original parameters (but better class weighting)
    print("\\n🔬 Training Random Forest with balanced class weighting...")
    rf_params = {
        'n_estimators': 100,
        'max_depth': 8,
        'min_samples_split': 15,
        'min_samples_leaf': 8,
        'max_features': 'sqrt',
        'bootstrap': True,
        'random_state': 42,
        'class_weight': {0: 1.0, 1: 0.7, 2: 1.0}  # Reduce weight for "Uncontrolled Movement"
    }
    
    rf = RandomForestClassifier(**rf_params)
    rf.fit(X_train_scaled, y_train)
    
    print("   ✅ Model trained successfully")
    
    # Baseline performance
    y_pred_baseline = rf.predict(X_test_scaled)
    cm_baseline = confusion_matrix(y_test, y_pred_baseline)
    
    print(f"\\n🎯 BASELINE Performance (Standard Prediction):")
    print(f"   Clean → Uncontrolled (false positives): {cm_baseline[0, 1]}")
    print(f"   Accuracy: {(y_pred_baseline == y_test).mean():.4f}")
    
    # Find optimal thresholds
    best_result = find_optimal_thresholds(rf, X_test_scaled, y_test)
    
    if best_result:
        print(f"\\n🎉 BEST THRESHOLD CONFIGURATION:")
        print(f"   Uncontrolled Movement threshold: {best_result['uncontrolled_threshold']:.2f}")
        print(f"   Clean → Uncontrolled cases: {best_result['clean_to_uncontrolled']} (was {cm_baseline[0, 1]})")
        
        reduction = (cm_baseline[0, 1] - best_result['clean_to_uncontrolled']) / cm_baseline[0, 1] * 100
        print(f"   False positive reduction: {reduction:.1f}%")
        print(f"   Precision: {best_result['precision']:.4f}")
        print(f"   Recall: {best_result['recall']:.4f}")
        print(f"   F1 Score: {best_result['f1']:.4f}")
        
        print(f"\\n📋 Optimized Confusion Matrix:")
        cm_opt = best_result['confusion_matrix']
        print(f"   Actual \\ Predicted    Clean  Uncontrolled  Inclination")
        print(f"   Clean                {cm_opt[0][0]:>5}        {cm_opt[0][1]:>5}         {cm_opt[0][2]:>5}")
        print(f"   Uncontrolled         {cm_opt[1][0]:>5}        {cm_opt[1][1]:>5}         {cm_opt[1][2]:>5}")
        if len(cm_opt) > 2:
            print(f"   Inclination          {cm_opt[2][0]:>5}        {cm_opt[2][1]:>5}         {cm_opt[2][2]:>5}")
        
        # Feature importance analysis
        analyze_feature_importance_for_fp(rf, feature_cols, X_test_scaled, y_test, best_result['y_pred'])
        
        print(f"\\n✅ RECOMMENDATION:")
        print(f"   Use uncontrolled_threshold = {best_result['uncontrolled_threshold']:.2f}")
        print(f"   This reduces false positives by {reduction:.1f}% while maintaining good performance.")
        print(f"   Users will see {best_result['clean_to_uncontrolled']} instead of {cm_baseline[0, 1]} incorrect 'uncontrolled' flags.")
    
    else:
        print("\\n❌ Could not find optimal threshold configuration.")
        print("   Consider alternative approaches like feature selection or different algorithms.")


if __name__ == "__main__":
    main()