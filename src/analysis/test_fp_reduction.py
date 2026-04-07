"""
Back Squats False Positive Testing Script
========================================
Tests the improved classifier parameters to verify false positive reduction.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_improved_parameters():
    """
    Test the improved RF parameters against the original ones
    """
    print("=" * 60)
    print("BACK SQUATS FALSE POSITIVE REDUCTION TEST")
    print("=" * 60)
    
    # Load the merged back squats dataset
    merged_path = project_root / 'output' / 'merged_datasets' / 'BACK_SQUATS_20260218_234157.csv'
    
    if not merged_path.exists():
        print(f"❌ Dataset not found: {merged_path}")
        print("Please run the dataset merger first.")
        return
    
    print(f"📊 Loading dataset: {merged_path.name}")
    df = pd.read_csv(merged_path)
    
    print(f"   Dataset shape: {df.shape}")
    print(f"   Class distribution:")
    print(df['target'].value_counts().sort_index())
    
    # Prepare features and target
    exclude_cols = ['target', 'source_file', 'participant', 'rep', 
                   'timestamp', 'timestamp_ms', 'equipment_code', 'exercise_code', 'quality_code',
                   'Unnamed: 0', 'index']
    
    feature_cols = []
    for col in df.columns:
        if col not in exclude_cols:
            # Check if column is numeric
            if pd.api.types.is_numeric_dtype(df[col]):
                feature_cols.append(col)
            else:
                print(f"   Skipping non-numeric column: {col}")
    
    X = df[feature_cols]
    y = df['target']
    
    print(f"   Feature count: {len(feature_cols)}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\\n   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    
    print("\\n" + "=" * 60)
    print("TESTING PARAMETER SETS")
    print("=" * 60)
    
    # Original parameters (problematic)
    original_params = {
        'n_estimators': 100,
        'max_depth': 8,
        'min_samples_split': 15,
        'min_samples_leaf': 8,
        'max_features': 'sqrt',
        'bootstrap': True,
        'random_state': 42,
        'class_weight': None
    }
    
    # Improved parameters (false positive reduction)
    improved_params = {
        'n_estimators': 150,
        'max_depth': 6,
        'min_samples_split': 25,
        'min_samples_leaf': 15,
        'max_features': 'sqrt',
        'bootstrap': True,
        'random_state': 42,
        'class_weight': 'balanced_subsample'
    }
    
    results = {}
    
    for name, params in [("ORIGINAL", original_params), ("IMPROVED", improved_params)]:
        print(f"\\n🔬 Testing {name} Parameters:")
        for key, value in params.items():
            print(f"   {key}: {value}")
        
        # Train model
        print(f"\\n   Training {name} model...")
        rf = RandomForestClassifier(**params)
        rf.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = rf.predict(X_test_scaled)
        
        # Calculate metrics
        accuracy = (y_pred == y_test).mean()
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0) 
        
        # Class-specific metrics
        per_class_precision = precision_score(y_test, y_pred, average=None, zero_division=0)
        per_class_recall = recall_score(y_test, y_pred, average=None, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'per_class_precision': per_class_precision,
            'per_class_recall': per_class_recall,
            'confusion_matrix': cm,
            'model': rf
        }
        
        print(f"\\n   📊 {name} Results:")
        print(f"      Accuracy:  {accuracy:.4f}")
        print(f"      Precision: {precision:.4f}")
        print(f"      Recall:    {recall:.4f}")
        print(f"      F1 Score:  {f1:.4f}")
        
        print(f"\\n   📋 Per-Class Metrics:")
        class_names = ['Clean', 'Uncontrolled Movement', 'Inclination Asymmetry']
        for i, class_name in enumerate(class_names):
            if i < len(per_class_precision):
                print(f"      {class_name}:")
                print(f"         Precision: {per_class_precision[i]:.4f}")
                print(f"         Recall:    {per_class_recall[i]:.4f}")
        
        print(f"\\n   🔥 Confusion Matrix:")
        print(f"      Actual \\ Predicted    Clean  Uncontrolled  Inclination")
        print(f"      Clean                {cm[0][0]:>5}        {cm[0][1]:>5}         {cm[0][2]:>5}")
        print(f"      Uncontrolled         {cm[1][0]:>5}        {cm[1][1]:>5}         {cm[1][2]:>5}")
        if len(cm) > 2:
            print(f"      Inclination          {cm[2][0]:>5}        {cm[2][1]:>5}         {cm[2][2]:>5}")
    
    print("\\n" + "=" * 60)
    print("COMPARISON ANALYSIS")
    print("=" * 60)
    
    # Calculate false positive rates for "Clean" class (class 0)
    orig_cm = results['ORIGINAL']['confusion_matrix']
    impr_cm = results['IMPROVED']['confusion_matrix']
    
    # False positives for Clean = other classes predicted as Clean
    orig_clean_fp = orig_cm[1, 0] + (orig_cm[2, 0] if len(orig_cm) > 2 else 0)
    impr_clean_fp = impr_cm[1, 0] + (impr_cm[2, 0] if len(impr_cm) > 2 else 0)
    
    # Clean misclassified as Uncontrolled (the main problem)
    orig_clean_as_uncontrolled = orig_cm[0, 1]
    impr_clean_as_uncontrolled = impr_cm[0, 1] 
    
    print(f"\\n🎯 KEY IMPROVEMENT METRICS:")
    print(f"   Clean → Uncontrolled (THE PROBLEM):")
    print(f"      Original: {orig_clean_as_uncontrolled} cases")
    print(f"      Improved: {impr_clean_as_uncontrolled} cases")
    if orig_clean_as_uncontrolled > 0:
        improvement = (orig_clean_as_uncontrolled - impr_clean_as_uncontrolled) / orig_clean_as_uncontrolled * 100
        print(f"      Reduction: {improvement:.1f}%")
    
    print(f"\\n   Overall Performance Comparison:")
    print(f"      Precision - Original: {results['ORIGINAL']['precision']:.4f}, Improved: {results['IMPROVED']['precision']:.4f}")
    print(f"      Recall    - Original: {results['ORIGINAL']['recall']:.4f}, Improved: {results['IMPROVED']['recall']:.4f}")
    print(f"      F1 Score  - Original: {results['ORIGINAL']['f1']:.4f}, Improved: {results['IMPROVED']['f1']:.4f}")
    
    print(f"\\n   'Clean' Class Performance (Most Important):")
    print(f"      Precision - Original: {results['ORIGINAL']['per_class_precision'][0]:.4f}, Improved: {results['IMPROVED']['per_class_precision'][0]:.4f}")
    print(f"      Recall    - Original: {results['ORIGINAL']['per_class_recall'][0]:.4f}, Improved: {results['IMPROVED']['per_class_recall'][0]:.4f}")
    
    print("\\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    
    if impr_clean_as_uncontrolled < orig_clean_as_uncontrolled:
        print("✅ IMPROVED PARAMETERS ARE BETTER!")
        print("   The improved parameters reduce false positives for clean reps.")
        print("   Recommend using the improved parameter set.")
    else:
        print("⚠️  PARAMETERS NEED FURTHER TUNING")
        print("   The improved parameters didn't reduce false positives enough.")
        print("   Consider additional adjustments or threshold tuning.")

if __name__ == "__main__":
    test_improved_parameters()