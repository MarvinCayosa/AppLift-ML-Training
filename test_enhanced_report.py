"""
Test script to generate enhanced classification report
using existing merged dataset without GUI interaction.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import joblib
import os

# Scikit-learn imports
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, balanced_accuracy_score
)
from sklearn.pipeline import Pipeline

def perform_cross_validation_simple(model, X_train, y_train, n_splits=5):
    """Simple cross-validation for testing."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', model)
    ])
    
    # Cross-validation scores
    cv_accuracy = cross_val_score(cv_pipe, X_train, y_train, cv=cv, scoring='accuracy')
    cv_balanced_accuracy = cross_val_score(cv_pipe, X_train, y_train, cv=cv, scoring='balanced_accuracy')
    cv_f1_weighted = cross_val_score(cv_pipe, X_train, y_train, cv=cv, scoring='f1_weighted')
    cv_f1_macro = cross_val_score(cv_pipe, X_train, y_train, cv=cv, scoring='f1_macro')
    cv_precision_macro = cross_val_score(cv_pipe, X_train, y_train, cv=cv, scoring='precision_macro')
    cv_recall_macro = cross_val_score(cv_pipe, X_train, y_train, cv=cv, scoring='recall_macro')
    
    # Training accuracy for generalization gap
    cv_pipe.fit(X_train, y_train)
    train_accuracy = cv_pipe.score(X_train, y_train)
    
    cv_results = {
        'accuracy': cv_accuracy,
        'balanced_accuracy': cv_balanced_accuracy,
        'f1_weighted': cv_f1_weighted,
        'f1_macro': cv_f1_macro,
        'precision_macro': cv_precision_macro,
        'recall_macro': cv_recall_macro,
        'accuracy_mean': cv_accuracy.mean(),
        'accuracy_std': cv_accuracy.std(),
        'balanced_accuracy_mean': cv_balanced_accuracy.mean(),
        'balanced_accuracy_std': cv_balanced_accuracy.std(),
        'f1_weighted_mean': cv_f1_weighted.mean(),
        'f1_weighted_std': cv_f1_weighted.std(),
        'f1_macro_mean': cv_f1_macro.mean(),
        'f1_macro_std': cv_f1_macro.std(),
        'precision_macro_mean': cv_precision_macro.mean(),
        'precision_macro_std': cv_precision_macro.std(),
        'recall_macro_mean': cv_recall_macro.mean(),
        'recall_macro_std': cv_recall_macro.std(),
        'train_accuracy': train_accuracy,
        'generalization_gap': train_accuracy - cv_accuracy.mean()
    }
    
    return cv_results

def generate_enhanced_report(model, results, cv_results, feature_names, output_folder, quality_names, dimensionality_reduction_summary):
    """Generate the enhanced classification report."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Calculate stability assessment
    cv_coeff_var = cv_results['accuracy_std'] / cv_results['accuracy_mean'] * 100
    stability_assessment = "Excellent" if cv_coeff_var < 2 else "Good" if cv_coeff_var < 5 else "Moderate" if cv_coeff_var < 10 else "Poor"
    
    report_path = output_folder / f'classification_report_{timestamp}.txt'
    with open(report_path, 'w') as f:
        # Executive Summary
        f.write("=" * 80 + "\n")
        f.write("RANDOM FOREST CLASSIFIER - COMPREHENSIVE EVALUATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("EXECUTIVE SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"This report presents a comprehensive evaluation of a Random Forest classifier\n")
        f.write(f"for exercise form quality assessment. The model achieved {results['accuracy']:.1%} test accuracy\n")
        f.write(f"with {cv_results['accuracy_mean']:.1%} ± {cv_results['accuracy_std']:.1%} cross-validation accuracy.\n")
        f.write(f"Dataset dimensionality was reduced from {dimensionality_reduction_summary.get('initial_feature_count', len(feature_names))} ")
        f.write(f"to {len(feature_names)} features ({dimensionality_reduction_summary.get('reduction_percent', 0):.1f}% reduction).\n")
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
        f.write(f"SMOTE Applied:           No\n")
        f.write(f"Class Imbalance Strategy: balanced_weights\n")
        f.write("\n")
        
        # Model Configuration
        f.write("MODEL CONFIGURATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Algorithm:               Random Forest Classifier\n")
        f.write(f"n_estimators:            {model.n_estimators}\n")
        f.write(f"max_depth:               {model.max_depth if model.max_depth else 'None'}\n")
        f.write(f"min_samples_split:       {model.min_samples_split}\n")
        f.write(f"min_samples_leaf:        {model.min_samples_leaf}\n")
        f.write(f"max_features:            {model.max_features}\n")
        f.write(f"bootstrap:               {model.bootstrap}\n")
        f.write(f"class_weight:            {model.class_weight}\n")
        f.write(f"random_state:            {model.random_state}\n")
        f.write("\n")
        
        # Dataset & Dimensionality Information
        f.write("DATASET & DIMENSIONALITY ANALYSIS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Original Feature Count:    {dimensionality_reduction_summary.get('initial_feature_count', len(feature_names))}\n")
        f.write(f"Final Feature Count:       {len(feature_names)}\n")
        f.write(f"Dimensionality Reduction:  {dimensionality_reduction_summary.get('method', 'none')}\n")
        f.write(f"Feature Reduction Rate:    {dimensionality_reduction_summary.get('reduction_percent', 0):.1f}%\n")
        f.write(f"Features Retained:         {100 - dimensionality_reduction_summary.get('reduction_percent', 0):.1f}%\n")
        
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
        f.write(f"  Accuracy:           {cv_results['accuracy_mean']:.4f} ± {cv_results['accuracy_std']:.4f}\n")
        f.write(f"  Balanced Accuracy:  {cv_results['balanced_accuracy_mean']:.4f} ± {cv_results['balanced_accuracy_std']:.4f}\n")
        f.write(f"  F1 Weighted:        {cv_results['f1_weighted_mean']:.4f} ± {cv_results['f1_weighted_std']:.4f}\n")
        f.write(f"  F1 Macro:           {cv_results['f1_macro_mean']:.4f} ± {cv_results['f1_macro_std']:.4f}\n")
        f.write(f"  Precision Macro:    {cv_results['precision_macro_mean']:.4f} ± {cv_results['precision_macro_std']:.4f}\n")
        f.write(f"  Recall Macro:       {cv_results['recall_macro_mean']:.4f} ± {cv_results['recall_macro_std']:.4f}\n")
        
        # Per-fold detailed results
        f.write("\nPer-Fold Detailed Results:\n")
        f.write("Fold    Accuracy  Bal.Acc   F1-Weighted F1-Macro  Prec-Macro Rec-Macro\n")
        f.write("-" * 80 + "\n")
        for i in range(5):  # 5-fold CV
            fold_num = i + 1
            acc = cv_results['accuracy'][i]
            bal_acc = cv_results['balanced_accuracy'][i]
            f1_w = cv_results['f1_weighted'][i]
            f1_m = cv_results['f1_macro'][i]
            prec_m = cv_results['precision_macro'][i]
            rec_m = cv_results['recall_macro'][i]
            
            f.write(f"{fold_num:>4}    {acc:>7.4f}   {bal_acc:>7.4f}   {f1_w:>9.4f}   {f1_m:>7.4f}   {prec_m:>8.4f}   {rec_m:>7.4f}\n")
        
        # Statistical analysis
        f.write("\nStatistical Analysis:\n")
        acc_min = cv_results['accuracy'].min()
        acc_max = cv_results['accuracy'].max()
        acc_range = acc_max - acc_min
        f.write(f"  Accuracy Range:     {acc_range:.4f} (min: {acc_min:.4f}, max: {acc_max:.4f})\n")
        f.write(f"  Training Accuracy:  {cv_results['train_accuracy']:.4f}\n")
        f.write(f"  Validation Accuracy: {cv_results['accuracy_mean']:.4f}\n") 
        f.write(f"  Generalization Gap: {cv_results['generalization_gap']:.4f}\n")
        f.write(f"  Coefficient of Variation: {cv_coeff_var:.2f}% (model stability indicator)\n")
        f.write(f"  Stability Assessment: {stability_assessment}\n")
        f.write("\n")

        # Test set performance
        f.write("TEST SET PERFORMANCE\n")
        f.write("-" * 80 + "\n")
        f.write(results['classification_report'])
        f.write("\n")
        
        # Confusion matrix
        f.write("CONFUSION MATRIX ANALYSIS\n")
        f.write("-" * 80 + "\n")
        
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
        
        for i, actual_class in enumerate(class_names):
            f.write(f"{actual_class:<20}")
            for j in range(len(class_names)):
                if i < cm.shape[0] and j < cm.shape[1]:
                    f.write(f"{cm[i, j]:>15}")
                else:
                    f.write(f"{'0':>15}")
            f.write("\n")
        f.write("\n")
        
        # Classification metrics
        f.write("CLASSIFICATION METRICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Standard Accuracy: {results['accuracy']:.4f}\n")
        f.write(f"Balanced Accuracy: {results['balanced_accuracy']:.4f}\n")
        f.write(f"F1 Weighted: {results['f1_weighted']:.4f}\n")
        f.write(f"F1 Macro: {results['f1_macro']:.4f}\n")
        f.write("\n")
        
        # Per-class performance
        if 'per_class_metrics' in results:
            f.write("PER-CLASS DETAILED PERFORMANCE\n")
            f.write("-" * 80 + "\n")
            
            f.write("Class                Precision    Recall    F1-Score    Support\n")
            f.write("-" * 80 + "\n")
            
            for class_name, metrics in results['per_class_metrics'].items():
                f.write(f"{class_name:<20} {metrics['precision']:>9.4f} {metrics['recall']:>9.4f} {metrics['f1']:>9.4f} {metrics['support']:>9}\n")
            
            f.write("\nDetailed Per-Class Analysis:\n")
            for class_name, metrics in results['per_class_metrics'].items():
                f.write(f"\n{class_name}:\n")
                f.write(f"  • Precision: {metrics['precision']:.4f} - Of all predicted {class_name}, {metrics['precision']:.1%} were correct\n")
                f.write(f"  • Recall: {metrics['recall']:.4f} - Of all actual {class_name}, {metrics['recall']:.1%} were identified\n")
                f.write(f"  • F1-Score: {metrics['f1']:.4f} - Harmonic mean of precision and recall\n")
                f.write(f"  • Support: {metrics['support']} samples in test set\n")
        f.write("\n")
        
        # Feature importance
        f.write("FEATURE IMPORTANCE ANALYSIS\n")
        f.write("-" * 80 + "\n")
        f.write("Top 20 Most Important Features:\n")
        f.write("Rank  Feature Name                     Importance    Cumulative %\n")
        f.write("-" * 80 + "\n")
        
        cumulative_importance = 0
        for idx, (_, row) in enumerate(results['feature_importance'].head(20).iterrows(), 1):
            cumulative_importance += row['importance']
            f.write(f"{idx:>4}  {row['feature']:<30} {row['importance']:>10.4f} {cumulative_importance:>11.1%}\n")
        
        f.write(f"\nTop 20 features account for {cumulative_importance:.1%} of total importance.\n\n")
        
        # Final Summary
        f.write("COMPREHENSIVE SUMMARY & CONCLUSIONS\n")
        f.write("=" * 80 + "\n")
        
        f.write("Model Performance Summary:\n")
        f.write("-" * 40 + "\n")
        f.write(f"• Overall Accuracy:      {results['accuracy']:.1%} (Test Set)\n")
        f.write(f"• Cross-Validation:      {cv_results['accuracy_mean']:.1%} ± {cv_results['accuracy_std']:.1%} (5-fold)\n")
        f.write(f"• Balanced Accuracy:     {results['balanced_accuracy']:.1%}\n")
        f.write(f"• F1-Score (Macro):      {results['f1_macro']:.1%}\n")
        f.write(f"• Model Stability:       {stability_assessment} (CV = {cv_coeff_var:.2f}%)\n")
        f.write(f"• Generalization Gap:    {cv_results['generalization_gap']:.4f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF COMPREHENSIVE EVALUATION REPORT\n")
        f.write("=" * 80 + "\n")
    
    return report_path

def test_enhanced_report():
    """Test the enhanced report generation on existing dataset."""
    
    print("=" * 60)
    print("TESTING ENHANCED CLASSIFICATION REPORT")
    print("=" * 60)
    
    # Find existing merged dataset
    datasets_dir = Path("output/merged_datasets")
    if not datasets_dir.exists():
        print("No merged datasets found. Please run the main pipeline first.")
        return
    
    # Get the latest merged dataset
    csv_files = list(datasets_dir.glob("*.csv"))
    if not csv_files:
        print("No CSV files found in merged datasets directory.")
        return
    
    # Use the latest file
    latest_file = max(csv_files, key=os.path.getctime)
    print(f"Using dataset: {latest_file.name}")
    
    # Load data
    df = pd.read_csv(latest_file)
    print(f"Loaded {len(df):,} rows x {len(df.columns)} columns")
    
    # Check for required columns
    target_column = 'target'
    if target_column not in df.columns:
        print(f"Target column '{target_column}' not found.")
        print(f"Available columns: {list(df.columns)}")
        return
    
    # Prepare features (exclude metadata columns) 
    exclude_columns = [
        'target', 'timestamp', 'timestamp_ms', 'exercise_code', 
        'participant', 'rep', 'rep_original', 'source_file', 'source_dataset',
        'equipment_code'
    ]
    feature_columns = [col for col in df.columns if col not in exclude_columns]
    
    X = df[feature_columns]
    y = df[target_column]
    
    print(f"Features: {len(feature_columns)}")
    print(f"Classes: {sorted(y.unique())}")
    
    # Quick feature cleaning
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.mean())
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Create Random Forest model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        class_weight='balanced'
    )
    
    # Train model
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\nTraining Random Forest...")
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    
    # Perform cross-validation
    print("\nPerforming cross-validation...")
    cv_results = perform_cross_validation_simple(model, X_train, y_train, n_splits=5)
    
    # Calculate per-class metrics
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    per_class_metrics = {}
    
    quality_names = {0: 'Clean', 1: 'Uncontrolled Movement', 2: 'Abrupt Initiation'}
    
    for class_idx in sorted(y.unique()):
        class_name = quality_names.get(class_idx, f'Class {class_idx}')
        if str(class_idx) in report_dict:
            per_class_metrics[class_name] = {
                'precision': report_dict[str(class_idx)]['precision'],
                'recall': report_dict[str(class_idx)]['recall'],
                'f1': report_dict[str(class_idx)]['f1-score'],
                'support': report_dict[str(class_idx)]['support']
            }
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Prepare results
    results = {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
        'classification_report': classification_report(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'feature_importance': feature_importance,
        'per_class_metrics': per_class_metrics,
        'y_test': y_test,
        'y_pred': y_pred
    }
    
    dimensionality_reduction_summary = {
        'method': 'feature_selection_test',
        'initial_feature_count': len(feature_columns),
        'final_feature_count': len(feature_columns),
        'reduction_percent': 0.0
    }
    
    # Create output directory
    output_dir = Path("output_fixed/models/Test_Enhanced_Report")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate enhanced report
    print(f"\nGenerating enhanced classification report...")
    report_path = generate_enhanced_report(
        model=model,
        results=results,
        cv_results=cv_results,
        feature_names=feature_columns,
        output_folder=output_dir,
        quality_names=quality_names,
        dimensionality_reduction_summary=dimensionality_reduction_summary
    )
    
    print(f"Enhanced report generated successfully!")
    print(f"Report saved to: {report_path}")
    
    return report_path

if __name__ == "__main__":
    test_enhanced_report()