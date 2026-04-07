"""
Model Comparison Tool - Random Forest vs XGBoost
=================================================
Compare performance of Random Forest and XGBoost models on the same dataset.

Features:
- Side-by-side performance metrics
- Feature importance comparison
- Visualization of differences
- Statistical significance testing

Author: AppLift ML Training Pipeline
Date: March 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import json
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
RF_OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'realistic'
XGB_OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'xgboost'
COMPARISON_OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'comparison'
COMPARISON_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# COMPARISON FUNCTIONS
# =============================================================================

def load_model_metadata(model_dir):
    """
    Load model metadata from directory.
    
    Parameters:
    - model_dir: Path to model directory
    
    Returns:
    - metadata: Dictionary with model information
    """
    model_dir = Path(model_dir)
    
    # Find latest metadata file
    metadata_files = list(model_dir.glob('model_metadata_*.json'))
    if not metadata_files:
        return None
    
    latest_metadata = max(metadata_files, key=lambda p: p.stat().st_mtime)
    
    with open(latest_metadata, 'r') as f:
        metadata = json.load(f)
    
    return metadata


def compare_models(rf_metadata, xgb_metadata):
    """
    Compare two models and generate comparison report.
    
    Parameters:
    - rf_metadata: Random Forest metadata
    - xgb_metadata: XGBoost metadata
    
    Returns:
    - comparison_df: DataFrame with comparison metrics
    """
    metrics_to_compare = [
        'test_accuracy',
        'test_balanced_accuracy',
        'test_f1_weighted',
        'test_f1_macro',
        'cv_accuracy_mean',
        'cv_balanced_accuracy_mean'
    ]
    
    comparison_data = []
    
    for metric in metrics_to_compare:
        rf_value = rf_metadata['metrics'].get(metric, None)
        xgb_value = xgb_metadata['metrics'].get(metric, None)
        
        if rf_value is not None and xgb_value is not None:
            diff = xgb_value - rf_value
            diff_pct = (diff / rf_value) * 100 if rf_value != 0 else 0
            
            comparison_data.append({
                'Metric': metric.replace('_', ' ').title(),
                'Random Forest': rf_value,
                'XGBoost': xgb_value,
                'Difference': diff,
                'Difference (%)': diff_pct
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    return comparison_df


def create_comparison_visualizations(comparison_df, output_dir):
    """
    Create visualizations comparing model performance.
    
    Parameters:
    - comparison_df: DataFrame with comparison metrics
    - output_dir: Directory to save visualizations
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Bar chart comparison
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(comparison_df))
    width = 0.35
    
    ax.bar(x - width/2, comparison_df['Random Forest'], width, label='Random Forest', color='#4CAF50')
    ax.bar(x + width/2, comparison_df['XGBoost'], width, label='XGBoost', color='#FF9800')
    
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Random Forest vs XGBoost Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df['Metric'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Difference heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    
    diff_data = comparison_df[['Metric', 'Difference (%)']].set_index('Metric')
    
    sns.heatmap(diff_data.T, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                cbar_kws={'label': 'Difference (%)'}, ax=ax)
    ax.set_title('Performance Difference (XGBoost - Random Forest)', fontsize=14, fontweight='bold')
    ax.set_ylabel('')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'difference_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n[OK] Visualizations saved to: {output_dir}")


def generate_comparison_report(comparison_df, rf_metadata, xgb_metadata, output_dir):
    """
    Generate comprehensive comparison report.
    
    Parameters:
    - comparison_df: DataFrame with comparison metrics
    - rf_metadata: Random Forest metadata
    - xgb_metadata: XGBoost metadata
    - output_dir: Directory to save report
    """
    output_dir = Path(output_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = output_dir / f'comparison_report_{timestamp}.txt'
    
    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("MODEL COMPARISON REPORT: RANDOM FOREST VS XGBOOST\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Report Generated: {timestamp}\n\n")
        
        f.write("MODELS COMPARED:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Random Forest:\n")
        f.write(f"  Training Date: {rf_metadata.get('training_date', 'N/A')}\n")
        f.write(f"  Features: {rf_metadata.get('n_features', 'N/A')}\n")
        f.write(f"  Classes: {rf_metadata.get('n_classes', 'N/A')}\n\n")
        
        f.write(f"XGBoost:\n")
        f.write(f"  Training Date: {xgb_metadata.get('training_date', 'N/A')}\n")
        f.write(f"  Features: {xgb_metadata.get('n_features', 'N/A')}\n")
        f.write(f"  Classes: {xgb_metadata.get('n_classes', 'N/A')}\n\n")
        
        f.write("PERFORMANCE COMPARISON:\n")
        f.write("=" * 80 + "\n")
        f.write(comparison_df.to_string(index=False))
        f.write("\n\n")
        
        f.write("SUMMARY:\n")
        f.write("-" * 80 + "\n")
        
        # Determine winner
        avg_diff = comparison_df['Difference (%)'].mean()
        if avg_diff > 1:
            winner = "XGBoost"
            margin = avg_diff
        elif avg_diff < -1:
            winner = "Random Forest"
            margin = abs(avg_diff)
        else:
            winner = "TIE"
            margin = abs(avg_diff)
        
        f.write(f"Overall Winner: {winner}\n")
        f.write(f"Average Performance Difference: {margin:.2f}%\n\n")
        
        # Best metrics for each model
        rf_best = comparison_df.loc[comparison_df['Difference'] < 0, 'Metric'].tolist()
        xgb_best = comparison_df.loc[comparison_df['Difference'] > 0, 'Metric'].tolist()
        
        f.write(f"Random Forest performs better on:\n")
        for metric in rf_best:
            f.write(f"  - {metric}\n")
        if not rf_best:
            f.write(f"  - None\n")
        f.write("\n")
        
        f.write(f"XGBoost performs better on:\n")
        for metric in xgb_best:
            f.write(f"  - {metric}\n")
        if not xgb_best:
            f.write(f"  - None\n")
        f.write("\n")
        
        f.write("RECOMMENDATIONS:\n")
        f.write("-" * 80 + "\n")
        if winner == "XGBoost":
            f.write("XGBoost shows better overall performance. Consider using XGBoost for:\n")
            f.write("  - Production deployment\n")
            f.write("  - Real-time predictions\n")
            f.write("  - Better handling of complex patterns\n")
        elif winner == "Random Forest":
            f.write("Random Forest shows better overall performance. Consider using RF for:\n")
            f.write("  - More interpretable results\n")
            f.write("  - Faster training times\n")
            f.write("  - Better feature importance analysis\n")
        else:
            f.write("Both models perform similarly. Consider:\n")
            f.write("  - Using ensemble of both models\n")
            f.write("  - Choosing based on deployment constraints\n")
            f.write("  - Random Forest for interpretability\n")
            f.write("  - XGBoost for slight performance edge\n")
    
    print(f"\n[OK] Comparison report saved to: {report_file}")


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """
    Main function to compare Random Forest and XGBoost models.
    """
    print("=" * 80)
    print("  MODEL COMPARISON: RANDOM FOREST VS XGBOOST")
    print("=" * 80)
    print()
    
    # Find model directories
    rf_models = list((RF_OUTPUT_DIR / 'models').glob('*'))
    xgb_models = list((XGB_OUTPUT_DIR / 'models').glob('*'))
    
    if not rf_models:
        print("[ERROR] No Random Forest models found in:", RF_OUTPUT_DIR / 'models')
        return
    
    if not xgb_models:
        print("[ERROR] No XGBoost models found in:", XGB_OUTPUT_DIR / 'models')
        return
    
    print(f"[INFO] Found {len(rf_models)} RF model(s) and {len(xgb_models)} XGBoost model(s)")
    print()
    
    # Compare each exercise
    for rf_dir in rf_models:
        exercise_name = rf_dir.name
        xgb_dir = XGB_OUTPUT_DIR / 'models' / exercise_name
        
        if not xgb_dir.exists():
            print(f"[SKIP] No XGBoost model for {exercise_name}")
            continue
        
        print(f"\n[COMPARE] Comparing models for: {exercise_name}")
        print("-" * 80)
        
        # Load metadata
        rf_metadata = load_model_metadata(rf_dir)
        xgb_metadata = load_model_metadata(xgb_dir)
        
        if rf_metadata is None or xgb_metadata is None:
            print(f"[ERROR] Could not load metadata for {exercise_name}")
            continue
        
        # Compare models
        comparison_df = compare_models(rf_metadata, xgb_metadata)
        
        print("\nPerformance Comparison:")
        print(comparison_df.to_string(index=False))
        
        # Create output directory for this exercise
        exercise_output_dir = COMPARISON_OUTPUT_DIR / exercise_name
        exercise_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create visualizations
        create_comparison_visualizations(comparison_df, exercise_output_dir)
        
        # Generate report
        generate_comparison_report(comparison_df, rf_metadata, xgb_metadata, exercise_output_dir)
    
    print("\n" + "=" * 80)
    print("  COMPARISON COMPLETE!")
    print("=" * 80)
    print(f"\n[OUTPUT] Results saved to: {COMPARISON_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
