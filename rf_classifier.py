"""
AppLift ML Training - Random Forest Classifier
================================================
A comprehensive classification pipeline for exercise execution quality.

Features:
- Interactive UI for column selection
- Feature engineering (rep-level aggregations)
- Proper train/test split to prevent data leakage
- Random Forest with hyperparameter tuning (Grid Search + Random Search)
- 5-Fold Cross-Validation
- Model export to .pkl file

Author: AppLift ML Training Pipeline
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    balanced_accuracy_score, precision_recall_curve, roc_curve, auc
)
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR
OUTPUT_DIR = PROJECT_ROOT / 'output'
MODELS_DIR = OUTPUT_DIR / 'models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Target column name
TARGET_COLUMN = 'target'

# Columns to always exclude (metadata, not features)
ALWAYS_EXCLUDE = [
    'source_file', 'target_warning', 'rep_original',
    'Unnamed: 0', 'index'
]

# Equipment types mapping
EQUIPMENT_TYPES = {
    0: 'Dumbbell',
    1: 'Barbell', 
    2: 'Weight Stack'
}

# Exercise types mapping
EXERCISE_TYPES = {
    0: 'Concentration Curls',
    1: 'Overhead Extension',
    2: 'Bench Press',
    3: 'Back Squat',
    4: 'Lateral Pulldown',
    5: 'Seated Leg Extension'
}

# Quality names for display - context-aware based on exercise type
QUALITY_NAMES_BY_EXERCISE = {
    0: {0: 'Clean', 1: 'Uncontrolled Movement', 2: 'Abrupt Initiation'},  # Concentration Curls
    1: {0: 'Clean', 1: 'Uncontrolled Movement', 2: 'Abrupt Initiation'},  # Overhead Extension
    2: {0: 'Clean', 1: 'Uncontrolled Movement', 2: 'Inclination Asymmetry'},  # Bench Press
    3: {0: 'Clean', 1: 'Uncontrolled Movement', 2: 'Inclination Asymmetry'},  # Back Squat
    4: {0: 'Clean', 1: 'Pulling Too Fast', 2: 'Releasing Too Fast'},  # Lateral Pulldown
    5: {0: 'Clean', 1: 'Pulling Too Fast', 2: 'Releasing Too Fast'}   # Seated Leg Extension
}

# Default quality names (fallback for unknown exercises)
QUALITY_NAMES = {
    0: 'Clean',
    1: 'Uncontrolled Movement',
    2: 'Abrupt Initiation'
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_quality_names(exercise_code=None, df=None):
    """
    Get appropriate quality names based on exercise type
    
    Parameters:
    - exercise_code: Specific exercise code to get quality names for
    - df: DataFrame to auto-detect exercise code from
    
    Returns:
    - Dictionary mapping quality codes to names
    """
    # If exercise_code is provided directly, use it
    if exercise_code is not None:
        return QUALITY_NAMES_BY_EXERCISE.get(exercise_code, QUALITY_NAMES)
    
    # Try to auto-detect from DataFrame
    if df is not None and 'exercise_code' in df.columns:
        unique_exercises = df['exercise_code'].unique()
        if len(unique_exercises) == 1:
            # Single exercise type - use specific quality names
            exercise_code = unique_exercises[0]
            return QUALITY_NAMES_BY_EXERCISE.get(exercise_code, QUALITY_NAMES)
        elif len(unique_exercises) > 1:
            # Multiple exercise types - use generic quality names
            print(f"  ℹ️ Multiple exercises detected: {[EXERCISE_TYPES.get(ex, f'Exercise {ex}') for ex in unique_exercises]}")
            print(f"  Using generic quality names for mixed exercise dataset")
            return QUALITY_NAMES
    
    # Fallback to default quality names
    return QUALITY_NAMES


def get_dataset_info(df):
    """
    Analyze and display dataset composition (equipment, exercises, qualities)
    
    Parameters:
    - df: DataFrame to analyze
    
    Returns:
    - Dictionary with dataset composition information
    """
    info = {
        'total_samples': len(df),
        'equipment_types': {},
        'exercise_types': {},
        'quality_distribution': {}
    }
    
    # Analyze equipment types
    if 'equipment_code' in df.columns:
        for eq_code in df['equipment_code'].unique():
            eq_name = EQUIPMENT_TYPES.get(eq_code, f'Equipment {eq_code}')
            count = len(df[df['equipment_code'] == eq_code])
            info['equipment_types'][eq_name] = count
    
    # Analyze exercise types  
    if 'exercise_code' in df.columns:
        for ex_code in df['exercise_code'].unique():
            ex_name = EXERCISE_TYPES.get(ex_code, f'Exercise {ex_code}')
            count = len(df[df['exercise_code'] == ex_code])
            info['exercise_types'][ex_name] = count
    
    # Analyze quality distribution per exercise
    if 'target' in df.columns and 'exercise_code' in df.columns:
        for ex_code in df['exercise_code'].unique():
            ex_name = EXERCISE_TYPES.get(ex_code, f'Exercise {ex_code}')
            ex_data = df[df['exercise_code'] == ex_code]
            quality_names = QUALITY_NAMES_BY_EXERCISE.get(ex_code, QUALITY_NAMES)
            
            quality_dist = {}
            for quality_code in ex_data['target'].unique():
                quality_name = quality_names.get(quality_code, f'Quality {quality_code}')
                count = len(ex_data[ex_data['target'] == quality_code])
                quality_dist[quality_name] = count
            
            info['quality_distribution'][ex_name] = quality_dist
    
    return info


def display_dataset_info(info):
    """Display dataset information in a formatted way"""
    print(f"\n📊 Dataset Composition Analysis:")
    print("=" * 60)
    print(f"  Total Samples: {info['total_samples']:,}")
    
    if info['equipment_types']:
        print(f"\n🏋️ Equipment Distribution:")
        for equipment, count in info['equipment_types'].items():
            percentage = (count / info['total_samples']) * 100
            print(f"    {equipment}: {count:,} samples ({percentage:.1f}%)")
    
    if info['exercise_types']:
        print(f"\n🤸 Exercise Distribution:")
        for exercise, count in info['exercise_types'].items():
            percentage = (count / info['total_samples']) * 100
            print(f"    {exercise}: {count:,} samples ({percentage:.1f}%)")
    
    if info['quality_distribution']:
        print(f"\n🎯 Quality Distribution by Exercise:")
        for exercise, qualities in info['quality_distribution'].items():
            print(f"    {exercise}:")
            total_ex_samples = sum(qualities.values())
            for quality, count in qualities.items():
                percentage = (count / total_ex_samples) * 100 if total_ex_samples > 0 else 0
                print(f"      • {quality}: {count:,} ({percentage:.1f}%)")


# =============================================================================
# FILE SELECTION UI
# =============================================================================

def select_csv_file():
    """Open a file dialog to select a CSV file"""
    root = tk.Tk()
    root.withdraw()
    
    file_path = filedialog.askopenfilename(
        title="Select Dataset CSV File",
        initialdir=str(OUTPUT_DIR),
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    
    root.destroy()
    return file_path


# =============================================================================
# COLUMN SELECTION UI
# =============================================================================

def select_columns_ui(df, target_column=TARGET_COLUMN):
    """
    Show a UI to select which columns to include/exclude for training
    Target column is automatically hidden from selection
    
    Returns:
    - selected_columns: List of columns to use as features
    - excluded_columns: List of columns that were excluded
    """
    root = tk.Tk()
    root.title("🎯 Feature Selection for Random Forest")
    root.geometry("900x700")
    root.configure(bg='#f5f5f5')
    
    result = {'selected': None, 'excluded': None}
    
    # Header
    header_frame = tk.Frame(root, bg='#4CAF50', pady=15)
    header_frame.pack(fill=tk.X)
    
    header = tk.Label(header_frame, text="🌲 Random Forest Feature Selection", 
                      font=('Arial', 18, 'bold'), bg='#4CAF50', fg='white')
    header.pack()
    
    subtitle = tk.Label(header_frame, 
                       text=f"Select features for classification (Target: '{target_column}' - hidden)",
                       font=('Arial', 11), bg='#4CAF50', fg='white')
    subtitle.pack()
    
    # Get all columns except target and always-excluded
    all_columns = [col for col in df.columns 
                   if col != target_column and col not in ALWAYS_EXCLUDE]
    
    # Categorize columns
    numeric_cols = df[all_columns].select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [col for col in all_columns if col not in numeric_cols]
    
    # Info frame
    info_frame = tk.Frame(root, bg='#f5f5f5')
    info_frame.pack(fill=tk.X, padx=20, pady=10)
    
    info_text = f"📊 Dataset: {len(df):,} samples | {len(all_columns)} potential features\n"
    info_text += f"🔢 Numeric: {len(numeric_cols)} | 📝 Categorical: {len(categorical_cols)}"
    
    # Add equipment and exercise information
    if 'equipment_code' in df.columns:
        equipment_dist = df['equipment_code'].value_counts().sort_index()
        info_text += f"\n🏋️ Equipment: "
        for eq_code, count in equipment_dist.items():
            eq_name = EQUIPMENT_TYPES.get(eq_code, f'Equipment {eq_code}')
            info_text += f"{eq_name}={count:,} "
    
    if 'exercise_code' in df.columns:
        exercise_dist = df['exercise_code'].value_counts().sort_index()
        info_text += f"\n🤸 Exercises: "
        for ex_code, count in exercise_dist.items():
            ex_name = EXERCISE_TYPES.get(ex_code, f'Exercise {ex_code}')
            info_text += f"{ex_name}={count:,} "
    
    if target_column in df.columns:
        # Get appropriate quality names for this dataset
        quality_names = get_quality_names(df=df)
        target_dist = df[target_column].value_counts().sort_index()
        info_text += f"\n🎯 Quality: "
        for val, count in target_dist.items():
            name = quality_names.get(val, f'Quality {val}')
            info_text += f"{name}={count:,} "
    
    info_label = tk.Label(info_frame, text=info_text, font=('Arial', 10), 
                         bg='#f5f5f5', fg='#333', justify='left')
    info_label.pack(anchor='w')
    
    # Main content frame with two listboxes
    content_frame = tk.Frame(root, bg='#f5f5f5')
    content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
    
    # Left side - Available columns (to exclude)
    left_frame = tk.LabelFrame(content_frame, text="📋 Available Columns (Check to EXCLUDE)", 
                               font=('Arial', 11, 'bold'), bg='#f5f5f5')
    left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
    
    # Search box
    search_frame = tk.Frame(left_frame, bg='#f5f5f5')
    search_frame.pack(fill=tk.X, padx=5, pady=5)
    
    tk.Label(search_frame, text="🔍 Search:", bg='#f5f5f5').pack(side=tk.LEFT)
    search_var = tk.StringVar()
    search_entry = tk.Entry(search_frame, textvariable=search_var, width=30)
    search_entry.pack(side=tk.LEFT, padx=5)
    
    # Scrollable frame for checkboxes
    canvas = tk.Canvas(left_frame, bg='white', highlightthickness=0)
    scrollbar = ttk.Scrollbar(left_frame, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas, bg='white')
    
    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    # Create checkboxes for each column
    checkbox_vars = {}
    checkbox_widgets = {}
    
    # Recommended columns to exclude (IDs, timestamps that might cause leakage)
    recommended_exclude = ['timestamp_ms', 'participant', 'rep', 'equipment_code', 
                          'exercise_code', 'sample_index']
    
    for col in all_columns:
        var = tk.BooleanVar(value=col in recommended_exclude)
        checkbox_vars[col] = var
        
        # Determine column type for display
        if col in numeric_cols:
            col_type = "🔢"
            dtype_str = f"({df[col].dtype})"
        else:
            col_type = "📝"
            dtype_str = "(categorical)"
        
        # Color code recommended exclusions
        if col in recommended_exclude:
            bg_color = '#FFECB3'  # Light yellow for recommended exclude
        else:
            bg_color = 'white'
        
        frame = tk.Frame(scrollable_frame, bg=bg_color)
        frame.pack(fill=tk.X, padx=2, pady=1)
        
        cb = tk.Checkbutton(frame, text=f"{col_type} {col} {dtype_str}", 
                           variable=var, bg=bg_color, anchor='w',
                           font=('Arial', 9))
        cb.pack(fill=tk.X)
        checkbox_widgets[col] = (frame, cb, bg_color)
    
    def filter_columns(*args):
        search_term = search_var.get().lower()
        for col, (frame, cb, bg_color) in checkbox_widgets.items():
            if search_term in col.lower():
                frame.pack(fill=tk.X, padx=2, pady=1)
            else:
                frame.pack_forget()
    
    search_var.trace('w', filter_columns)
    
    # Right side - Summary and quick actions
    right_frame = tk.LabelFrame(content_frame, text="⚙️ Quick Actions & Summary", 
                                font=('Arial', 11, 'bold'), bg='#f5f5f5', width=300)
    right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
    right_frame.pack_propagate(False)  # Maintain width
    
    # Quick action buttons
    btn_frame = tk.Frame(right_frame, bg='#f5f5f5')
    btn_frame.pack(fill=tk.X, padx=10, pady=10)
    
    def select_all():
        for var in checkbox_vars.values():
            var.set(True)
        update_summary()
    
    def deselect_all():
        for var in checkbox_vars.values():
            var.set(False)
        update_summary()
    
    def select_recommended():
        for col, var in checkbox_vars.items():
            var.set(col in recommended_exclude)
        update_summary()
    
    def select_non_numeric():
        for col, var in checkbox_vars.items():
            var.set(col not in numeric_cols)
        update_summary()
    
    tk.Button(btn_frame, text="❌ Exclude All", command=select_all,
             bg='#f44336', fg='white', width=15).pack(pady=2)
    tk.Button(btn_frame, text="✅ Include All", command=deselect_all,
             bg='#4CAF50', fg='white', width=15).pack(pady=2)
    tk.Button(btn_frame, text="⭐ Recommended", command=select_recommended,
             bg='#FF9800', fg='white', width=15).pack(pady=2)
    tk.Button(btn_frame, text="🔢 Only Numeric", command=select_non_numeric,
             bg='#2196F3', fg='white', width=15).pack(pady=2)
    
    # Summary display
    summary_frame = tk.Frame(right_frame, bg='#f5f5f5')
    summary_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    summary_text = tk.Text(summary_frame, height=15, width=35, font=('Courier', 9),
                          bg='#f0f0f0', state='disabled')
    summary_text.pack(fill=tk.BOTH, expand=True)
    
    def update_summary():
        excluded = [col for col, var in checkbox_vars.items() if var.get()]
        included = [col for col, var in checkbox_vars.items() if not var.get()]
        
        summary_text.config(state='normal')
        summary_text.delete(1.0, tk.END)
        
        summary_text.insert(tk.END, f"📊 FEATURE SUMMARY\n")
        summary_text.insert(tk.END, "=" * 30 + "\n\n")
        summary_text.insert(tk.END, f"✅ Included: {len(included)} features\n")
        summary_text.insert(tk.END, f"❌ Excluded: {len(excluded)} columns\n")
        summary_text.insert(tk.END, f"🎯 Target: {target_column}\n\n")
        
        summary_text.insert(tk.END, "📋 INCLUDED FEATURES:\n")
        summary_text.insert(tk.END, "-" * 30 + "\n")
        for col in included[:15]:
            summary_text.insert(tk.END, f"  • {col}\n")
        if len(included) > 15:
            summary_text.insert(tk.END, f"  ... and {len(included)-15} more\n")
        
        summary_text.insert(tk.END, "\n❌ EXCLUDED COLUMNS:\n")
        summary_text.insert(tk.END, "-" * 30 + "\n")
        for col in excluded[:10]:
            summary_text.insert(tk.END, f"  • {col}\n")
        if len(excluded) > 10:
            summary_text.insert(tk.END, f"  ... and {len(excluded)-10} more\n")
        
        summary_text.config(state='disabled')
    
    # Bind checkbox changes to summary update
    for var in checkbox_vars.values():
        var.trace('w', lambda *args: update_summary())
    
    update_summary()
    
    # Bottom buttons
    bottom_frame = tk.Frame(root, bg='#f5f5f5')
    bottom_frame.pack(fill=tk.X, padx=20, pady=15)
    
    def confirm():
        excluded = [col for col, var in checkbox_vars.items() if var.get()]
        included = [col for col, var in checkbox_vars.items() if not var.get()]
        
        if len(included) == 0:
            messagebox.showerror("Error", "You must include at least one feature!")
            return
        
        result['selected'] = included
        result['excluded'] = excluded
        root.destroy()
    
    def cancel():
        root.destroy()
    
    tk.Button(bottom_frame, text="🚀 Train Model with Selected Features", 
             command=confirm, font=('Arial', 12, 'bold'), bg='#4CAF50', fg='white',
             padx=30, pady=10, cursor='hand2').pack(side=tk.LEFT, padx=5)
    
    tk.Button(bottom_frame, text="❌ Cancel", command=cancel,
             font=('Arial', 12), bg='#f44336', fg='white',
             padx=30, pady=10, cursor='hand2').pack(side=tk.LEFT, padx=5)
    
    # Center window
    root.update_idletasks()
    x = (root.winfo_screenwidth() - root.winfo_width()) // 2
    y = (root.winfo_screenheight() - root.winfo_height()) // 2
    root.geometry(f"+{x}+{y}")
    
    root.mainloop()
    
    return result['selected'], result['excluded']


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def compute_rep_features(df, signal_columns=None):
    """
    Compute aggregate features per rep to prevent data leakage.
    Each rep becomes one sample for the model.
    
    This approach:
    1. Groups data by participant, source_file, and rep
    2. Computes statistical features for each rep
    3. Returns one row per rep with computed features
    
    Parameters:
    - df: Raw DataFrame with sensor readings
    - signal_columns: List of signal columns to compute features from
    
    Returns:
    - features_df: DataFrame with one row per rep and computed features
    """
    print("\n📐 Computing rep-level features...")
    
    # Default signal columns if not specified
    if signal_columns is None:
        signal_columns = ['filteredMag', 'filteredX', 'filteredY', 'filteredZ',
                         'accelMag', 'accelX', 'accelY', 'accelZ',
                         'gyroMag', 'gyroX', 'gyroY', 'gyroZ']
        signal_columns = [col for col in signal_columns if col in df.columns]
    
    print(f"  Signal columns: {signal_columns}")
    
    # Group by participant, source_file, and rep
    group_cols = ['participant', 'source_file', 'rep']
    group_cols = [col for col in group_cols if col in df.columns]
    
    if 'rep' not in df.columns:
        print("  ⚠️ No 'rep' column found. Using entire dataset as single sample.")
        group_cols = ['source_file'] if 'source_file' in df.columns else []
    
    # Filter out rep 0 (usually incomplete data)
    if 'rep' in df.columns:
        df = df[df['rep'] > 0].copy()
    
    all_features = []
    
    # Group and compute features
    if group_cols:
        grouped = df.groupby(group_cols)
        total_groups = len(grouped)
        print(f"  Computing features for {total_groups} reps...")
        
        for i, (group_key, group_df) in enumerate(grouped):
            features = {}
            
            # Add group identifiers
            if isinstance(group_key, tuple):
                for j, col in enumerate(group_cols):
                    features[col] = group_key[j]
            else:
                features[group_cols[0]] = group_key
            
            # Get target (should be same for all rows in rep)
            if TARGET_COLUMN in group_df.columns:
                features[TARGET_COLUMN] = group_df[TARGET_COLUMN].iloc[0]
            
            # Get metadata (same for all rows in rep)
            for meta_col in ['equipment_code', 'exercise_code']:
                if meta_col in group_df.columns:
                    features[meta_col] = group_df[meta_col].iloc[0]
            
            # Compute time-based features
            if 'timestamp_ms' in group_df.columns:
                timestamps = group_df['timestamp_ms'].values
                features['rep_duration_ms'] = timestamps[-1] - timestamps[0]
                features['sample_count'] = len(group_df)
                if len(timestamps) > 1:
                    features['avg_sample_rate'] = 1000 / np.mean(np.diff(timestamps))
            
            # Compute statistical features for each signal column
            for col in signal_columns:
                if col in group_df.columns:
                    signal = group_df[col].dropna().values
                    
                    if len(signal) > 0:
                        # Basic statistics
                        features[f'{col}_mean'] = np.mean(signal)
                        features[f'{col}_std'] = np.std(signal)
                        features[f'{col}_min'] = np.min(signal)
                        features[f'{col}_max'] = np.max(signal)
                        features[f'{col}_range'] = np.max(signal) - np.min(signal)
                        features[f'{col}_median'] = np.median(signal)
                        
                        # Percentiles
                        features[f'{col}_p25'] = np.percentile(signal, 25)
                        features[f'{col}_p75'] = np.percentile(signal, 75)
                        features[f'{col}_iqr'] = features[f'{col}_p75'] - features[f'{col}_p25']
                        
                        # Shape statistics
                        if len(signal) > 2:
                            features[f'{col}_skew'] = pd.Series(signal).skew()
                            features[f'{col}_kurtosis'] = pd.Series(signal).kurtosis()
                        
                        # Energy and power
                        features[f'{col}_energy'] = np.sum(signal ** 2)
                        features[f'{col}_rms'] = np.sqrt(np.mean(signal ** 2))
                        
                        # Rate of change (first derivative stats)
                        if len(signal) > 1:
                            diff = np.diff(signal)
                            features[f'{col}_diff_mean'] = np.mean(diff)
                            features[f'{col}_diff_std'] = np.std(diff)
                            features[f'{col}_diff_max'] = np.max(np.abs(diff))
                        
                        # Peak-related features
                        peak_idx = np.argmax(signal)
                        features[f'{col}_peak_position'] = peak_idx / len(signal) if len(signal) > 0 else 0
                        features[f'{col}_peak_value'] = signal[peak_idx]
            
            all_features.append(features)
            
            if (i + 1) % 100 == 0:
                print(f"    Processed {i + 1}/{total_groups} reps...")
    
    features_df = pd.DataFrame(all_features)
    
    print(f"  ✓ Created {len(features_df)} samples with {len(features_df.columns)} features")
    
    return features_df



# =============================================================================
# DATA PREPARATION
# =============================================================================

def analyze_class_distribution(y, title="Dataset", exercise_code=None, df=None):
    """
    Analyze and display class distribution
    """
    print(f"\n📊 {title} Class Distribution:")
    print("=" * 40)
    
    # Get appropriate quality names
    quality_names = get_quality_names(exercise_code=exercise_code, df=df)
    
    class_counts = Counter(y)
    total = len(y)
    
    for class_id in sorted(class_counts.keys()):
        count = class_counts[class_id]
        percentage = count / total * 100
        class_name = quality_names.get(class_id, f'Class {class_id}')
        print(f"  {class_name}: {count:,} samples ({percentage:.1f}%)")
    
    # Calculate imbalance ratio
    max_count = max(class_counts.values())
    min_count = min(class_counts.values())
    imbalance_ratio = max_count / min_count
    
    print(f"\n  Imbalance Ratio: {imbalance_ratio:.2f}:1")
    
    if imbalance_ratio > 3:
        print(f"  ⚠️ HIGH IMBALANCE detected (ratio > 3:1)")
    elif imbalance_ratio > 1.5:
        print(f"  ⚠️ MODERATE IMBALANCE detected (ratio > 1.5:1)")
    else:
        print(f"  ✅ BALANCED dataset (ratio ≤ 1.5:1)")
    
    return class_counts, imbalance_ratio


def prepare_data(df, selected_features, target_column=TARGET_COLUMN):
    """
    Prepare data for training: separate features and target, handle missing values
    
    Parameters:
    - df: DataFrame with computed features
    - selected_features: List of feature columns to use
    - target_column: Name of target column
    
    Returns:
    - X: Feature matrix
    - y: Target vector
    - feature_names: List of feature names used
    """
    print("\n� Preparing data for training...")
    
    # Filter to only selected features that exist in the dataframe
    available_features = [col for col in selected_features if col in df.columns]
    missing_features = [col for col in selected_features if col not in df.columns]
    
    if missing_features:
        print(f"  ⚠️ Features not found (skipping): {missing_features[:5]}...")
    
    print(f"  Using {len(available_features)} features")
    
    # Extract features and target
    X = df[available_features].copy()
    y = df[target_column].copy()
    
    # Handle missing values in features
    missing_before = X.isnull().sum().sum()
    if missing_before > 0:
        print(f"  Filling {missing_before} missing values with column medians")
        X = X.fillna(X.median())
        X = X.fillna(0)  # Fallback for columns that are all NaN
    
    # Handle infinite values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    X = X.fillna(0)
    
    print(f"  ✓ X shape: {X.shape}, y shape: {y.shape}")
    
    # Analyze class distribution
    analyze_class_distribution(y, "Final Dataset", df=df)
    
    print(f"  ✓ Target distribution: {dict(Counter(y))}")
    
    return X, y, available_features


# =============================================================================
# HYPERPARAMETER OPTIMIZATION
# =============================================================================

def get_hyperparameter_grid():
    """
    Define comprehensive hyperparameter grid for Random Forest
    """
    param_grid = {
        'n_estimators': [100, 200, 500, 1000],
        'max_depth': [10, 15, 20, 25, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False],
        'criterion': ['gini', 'entropy'],
        'class_weight': ['balanced', 'balanced_subsample', None]  # Handle class imbalance
    }
    
    return param_grid


def perform_grid_search(X_train, y_train, cv_folds=5, n_jobs=-1, verbose=1):
    """
    Perform comprehensive Grid Search with Cross-Validation to find best hyperparameters
    
    Parameters:
    - X_train: Training features
    - y_train: Training targets
    - cv_folds: Number of cross-validation folds
    - n_jobs: Number of parallel jobs (-1 for all cores)
    - verbose: Verbosity level
    
    Returns:
    - best_params: Dictionary of best hyperparameters
    - grid_search: GridSearchCV object with results
    """
    print(f"\n� Performing Grid Search for optimal hyperparameters...")
    print(f"  CV Folds: {cv_folds} | Parallel Jobs: {n_jobs}")
    
    # Get parameter grid
    param_grid = get_hyperparameter_grid()
    
    print(f"  Parameter combinations to test: {np.prod([len(v) for v in param_grid.values()]):,}")
    print(f"  Total fits: {np.prod([len(v) for v in param_grid.values()]) * cv_folds:,}")
    
    # Create base Random Forest
    rf = RandomForestClassifier(
        random_state=42,
        n_jobs=1  # Individual trees use 1 job, GridSearch uses n_jobs
    )
    
    # Create GridSearchCV
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
        scoring='f1_weighted',  # Use weighted F1 as primary metric
        n_jobs=n_jobs,
        verbose=verbose,
        return_train_score=True,
        refit=True
    )
    
    print(f"  🚀 Starting Grid Search (this may take a while)...")
    
    # Fit Grid Search
    start_time = pd.Timestamp.now()
    grid_search.fit(X_train, y_train)
    end_time = pd.Timestamp.now()
    
    duration = (end_time - start_time).total_seconds()
    print(f"  ✅ Grid Search completed in {duration:.1f} seconds ({duration/60:.1f} minutes)")
    
    # Extract results
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    print(f"\n🏆 Best Hyperparameters Found:")
    print("=" * 50)
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    print(f"\n📊 Best Cross-Validation Score:")
    print(f"  F1 Weighted: {best_score:.4f}")
    
    # Display top 5 parameter combinations
    results_df = pd.DataFrame(grid_search.cv_results_)
    top_5 = results_df.nlargest(5, 'mean_test_score')[['mean_test_score', 'std_test_score', 'params']]
    
    print(f"\n🥇 Top 5 Parameter Combinations:")
    print("-" * 70)
    for i, (idx, row) in enumerate(top_5.iterrows(), 1):
        print(f"  {i}. Score: {row['mean_test_score']:.4f} (±{row['std_test_score']:.4f})")
        print(f"     Params: {row['params']}")
        print()
    
    return best_params, grid_search


def get_hyperparameter_distributions():
    """
    Define hyperparameter distributions for Random Search
    Uses scipy.stats distributions for more comprehensive sampling
    """
    from scipy.stats import randint, uniform
    
    param_distributions = {
        'n_estimators': randint(50, 1000),  # Random integers between 50-1000
        'max_depth': [10, 15, 20, 25, 30, None],  # Discrete choices
        'min_samples_split': randint(2, 20),  # Random integers between 2-20
        'min_samples_leaf': randint(1, 10),  # Random integers between 1-10
        'max_features': ['sqrt', 'log2', None, 0.1, 0.3, 0.5, 0.7, 0.9],  # Mix of strings and floats
        'bootstrap': [True, False],  # Boolean choices
        'criterion': ['gini', 'entropy'],  # Discrete choices
        'max_samples': [None, 0.5, 0.7, 0.8, 0.9]  # Bootstrap sample size
    }
    
    return param_distributions


def perform_random_search(X_train, y_train, n_iter=100, cv_folds=5, n_jobs=-1, verbose=1):
    """
    Perform Random Search with Cross-Validation to find good hyperparameters
    
    Parameters:
    - X_train: Training features
    - y_train: Training targets
    - n_iter: Number of parameter combinations to try
    - cv_folds: Number of cross-validation folds
    - n_jobs: Number of parallel jobs (-1 for all cores)
    - verbose: Verbosity level
    
    Returns:
    - best_params: Dictionary of best hyperparameters
    - random_search: RandomizedSearchCV object with results
    """
    print(f"\n🎲 Performing Random Search for optimal hyperparameters...")
    print(f"  Iterations: {n_iter} | CV Folds: {cv_folds} | Parallel Jobs: {n_jobs}")
    
    # Get parameter distributions
    param_distributions = get_hyperparameter_distributions()
    
    print(f"  Parameter combinations to sample: {n_iter:,}")
    print(f"  Total fits: {n_iter * cv_folds:,}")
    print(f"  Expected time: ~{n_iter * cv_folds * 0.5 / 60:.1f}-{n_iter * cv_folds * 2 / 60:.1f} minutes")
    
    # Create base Random Forest
    rf = RandomForestClassifier(
        random_state=42,
        n_jobs=1  # Individual trees use 1 job, RandomizedSearch uses n_jobs
    )
    
    # Create RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
        scoring='f1_weighted',  # Use weighted F1 as primary metric
        n_jobs=n_jobs,
        verbose=verbose,
        return_train_score=True,
        refit=True,
        random_state=42
    )
    
    print(f"  🚀 Starting Random Search...")
    
    # Fit Random Search
    start_time = pd.Timestamp.now()
    random_search.fit(X_train, y_train)
    end_time = pd.Timestamp.now()
    
    duration = (end_time - start_time).total_seconds()
    print(f"  ✅ Random Search completed in {duration:.1f} seconds ({duration/60:.1f} minutes)")
    
    # Extract results
    best_params = random_search.best_params_
    best_score = random_search.best_score_
    
    print(f"\n🏆 Best Hyperparameters Found:")
    print("=" * 50)
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    print(f"\n📊 Best Cross-Validation Score:")
    print(f"  F1 Weighted: {best_score:.4f}")
    
    # Display top 5 parameter combinations
    results_df = pd.DataFrame(random_search.cv_results_)
    top_5 = results_df.nlargest(5, 'mean_test_score')[['mean_test_score', 'std_test_score', 'params']]
    
    print(f"\n🥇 Top 5 Parameter Combinations:")
    print("-" * 70)
    for i, (idx, row) in enumerate(top_5.iterrows(), 1):
        print(f"  {i}. Score: {row['mean_test_score']:.4f} (±{row['std_test_score']:.4f})")
        print(f"     Params: {row['params']}")
        print()
    
    return best_params, random_search


def create_optimized_model(best_params):
    """
    Create Random Forest model with optimized hyperparameters
    
    Parameters:
    - best_params: Dictionary of best hyperparameters from grid search
    
    Returns:
    - model: Optimized RandomForestClassifier
    """
    print(f"\n🌲 Creating optimized Random Forest model...")
    
    # Add oob_score=True if bootstrap=True (which it is by default or explicitly set)
    model_params = best_params.copy()
    if model_params.get('bootstrap', True):  # Default is True if not specified
        model_params['oob_score'] = True
    
    model = RandomForestClassifier(
        **model_params,
        random_state=42,
        n_jobs=-1
    )
    
    print(f"  ✅ Model created with optimized parameters")
    return model


# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_random_forest(X_train, y_train, X_test, y_test, feature_names, best_params=None, exercise_code=None, df=None):
    """
    Train a Random Forest classifier with optional optimized hyperparameters
    
    Parameters:
    - X_train, X_test: Feature matrices
    - y_train, y_test: Target vectors  
    - feature_names: List of feature names
    - best_params: Optional dictionary of optimized hyperparameters from grid search
    - exercise_code: Optional exercise code for context-aware quality names
    - df: Optional dataframe for auto-detecting exercise type
    
    Returns:
    - model: Trained RandomForestClassifier
    - scaler: Fitted StandardScaler
    - results: Dictionary with evaluation metrics
    """
    print("\n🌲 Training Random Forest Classifier...")
    
    # Analyze class distribution
    analyze_class_distribution(y_train, "Training Set", exercise_code=exercise_code, df=df)
    
    # Get appropriate quality names for this dataset
    quality_names = get_quality_names(exercise_code=exercise_code, df=df)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create model with optimized parameters or defaults
    if best_params:
        print(f"  🎯 Using optimized hyperparameters from Grid Search")
        model = create_optimized_model(best_params)
    else:
        print(f"  🔧 Using default hyperparameters")
        # Default Random Forest parameters with class_weight='balanced' for imbalanced data
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            criterion='gini',
            class_weight='balanced',  # Handle class imbalance
            oob_score=True,
            random_state=42,
            n_jobs=-1
        )
    
    print("  🚀 Training model...")
    model.fit(X_train_scaled, y_train)
    
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
        'y_test': y_test,  # Store test labels for confusion matrix
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'classification_report': classification_report(y_test, y_pred, 
                                                       target_names=[quality_names.get(i, f'Class {i}') 
                                                                    for i in sorted(y_test.unique())],
                                                       zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    # Add OOB score if available
    if hasattr(model, 'oob_score_'):
        results['oob_score'] = model.oob_score_
    
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
    
    print(f"\n📊 Model Performance (Imbalanced-Aware Metrics):")
    print(f"  • Standard Accuracy: {results['accuracy']:.4f}")
    print(f"  • Balanced Accuracy: {results['balanced_accuracy']:.4f}")
    print(f"  • F1 Score (Weighted): {results['f1_weighted']:.4f}")
    print(f"  • F1 Score (Macro): {results['f1_macro']:.4f}")
    if 'oob_score' in results:
        print(f"  • OOB Score: {results['oob_score']:.4f}")
    else:
        print(f"  • OOB Score: N/A (oob_score=False)")
    
    print(f"\n📋 Per-Class Performance:")
    for class_name, metrics in results['per_class_metrics'].items():
        print(f"  • {class_name}: F1={metrics['f1']:.3f}, Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f} (n={metrics['support']})")
    
    return model, scaler, results


# =============================================================================
# CROSS-VALIDATION
# =============================================================================

def perform_cross_validation(X, y, n_splits=5, best_params=None):
    """
    Perform stratified k-fold cross-validation
    
    Parameters:
    - X: Feature matrix
    - y: Target vector
    - n_splits: Number of CV folds
    - best_params: Optional optimized hyperparameters
    
    Returns:
    - cv_results: Dictionary with CV scores and statistics
    """
    print(f"\n🔄 Performing {n_splits}-Fold Stratified Cross-Validation...")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create model with optimized or default parameters
    if best_params:
        model = create_optimized_model(best_params)
    else:
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced',  # Handle class imbalance
            oob_score=True,
            random_state=42,
            n_jobs=-1
        )
    
    # Stratified K-Fold
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Cross-validation scores
    cv_accuracy = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
    cv_balanced_accuracy = cross_val_score(model, X_scaled, y, cv=cv, scoring='balanced_accuracy')
    cv_precision_weighted = cross_val_score(model, X_scaled, y, cv=cv, scoring='precision_weighted')
    cv_recall_weighted = cross_val_score(model, X_scaled, y, cv=cv, scoring='recall_weighted')
    cv_f1_weighted = cross_val_score(model, X_scaled, y, cv=cv, scoring='f1_weighted')
    cv_precision_macro = cross_val_score(model, X_scaled, y, cv=cv, scoring='precision_macro')
    cv_recall_macro = cross_val_score(model, X_scaled, y, cv=cv, scoring='recall_macro')
    cv_f1_macro = cross_val_score(model, X_scaled, y, cv=cv, scoring='f1_macro')
    
    cv_results = {
        'accuracy': cv_accuracy,
        'balanced_accuracy': cv_balanced_accuracy,
        'precision_weighted': cv_precision_weighted,
        'recall_weighted': cv_recall_weighted,
        'f1_weighted': cv_f1_weighted,
        'precision_macro': cv_precision_macro,
        'recall_macro': cv_recall_macro,
        'f1_macro': cv_f1_macro,
        
        # Means and stds
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
    
    print(f"\n📊 Cross-Validation Results ({n_splits}-Fold):")
    print("=" * 60)
    print(f"  Standard Accuracy:  {cv_results['accuracy_mean']:.4f} ± {cv_results['accuracy_std']:.4f}")
    print(f"  Balanced Accuracy:  {cv_results['balanced_accuracy_mean']:.4f} ± {cv_results['balanced_accuracy_std']:.4f}")
    print(f"  F1 Weighted:        {cv_results['f1_weighted_mean']:.4f} ± {cv_results['f1_weighted_std']:.4f}")
    print(f"  F1 Macro:           {cv_results['f1_macro_mean']:.4f} ± {cv_results['f1_macro_std']:.4f}")
    print(f"  Precision Macro:    {cv_results['precision_macro_mean']:.4f} ± {cv_results['precision_macro_std']:.4f}")
    print(f"  Recall Macro:       {cv_results['recall_macro_mean']:.4f} ± {cv_results['recall_macro_std']:.4f}")
    print("=" * 60)
    
    print(f"\n  Per-fold Balanced Accuracy: {[f'{x:.4f}' for x in cv_balanced_accuracy]}")
    
    # Check for overfitting/underfitting using balanced accuracy
    print(f"\n🔍 Model Fit Analysis:")
    
    # Train on full data to get training score
    model.fit(X_scaled, y)
    train_balanced_accuracy = balanced_accuracy_score(y, model.predict(X_scaled))
    
    gap = train_balanced_accuracy - cv_results['balanced_accuracy_mean']
    
    print(f"  • Training Balanced Accuracy: {train_balanced_accuracy:.4f}")
    print(f"  • CV Balanced Accuracy (mean): {cv_results['balanced_accuracy_mean']:.4f}")
    print(f"  • Gap (Train - CV): {gap:.4f}")
    
    if gap > 0.15:
        print("  ⚠️ WARNING: Possible OVERFITTING detected (gap > 0.15)")
        print("     Consider: reducing max_depth, increasing min_samples_split")
    elif cv_results['balanced_accuracy_mean'] < 0.6:
        print("  ⚠️ WARNING: Possible UNDERFITTING detected (CV balanced accuracy < 0.60)")
        print("     Consider: increasing n_estimators, max_depth, or adding more features")
    else:
        print("  ✅ Model appears well-fitted (gap is reasonable)")
    
    cv_results['train_balanced_accuracy'] = train_balanced_accuracy
    cv_results['generalization_gap'] = gap
    
    return cv_results


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_model_visualizations(y_test, results, cv_results, output_folder, exercise_code=None, df=None):
    """
    Create enhanced visualizations for imbalanced classification evaluation
    """
    print("\n📊 Creating enhanced model visualizations...")
    
    # Get appropriate quality names for this dataset
    quality_names = get_quality_names(exercise_code=exercise_code, df=df)
    
    viz_folder = Path(output_folder)
    viz_folder.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('🌲 Random Forest - Imbalanced Classification Analysis', fontsize=16, fontweight='bold', y=1.02)
    
    # 1. Confusion Matrix
    cm = results['confusion_matrix']
    class_names = [quality_names.get(i, f'Class {i}') for i in sorted(y_test.unique())]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
               xticklabels=class_names, yticklabels=class_names)
    axes[0, 0].set_title('Confusion Matrix', fontweight='bold')
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('Actual')
    
    # 2. Feature Importance (Top 15)
    feature_imp = results['feature_importance'].head(15)
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(feature_imp)))[::-1]
    bars = axes[0, 1].barh(feature_imp['feature'], feature_imp['importance'], color=colors)
    axes[0, 1].set_title('Top 15 Feature Importances', fontweight='bold')
    axes[0, 1].set_xlabel('Importance')
    axes[0, 1].invert_yaxis()
    
    # 3. Class Distribution
    y_combined = np.concatenate([y_test])  # Could add train data if needed
    class_counts = Counter(y_combined)
    class_labels = [quality_names.get(i, f'Class {i}') for i in sorted(class_counts.keys())]
    counts = [class_counts[i] for i in sorted(class_counts.keys())]
    
    colors_dist = ['#4CAF50', '#FF9800', '#f44336'][:len(counts)]
    bars = axes[0, 2].bar(class_labels, counts, color=colors_dist, alpha=0.7)
    axes[0, 2].set_title('Test Set Class Distribution', fontweight='bold')
    axes[0, 2].set_ylabel('Count')
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        axes[0, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       str(count), ha='center', va='bottom', fontweight='bold')
    
    # 4. Cross-Validation Scores (Enhanced)
    metrics = ['Accuracy', 'Balanced Acc.', 'F1 Weighted', 'F1 Macro']
    means = [cv_results['accuracy_mean'], 
             cv_results.get('balanced_accuracy_mean', cv_results['accuracy_mean']),
             cv_results.get('f1_weighted_mean', 0), 
             cv_results.get('f1_macro_mean', 0)]
    stds = [cv_results['accuracy_std'], 
            cv_results.get('balanced_accuracy_std', cv_results['accuracy_std']),
            cv_results.get('f1_weighted_std', 0), 
            cv_results.get('f1_macro_std', 0)]
    
    x_pos = np.arange(len(metrics))
    bars = axes[1, 0].bar(x_pos, means, yerr=stds, capsize=5, 
                         color=['#4CAF50', '#2196F3', '#FF9800', '#9C27B0'], alpha=0.8)
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(metrics, rotation=15)
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_title('Cross-Validation Scores', fontweight='bold')
    axes[1, 0].set_ylabel('Score')
    
    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.02,
                       f'{mean:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 5. Per-Class Performance
    if 'per_class_metrics' in results:
        classes = list(results['per_class_metrics'].keys())
        f1_scores = [results['per_class_metrics'][cls]['f1'] for cls in classes]
        precisions = [results['per_class_metrics'][cls]['precision'] for cls in classes]
        recalls = [results['per_class_metrics'][cls]['recall'] for cls in classes]
        
        x = np.arange(len(classes))
        width = 0.25
        
        axes[1, 1].bar(x - width, f1_scores, width, label='F1', color='#4CAF50', alpha=0.8)
        axes[1, 1].bar(x, precisions, width, label='Precision', color='#2196F3', alpha=0.8)
        axes[1, 1].bar(x + width, recalls, width, label='Recall', color='#FF9800', alpha=0.8)
        
        axes[1, 1].set_xlabel('Class')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('Per-Class Performance', fontweight='bold')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(classes, rotation=15)
        axes[1, 1].legend()
        axes[1, 1].set_ylim(0, 1)
    
    # 6. Balanced vs Standard Accuracy Comparison
    folds = list(range(1, len(cv_results['accuracy']) + 1))
    axes[1, 2].plot(folds, cv_results['accuracy'], 'o-', linewidth=2, markersize=8, 
                   color='#4CAF50', label='Standard Accuracy')
    
    # Only plot balanced accuracy if it exists
    if 'balanced_accuracy' in cv_results:
        axes[1, 2].plot(folds, cv_results['balanced_accuracy'], 's-', linewidth=2, markersize=8,
                       color='#2196F3', label='Balanced Accuracy')
    
    # Add mean lines
    axes[1, 2].axhline(y=cv_results['accuracy_mean'], color='#4CAF50', linestyle='--', alpha=0.7)
    if 'balanced_accuracy_mean' in cv_results:
        axes[1, 2].axhline(y=cv_results['balanced_accuracy_mean'], color='#2196F3', linestyle='--', alpha=0.7)
    
    axes[1, 2].set_xlabel('Fold')
    axes[1, 2].set_ylabel('Accuracy')
    axes[1, 2].set_title('Standard vs Balanced Accuracy', fontweight='bold')
    axes[1, 2].legend()
    axes[1, 2].set_xticks(folds)
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = viz_folder / 'rf_imbalanced_evaluation.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ Saved: {output_path}")
    
    return output_path


def create_detailed_analysis_plots(results, cv_results, y_test, output_folder, exercise_code=None, df=None):
    """
    Create detailed analysis plots for comprehensive model evaluation
    """
    print("\n📈 Creating detailed analysis plots...")
    
    # Get appropriate quality names for this dataset
    quality_names = get_quality_names(exercise_code=exercise_code, df=df)
    
    viz_folder = Path(output_folder)
    viz_folder.mkdir(parents=True, exist_ok=True)
    
    # Create multiple figure sets
    
    # =============================================================================
    # PLOT SET 1: Feature Importance Analysis
    # =============================================================================
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('🔍 Feature Importance Analysis', fontsize=16, fontweight='bold', y=1.02)
    
    # 1. Top 20 Feature Importance (Horizontal Bar)
    feature_imp = results['feature_importance'].head(20)
    colors = plt.cm.viridis(np.linspace(0, 1, len(feature_imp)))
    
    axes[0, 0].barh(range(len(feature_imp)), feature_imp['importance'], color=colors)
    axes[0, 0].set_yticks(range(len(feature_imp)))
    axes[0, 0].set_yticklabels(feature_imp['feature'], fontsize=9)
    axes[0, 0].set_xlabel('Importance Score')
    axes[0, 0].set_title('Top 20 Most Important Features', fontweight='bold')
    axes[0, 0].invert_yaxis()
    axes[0, 0].grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (_, row) in enumerate(feature_imp.iterrows()):
        axes[0, 0].text(row['importance'] + 0.0005, i, f'{row["importance"]:.4f}', 
                       va='center', fontsize=8, fontweight='bold')
    
    # 2. Feature Importance Distribution
    all_importances = results['feature_importance']['importance']
    axes[0, 1].hist(all_importances, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 1].axvline(all_importances.mean(), color='red', linestyle='--', 
                      label=f'Mean: {all_importances.mean():.4f}')
    axes[0, 1].axvline(all_importances.median(), color='green', linestyle='--', 
                      label=f'Median: {all_importances.median():.4f}')
    axes[0, 1].set_xlabel('Importance Score')
    axes[0, 1].set_ylabel('Number of Features')
    axes[0, 1].set_title('Distribution of Feature Importances', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # 3. Cumulative Feature Importance
    sorted_importance = results['feature_importance'].sort_values('importance', ascending=False)
    cumulative_importance = np.cumsum(sorted_importance['importance'])
    
    axes[1, 0].plot(range(1, len(cumulative_importance) + 1), cumulative_importance, 
                   marker='o', markersize=2, linewidth=2, color='purple')
    axes[1, 0].axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='80% Threshold')
    axes[1, 0].axhline(y=0.9, color='orange', linestyle='--', alpha=0.7, label='90% Threshold')
    axes[1, 0].set_xlabel('Number of Features')
    axes[1, 0].set_ylabel('Cumulative Importance')
    axes[1, 0].set_title('Cumulative Feature Importance', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Find features needed for 80% and 90% importance
    idx_80 = np.argmax(cumulative_importance >= 0.8) + 1
    idx_90 = np.argmax(cumulative_importance >= 0.9) + 1
    axes[1, 0].annotate(f'80%: {idx_80} features', xy=(idx_80, 0.8), xytext=(idx_80+20, 0.85),
                       arrowprops=dict(arrowstyle='->', color='red'), fontweight='bold')
    axes[1, 0].annotate(f'90%: {idx_90} features', xy=(idx_90, 0.9), xytext=(idx_90+20, 0.95),
                       arrowprops=dict(arrowstyle='->', color='orange'), fontweight='bold')
    
    # 4. Feature Categories Analysis
    feature_names = results['feature_importance']['feature']
    categories = {
        'Filtered Signals': [f for f in feature_names if 'filtered' in f.lower()],
        'Acceleration': [f for f in feature_names if 'accel' in f.lower() and 'filtered' not in f.lower()],
        'Gyroscope': [f for f in feature_names if 'gyro' in f.lower()],
        'Statistical': [f for f in feature_names if any(stat in f.lower() for stat in ['mean', 'std', 'max', 'min', 'median', 'p25', 'p75'])],
        'Time/Duration': [f for f in feature_names if any(time in f.lower() for time in ['duration', 'time', 'sample_count'])],
        'Other': []
    }
    
    # Assign uncategorized features to 'Other'
    categorized = set()
    for cat_features in categories.values():
        categorized.update(cat_features)
    categories['Other'] = [f for f in feature_names if f not in categorized]
    
    # Calculate average importance per category
    category_importance = {}
    for category, features in categories.items():
        if features:
            cat_importances = [results['feature_importance'][results['feature_importance']['feature'] == f]['importance'].iloc[0] 
                             for f in features if f in feature_names.values]
            category_importance[category] = np.mean(cat_importances) if cat_importances else 0
        else:
            category_importance[category] = 0
    
    # Remove empty categories
    category_importance = {k: v for k, v in category_importance.items() if v > 0}
    
    if category_importance:
        categories_list = list(category_importance.keys())
        importance_values = list(category_importance.values())
        colors_cat = plt.cm.Set3(np.linspace(0, 1, len(categories_list)))
        
        bars = axes[1, 1].bar(categories_list, importance_values, color=colors_cat, alpha=0.8)
        axes[1, 1].set_ylabel('Average Importance')
        axes[1, 1].set_title('Average Importance by Feature Category', fontweight='bold')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, importance_values):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0002,
                           f'{value:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    feature_importance_path = viz_folder / 'feature_importance_analysis.png'
    plt.savefig(feature_importance_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {feature_importance_path}")
    
    # =============================================================================
    # PLOT SET 2: Model Performance Deep Dive
    # =============================================================================
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('📊 Model Performance Deep Dive', fontsize=16, fontweight='bold', y=1.02)
    
    # 1. Detailed Confusion Matrix with Percentages
    cm = results['confusion_matrix']
    class_names = [quality_names.get(i, f'Class {i}') for i in sorted(y_test.unique())]
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create annotation text combining counts and percentages
    annot_text = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot_text[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'
    
    sns.heatmap(cm_percent, annot=annot_text, fmt='', cmap='Blues', ax=axes[0, 0],
               xticklabels=class_names, yticklabels=class_names, cbar_kws={'label': 'Percentage'})
    axes[0, 0].set_title('Confusion Matrix with Percentages', fontweight='bold')
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('Actual')
    
    # 2. Per-Class Metrics Radar Chart
    if 'per_class_metrics' in results:
        classes = list(results['per_class_metrics'].keys())
        metrics = ['precision', 'recall', 'f1']
        
        # Setup for radar chart
        angles = np.linspace(0, 2 * np.pi, len(classes), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))  # Close the circle
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for i, metric in enumerate(metrics):
            values = [results['per_class_metrics'][cls][metric] for cls in classes]
            values += [values[0]]  # Close the circle
            
            axes[0, 1].plot(angles, values, 'o-', linewidth=2, label=metric.title(), color=colors[i])
            axes[0, 1].fill(angles, values, alpha=0.15, color=colors[i])
        
        axes[0, 1].set_xticks(angles[:-1])
        axes[0, 1].set_xticklabels(classes)
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].set_title('Per-Class Metrics (Radar Chart)', fontweight='bold')
        axes[0, 1].legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        axes[0, 1].grid(True)
    
    # 3. Cross-Validation Score Distributions
    metrics_cv = ['accuracy', 'balanced_accuracy', 'f1_weighted', 'f1_macro']
    cv_data = []
    labels = []
    
    for metric in metrics_cv:
        if metric in cv_results:
            cv_data.append(cv_results[metric])
            labels.append(metric.replace('_', ' ').title())
    
    if cv_data:
        box_plot = axes[0, 2].boxplot(cv_data, labels=labels, patch_artist=True)
        colors = ['#FFB6C1', '#87CEEB', '#98FB98', '#DDA0DD']
        for patch, color in zip(box_plot['boxes'], colors[:len(cv_data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        axes[0, 2].set_title('Cross-Validation Score Distributions', fontweight='bold')
        axes[0, 2].set_ylabel('Score')
        axes[0, 2].tick_params(axis='x', rotation=45)
        axes[0, 2].grid(axis='y', alpha=0.3)
    
    # 4. Learning Curve Simulation (using CV results as proxy)
    folds = range(1, len(cv_results['accuracy']) + 1)
    
    axes[1, 0].plot(folds, cv_results['accuracy'], 'o-', linewidth=2, markersize=8, 
                   color='blue', label='Validation Accuracy')
    if 'balanced_accuracy' in cv_results:
        axes[1, 0].plot(folds, cv_results['balanced_accuracy'], 's-', linewidth=2, markersize=8,
                       color='red', label='Validation Balanced Accuracy')
    
    # Add confidence intervals
    acc_mean = cv_results['accuracy_mean']
    acc_std = cv_results['accuracy_std']
    axes[1, 0].fill_between(folds, acc_mean - acc_std, acc_mean + acc_std, alpha=0.2, color='blue')
    
    axes[1, 0].set_xlabel('CV Fold')
    axes[1, 0].set_ylabel('Accuracy Score')
    axes[1, 0].set_title('Cross-Validation Learning Curve', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, 1)
    
    # 5. Feature Importance vs Model Performance Correlation
    top_features = results['feature_importance'].head(10)
    feature_names_short = [name[:15] + '...' if len(name) > 15 else name for name in top_features['feature']]
    
    bars = axes[1, 1].bar(range(len(top_features)), top_features['importance'], 
                         color=plt.cm.plasma(np.linspace(0, 1, len(top_features))))
    axes[1, 1].set_xticks(range(len(top_features)))
    axes[1, 1].set_xticklabels(feature_names_short, rotation=45, ha='right')
    axes[1, 1].set_ylabel('Importance Score')
    axes[1, 1].set_title('Top 10 Features Impact', fontweight='bold')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (bar, importance) in enumerate(zip(bars, top_features['importance'])):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                       f'{importance:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # 6. Model Stability Analysis
    if len(cv_results['accuracy']) > 1:
        metrics_stability = ['accuracy', 'f1_weighted']
        if 'balanced_accuracy' in cv_results:
            metrics_stability.append('balanced_accuracy')
        if 'f1_macro' in cv_results:
            metrics_stability.append('f1_macro')
        
        stability_data = []
        stability_labels = []
        
        for metric in metrics_stability:
            if metric in cv_results:
                scores = cv_results[metric]
                cv_coefficient = np.std(scores) / np.mean(scores) if np.mean(scores) != 0 else 0
                stability_data.append(cv_coefficient)
                stability_labels.append(metric.replace('_', ' ').title())
        
        if stability_data:
            bars = axes[1, 2].bar(stability_labels, stability_data, 
                                 color=['green' if x < 0.1 else 'yellow' if x < 0.2 else 'red' for x in stability_data])
            axes[1, 2].set_ylabel('Coefficient of Variation')
            axes[1, 2].set_title('Model Stability Analysis\n(Lower = More Stable)', fontweight='bold')
            axes[1, 2].tick_params(axis='x', rotation=45)
            axes[1, 2].grid(axis='y', alpha=0.3)
            
            # Add interpretation
            for bar, cv_val in zip(bars, stability_data):
                stability_text = 'Stable' if cv_val < 0.1 else 'Moderate' if cv_val < 0.2 else 'Unstable'
                axes[1, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                               f'{cv_val:.3f}\n{stability_text}', ha='center', va='bottom', 
                               fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    performance_path = viz_folder / 'model_performance_deep_dive.png'
    plt.savefig(performance_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {performance_path}")
    
    # =============================================================================
    # PLOT SET 3: Class Imbalance Analysis
    # =============================================================================
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('⚖️ Class Imbalance Analysis', fontsize=16, fontweight='bold', y=1.02)
    
    # 1. Class Distribution Pie Chart
    class_counts = Counter(y_test)
    class_labels = [quality_names.get(i, f'Class {i}') for i in sorted(class_counts.keys())]
    counts = [class_counts[i] for i in sorted(class_counts.keys())]
    colors = ['#FF9999', '#66B2FF', '#99FF99'][:len(counts)]
    
    wedges, texts, autotexts = axes[0, 0].pie(counts, labels=class_labels, autopct='%1.1f%%', 
                                             colors=colors, explode=[0.05]*len(counts))
    axes[0, 0].set_title('Class Distribution in Test Set', fontweight='bold')
    
    # Enhance the text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    # 2. Per-Class Performance vs Sample Size
    if 'per_class_metrics' in results:
        classes = list(results['per_class_metrics'].keys())
        sample_sizes = [results['per_class_metrics'][cls]['support'] for cls in classes]
        f1_scores = [results['per_class_metrics'][cls]['f1'] for cls in classes]
        
        # Create bubble plot
        bubble_sizes = [size * 3 for size in sample_sizes]  # Scale for visibility
        scatter = axes[0, 1].scatter(sample_sizes, f1_scores, s=bubble_sizes, 
                                   c=range(len(classes)), cmap='viridis', alpha=0.7)
        
        # Add class labels
        for i, (size, f1, cls) in enumerate(zip(sample_sizes, f1_scores, classes)):
            axes[0, 1].annotate(cls, (size, f1), xytext=(5, 5), textcoords='offset points',
                               fontweight='bold', fontsize=10)
        
        axes[0, 1].set_xlabel('Sample Size (Test Set)')
        axes[0, 1].set_ylabel('F1 Score')
        axes[0, 1].set_title('Performance vs Sample Size', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add trend line
        if len(sample_sizes) > 1:
            z = np.polyfit(sample_sizes, f1_scores, 1)
            p = np.poly1d(z)
            axes[0, 1].plot(sample_sizes, p(sample_sizes), "r--", alpha=0.8, linewidth=2, label='Trend')
            axes[0, 1].legend()
    
    # 3. Precision vs Recall Trade-off
    if 'per_class_metrics' in results:
        classes = list(results['per_class_metrics'].keys())
        precisions = [results['per_class_metrics'][cls]['precision'] for cls in classes]
        recalls = [results['per_class_metrics'][cls]['recall'] for cls in classes]
        
        colors_pr = ['#FF6B6B', '#4ECDC4', '#45B7D1'][:len(classes)]
        
        for i, (prec, rec, cls, color) in enumerate(zip(precisions, recalls, classes, colors_pr)):
            axes[1, 0].scatter(rec, prec, s=200, c=color, alpha=0.8, label=cls, edgecolors='black')
            axes[1, 0].annotate(f'{cls}\nF1: {results["per_class_metrics"][cls]["f1"]:.3f}', 
                               (rec, prec), xytext=(10, 10), textcoords='offset points',
                               fontweight='bold', fontsize=9, ha='left')
        
        # Add diagonal line (F1 iso-lines)
        x = np.linspace(0, 1, 100)
        for f1_line in [0.5, 0.7, 0.9]:
            y = (f1_line * x) / (2 * x - f1_line)
            y = np.where((x > f1_line/2) & (y >= 0) & (y <= 1), y, np.nan)
            axes[1, 0].plot(x, y, '--', alpha=0.5, label=f'F1={f1_line}')
        
        axes[1, 0].set_xlabel('Recall')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].set_title('Precision vs Recall Trade-off', fontweight='bold')
        axes[1, 0].set_xlim(0, 1)
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 4. Class-wise Error Analysis
    cm = results['confusion_matrix']
    class_names = [quality_names.get(i, f'Class {i}') for i in sorted(y_test.unique())]
    
    # Calculate error types for each class
    error_data = []
    error_labels = []
    
    for i, class_name in enumerate(class_names):
        true_positives = cm[i, i]
        false_negatives = cm[i, :].sum() - true_positives
        false_positives = cm[:, i].sum() - true_positives
        
        error_data.append([true_positives, false_negatives, false_positives])
        error_labels.append(class_name)
    
    error_data = np.array(error_data)
    
    x = np.arange(len(error_labels))
    width = 0.25
    
    bars1 = axes[1, 1].bar(x - width, error_data[:, 0], width, label='True Positives', 
                          color='green', alpha=0.7)
    bars2 = axes[1, 1].bar(x, error_data[:, 1], width, label='False Negatives', 
                          color='red', alpha=0.7)
    bars3 = axes[1, 1].bar(x + width, error_data[:, 2], width, label='False Positives', 
                          color='orange', alpha=0.7)
    
    axes[1, 1].set_xlabel('Classes')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Error Analysis by Class', fontweight='bold')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(error_labels)
    axes[1, 1].legend()
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                               f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    imbalance_path = viz_folder / 'class_imbalance_analysis.png'
    plt.savefig(imbalance_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {imbalance_path}")
    
    return [feature_importance_path, performance_path, imbalance_path]


# =============================================================================
# MODEL EXPORT
# =============================================================================

def export_model(model, scaler, feature_names, results, cv_results, output_folder, smote_applied=False, exercise_code=None, df=None):
    """
    Export the trained model and associated objects to a .pkl file
    """
    print("\n💾 Exporting model...")
    
    # Get appropriate quality names for this dataset
    quality_names = get_quality_names(exercise_code=exercise_code, df=df)
    
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create model package
    model_package = {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names,
        'model_type': 'RandomForestClassifier',
        'training_date': timestamp,
        'metrics': {
            'test_accuracy': results['accuracy'],
            'test_balanced_accuracy': results.get('balanced_accuracy', results['accuracy']),
            'test_precision': results['precision_weighted'],
            'test_recall': results['recall_weighted'],
            'test_f1': results['f1_weighted'],
            'test_f1_macro': results.get('f1_macro', results['f1_weighted']),
            'oob_score': results.get('oob_score', None),
            'cv_accuracy_mean': cv_results['accuracy_mean'],
            'cv_accuracy_std': cv_results['accuracy_std'],
            'cv_balanced_accuracy_mean': cv_results.get('balanced_accuracy_mean', cv_results['accuracy_mean']),
            'cv_balanced_accuracy_std': cv_results.get('balanced_accuracy_std', cv_results['accuracy_std']),
            'cv_f1_mean': cv_results.get('f1_weighted_mean', cv_results.get('f1_mean', 0)),
            'cv_f1_std': cv_results.get('f1_weighted_std', cv_results.get('f1_std', 0)),
            'cv_f1_macro_mean': cv_results.get('f1_macro_mean', 0),
            'cv_f1_macro_std': cv_results.get('f1_macro_std', 0),
            'generalization_gap': cv_results['generalization_gap']
        },
        'feature_importance': results['feature_importance'].to_dict(),
        'class_names': quality_names,
        'equipment_types': EQUIPMENT_TYPES,
        'exercise_types': EXERCISE_TYPES,
        'quality_names_by_exercise': QUALITY_NAMES_BY_EXERCISE,
        'exercise_code': exercise_code,
        'per_class_metrics': results.get('per_class_metrics', {}),
        'smote_applied': smote_applied
    }
    
    # Save model
    model_path = output_folder / f'rf_classifier_{timestamp}.pkl'
    joblib.dump(model_package, model_path)
    
    print(f"  ✓ Model saved: {model_path}")
    
    # Save feature importance to CSV
    importance_path = output_folder / f'feature_importance_{timestamp}.csv'
    results['feature_importance'].to_csv(importance_path, index=False)
    print(f"  ✓ Feature importance saved: {importance_path}")
    
    # Save classification report
    report_path = output_folder / f'classification_report_{timestamp}.txt'
    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("RANDOM FOREST CLASSIFIER - IMBALANCED DATA EVALUATION\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Training Date: {timestamp}\n")
        f.write(f"Number of Features: {len(feature_names)}\n")
        f.write(f"SMOTE Applied: {'Yes' if smote_applied else 'No'}\n\n")
        
        f.write("-" * 60 + "\n")
        f.write("TEST SET PERFORMANCE\n")
        f.write("-" * 60 + "\n")
        f.write(results['classification_report'])
        f.write("\n")
        
        # Add confusion matrix section
        f.write("-" * 60 + "\n")
        f.write("CONFUSION MATRIX\n")
        f.write("-" * 60 + "\n")
        
        # Get class names and confusion matrix
        cm = results['confusion_matrix']
        unique_classes = sorted(set(results.get('y_test', [])) if 'y_test' in results else [0, 1, 2])
        class_names = [quality_names.get(i, f'Class {i}') for i in unique_classes]
        
        # Write confusion matrix header
        f.write("Actual \\ Predicted")
        for class_name in class_names:
            f.write(f"{class_name:>12}")
        f.write("\n")
        f.write("-" * (17 + 12 * len(class_names)) + "\n")
        
        # Write confusion matrix rows
        for i, actual_class in enumerate(class_names):
            f.write(f"{actual_class:<17}")
            for j in range(len(class_names)):
                if i < cm.shape[0] and j < cm.shape[1]:
                    f.write(f"{cm[i, j]:>12}")
                else:
                    f.write(f"{'0':>12}")
            f.write("\n")
        
        # Calculate and write confusion matrix percentages
        f.write("\nConfusion Matrix (Percentages by Actual Class):\n")
        f.write("-" * 50 + "\n")
        f.write("Actual \\ Predicted")
        for class_name in class_names:
            f.write(f"{class_name:>12}")
        f.write("\n")
        f.write("-" * (17 + 12 * len(class_names)) + "\n")
        
        for i, actual_class in enumerate(class_names):
            f.write(f"{actual_class:<17}")
            if i < cm.shape[0]:
                row_sum = cm[i, :].sum()
                for j in range(len(class_names)):
                    if j < cm.shape[1] and row_sum > 0:
                        percentage = (cm[i, j] / row_sum) * 100
                        f.write(f"{percentage:>11.1f}%")
                    else:
                        f.write(f"{'0.0%':>12}")
            else:
                for j in range(len(class_names)):
                    f.write(f"{'0.0%':>12}")
            f.write("\n")
        f.write("\n")
        
        f.write("-" * 60 + "\n")
        f.write("IMBALANCE-AWARE METRICS\n")
        f.write("-" * 60 + "\n")
        f.write(f"Standard Accuracy: {results['accuracy']:.4f}\n")
        f.write(f"Balanced Accuracy: {results.get('balanced_accuracy', 'N/A'):.4f}\n" if isinstance(results.get('balanced_accuracy'), float) else f"Balanced Accuracy: N/A\n")
        f.write(f"F1 Weighted: {results['f1_weighted']:.4f}\n")
        f.write(f"F1 Macro: {results.get('f1_macro', 'N/A'):.4f}\n" if isinstance(results.get('f1_macro'), float) else f"F1 Macro: N/A\n")
        f.write("\n")
        
        if 'per_class_metrics' in results:
            f.write("-" * 60 + "\n")
            f.write("PER-CLASS PERFORMANCE\n")
            f.write("-" * 60 + "\n")
            for class_name, metrics in results['per_class_metrics'].items():
                f.write(f"  {class_name}:\n")
                f.write(f"    Precision: {metrics['precision']:.4f}\n")
                f.write(f"    Recall:    {metrics['recall']:.4f}\n")
                f.write(f"    F1 Score:  {metrics['f1']:.4f}\n")
                f.write(f"    Support:   {metrics['support']}\n\n")
        
        f.write("-" * 60 + "\n")
        f.write("5-FOLD CROSS-VALIDATION\n")
        f.write("-" * 60 + "\n")
        f.write(f"Standard Accuracy:  {cv_results['accuracy_mean']:.4f} ± {cv_results['accuracy_std']:.4f}\n")
        f.write(f"Balanced Accuracy:  {cv_results.get('balanced_accuracy_mean', 'N/A'):.4f} ± {cv_results.get('balanced_accuracy_std', 'N/A'):.4f}\n" if isinstance(cv_results.get('balanced_accuracy_mean'), float) else f"Balanced Accuracy:  N/A ± N/A\n")
        f.write(f"F1 Weighted:        {cv_results.get('f1_weighted_mean', cv_results.get('f1_mean', 'N/A')):.4f} ± {cv_results.get('f1_weighted_std', cv_results.get('f1_std', 'N/A')):.4f}\n" if isinstance(cv_results.get('f1_weighted_mean', cv_results.get('f1_mean')), float) else f"F1 Weighted:        N/A ± N/A\n")
        f.write(f"F1 Macro:           {cv_results.get('f1_macro_mean', 'N/A'):.4f} ± {cv_results.get('f1_macro_std', 'N/A'):.4f}\n" if isinstance(cv_results.get('f1_macro_mean'), float) else f"F1 Macro:           N/A ± N/A\n")
        f.write("\n")
        
        balanced_acc_key = 'train_balanced_accuracy' if 'train_balanced_accuracy' in cv_results else 'train_accuracy'
        f.write(f"Training Accuracy: {cv_results[balanced_acc_key]:.4f}\n")
        f.write(f"Generalization Gap: {cv_results['generalization_gap']:.4f}\n\n")
        
        f.write("-" * 60 + "\n")
        f.write("TOP 20 FEATURES\n")
        f.write("-" * 60 + "\n")
        for _, row in results['feature_importance'].head(20).iterrows():
            f.write(f"  {row['feature']}: {row['importance']:.4f}\n")
        
        f.write("\n" + "=" * 60 + "\n")
    
    print(f"  ✓ Report saved: {report_path}")
    
    return model_path


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_classification_pipeline():
    """
    Main function to run the complete Random Forest classification pipeline
    """
    print("\n" + "=" * 70)
    print("     🌲 RANDOM FOREST CLASSIFICATION PIPELINE")
    print("=" * 70)
    print(f"\n📂 Project Root: {PROJECT_ROOT}")
    print(f"📁 Models Directory: {MODELS_DIR}")
    
    # =========================================================================
    # STEP 1: Select CSV file
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: SELECT DATASET")
    print("=" * 70)
    
    print("\n📂 Opening file selection dialog...")
    csv_file = select_csv_file()
    
    if not csv_file:
        print("\n❌ No file selected. Exiting.")
        return
    
    print(f"\n✓ Selected file: {Path(csv_file).name}")
    
    # Load data
    print("\n📖 Loading dataset...")
    df = pd.read_csv(csv_file)
    print(f"  ✓ Loaded {len(df):,} rows × {len(df.columns)} columns")
    
    # Analyze dataset composition
    dataset_info = get_dataset_info(df)
    display_dataset_info(dataset_info)
    
    # Auto-detect exercise type for quality names
    exercise_code = None
    if 'exercise_code' in df.columns:
        unique_exercises = df['exercise_code'].unique()
        if len(unique_exercises) == 1:
            exercise_code = unique_exercises[0]
            exercise_name = EXERCISE_TYPES.get(exercise_code, f'Exercise {exercise_code}')
            print(f"\n🎯 Detected single exercise: {exercise_name}")
    
    # Check for target column
    if TARGET_COLUMN not in df.columns:
        print(f"\n❌ Error: Target column '{TARGET_COLUMN}' not found in dataset!")
        print(f"  Available columns: {list(df.columns)}")
        return
    
    # =========================================================================
    # STEP 2: Feature Engineering
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: FEATURE ENGINEERING")
    print("=" * 70)
    
    # Compute rep-level features to prevent data leakage
    features_df = compute_rep_features(df)
    
    if len(features_df) == 0:
        print("\n❌ Error: No features computed. Check your data.")
        return
    
    # =========================================================================
    # STEP 3: Column Selection UI
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: SELECT FEATURES")
    print("=" * 70)
    
    print("\n🎯 Opening feature selection dialog...")
    selected_features, excluded_columns = select_columns_ui(features_df)
    
    if selected_features is None:
        print("\n❌ Feature selection cancelled. Exiting.")
        return
    
    print(f"\n✓ Selected {len(selected_features)} features")
    print(f"✓ Excluded {len(excluded_columns)} columns")
    
    # =========================================================================
    # STEP 4: Prepare Data
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 4: PREPARE DATA FOR TRAINING")
    print("=" * 70)
    
    # Prepare data (no SMOTE needed since you have balanced Clean samples)
    X, y, feature_names = prepare_data(features_df, selected_features)
    
    # Split data BEFORE any training to prevent leakage
    print("\n📊 Splitting data (stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"  • Training set: {len(X_train)} samples")
    print(f"  • Test set: {len(X_test)} samples")
    print(f"  • Train target distribution: {dict(y_train.value_counts().sort_index())}")
    print(f"  • Test target distribution: {dict(y_test.value_counts().sort_index())}")
    
    # =========================================================================
    # STEP 5: Hyperparameter Optimization (Optional)
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 5: HYPERPARAMETER OPTIMIZATION")
    print("=" * 70)
    
    print("\n🤖 HYPERPARAMETER OPTIMIZATION")
    print("Choose your hyperparameter optimization strategy:")
    print("1. Use default parameters (fastest - ~30 seconds)")
    print("2. Random Search optimization (fast & effective - ~5-15 minutes)")
    print("3. Grid Search optimization (thorough but slow - ~30-60 minutes)")
    print("\nℹ️ Random Search often finds good parameters much faster than Grid Search!")
    
    while True:
        try:
            choice = input("\nChoose (1-3): ").strip()
            if choice in ['1', '2', '3']:
                break
            else:
                print("Please enter 1, 2, or 3")
        except KeyboardInterrupt:
            print("\n❌ Operation cancelled.")
            return
        except:
            print("Please enter 1, 2, or 3")
    
    best_params = None
    search_results = None
    
    if choice == '2':
        print(f"\n🎲 Performing Random Search optimization...")
        print("⚡ This is much faster than Grid Search and often finds excellent parameters!")
        
        # Ask for number of iterations
        print("\nHow many random combinations to try?")
        print("  50  - Quick search (~3-5 minutes)")
        print("  100 - Balanced search (~5-10 minutes) [Recommended]")
        print("  200 - Thorough search (~10-20 minutes)")
        
        while True:
            try:
                n_iter_choice = input("\nEnter number (50, 100, 200) or custom: ").strip()
                if n_iter_choice in ['50', '100', '200']:
                    n_iter = int(n_iter_choice)
                    break
                else:
                    try:
                        n_iter = int(n_iter_choice)
                        if 10 <= n_iter <= 500:
                            break
                        else:
                            print("Please enter a number between 10 and 500")
                    except ValueError:
                        print("Please enter a valid number")
            except KeyboardInterrupt:
                print("\n❌ Operation cancelled.")
                return
        
        best_params, search_results = perform_random_search(X_train, y_train, n_iter=n_iter)
        print(f"✅ Random Search completed! Using optimized parameters.")
        
    elif choice == '3':
        print(f"\n🔍 Performing Grid Search optimization...")
        print("⚠️ This tests ALL combinations and may take 30-60 minutes...")
        best_params, search_results = perform_grid_search(X_train, y_train)
        print(f"✅ Grid Search completed! Using optimized parameters.")
    else:
        print(f"\n⚡ Using default Random Forest parameters for faster training.")
    
    # =========================================================================
    # STEP 6: Train Model
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 6: TRAIN MODEL")
    print("=" * 70)
    
    model, scaler, results = train_random_forest(X_train, y_train, X_test, y_test, feature_names, best_params, exercise_code=exercise_code, df=features_df)
    
    print(f"\n📋 Classification Report:\n")
    print(results['classification_report'])
    
    # =========================================================================
    # STEP 7: Cross-Validation
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 7: 5-FOLD CROSS-VALIDATION")
    print("=" * 70)
    
    cv_results = perform_cross_validation(X, y, n_splits=5, best_params=best_params)
    
    # =========================================================================
    # STEP 7: Visualizations
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 7: CREATE VISUALIZATIONS")
    print("=" * 70)
    
    viz_folder = MODELS_DIR / 'visualizations'
    create_model_visualizations(y_test, results, cv_results, viz_folder, exercise_code=exercise_code, df=features_df)
    
    # =========================================================================
    # STEP 8: Detailed Analysis Plots
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 8: DETAILED ANALYSIS PLOTS")
    print("=" * 70)
    
    detailed_plots = create_detailed_analysis_plots(results, cv_results, y_test, viz_folder, exercise_code=exercise_code, df=features_df)
    
    # =========================================================================
    # STEP 9: Export Model
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 9: EXPORT MODEL")
    print("=" * 70)
    
    model_path = export_model(model, scaler, feature_names, results, cv_results, MODELS_DIR, smote_applied=False, exercise_code=exercise_code, df=features_df)
    
    # =========================================================================
    # Final Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("📋 PIPELINE COMPLETE - SUMMARY")
    print("=" * 70)
    
    print(f"\n🎯 Model Performance Summary:")
    print(f"  • Test Accuracy: {results['accuracy']:.4f}")
    print(f"  • Test Balanced Accuracy: {results.get('balanced_accuracy', results['accuracy']):.4f}")
    print(f"  • Test F1 Score (Weighted): {results['f1_weighted']:.4f}")
    print(f"  • Test F1 Score (Macro): {results.get('f1_macro', results['f1_weighted']):.4f}")
    print(f"  • CV Accuracy: {cv_results['accuracy_mean']:.4f} ± {cv_results['accuracy_std']:.4f}")
    print(f"  • CV Balanced Accuracy: {cv_results.get('balanced_accuracy_mean', cv_results['accuracy_mean']):.4f} ± {cv_results.get('balanced_accuracy_std', cv_results['accuracy_std']):.4f}")
    print(f"  • CV F1 Score (Weighted): {cv_results.get('f1_weighted_mean', cv_results.get('f1_mean', 0)):.4f} ± {cv_results.get('f1_weighted_std', cv_results.get('f1_std', 0)):.4f}")
    print(f"  • CV F1 Score (Macro): {cv_results.get('f1_macro_mean', 0):.4f} ± {cv_results.get('f1_macro_std', 0):.4f}")
    print(f"  • Generalization Gap: {cv_results['generalization_gap']:.4f}")
    
    print(f"\n📁 Output Files:")
    print(f"  • Model: {model_path}")
    print(f"  • Main Visualization: {viz_folder / 'rf_imbalanced_evaluation.png'}")
    print(f"  • Detailed Analysis Plots:")
    for plot_path in detailed_plots:
        print(f"    - {plot_path.name}")
    
    print(f"\n🔝 Top 5 Most Important Features:")
    for _, row in results['feature_importance'].head(5).iterrows():
        print(f"  • {row['feature']}: {row['importance']:.4f}")
    
    # Display class-specific performance
    if 'per_class_metrics' in results:
        print(f"\n📋 Per-Class Performance Summary:")
        for class_name, metrics in results['per_class_metrics'].items():
            print(f"  • {class_name}:")
            print(f"    - Precision: {metrics['precision']:.3f}")
            print(f"    - Recall: {metrics['recall']:.3f}")
            print(f"    - F1 Score: {metrics['f1']:.3f}")
            print(f"    - Support: {metrics['support']} samples")
    
    print("\n" + "=" * 70)
    print("✅ CLASSIFICATION PIPELINE COMPLETE!")
    print("📊 Check the visualizations folder for detailed analysis plots!")
    print("=" * 70 + "\n")
    
    return model, scaler, feature_names, results, cv_results


def test_smote_methods_disabled():
    """
    DISABLED: This function was removed because SMOTE handling was removed.
    Since you now have more balanced Clean samples, focus on hyperparameter tuning instead.
    """
    print("⚠️ SMOTE testing has been disabled since you now have balanced data.")
    print("   Use the grid search functionality to optimize your model instead.")
    return None


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    run_classification_pipeline()
