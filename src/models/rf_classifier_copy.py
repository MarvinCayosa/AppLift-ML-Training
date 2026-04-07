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
from sklearn.feature_selection import SelectKBest, f_classif, RFE

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

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
            print(f"  [INFO] Multiple exercises detected: {[EXERCISE_TYPES.get(ex, f'Exercise {ex}') for ex in unique_exercises]}")
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
    print(f"\n[STATS] Dataset Composition Analysis:")
    print("=" * 60)
    print(f"  Total Samples: {info['total_samples']:,}")
    
    if info['equipment_types']:
        print(f"\n[EQUIP] Equipment Distribution:")
        for equipment, count in info['equipment_types'].items():
            percentage = (count / info['total_samples']) * 100
            print(f"    {equipment}: {count:,} samples ({percentage:.1f}%)")
    
    if info['exercise_types']:
        print(f"\n[EXERCISE] Exercise Distribution:")
        for exercise, count in info['exercise_types'].items():
            percentage = (count / info['total_samples']) * 100
            print(f"    {exercise}: {count:,} samples ({percentage:.1f}%)")
    
    if info['quality_distribution']:
        print(f"\n[TARGET] Quality Distribution by Exercise:")
        for exercise, qualities in info['quality_distribution'].items():
            print(f"    {exercise}:")
            total_ex_samples = sum(qualities.values())
            for quality, count in qualities.items():
                percentage = (count / total_ex_samples) * 100 if total_ex_samples > 0 else 0
                print(f"      * {quality}: {count:,} ({percentage:.1f}%)")


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
    root.title("[TARGET] Feature Selection for Random Forest")
    root.geometry("900x700")
    root.configure(bg='#f5f5f5')
    
    result = {'selected': None, 'excluded': None}
    
    # Header
    header_frame = tk.Frame(root, bg='#4CAF50', pady=15)
    header_frame.pack(fill=tk.X)
    
    header = tk.Label(header_frame, text="[RF] Random Forest Feature Selection", 
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
    
    info_text = f"[STATS] Dataset: {len(df):,} samples | {len(all_columns)} potential features\n"
    info_text += f"[NUM] Numeric: {len(numeric_cols)} | [MEMO] Categorical: {len(categorical_cols)}"
    
    # Add equipment and exercise information
    if 'equipment_code' in df.columns:
        equipment_dist = df['equipment_code'].value_counts().sort_index()
        info_text += f"\n[EQUIP] Equipment: "
        for eq_code, count in equipment_dist.items():
            eq_name = EQUIPMENT_TYPES.get(eq_code, f'Equipment {eq_code}')
            info_text += f"{eq_name}={count:,} "
    
    if 'exercise_code' in df.columns:
        exercise_dist = df['exercise_code'].value_counts().sort_index()
        info_text += f"\n[EXERCISE] Exercises: "
        for ex_code, count in exercise_dist.items():
            ex_name = EXERCISE_TYPES.get(ex_code, f'Exercise {ex_code}')
            info_text += f"{ex_name}={count:,} "
    
    if target_column in df.columns:
        # Get appropriate quality names for this dataset
        quality_names = get_quality_names(df=df)
        target_dist = df[target_column].value_counts().sort_index()
        info_text += f"\n[TARGET] Quality: "
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
    left_frame = tk.LabelFrame(content_frame, text="[LIST] Available Columns (Check to EXCLUDE)", 
                               font=('Arial', 11, 'bold'), bg='#f5f5f5')
    left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
    
    # Search box
    search_frame = tk.Frame(left_frame, bg='#f5f5f5')
    search_frame.pack(fill=tk.X, padx=5, pady=5)
    
    tk.Label(search_frame, text="[SEARCH] Search:", bg='#f5f5f5').pack(side=tk.LEFT)
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
            col_type = "[NUM]"
            dtype_str = f"({df[col].dtype})"
        else:
            col_type = "[MEMO]"
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
    right_frame = tk.LabelFrame(content_frame, text="[SETTINGS] Quick Actions & Summary", 
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
    
    tk.Button(btn_frame, text="[ERROR] Exclude All", command=select_all,
             bg='#f44336', fg='white', width=15).pack(pady=2)
    tk.Button(btn_frame, text="[OK] Include All", command=deselect_all,
             bg='#4CAF50', fg='white', width=15).pack(pady=2)
    tk.Button(btn_frame, text="[STAR] Recommended", command=select_recommended,
             bg='#FF9800', fg='white', width=15).pack(pady=2)
    tk.Button(btn_frame, text="[NUM] Only Numeric", command=select_non_numeric,
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
        
        summary_text.insert(tk.END, f"[STATS] FEATURE SUMMARY\n")
        summary_text.insert(tk.END, "=" * 30 + "\n\n")
        summary_text.insert(tk.END, f"[OK] Included: {len(included)} features\n")
        summary_text.insert(tk.END, f"[ERROR] Excluded: {len(excluded)} columns\n")
        summary_text.insert(tk.END, f"[TARGET] Target: {target_column}\n\n")
        
        summary_text.insert(tk.END, "[LIST] INCLUDED FEATURES:\n")
        summary_text.insert(tk.END, "-" * 30 + "\n")
        for col in included[:15]:
            summary_text.insert(tk.END, f"  * {col}\n")
        if len(included) > 15:
            summary_text.insert(tk.END, f"  ... and {len(included)-15} more\n")
        
        summary_text.insert(tk.END, "\n[ERROR] EXCLUDED COLUMNS:\n")
        summary_text.insert(tk.END, "-" * 30 + "\n")
        for col in excluded[:10]:
            summary_text.insert(tk.END, f"  * {col}\n")
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
    
    tk.Button(bottom_frame, text="[START] Train Model with Selected Features", 
             command=confirm, font=('Arial', 12, 'bold'), bg='#4CAF50', fg='white',
             padx=30, pady=10, cursor='hand2').pack(side=tk.LEFT, padx=5)
    
    tk.Button(bottom_frame, text="[ERROR] Cancel", command=cancel,
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
    print("\n[DR] Computing rep-level features...")
    
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
        print("  [WARNING] No 'rep' column found. Using entire dataset as single sample.")
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
                        
                        # Jerk features (third derivative - rate of change of acceleration)
                        if len(signal) > 2:
                            # Compute jerk as third derivative (second diff)
                            jerk = np.diff(signal, n=2)  # Second order difference approximates jerk
                            features[f'{col}_jerk_mean'] = np.mean(jerk)
                            features[f'{col}_jerk_std'] = np.std(jerk)
                            features[f'{col}_jerk_max'] = np.max(np.abs(jerk))
                            features[f'{col}_jerk_rms'] = np.sqrt(np.mean(jerk ** 2))
                        
                        # Peak-related features
                        peak_idx = np.argmax(signal)
                        features[f'{col}_peak_position'] = peak_idx / len(signal) if len(signal) > 0 else 0
                        features[f'{col}_peak_value'] = signal[peak_idx]
            
            # Compute Log Dimensionless Jerk (LDLJ) for movement smoothness
            # LDLJ = -ln((duration/a_peak^2) * integraljerk^2dt)
            # Values closer to 0 = smoother, more negative = jerkier
            accel_cols = [col for col in ['accelX', 'accelY', 'accelZ'] if col in group_df.columns]
            if len(accel_cols) == 3 and 'timestamp_ms' in group_df.columns:
                try:
                    timestamps = group_df['timestamp_ms'].values
                    duration = (timestamps[-1] - timestamps[0]) / 1000.0  # Convert to seconds
                    
                    if duration > 0 and len(group_df) > 3:
                        # Get acceleration signals
                        accel_x = group_df['accelX'].dropna().values
                        accel_y = group_df['accelY'].dropna().values
                        accel_z = group_df['accelZ'].dropna().values
                        
                        # Compute total acceleration magnitude
                        accel_mag = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
                        
                        # Calculate a_peak (peak accel magnitude - mean accel magnitude)
                        a_peak = np.max(accel_mag) - np.mean(accel_mag)
                        
                        if a_peak > 0.01:  # Avoid division by very small numbers
                            # Compute jerk (third derivative) for each axis
                            jerk_x = np.diff(accel_x, n=2)
                            jerk_y = np.diff(accel_y, n=2)
                            jerk_z = np.diff(accel_z, n=2)
                            
                            # Compute squared jerk magnitude
                            jerk_squared = jerk_x**2 + jerk_y**2 + jerk_z**2
                            
                            # Integrate jerk^2 over time (sum approximation)
                            jerk_integral = np.sum(jerk_squared)
                            
                            # Compute LDLJ
                            ldlj_term = (duration / (a_peak**2)) * jerk_integral
                            
                            if ldlj_term > 0:
                                features['ldlj'] = -np.log(ldlj_term)
                                features['smoothness_score'] = -features['ldlj']  # Inverted: higher = smoother
                            else:
                                features['ldlj'] = 0
                                features['smoothness_score'] = 0
                        else:
                            features['ldlj'] = 0
                            features['smoothness_score'] = 0
                except Exception as e:
                    # If LDLJ computation fails, set to 0
                    features['ldlj'] = 0
                    features['smoothness_score'] = 0
            
            all_features.append(features)
            
            if (i + 1) % 100 == 0:
                print(f"    Processed {i + 1}/{total_groups} reps...")
    
    features_df = pd.DataFrame(all_features)
    
    print(f"  [OK] Created {len(features_df)} samples with {len(features_df.columns)} features")
    
    return features_df



# =============================================================================
# DATA PREPARATION
# =============================================================================

def analyze_class_distribution(y, title="Dataset", exercise_code=None, df=None):
    """
    Analyze and display class distribution
    """
    print(f"\n[STATS] {title} Class Distribution:")
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
        print(f"  [WARNING] HIGH IMBALANCE detected (ratio > 3:1)")
    elif imbalance_ratio > 1.5:
        print(f"  [WARNING] MODERATE IMBALANCE detected (ratio > 1.5:1)")
    else:
        print(f"  [OK] BALANCED dataset (ratio <= 1.5:1)")
    
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
    print("\n Preparing data for training...")
    
    # Filter to only selected features that exist in the dataframe
    available_features = [col for col in selected_features if col in df.columns]
    missing_features = [col for col in selected_features if col not in df.columns]
    
    if missing_features:
        print(f"  [WARNING] Features not found (skipping): {missing_features[:5]}...")
    
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
    
    print(f"  [OK] X shape: {X.shape}, y shape: {y.shape}")
    
    # Analyze class distribution
    analyze_class_distribution(y, "Final Dataset", df=df)
    
    print(f"  [OK] Target distribution: {dict(Counter(y))}")
    
    return X, y, available_features


# =============================================================================
# TRAINING CONFIGURATION & DIMENSIONALITY REDUCTION
# =============================================================================

def calculate_imbalance_ratio(y):
    """
    Calculate class imbalance ratio (majority/minority).
    """
    class_counts = Counter(y)
    if not class_counts:
        return 1.0
    
    max_count = max(class_counts.values())
    min_count = min(class_counts.values())
    if min_count == 0:
        return float('inf')
    
    return max_count / min_count


def configure_class_imbalance_strategy(y_train, quality_names=None):
    """
    Display class distribution and let user choose whether to apply class imbalance handling.
    
    Parameters:
    - y_train: Training target vector
    - quality_names: Optional dict mapping class codes to names
    
    Returns:
    - config: Dictionary with imbalance strategy settings
    """
    if quality_names is None:
        quality_names = QUALITY_NAMES
    
    class_counts = Counter(y_train)
    total_samples = len(y_train)
    imbalance_ratio = calculate_imbalance_ratio(y_train)
    
    print("\n" + "=" * 70)
    print("                    CLASS IMBALANCE ANALYSIS")
    print("=" * 70)
    
    # Display detailed class distribution
    print("\n[STATS] TRAINING SET CLASS DISTRIBUTION:")
    print("-" * 50)
    
    sorted_classes = sorted(class_counts.keys())
    max_count = max(class_counts.values())
    min_count = min(class_counts.values())
    
    for class_id in sorted_classes:
        count = class_counts[class_id]
        percentage = (count / total_samples) * 100
        class_name = quality_names.get(class_id, f'Class {class_id}')
        
        # Visual bar representation
        bar_length = int((count / max_count) * 30)
        bar = "#" * bar_length + "." * (30 - bar_length)
        
        # Mark majority/minority
        if count == max_count:
            label = " [MAJORITY]"
        elif count == min_count:
            label = " [MINORITY]"
        else:
            label = ""
        
        print(f"  {class_name}:")
        print(f"    {bar} {count:,} samples ({percentage:.1f}%){label}")
    
    print("-" * 50)
    print(f"  Total Training Samples: {total_samples:,}")
    
    # Display imbalance assessment
    print("\n[CHART] IMBALANCE ASSESSMENT:")
    print("-" * 50)
    print(f"  Imbalance Ratio: {imbalance_ratio:.2f}:1 (majority/minority)")
    
    # Clear verdict with explanation
    if imbalance_ratio > 3:
        verdict = "[WARNING]  HIGH IMBALANCE"
        explanation = "The majority class has 3x+ more samples than minority."
        recommendation = "STRONGLY RECOMMENDED to enable class imbalance strategy."
        default_choice = '1'
    elif imbalance_ratio > 1.5:
        verdict = "[WARNING]  MODERATE IMBALANCE"
        explanation = "Classes are noticeably imbalanced (1.5x-3x difference)."
        recommendation = "RECOMMENDED to enable class imbalance strategy."
        default_choice = '1'
    else:
        verdict = "[OK] BALANCED"
        explanation = "Class distribution is reasonably balanced (<1.5x difference)."
        recommendation = "Class imbalance strategy is optional."
        default_choice = '2'
    
    print(f"\n  Verdict: {verdict}")
    print(f"  {explanation}")
    print(f"  [TIP] {recommendation}")
    
    # Ask user for choice
    print("\n" + "-" * 50)
    print("CHOOSE CLASS IMBALANCE HANDLING:")
    print("-" * 50)
    print("\n1. [OK] ENABLE class_weight='balanced' (cost-sensitive learning)")
    print("   -> Adjusts class weights inversely proportional to frequencies")
    print("   -> No synthetic samples created, just re-weights the loss")
    print("   -> Fast and effective for moderate imbalance")
    print("\n2. [X] DISABLE class imbalance strategy (class_weight=None)")
    print("   -> Model treats all samples equally regardless of class frequency")
    print("   -> May bias predictions toward majority class")
    print("   -> Use only if classes are naturally balanced")
    
    smote_available_msg = "" if SMOTE_AVAILABLE else " [NOT INSTALLED - pip install imbalanced-learn]"
    print(f"\n3. [SMOTE] Apply SMOTE oversampling on training data{smote_available_msg}")
    print("   -> Creates synthetic minority samples to balance classes")
    print("   -> Applied ONLY to training set (test set stays untouched)")
    print("   -> Best for significant imbalance (>2:1 ratio)")
    print("\n4. [SMOTE+W] SMOTE + class_weight='balanced' (combined)")
    print("   -> Oversample with SMOTE AND apply cost-sensitive weighting")
    print("   -> Most aggressive imbalance handling")
    print("   -> Use for severe imbalance (>3:1 ratio)")
    
    print(f"\nDefault recommendation: Option {default_choice}")
    
    valid_choices = ['1', '2', '3', '4']
    while True:
        try:
            choice = input(f"\nEnter choice (1-4) [default={default_choice}]: ").strip()
            if choice == "":
                choice = default_choice
            
            if choice in valid_choices:
                if choice in ['3', '4'] and not SMOTE_AVAILABLE:
                    print("[ERROR] SMOTE is not installed. Install with: pip install imbalanced-learn")
                    print("        Falling back to class_weight='balanced' (option 1).")
                    choice = '1'
                break
            print("Please enter 1, 2, 3, or 4")
        except KeyboardInterrupt:
            print("\n[ERROR] Operation cancelled.")
            return None
        except Exception:
            print("Please enter 1, 2, 3, or 4")
    
    use_smote = choice in ['3', '4']
    use_class_weight = choice in ['1', '4']
    class_weight = 'balanced' if use_class_weight else None
    
    print("\n" + "-" * 50)
    if choice == '1':
        print("[OK] CLASS IMBALANCE STRATEGY: class_weight='balanced'")
    elif choice == '2':
        print("[X] CLASS IMBALANCE STRATEGY: DISABLED")
        print("   Using class_weight=None (no adjustment)")
    elif choice == '3':
        print("[OK] CLASS IMBALANCE STRATEGY: SMOTE oversampling")
        print("   Will apply SMOTE to training data before model training")
    elif choice == '4':
        print("[OK] CLASS IMBALANCE STRATEGY: SMOTE + class_weight='balanced'")
        print("   Will apply SMOTE AND cost-sensitive weighting")
    print("-" * 50)
    
    return {
        'use_imbalance_strategy': use_class_weight,
        'use_smote': use_smote,
        'class_weight': class_weight,
        'imbalance_ratio': imbalance_ratio,
        'class_distribution': dict(class_counts),
        'verdict': verdict
    }


def apply_smote(X_train, y_train, quality_names=None):
    """
    Apply SMOTE oversampling to balance the training set.
    Only applied to training data to prevent data leakage.
    
    Parameters:
    - X_train: Training feature matrix
    - y_train: Training target vector
    - quality_names: Optional dict mapping class codes to names
    
    Returns:
    - X_resampled: Resampled training features
    - y_resampled: Resampled training targets
    - smote_summary: Dictionary with SMOTE details
    """
    if not SMOTE_AVAILABLE:
        print("  [ERROR] imbalanced-learn not installed. Skipping SMOTE.")
        return X_train, y_train, {'applied': False, 'reason': 'not installed'}
    
    if quality_names is None:
        quality_names = QUALITY_NAMES
    
    print("\n[SMOTE] Applying SMOTE oversampling to training data...")
    print("-" * 50)
    
    # Display before distribution
    before_counts = Counter(y_train)
    total_before = len(y_train)
    print("  BEFORE SMOTE:")
    for class_id in sorted(before_counts.keys()):
        count = before_counts[class_id]
        class_name = quality_names.get(class_id, f'Class {class_id}')
        print(f"    {class_name}: {count:,} samples ({count/total_before*100:.1f}%)")
    
    # Determine k_neighbors: use min(5, smallest_class_count - 1)
    min_class_count = min(before_counts.values())
    k_neighbors = min(5, min_class_count - 1)
    
    if k_neighbors < 1:
        print("  [WARNING] Smallest class has too few samples for SMOTE (need >= 2).")
        print("  Skipping SMOTE.")
        return X_train, y_train, {'applied': False, 'reason': 'too few samples'}
    
    try:
        smote = SMOTE(
            random_state=42,
            k_neighbors=k_neighbors,
            n_jobs=-1
        )
        
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        
        # Display after distribution
        after_counts = Counter(y_resampled)
        total_after = len(y_resampled)
        print("\n  AFTER SMOTE:")
        for class_id in sorted(after_counts.keys()):
            count = after_counts[class_id]
            class_name = quality_names.get(class_id, f'Class {class_id}')
            added = count - before_counts.get(class_id, 0)
            added_str = f" (+{added} synthetic)" if added > 0 else ""
            print(f"    {class_name}: {count:,} samples ({count/total_after*100:.1f}%){added_str}")
        
        print(f"\n  Total samples: {total_before:,} -> {total_after:,} (+{total_after - total_before:,})")
        print(f"  k_neighbors used: {k_neighbors}")
        print("-" * 50)
        
        smote_summary = {
            'applied': True,
            'method': 'SMOTE',
            'k_neighbors': k_neighbors,
            'before_distribution': dict(before_counts),
            'after_distribution': dict(after_counts),
            'samples_before': total_before,
            'samples_after': total_after,
            'synthetic_samples': total_after - total_before
        }
        
        # Convert back to DataFrame if input was DataFrame
        if isinstance(X_train, pd.DataFrame):
            X_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
        if isinstance(y_train, pd.Series):
            y_resampled = pd.Series(y_resampled, name=y_train.name)
        
        return X_resampled, y_resampled, smote_summary
        
    except Exception as e:
        print(f"  [ERROR] SMOTE failed: {e}")
        print("  Continuing without SMOTE.")
        return X_train, y_train, {'applied': False, 'reason': str(e)}


def get_recommended_top_k(total_features):
    """
    Recommend a top-K range based on practical RF pruning guidance.
    """
    # Practical baseline suggested by user guidance: 40-60
    if total_features <= 60:
        return max(20, min(total_features, int(total_features * 0.7)))
    return min(60, total_features)


def configure_dimensionality_reduction(total_features):
    """
    Configure optional dimensionality reduction strategy with detailed guidance.
    
    Returns:
    - config: Dictionary with dimensionality reduction settings
    """
    recommended_top_k = get_recommended_top_k(total_features)
    
    print("\n" + "=" * 70)
    print("                 DIMENSIONALITY REDUCTION STRATEGY")
    print("=" * 70)
    
    # Display current feature count with assessment
    print(f"\n[STATS] CURRENT FEATURE COUNT: {total_features}")
    
    if total_features > 150:
        complexity = "[WARNING]  HIGH DIMENSIONALITY"
        complexity_desc = "Model may be too complex, risk of overfitting"
        recommended_method = "4"  # Correlation + RF importance
    elif total_features > 80:
        complexity = "[WARNING]  MODERATE DIMENSIONALITY"
        complexity_desc = "Some reduction may improve generalization"
        recommended_method = "2"  # RF importance
    else:
        complexity = "[OK] MANAGEABLE DIMENSIONALITY"
        complexity_desc = "Feature count is reasonable"
        recommended_method = "1"  # No reduction
    
    print(f"   Status: {complexity}")
    print(f"   {complexity_desc}")
    print(f"   [TIP] Recommended approach: Option {recommended_method}")
    
    # Display options with explanations
    print("\n" + "-" * 70)
    print("AVAILABLE STRATEGIES:")
    print("-" * 70)
    
    print("\n1. [X] NO DIMENSIONALITY REDUCTION")
    print("   -> Keep all features as-is")
    print("   -> Best for: Small feature sets (<60 features)")
    print("   -> Risk: May overfit with too many features")
    
    print("\n2. [TARGET] RF IMPORTANCE PRUNING (top-K features)")
    print("   -> Train RF, rank features by importance, keep top K")
    print("   -> Best for: Quick baseline, moderate reduction")
    print("   -> Practical rule: Keep 40-60 features, retrain")
    print(f"   -> Recommended K: {recommended_top_k}")
    
    print("\n3. [LINK] CORRELATION PRUNING ONLY (|rho| > threshold)")
    print("   -> Remove highly correlated features (redundant info)")
    print("   -> Best for: IMU data with many correlated signals")
    print("   -> Keeps: peak_position over peak_value, skew/kurtosis over raw max/min")
    print("   -> Default threshold: 0.90")
    
    print("\n4. [LINK]+[TARGET] CORRELATION PRUNING + RF IMPORTANCE [RECOMMENDED FOR HIGH DIM]")
    print("   -> First removes correlated features, then RF importance pruning")
    print("   -> Best for: High dimensionality (>100 features)")
    print("   -> Most thorough approach for IMU data")
    
    print("\n5. [LINK]+[CYCLE] CORRELATION PRUNING + RFE")
    print("   -> Correlation pruning followed by Recursive Feature Elimination")
    print("   -> Best for: Academic/systematic approach")
    print("   -> Slower but more thorough than pure importance pruning")
    
    print("\n6. [CYCLE] RFE ONLY (Recursive Feature Elimination)")
    print("   -> Iteratively removes least important features")
    print("   -> Best for: Systematic, defensible feature selection")
    print("   -> Slower than importance pruning but more rigorous")
    
    print("\n" + "-" * 70)
    print(f"[PIN] DEFAULT RECOMMENDATION: Option {recommended_method}")
    print("-" * 70)
    
    while True:
        try:
            choice = input(f"\nEnter choice (1-6) [default={recommended_method}]: ").strip()
            if choice == "":
                choice = recommended_method
            if choice in ['1', '2', '3', '4', '5', '6']:
                break
            print("Please enter a number from 1 to 6")
        except KeyboardInterrupt:
            print("\n[ERROR] Operation cancelled.")
            return None
        except Exception:
            print("Please enter a number from 1 to 6")
    
    method_map = {
        '1': 'none',
        '2': 'rf_importance',
        '3': 'correlation',
        '4': 'correlation_rf_importance',
        '5': 'correlation_rfe',
        '6': 'rfe'
    }
    
    config = {
        'method': method_map[choice],
        'correlation_threshold': 0.90,
        'top_k': recommended_top_k,
        'rfe_n_features': recommended_top_k,
        'rfe_step': 10
    }
    
    # Correlation threshold configuration
    if config['method'] in ['correlation', 'correlation_rf_importance', 'correlation_rfe']:
        print("\n[LINK] CORRELATION PRUNING CONFIGURATION:")
        print("   Higher threshold (0.95) = Keep more features (less aggressive)")
        print("   Lower threshold (0.85) = Remove more features (more aggressive)")
        default_threshold = "0.90"
        while True:
            try:
                threshold_input = input(
                    f"   Correlation threshold (0.80-0.99) [default={default_threshold}]: "
                ).strip()
                if threshold_input == "":
                    threshold_input = default_threshold
                
                threshold = float(threshold_input)
                if 0.80 <= threshold <= 0.99:
                    config['correlation_threshold'] = threshold
                    break
                print("   Please enter a value between 0.80 and 0.99")
            except Exception:
                print("   Please enter a valid decimal number (e.g., 0.90)")
    
    # Top-K configuration
    if config['method'] in ['rf_importance', 'correlation_rf_importance', 'rfe', 'correlation_rfe']:
        print("\n[TARGET] FEATURE COUNT CONFIGURATION:")
        print("   Practical guidance: 40-60 features usually preserve >99% performance")
        print("   Rule: If performance drops <1% after reduction, keep reduced model")
        default_k = str(recommended_top_k)
        while True:
            try:
                top_k_input = input(
                    f"   Number of features to keep (5-{total_features}) [default={default_k}]: "
                ).strip()
                if top_k_input == "":
                    top_k_input = default_k
                
                top_k = int(top_k_input)
                if 5 <= top_k <= total_features:
                    config['top_k'] = top_k
                    config['rfe_n_features'] = top_k
                    break
                print(f"   Please enter an integer between 5 and {total_features}")
            except Exception:
                print("   Please enter a valid integer")
    
    # RFE step configuration
    if config['method'] in ['rfe', 'correlation_rfe']:
        print("\n[CYCLE] RFE CONFIGURATION:")
        print("   Step size = features eliminated per iteration")
        print("   Larger step = Faster but less precise")
        default_step = "10"
        while True:
            try:
                step_input = input(f"   RFE elimination step size [default={default_step}]: ").strip()
                if step_input == "":
                    step_input = default_step
                
                step = int(step_input)
                if 1 <= step <= 50:
                    config['rfe_step'] = step
                    break
                print("   Please enter an integer between 1 and 50")
            except Exception:
                print("   Please enter a valid integer")
    
    # Display summary
    print("\n" + "-" * 70)
    print("DIMENSIONALITY REDUCTION SUMMARY:")
    print("-" * 70)
    method_names = {
        'none': 'No reduction',
        'rf_importance': 'RF Importance Pruning',
        'correlation': 'Correlation Pruning',
        'correlation_rf_importance': 'Correlation + RF Importance Pruning',
        'correlation_rfe': 'Correlation + RFE',
        'rfe': 'Recursive Feature Elimination (RFE)'
    }
    print(f"  Method: {method_names[config['method']]}")
    if config['method'] != 'none':
        if config['method'] in ['correlation', 'correlation_rf_importance', 'correlation_rfe']:
            print(f"  Correlation threshold: {config['correlation_threshold']}")
        if config['method'] in ['rf_importance', 'correlation_rf_importance', 'rfe', 'correlation_rfe']:
            print(f"  Target features: {config['top_k']}")
        if config['method'] in ['rfe', 'correlation_rfe']:
            print(f"  RFE step size: {config['rfe_step']}")
    print("-" * 70)
    
    return config


def get_feature_preference_score(feature_name):
    """
    Preference heuristic for correlated-feature pruning.
    Higher score means "prefer to keep".
    """
    name = feature_name.lower()
    score = 0
    
    # Prefer features that often capture robust shape/timing information.
    if 'peak_position' in name:
        score += 3
    if 'skew' in name:
        score += 2
    if 'kurtosis' in name:
        score += 2
    
    # De-prioritize features that are often noisy/extreme-value sensitive.
    if 'peak_value' in name:
        score -= 2
    if '_max' in name:
        score -= 1
    if '_min' in name:
        score -= 1
    
    return score


def choose_feature_from_correlated_pair(feature_a, feature_b, X_train):
    """
    Choose which feature to keep from a highly correlated pair.
    """
    score_a = get_feature_preference_score(feature_a)
    score_b = get_feature_preference_score(feature_b)
    
    if score_a > score_b:
        return feature_a, feature_b
    if score_b > score_a:
        return feature_b, feature_a
    
    # Tie-breaker: keep the higher-variance feature.
    var_a = X_train[feature_a].var()
    var_b = X_train[feature_b].var()
    
    if np.isnan(var_a):
        var_a = 0.0
    if np.isnan(var_b):
        var_b = 0.0
    
    if var_a >= var_b:
        return feature_a, feature_b
    return feature_b, feature_a


def apply_correlation_pruning(X_train, X_test, threshold=0.90):
    """
    Remove highly correlated features based on train-set correlation matrix.
    """
    print(f"\n[CORR] Applying correlation pruning (|rho| > {threshold:.2f})...")
    
    corr_matrix = X_train.corr().abs().fillna(0.0)
    features = list(X_train.columns)
    to_drop = set()
    high_corr_pairs = 0
    
    for i, feature_a in enumerate(features):
        if feature_a in to_drop:
            continue
        
        for j in range(i + 1, len(features)):
            feature_b = features[j]
            if feature_b in to_drop:
                continue
            
            corr_value = corr_matrix.iloc[i, j]
            if corr_value > threshold:
                high_corr_pairs += 1
                keep_feature, drop_feature = choose_feature_from_correlated_pair(
                    feature_a, feature_b, X_train
                )
                to_drop.add(drop_feature)
                
                # If current anchor feature is dropped, move to next anchor.
                if drop_feature == feature_a:
                    break
    
    kept_features = [feature for feature in features if feature not in to_drop]
    X_train_reduced = X_train[kept_features].copy()
    X_test_reduced = X_test[kept_features].copy()
    
    print(f"  [OK] High-correlation pairs found: {high_corr_pairs}")
    print(f"  [OK] Features dropped: {len(to_drop)}")
    print(f"  [OK] Features kept: {len(kept_features)}")
    
    summary = {
        'step': 'correlation_pruning',
        'threshold': threshold,
        'high_correlation_pairs': high_corr_pairs,
        'dropped_features': len(to_drop),
        'kept_features': len(kept_features)
    }
    
    return X_train_reduced, X_test_reduced, kept_features, summary


def get_default_rf_params(class_weight='balanced'):
    """
    Default Random Forest parameters used when hyperparameter search is skipped.
    """
    return {
        'n_estimators': 200,
        'max_depth': 20,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'bootstrap': True,
        'criterion': 'gini',
        'class_weight': class_weight
    }


def create_optimized_model(best_params=None, class_weight_setting='balanced'):
    """
    Create Random Forest model with optimized hyperparameters or defaults.
    
    Parameters:
    - best_params: Optional dictionary of best hyperparameters from search
    - class_weight_setting: Default class_weight to use if absent in params
    
    Returns:
    - model: RandomForestClassifier
    """
    if best_params:
        print(f"\n[RF] Creating optimized Random Forest model...")
        model_params = best_params.copy()
        if 'class_weight' not in model_params:
            model_params['class_weight'] = class_weight_setting
    else:
        print(f"\n[RF] Creating default Random Forest model...")
        model_params = get_default_rf_params(class_weight=class_weight_setting)
    
    # Enable OOB only when bootstrap sampling is enabled.
    if model_params.get('bootstrap', True):
        model_params['oob_score'] = True
    else:
        model_params.pop('oob_score', None)
    
    model = RandomForestClassifier(
        **model_params,
        random_state=42,
        n_jobs=-1
    )
    
    print("  [OK] Model created")
    return model



def apply_rf_importance_pruning(
    X_train,
    X_test,
    y_train,
    best_params=None,
    class_weight_setting='balanced',
    top_k=50
):
    """
    Train RF, rank by feature importance, and keep top-K features.
    """
    print(f"\n[RF] Applying RF importance pruning (top {top_k})...")
    
    selector_model = create_optimized_model(
        best_params=best_params,
        class_weight_setting=class_weight_setting
    )
    selector_model.fit(X_train, y_train)
    
    importances = pd.DataFrame({
        'feature': X_train.columns,
        'importance': selector_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    top_k = min(max(5, top_k), len(importances))
    selected_features = importances.head(top_k)['feature'].tolist()
    
    X_train_reduced = X_train[selected_features].copy()
    X_test_reduced = X_test[selected_features].copy()
    
    print(f"  [OK] Features kept: {len(selected_features)} / {X_train.shape[1]}")
    
    summary = {
        'step': 'rf_importance_pruning',
        'top_k': top_k,
        'kept_features': len(selected_features),
        'top_features_preview': importances.head(10).to_dict(orient='records')
    }
    
    return X_train_reduced, X_test_reduced, selected_features, summary


def apply_rfe_pruning(
    X_train,
    X_test,
    y_train,
    best_params=None,
    class_weight_setting='balanced',
    n_features_to_select=50,
    step=10
):
    """
    Apply Recursive Feature Elimination (RFE) using Random Forest estimator.
    """
    n_features_to_select = min(max(5, n_features_to_select), X_train.shape[1])
    step = max(1, step)
    
    print(
        f"\n[RFE] Applying RFE pruning "
        f"(target features={n_features_to_select}, step={step})..."
    )
    
    estimator = create_optimized_model(
        best_params=best_params,
        class_weight_setting=class_weight_setting
    )
    rfe = RFE(
        estimator=estimator,
        n_features_to_select=n_features_to_select,
        step=step
    )
    rfe.fit(X_train, y_train)
    
    selected_features = [
        feature
        for feature, is_selected in zip(X_train.columns, rfe.support_)
        if is_selected
    ]
    
    X_train_reduced = X_train[selected_features].copy()
    X_test_reduced = X_test[selected_features].copy()
    
    print(f"  [OK] Features kept: {len(selected_features)} / {X_train.shape[1]}")
    
    summary = {
        'step': 'rfe_pruning',
        'n_features_to_select': n_features_to_select,
        'step_size': step,
        'kept_features': len(selected_features)
    }
    
    return X_train_reduced, X_test_reduced, selected_features, summary


def apply_dimensionality_reduction(
    X_train,
    X_test,
    y_train,
    reduction_config,
    best_params=None,
    class_weight_setting='balanced'
):
    """
    Apply selected dimensionality reduction workflow on train/test sets.
    
    Returns:
    - X_train_reduced, X_test_reduced
    - reduced_feature_names
    - reduction_summary
    """
    method = reduction_config.get('method', 'none')
    initial_feature_count = X_train.shape[1]
    
    if method == 'none':
        summary = {
            'method': 'none',
            'initial_feature_count': initial_feature_count,
            'final_feature_count': initial_feature_count,
            'reduction_percent': 0.0,
            'steps': []
        }
        return X_train.copy(), X_test.copy(), list(X_train.columns), summary
    
    X_train_current = X_train.copy()
    X_test_current = X_test.copy()
    step_summaries = []
    
    if method in ['correlation', 'correlation_rf_importance', 'correlation_rfe']:
        X_train_current, X_test_current, _, corr_summary = apply_correlation_pruning(
            X_train_current,
            X_test_current,
            threshold=reduction_config.get('correlation_threshold', 0.90)
        )
        step_summaries.append(corr_summary)
    
    if method in ['rf_importance', 'correlation_rf_importance']:
        X_train_current, X_test_current, _, rf_summary = apply_rf_importance_pruning(
            X_train_current,
            X_test_current,
            y_train,
            best_params=best_params,
            class_weight_setting=class_weight_setting,
            top_k=reduction_config.get('top_k', 50)
        )
        step_summaries.append(rf_summary)
    
    if method in ['rfe', 'correlation_rfe']:
        X_train_current, X_test_current, _, rfe_summary = apply_rfe_pruning(
            X_train_current,
            X_test_current,
            y_train,
            best_params=best_params,
            class_weight_setting=class_weight_setting,
            n_features_to_select=reduction_config.get('rfe_n_features', reduction_config.get('top_k', 50)),
            step=reduction_config.get('rfe_step', 10)
        )
        step_summaries.append(rfe_summary)
    
    final_feature_count = X_train_current.shape[1]
    reduction_percent = (
        (initial_feature_count - final_feature_count) / initial_feature_count * 100
        if initial_feature_count > 0 else 0
    )
    
    summary = {
        'method': method,
        'initial_feature_count': initial_feature_count,
        'final_feature_count': final_feature_count,
        'reduction_percent': reduction_percent,
        'steps': step_summaries
    }
    
    print(
        f"\n[OK] Dimensionality reduction complete: "
        f"{initial_feature_count} -> {final_feature_count} features "
        f"({reduction_percent:.1f}% reduced)"
    )
    
    return X_train_current, X_test_current, list(X_train_current.columns), summary


# =============================================================================
# HYPERPARAMETER OPTIMIZATION
# =============================================================================

def get_hyperparameter_grid(use_imbalance_strategy=True):
    """
    Define comprehensive hyperparameter grid for Random Forest.
    """
    class_weight_options = ['balanced', 'balanced_subsample'] if use_imbalance_strategy else [None]
    
    param_grid = {
        'n_estimators': [100, 200, 500, 1000],
        'max_depth': [10, 15, 20, 25, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False],
        'criterion': ['gini', 'entropy'],
        'class_weight': class_weight_options
    }
    
    return param_grid


def perform_grid_search(X_train, y_train, cv_folds=5, n_jobs=-1, verbose=1, use_imbalance_strategy=True):
    """
    Perform comprehensive Grid Search with Cross-Validation to find best hyperparameters
    
    Parameters:
    - X_train: Training features
    - y_train: Training targets
    - cv_folds: Number of cross-validation folds
    - n_jobs: Number of parallel jobs (-1 for all cores)
    - verbose: Verbosity level
    - use_imbalance_strategy: If False, class_weight is fixed to None during search
    
    Returns:
    - best_params: Dictionary of best hyperparameters
    - grid_search: GridSearchCV object with results
    """
    print(f"\n Performing Grid Search for optimal hyperparameters...")
    print(f"  CV Folds: {cv_folds} | Parallel Jobs: {n_jobs}")
    
    # Get parameter grid
    param_grid = get_hyperparameter_grid(use_imbalance_strategy=use_imbalance_strategy)
    
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
    
    print(f"  [START] Starting Grid Search (this may take a while)...")
    
    # Fit Grid Search
    start_time = pd.Timestamp.now()
    grid_search.fit(X_train, y_train)
    end_time = pd.Timestamp.now()
    
    duration = (end_time - start_time).total_seconds()
    print(f"  [OK] Grid Search completed in {duration:.1f} seconds ({duration/60:.1f} minutes)")
    
    # Extract results
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    print(f"\n[BEST] Best Hyperparameters Found:")
    print("=" * 50)
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    print(f"\n[STATS] Best Cross-Validation Score:")
    print(f"  F1 Weighted: {best_score:.4f}")
    
    # Display top 5 parameter combinations
    results_df = pd.DataFrame(grid_search.cv_results_)
    top_5 = results_df.nlargest(5, 'mean_test_score')[['mean_test_score', 'std_test_score', 'params']]
    
    print(f"\n[TOP] Top 5 Parameter Combinations:")
    print("-" * 70)
    for i, (idx, row) in enumerate(top_5.iterrows(), 1):
        print(f"  {i}. Score: {row['mean_test_score']:.4f} (+/-{row['std_test_score']:.4f})")
        print(f"     Params: {row['params']}")
        print()
    
    return best_params, grid_search


def get_hyperparameter_distributions(use_imbalance_strategy=True):
    """
    Define hyperparameter distributions for Random Search.
    Uses scipy.stats distributions for more comprehensive sampling.
    """
    from scipy.stats import randint
    
    class_weight_options = ['balanced', 'balanced_subsample'] if use_imbalance_strategy else [None]
    
    param_distributions = {
        'n_estimators': randint(50, 1000),  # Random integers between 50-1000
        'max_depth': [10, 15, 20, 25, 30, None],  # Discrete choices
        'min_samples_split': randint(2, 20),  # Random integers between 2-20
        'min_samples_leaf': randint(1, 10),  # Random integers between 1-10
        'max_features': ['sqrt', 'log2', None, 0.1, 0.3, 0.5, 0.7, 0.9],  # Mix of strings and floats
        'bootstrap': [True, False],  # Boolean choices
        'criterion': ['gini', 'entropy'],  # Discrete choices
        'max_samples': [None, 0.5, 0.7, 0.8, 0.9],  # Bootstrap sample size
        'class_weight': class_weight_options
    }
    
    return param_distributions


def perform_random_search(
    X_train,
    y_train,
    n_iter=100,
    cv_folds=5,
    n_jobs=-1,
    verbose=1,
    use_imbalance_strategy=True
):
    """
    Perform Random Search with Cross-Validation to find good hyperparameters
    
    Parameters:
    - X_train: Training features
    - y_train: Training targets
    - n_iter: Number of parameter combinations to try
    - cv_folds: Number of cross-validation folds
    - n_jobs: Number of parallel jobs (-1 for all cores)
    - verbose: Verbosity level
    - use_imbalance_strategy: If False, class_weight is fixed to None during search
    
    Returns:
    - best_params: Dictionary of best hyperparameters
    - random_search: RandomizedSearchCV object with results
    """
    print(f"\n[RANDOM] Performing Random Search for optimal hyperparameters...")
    print(f"  Iterations: {n_iter} | CV Folds: {cv_folds} | Parallel Jobs: {n_jobs}")
    
    # Get parameter distributions
    param_distributions = get_hyperparameter_distributions(
        use_imbalance_strategy=use_imbalance_strategy
    )
    
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
    
    print(f"  [START] Starting Random Search...")
    
    # Fit Random Search
    start_time = pd.Timestamp.now()
    random_search.fit(X_train, y_train)
    end_time = pd.Timestamp.now()
    
    duration = (end_time - start_time).total_seconds()
    print(f"  [OK] Random Search completed in {duration:.1f} seconds ({duration/60:.1f} minutes)")
    
    # Extract results
    best_params = random_search.best_params_
    best_score = random_search.best_score_
    
    print(f"\n[BEST] Best Hyperparameters Found:")
    print("=" * 50)
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    print(f"\n[STATS] Best Cross-Validation Score:")
    print(f"  F1 Weighted: {best_score:.4f}")
    
    # Display top 5 parameter combinations
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

def train_random_forest(
    X_train,
    y_train,
    X_test,
    y_test,
    feature_names,
    best_params=None,
    class_weight_setting='balanced',
    exercise_code=None,
    df=None
):
    """
    Train a Random Forest classifier with optional optimized hyperparameters
    
    Parameters:
    - X_train, X_test: Feature matrices
    - y_train, y_test: Target vectors  
    - feature_names: List of feature names
    - best_params: Optional dictionary of optimized hyperparameters from grid search
    - class_weight_setting: Class imbalance strategy for default/model fallback
    - exercise_code: Optional exercise code for context-aware quality names
    - df: Optional dataframe for auto-detecting exercise type
    
    Returns:
    - model: Trained RandomForestClassifier
    - scaler: Fitted StandardScaler
    - results: Dictionary with evaluation metrics
    """
    print("\n[RF] Training Random Forest Classifier...")
    
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
        print("  Using optimized hyperparameters from search")
    else:
        print("  Using default hyperparameters")
    
    model = create_optimized_model(
        best_params=best_params,
        class_weight_setting=class_weight_setting
    )
    
    print("  Training model...")
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
    
    print(f"\n[STATS] Model Performance (Imbalanced-Aware Metrics):")
    print(f"  * Standard Accuracy: {results['accuracy']:.4f}")
    print(f"  * Balanced Accuracy: {results['balanced_accuracy']:.4f}")
    print(f"  * F1 Score (Weighted): {results['f1_weighted']:.4f}")
    print(f"  * F1 Score (Macro): {results['f1_macro']:.4f}")
    if 'oob_score' in results:
        print(f"  * OOB Score: {results['oob_score']:.4f}")
    else:
        print(f"  * OOB Score: N/A (oob_score=False)")
    
    print(f"\n[LIST] Per-Class Performance:")
    for class_name, metrics in results['per_class_metrics'].items():
        print(f"  * {class_name}: F1={metrics['f1']:.3f}, Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f} (n={metrics['support']})")
    
    return model, scaler, results


# =============================================================================
# CROSS-VALIDATION
# =============================================================================

def perform_cross_validation(X, y, n_splits=5, best_params=None, class_weight_setting='balanced'):
    """
    Perform stratified k-fold cross-validation
    
    Parameters:
    - X: Feature matrix
    - y: Target vector
    - n_splits: Number of CV folds
    - best_params: Optional optimized hyperparameters
    - class_weight_setting: Class imbalance strategy for default/model fallback
    
    Returns:
    - cv_results: Dictionary with CV scores and statistics
    """
    print(f"\n[CV] Performing {n_splits}-Fold Stratified Cross-Validation...")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create model with optimized or default parameters
    model = create_optimized_model(
        best_params=best_params,
        class_weight_setting=class_weight_setting
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
    
    print(f"\n[STATS] Cross-Validation Results ({n_splits}-Fold):")
    print("=" * 60)
    print(f"  Standard Accuracy:  {cv_results['accuracy_mean']:.4f} +/- {cv_results['accuracy_std']:.4f}")
    print(f"  Balanced Accuracy:  {cv_results['balanced_accuracy_mean']:.4f} +/- {cv_results['balanced_accuracy_std']:.4f}")
    print(f"  F1 Weighted:        {cv_results['f1_weighted_mean']:.4f} +/- {cv_results['f1_weighted_std']:.4f}")
    print(f"  F1 Macro:           {cv_results['f1_macro_mean']:.4f} +/- {cv_results['f1_macro_std']:.4f}")
    print(f"  Precision Macro:    {cv_results['precision_macro_mean']:.4f} +/- {cv_results['precision_macro_std']:.4f}")
    print(f"  Recall Macro:       {cv_results['recall_macro_mean']:.4f} +/- {cv_results['recall_macro_std']:.4f}")
    print("=" * 60)
    
    print(f"\n  Per-fold Balanced Accuracy: {[f'{x:.4f}' for x in cv_balanced_accuracy]}")
    
    # Check for overfitting/underfitting using balanced accuracy
    print(f"\n[SEARCH] Model Fit Analysis:")
    
    # Train on full data to get training score
    model.fit(X_scaled, y)
    train_balanced_accuracy = balanced_accuracy_score(y, model.predict(X_scaled))
    
    gap = train_balanced_accuracy - cv_results['balanced_accuracy_mean']
    
    print(f"  * Training Balanced Accuracy: {train_balanced_accuracy:.4f}")
    print(f"  * CV Balanced Accuracy (mean): {cv_results['balanced_accuracy_mean']:.4f}")
    print(f"  * Gap (Train - CV): {gap:.4f}")
    
    if gap > 0.15:
        print("  [WARNING] WARNING: Possible OVERFITTING detected (gap > 0.15)")
        print("     Consider: reducing max_depth, increasing min_samples_split")
    elif cv_results['balanced_accuracy_mean'] < 0.6:
        print("  [WARNING] WARNING: Possible UNDERFITTING detected (CV balanced accuracy < 0.60)")
        print("     Consider: increasing n_estimators, max_depth, or adding more features")
    else:
        print("  [OK] Model appears well-fitted (gap is reasonable)")
    
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
    print("\n[STATS] Creating enhanced model visualizations...")
    
    # Get appropriate quality names for this dataset
    quality_names = get_quality_names(exercise_code=exercise_code, df=df)
    
    viz_folder = Path(output_folder)
    viz_folder.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('[RF] Random Forest - Imbalanced Classification Analysis', fontsize=16, fontweight='bold', y=1.02)
    
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
    
    print(f"  [OK] Saved: {output_path}")
    
    return output_path


def create_detailed_analysis_plots(results, cv_results, y_test, output_folder, exercise_code=None, df=None):
    """
    Create detailed analysis plots for comprehensive model evaluation
    """
    print("\n[CHART] Creating detailed analysis plots...")
    
    # Get appropriate quality names for this dataset
    quality_names = get_quality_names(exercise_code=exercise_code, df=df)
    
    viz_folder = Path(output_folder)
    viz_folder.mkdir(parents=True, exist_ok=True)
    
    # Create multiple figure sets
    
    # =============================================================================
    # PLOT SET 1: Feature Importance Analysis
    # =============================================================================
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('[SEARCH] Feature Importance Analysis', fontsize=16, fontweight='bold', y=1.02)
    
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
    print(f"  [OK] Saved: {feature_importance_path}")
    
    # =============================================================================
    # PLOT SET 2: Model Performance Deep Dive
    # =============================================================================
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('[STATS] Model Performance Deep Dive', fontsize=16, fontweight='bold', y=1.02)
    
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
    print(f"  [OK] Saved: {performance_path}")
    
    # =============================================================================
    # PLOT SET 3: Class Imbalance Analysis
    # =============================================================================
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('[IMBAL] Class Imbalance Analysis', fontsize=16, fontweight='bold', y=1.02)
    
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
    print(f"  [OK] Saved: {imbalance_path}")
    
    return [feature_importance_path, performance_path, imbalance_path]


# =============================================================================
# MODEL EXPORT
# =============================================================================

def export_model(
    model,
    scaler,
    feature_names,
    results,
    cv_results,
    output_folder,
    smote_applied=False,
    exercise_code=None,
    df=None,
    class_imbalance_strategy=None,
    dimensionality_reduction_summary=None
):
    """
    Export the trained model and associated objects to a .pkl file
    """
    print("\n[SAVE] Exporting model...")
    
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
        'smote_applied': smote_applied,
        'class_imbalance_strategy': class_imbalance_strategy,
        'dimensionality_reduction': dimensionality_reduction_summary
    }
    
    # Save model
    model_path = output_folder / f'rf_classifier_{timestamp}.pkl'
    joblib.dump(model_package, model_path)
    
    print(f"  [OK] Model saved: {model_path}")
    
    # Save feature importance to CSV
    importance_path = output_folder / f'feature_importance_{timestamp}.csv'
    results['feature_importance'].to_csv(importance_path, index=False)
    print(f"  [OK] Feature importance saved: {importance_path}")
    
    # Save classification report
    report_path = output_folder / f'classification_report_{timestamp}.txt'
    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("RANDOM FOREST CLASSIFIER - IMBALANCED DATA EVALUATION\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Training Date: {timestamp}\n")
        f.write(f"Number of Features: {len(feature_names)}\n")
        f.write(f"SMOTE Applied: {'Yes' if smote_applied else 'No'}\n")
        f.write(
            f"Class Imbalance Strategy: "
            f"{class_imbalance_strategy if class_imbalance_strategy else 'None'}\n"
        )
        if dimensionality_reduction_summary:
            f.write(
                f"Dimensionality Reduction: "
                f"{dimensionality_reduction_summary.get('method', 'none')}\n"
            )
            f.write(
                f"Features Before/After: "
                f"{dimensionality_reduction_summary.get('initial_feature_count', len(feature_names))}"
                f"/"
                f"{dimensionality_reduction_summary.get('final_feature_count', len(feature_names))}\n"
            )
            f.write(
                f"Feature Reduction: "
                f"{dimensionality_reduction_summary.get('reduction_percent', 0):.1f}%\n"
            )
        f.write("\n")
        
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
        f.write(f"Standard Accuracy:  {cv_results['accuracy_mean']:.4f} +/- {cv_results['accuracy_std']:.4f}\n")
        f.write(f"Balanced Accuracy:  {cv_results.get('balanced_accuracy_mean', 'N/A'):.4f} +/- {cv_results.get('balanced_accuracy_std', 'N/A'):.4f}\n" if isinstance(cv_results.get('balanced_accuracy_mean'), float) else f"Balanced Accuracy:  N/A +/- N/A\n")
        f.write(f"F1 Weighted:        {cv_results.get('f1_weighted_mean', cv_results.get('f1_mean', 'N/A')):.4f} +/- {cv_results.get('f1_weighted_std', cv_results.get('f1_std', 'N/A')):.4f}\n" if isinstance(cv_results.get('f1_weighted_mean', cv_results.get('f1_mean')), float) else f"F1 Weighted:        N/A +/- N/A\n")
        f.write(f"F1 Macro:           {cv_results.get('f1_macro_mean', 'N/A'):.4f} +/- {cv_results.get('f1_macro_std', 'N/A'):.4f}\n" if isinstance(cv_results.get('f1_macro_mean'), float) else f"F1 Macro:           N/A +/- N/A\n")
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
    
    print(f"  [OK] Report saved: {report_path}")
    
    return model_path


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_classification_pipeline():
    """
    Main function to run the complete Random Forest classification pipeline.
    """
    print("\n" + "=" * 70)
    print("     RANDOM FOREST CLASSIFICATION PIPELINE")
    print("=" * 70)
    print(f"\nProject Root: {PROJECT_ROOT}")
    print(f"Models Directory: {MODELS_DIR}")
    
    # =========================================================================
    # STEP 1: Select CSV file
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: SELECT DATASET")
    print("=" * 70)
    
    print("\nOpening file selection dialog...")
    csv_file = select_csv_file()
    
    if not csv_file:
        print("\nNo file selected. Exiting.")
        return
    
    print(f"\nSelected file: {Path(csv_file).name}")
    
    # Load data
    print("\nLoading dataset...")
    df = pd.read_csv(csv_file)
    print(f"  Loaded {len(df):,} rows x {len(df.columns)} columns")
    
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
            print(f"\nDetected single exercise: {exercise_name}")
    
    # Check for target column
    if TARGET_COLUMN not in df.columns:
        print(f"\nError: Target column '{TARGET_COLUMN}' not found in dataset!")
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
        print("\nError: No features computed. Check your data.")
        return
    
    # =========================================================================
    # STEP 3: Column Selection UI
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: SELECT FEATURES")
    print("=" * 70)
    
    print("\nOpening feature selection dialog...")
    selected_features, excluded_columns = select_columns_ui(features_df)
    
    if selected_features is None:
        print("\nFeature selection cancelled. Exiting.")
        return
    
    print(f"\nSelected {len(selected_features)} features")
    print(f"Excluded {len(excluded_columns)} columns")
    
    # =========================================================================
    # STEP 4: Prepare Data
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 4: PREPARE DATA FOR TRAINING")
    print("=" * 70)
    
    X, y, feature_names = prepare_data(features_df, selected_features)
    
    # Split data BEFORE any training to prevent leakage
    print("\nSplitting data (stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"  Training set: {len(X_train)} samples")
    print(f"  Test set: {len(X_test)} samples")
    print(f"  Train target distribution: {dict(y_train.value_counts().sort_index())}")
    print(f"  Test target distribution: {dict(y_test.value_counts().sort_index())}")
    
    # Get quality names for display
    quality_names = get_quality_names(exercise_code=exercise_code, df=features_df)
    
    # =========================================================================
    # STEP 5: Class Imbalance Strategy
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 5: CLASS IMBALANCE STRATEGY")
    print("=" * 70)
    
    imbalance_config = configure_class_imbalance_strategy(y_train, quality_names=quality_names)
    if imbalance_config is None:
        return
    
    class_weight_setting = imbalance_config['class_weight']
    smote_applied = False
    smote_summary = None
    
    # =========================================================================
    # STEP 6: Hyperparameter Optimization (Optional)
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 6: HYPERPARAMETER OPTIMIZATION")
    print("=" * 70)
    
    print("\nHYPERPARAMETER OPTIMIZATION")
    print("Choose your hyperparameter optimization strategy:")
    print("1. Use default parameters (fastest - ~30 seconds)")
    print("2. Random Search optimization (fast & effective - ~5-15 minutes)")
    print("3. Grid Search optimization (thorough but slow - ~30-60 minutes)")
    print("\nRandom Search often finds good parameters much faster than Grid Search.")
    
    while True:
        try:
            choice = input("\nChoose (1-3): ").strip()
            if choice in ['1', '2', '3']:
                break
            print("Please enter 1, 2, or 3")
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
            return
        except Exception:
            print("Please enter 1, 2, or 3")
    
    best_params = None
    search_results = None
    
    if choice == '2':
        print("\nPerforming Random Search optimization...")
        print("This is much faster than Grid Search and often finds excellent parameters.")
        
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
                
                try:
                    n_iter = int(n_iter_choice)
                    if 10 <= n_iter <= 500:
                        break
                    print("Please enter a number between 10 and 500")
                except ValueError:
                    print("Please enter a valid number")
            except KeyboardInterrupt:
                print("\nOperation cancelled.")
                return
        
        best_params, search_results = perform_random_search(
            X_train,
            y_train,
            n_iter=n_iter,
            use_imbalance_strategy=imbalance_config['use_imbalance_strategy']
        )
        print("Random Search completed. Using optimized parameters.")
        
    elif choice == '3':
        print("\nPerforming Grid Search optimization...")
        print("This tests all combinations and may take 30-60 minutes.")
        best_params, search_results = perform_grid_search(
            X_train,
            y_train,
            use_imbalance_strategy=imbalance_config['use_imbalance_strategy']
        )
        print("Grid Search completed. Using optimized parameters.")
    else:
        print("\nUsing default Random Forest parameters for faster training.")
    
    # =========================================================================
    # STEP 7: Dimensionality Reduction (Optional)
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 7: DIMENSIONALITY REDUCTION")
    print("=" * 70)
    
    reduction_config = configure_dimensionality_reduction(len(feature_names))
    if reduction_config is None:
        return
    
    X_train_model, X_test_model, feature_names, reduction_summary = apply_dimensionality_reduction(
        X_train,
        X_test,
        y_train,
        reduction_config,
        best_params=best_params,
        class_weight_setting=class_weight_setting
    )
    
    # Use reduced feature set for CV to match training feature space.
    X_for_cv = X[feature_names].copy()
    
    # Apply SMOTE if selected (after dimensionality reduction, before training)
    y_train_model = y_train
    if imbalance_config.get('use_smote', False):
        print("\n" + "=" * 70)
        print("STEP 7b: SMOTE OVERSAMPLING")
        print("=" * 70)
        
        X_train_model, y_train_model, smote_summary = apply_smote(
            X_train_model, y_train, quality_names=quality_names
        )
        smote_applied = smote_summary.get('applied', False)
    
    # =========================================================================
    # STEP 8: Train Model
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 8: TRAIN MODEL")
    print("=" * 70)
    
    model, scaler, results = train_random_forest(
        X_train_model,
        y_train_model,
        X_test_model,
        y_test,
        feature_names,
        best_params=best_params,
        class_weight_setting=class_weight_setting,
        exercise_code=exercise_code,
        df=features_df
    )
    
    print("\nClassification Report:\n")
    print(results['classification_report'])
    
    # =========================================================================
    # STEP 9: Cross-Validation
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 9: 5-FOLD CROSS-VALIDATION")
    print("=" * 70)
    
    cv_results = perform_cross_validation(
        X_for_cv,
        y,
        n_splits=5,
        best_params=best_params,
        class_weight_setting=class_weight_setting
    )
    
    # =========================================================================
    # STEP 10: Visualizations
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 10: CREATE VISUALIZATIONS")
    print("=" * 70)
    
    viz_folder = MODELS_DIR / 'visualizations'
    create_model_visualizations(y_test, results, cv_results, viz_folder, exercise_code=exercise_code, df=features_df)
    
    # =========================================================================
    # STEP 11: Detailed Analysis Plots
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 11: DETAILED ANALYSIS PLOTS")
    print("=" * 70)
    
    detailed_plots = create_detailed_analysis_plots(results, cv_results, y_test, viz_folder, exercise_code=exercise_code, df=features_df)
    
    # =========================================================================
    # STEP 12: Export Model
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 12: EXPORT MODEL")
    print("=" * 70)
    
    model_path = export_model(
        model,
        scaler,
        feature_names,
        results,
        cv_results,
        MODELS_DIR,
        smote_applied=smote_applied,
        exercise_code=exercise_code,
        df=features_df,
        class_imbalance_strategy=class_weight_setting,
        dimensionality_reduction_summary=reduction_summary
    )
    
    # =========================================================================
    # Final Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE - SUMMARY")
    print("=" * 70)
    
    print("\nModel Performance Summary:")
    print(f"  Test Accuracy: {results['accuracy']:.4f}")
    print(f"  Test Balanced Accuracy: {results.get('balanced_accuracy', results['accuracy']):.4f}")
    print(f"  Test F1 Score (Weighted): {results['f1_weighted']:.4f}")
    print(f"  Test F1 Score (Macro): {results.get('f1_macro', results['f1_weighted']):.4f}")
    print(f"  CV Accuracy: {cv_results['accuracy_mean']:.4f} +/- {cv_results['accuracy_std']:.4f}")
    print(
        f"  CV Balanced Accuracy: "
        f"{cv_results.get('balanced_accuracy_mean', cv_results['accuracy_mean']):.4f} "
        f"+/- {cv_results.get('balanced_accuracy_std', cv_results['accuracy_std']):.4f}"
    )
    print(
        f"  CV F1 Score (Weighted): "
        f"{cv_results.get('f1_weighted_mean', cv_results.get('f1_mean', 0)):.4f} "
        f"+/- {cv_results.get('f1_weighted_std', cv_results.get('f1_std', 0)):.4f}"
    )
    print(
        f"  CV F1 Score (Macro): "
        f"{cv_results.get('f1_macro_mean', 0):.4f} "
        f"+/- {cv_results.get('f1_macro_std', 0):.4f}"
    )
    print(f"  Generalization Gap: {cv_results['generalization_gap']:.4f}")
    
    print("\nConfiguration Summary:")
    print(f"  Class imbalance ratio: {imbalance_config['imbalance_ratio']:.2f}:1")
    print(f"  Imbalance verdict: {imbalance_config.get('verdict', 'N/A')}")
    print(f"  Class imbalance strategy: {class_weight_setting if class_weight_setting else 'None'}")
    print(f"  SMOTE applied: {'Yes' if smote_applied else 'No'}")
    if smote_applied and smote_summary:
        print(f"  SMOTE synthetic samples: {smote_summary.get('synthetic_samples', 0):,}")
    print(
        f"  Dimensionality reduction: {reduction_summary.get('method', 'none')} "
        f"({reduction_summary.get('initial_feature_count', len(feature_names))} -> "
        f"{reduction_summary.get('final_feature_count', len(feature_names))} features)"
    )
    
    if 'per_class_metrics' in results:
        priority_names = ['Pulling Too Fast', 'Uncontrolled Movement', 'Abrupt Initiation']
        focus_class = None
        for priority in priority_names:
            focus_class = next(
                (name for name in results['per_class_metrics'].keys() if priority.lower() in name.lower()),
                None
            )
            if focus_class:
                break
        
        if focus_class:
            print(f"  Focus Recall ({focus_class}): {results['per_class_metrics'][focus_class]['recall']:.4f}")
    
    print("\nOutput Files:")
    print(f"  Model: {model_path}")
    print(f"  Main Visualization: {viz_folder / 'rf_imbalanced_evaluation.png'}")
    print("  Detailed Analysis Plots:")
    for plot_path in detailed_plots:
        print(f"    - {plot_path.name}")
    
    print("\nTop 5 Most Important Features:")
    for _, row in results['feature_importance'].head(5).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Display class-specific performance
    if 'per_class_metrics' in results:
        print("\nPer-Class Performance Summary:")
        for class_name, metrics in results['per_class_metrics'].items():
            print(f"  {class_name}:")
            print(f"    Precision: {metrics['precision']:.3f}")
            print(f"    Recall: {metrics['recall']:.3f}")
            print(f"    F1 Score: {metrics['f1']:.3f}")
            print(f"    Support: {metrics['support']} samples")
    
    print("\n" + "=" * 70)
    print("CLASSIFICATION PIPELINE COMPLETE")
    print("Check the visualizations folder for detailed analysis plots.")
    print("=" * 70 + "\n")
    
    return model, scaler, feature_names, results, cv_results





# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    run_classification_pipeline()

