"""
AppLift ML Training - Preprocessing Pipeline
=============================================
A comprehensive preprocessing pipeline for exercise sensor data.

Phase 1: Data Merging & Cleaning
- Select a folder (equipment type)
- Merge all CSV files into one
- Handle missing values
- Map quality codes to target labels
- Infer labels from folder structure when quality_code is missing

Phase 2: Rep Resegmentation  
- Re-align rep boundaries using valley detection
- Fix miscounted windows

Output:
- Cleaned merged dataset
- Resegmented dataset
- Summary reports
- Visualizations

Author: AppLift ML Training Pipeline
"""

import pandas as pd
import numpy as np
from scipy.signal import find_peaks, savgol_filter
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Get the script's directory - should be "AppLift ML Training"
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR

# Output directories
OUTPUT_DIR = PROJECT_ROOT / 'output'
MERGED_DIR = OUTPUT_DIR / 'phase1_merged'
RESEGMENTED_DIR = OUTPUT_DIR / 'phase2_resegmented'
VISUALIZATIONS_DIR = OUTPUT_DIR / 'visualizations'
REPORTS_DIR = OUTPUT_DIR / 'reports'

# Create directories
for dir_path in [OUTPUT_DIR, MERGED_DIR, RESEGMENTED_DIR, VISUALIZATIONS_DIR, REPORTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# =============================================================================
# LABEL MAPPINGS
# =============================================================================

# Equipment mapping
EQUIPMENT_MAP = {
    'Dumbbell': 0,
    'Barbell': 1,
    'Weight Stack': 2,
    'WeightStack': 2
}

# Exercise mapping
EXERCISE_MAP = {
    'Concentration_Curls': 0,
    'Concentration Curls': 0,
    'Overhead_Extension': 1,
    'Overhead Extension': 1,
    'Bench_Press': 2,
    'Bench Press': 2,
    'Back_Squat': 3,
    'Back Squat': 3,
    'Lateral_Pulldown': 4,
    'Lateral Pulldown': 4,
    'Seated_Leg_Extension': 5,
    'Seated Leg Extension': 5
}

# Quality code mapping based on folder names
QUALITY_FOLDER_MAP = {
    # Clean variations
    'Clean': 0,
    'clean': 0,
    
    # Dumbbell errors (Concentration Curls & Overhead Extension)
    'Uncontrolled Movement': 1,
    'Uncontrolled_Movement': 1,
    'Uncontrolled': 1,
    'Abrupt Initiation': 2,
    'Abrupt_Initiation': 2,
    'Abrupt': 2,
    
    # Barbell errors (Bench Press & Back Squat)
    'Inclination Asymmetry': 2,
    'Inclination_Asymmetry': 2,
    
    # Weight Stack errors (Lateral Pulldown & Seated Leg Extension)
    'Pulling Too Fast': 1,
    'Pulling_Too_Fast': 1,
    'Releasing Too Fast': 2,
    'Releasing_Too_Fast': 2
}

# Quality code names for display
QUALITY_NAMES = {
    0: 'Clean',
    1: 'Error Type 1',
    2: 'Error Type 2'
}


# =============================================================================
# PHASE 1: DATA MERGING & CLEANING
# =============================================================================

def select_folder_ui():
    """
    Show a UI to select a folder for processing
    """
    root = tk.Tk()
    root.title("📊 AppLift ML - Select Data Folder")
    root.geometry("700x550")
    root.configure(bg='#f5f5f5')
    
    selected_folder = [None]
    
    # Header
    header_frame = tk.Frame(root, bg='#2196F3', pady=15)
    header_frame.pack(fill=tk.X)
    
    header = tk.Label(header_frame, text="🏋️ AppLift ML Preprocessing Pipeline", 
                      font=('Arial', 18, 'bold'), bg='#2196F3', fg='white')
    header.pack()
    
    subtitle = tk.Label(header_frame, text="Phase 1: Select Data Folder to Process",
                       font=('Arial', 11), bg='#2196F3', fg='white')
    subtitle.pack()
    
    # Instructions
    instructions = tk.Label(root, 
                           text="Select an equipment folder (Dumbbell, Barbell, or Weight Stack)\n"
                                "or a specific exercise folder to process:",
                           font=('Arial', 10), bg='#f5f5f5', fg='#666', justify='center')
    instructions.pack(pady=10)
    
    # Frame for treeview
    tree_frame = tk.Frame(root, bg='#f5f5f5')
    tree_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
    
    # Scrollbars
    y_scrollbar = ttk.Scrollbar(tree_frame)
    y_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    x_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL)
    x_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
    
    # Treeview
    tree = ttk.Treeview(tree_frame, yscrollcommand=y_scrollbar.set, xscrollcommand=x_scrollbar.set)
    tree.pack(fill=tk.BOTH, expand=True)
    y_scrollbar.config(command=tree.yview)
    x_scrollbar.config(command=tree.xview)
    
    tree.heading('#0', text='📁 Project Folders', anchor='w')
    tree.column('#0', width=600)
    
    def populate_tree(parent, path, depth=0):
        """Recursively populate tree with folder contents (folders only, max depth 3)"""
        if depth > 3:
            return
        try:
            items = sorted(os.listdir(path))
            for item in items:
                item_path = os.path.join(path, item)
                if os.path.isdir(item_path) and not item.startswith('.'):
                    # Count CSV files in this folder
                    csv_count = len([f for f in os.listdir(item_path) if f.endswith('.csv')])
                    subfolders = len([f for f in os.listdir(item_path) if os.path.isdir(os.path.join(item_path, f))])
                    
                    # Create label
                    if csv_count > 0:
                        label = f"📁 {item} ({csv_count} CSV files)"
                    elif subfolders > 0:
                        label = f"📁 {item}"
                    else:
                        label = f"📁 {item}"
                    
                    node = tree.insert(parent, 'end', text=label, open=(depth < 1), values=(item_path,))
                    populate_tree(node, item_path, depth + 1)
        except PermissionError:
            pass
    
    # Populate tree from project root
    root_node = tree.insert('', 'end', text=f'📁 {PROJECT_ROOT.name}', open=True, values=(str(PROJECT_ROOT),))
    populate_tree(root_node, str(PROJECT_ROOT))
    
    # Selected folder label
    selected_label = tk.Label(root, text="Selected: None", 
                              font=('Arial', 10, 'bold'), bg='#f5f5f5', fg='#666')
    selected_label.pack(pady=5)
    
    def on_select(event):
        selection = tree.selection()
        if selection:
            item = tree.item(selection[0])
            values = item.get('values', [])
            if values:
                path = values[0]
                selected_folder[0] = path
                folder_name = os.path.basename(path)
                selected_label.config(text=f"✓ Selected: {folder_name}", fg='#4CAF50')
    
    def on_double_click(event):
        selection = tree.selection()
        if selection:
            item = tree.item(selection[0])
            values = item.get('values', [])
            if values:
                selected_folder[0] = values[0]
                root.destroy()
    
    tree.bind('<<TreeviewSelect>>', on_select)
    tree.bind('<Double-1>', on_double_click)
    
    # Button frame
    btn_frame = tk.Frame(root, bg='#f5f5f5')
    btn_frame.pack(pady=15)
    
    def browse_folder():
        folder_path = filedialog.askdirectory(
            title="Select Data Folder",
            initialdir=str(PROJECT_ROOT)
        )
        if folder_path:
            selected_folder[0] = folder_path
            root.destroy()
    
    def confirm_selection():
        if selected_folder[0]:
            root.destroy()
        else:
            messagebox.showwarning("No Selection", "Please select a folder first!")
    
    def cancel():
        selected_folder[0] = None
        root.destroy()
    
    browse_btn = tk.Button(btn_frame, text="📂 Browse...", command=browse_folder,
                          font=('Arial', 10), bg='#757575', fg='white', 
                          padx=20, pady=8, cursor='hand2', relief='flat')
    browse_btn.pack(side=tk.LEFT, padx=5)
    
    confirm_btn = tk.Button(btn_frame, text="✅ Process Selected Folder", command=confirm_selection,
                           font=('Arial', 10, 'bold'), bg='#4CAF50', fg='white',
                           padx=20, pady=8, cursor='hand2', relief='flat')
    confirm_btn.pack(side=tk.LEFT, padx=5)
    
    cancel_btn = tk.Button(btn_frame, text="❌ Cancel", command=cancel,
                          font=('Arial', 10), bg='#f44336', fg='white',
                          padx=20, pady=8, cursor='hand2', relief='flat')
    cancel_btn.pack(side=tk.LEFT, padx=5)
    
    # Center window
    root.update_idletasks()
    x = (root.winfo_screenwidth() - root.winfo_width()) // 2
    y = (root.winfo_screenheight() - root.winfo_height()) // 2
    root.geometry(f"+{x}+{y}")
    
    root.mainloop()
    
    return selected_folder[0]


def infer_labels_from_path(file_path):
    """
    Infer equipment_code, exercise_code, and quality_code from file path
    
    Returns: dict with inferred labels
    """
    path_parts = Path(file_path).parts
    path_str = str(file_path)
    
    labels = {
        'equipment_code': None,
        'exercise_code': None,
        'quality_code': None,
        'equipment_name': None,
        'exercise_name': None,
        'quality_name': None
    }
    
    # Find equipment
    for part in path_parts:
        for equip_name, equip_code in EQUIPMENT_MAP.items():
            if equip_name.lower() in part.lower():
                labels['equipment_code'] = equip_code
                labels['equipment_name'] = equip_name
                break
        if labels['equipment_code'] is not None:
            break
    
    # Find exercise
    for part in path_parts:
        for ex_name, ex_code in EXERCISE_MAP.items():
            if ex_name.lower().replace('_', ' ') in part.lower().replace('_', ' '):
                labels['exercise_code'] = ex_code
                labels['exercise_name'] = ex_name.replace('_', ' ')
                break
        if labels['exercise_code'] is not None:
            break
    
    # Find quality from folder name
    for part in path_parts:
        for quality_name, quality_code in QUALITY_FOLDER_MAP.items():
            if quality_name.lower() in part.lower():
                labels['quality_code'] = quality_code
                labels['quality_name'] = quality_name
                break
        if labels['quality_code'] is not None:
            break
    
    return labels


def load_and_merge_csv_files(folder_path, verbose=True):
    """
    Load all CSV files from the exact selected folder (non-recursive) and merge them
    
    Parameters:
    - folder_path: Path to the folder containing CSV files
    - verbose: Print progress information
    
    Returns:
    - merged_df: Merged DataFrame
    - file_info: List of dicts with information about each file
    """
    folder_path = Path(folder_path)
    csv_files = list(folder_path.glob('*.csv'))  # Only direct CSV files, not recursive
    
    # Filter out already processed files
    csv_files = [f for f in csv_files if not any(x in f.name for x in ['_merged', '_resegmented', '_boundaries'])]
    
    if verbose:
        print(f"\n📂 Found {len(csv_files)} CSV files in '{folder_path.name}'")
    
    all_dfs = []
    file_info = []
    
    for i, csv_file in enumerate(csv_files):
        try:
            df = pd.read_csv(csv_file)
            original_shape = df.shape
            
            # Infer labels from path
            inferred = infer_labels_from_path(csv_file)
            
            # Add source file info
            df['source_file'] = csv_file.name
            
            # Handle missing columns by inferring from folder structure
            if 'equipment_code' not in df.columns and inferred['equipment_code'] is not None:
                df['equipment_code'] = inferred['equipment_code']
            
            if 'exercise_code' not in df.columns and inferred['exercise_code'] is not None:
                df['exercise_code'] = inferred['exercise_code']
            
            # Handle quality_code - CRITICAL
            if 'quality_code' not in df.columns:
                if inferred['quality_code'] is not None:
                    df['quality_code'] = inferred['quality_code']
                    if verbose:
                        print(f"  ⚠️  Inferred quality_code={inferred['quality_code']} ({inferred['quality_name']}) for {csv_file.name}")
                else:
                    df['quality_code'] = -1  # Unknown
                    if verbose:
                        print(f"  ❌ Could not infer quality_code for {csv_file.name}")
            
            all_dfs.append(df)
            
            file_info.append({
                'file': csv_file.name,
                'path': str(csv_file),
                'rows': original_shape[0],
                'columns': original_shape[1],
                'equipment': inferred['equipment_name'],
                'exercise': inferred['exercise_name'],
                'quality': inferred['quality_name'],
                'quality_code': df['quality_code'].iloc[0] if len(df) > 0 else None,
                'had_quality_code': 'quality_code' in pd.read_csv(csv_file).columns
            })
            
            if verbose and (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(csv_files)} files...")
                
        except Exception as e:
            print(f"  ❌ Error loading {csv_file.name}: {e}")
    
    if not all_dfs:
        return None, []
    
    # Merge all DataFrames
    merged_df = pd.concat(all_dfs, ignore_index=True)
    
    if verbose:
        print(f"\n✅ Merged {len(all_dfs)} files → {len(merged_df)} total rows")
    
    return merged_df, file_info


def clean_dataset(df, verbose=True):
    """
    Clean the merged dataset
    
    - Handle missing values
    - Rename quality_code to 'target'
    - Validate data types
    
    Returns:
    - cleaned_df: Cleaned DataFrame
    - cleaning_report: Dict with cleaning statistics
    """
    if verbose:
        print("\n🧹 Cleaning dataset...")
    
    cleaning_report = {
        'original_shape': df.shape,
        'missing_values_before': df.isnull().sum().to_dict(),
        'duplicates_removed': 0,
        'invalid_targets_fixed': 0,
        'columns_standardized': []
    }
    
    df_clean = df.copy()
    
    # 1. Check for missing values
    missing_counts = df_clean.isnull().sum()
    total_missing = missing_counts.sum()
    
    if verbose and total_missing > 0:
        print(f"  Found {total_missing} missing values across columns:")
        for col, count in missing_counts[missing_counts > 0].items():
            print(f"    - {col}: {count} missing")
    
    # 2. Handle missing values in numeric columns (interpolate)
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_clean[col].isnull().any():
            df_clean[col] = df_clean[col].interpolate(method='linear', limit_direction='both')
            df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
    
    # 3. Rename quality_code to target
    if 'quality_code' in df_clean.columns:
        df_clean = df_clean.rename(columns={'quality_code': 'target'})
        cleaning_report['columns_standardized'].append('quality_code → target')
        if verbose:
            print("  ✓ Renamed 'quality_code' to 'target'")
    
    # 4. Validate target values (should be 0, 1, or 2)
    if 'target' in df_clean.columns:
        invalid_targets = df_clean[~df_clean['target'].isin([0, 1, 2])]
        if len(invalid_targets) > 0:
            cleaning_report['invalid_targets_fixed'] = len(invalid_targets)
            if verbose:
                print(f"  ⚠️  Found {len(invalid_targets)} rows with invalid target values")
            # For now, keep them but flag them
            df_clean.loc[~df_clean['target'].isin([0, 1, 2]), 'target_warning'] = True
    
    # 5. Remove exact duplicates
    before_dedup = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    cleaning_report['duplicates_removed'] = before_dedup - len(df_clean)
    
    if verbose and cleaning_report['duplicates_removed'] > 0:
        print(f"  ✓ Removed {cleaning_report['duplicates_removed']} duplicate rows")
    
    # 6. Ensure proper data types
    int_columns = ['participant', 'rep', 'equipment_code', 'exercise_code', 'target', 'timestamp_ms']
    for col in int_columns:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(int)
    
    cleaning_report['final_shape'] = df_clean.shape
    cleaning_report['missing_values_after'] = df_clean.isnull().sum().sum()
    
    if verbose:
        print(f"  ✓ Final shape: {df_clean.shape}")
    
    return df_clean, cleaning_report


# =============================================================================
# PHASE 2: RESEGMENTATION
# =============================================================================

def find_valleys(signal, distance=15, prominence=0.3):
    """
    Find valleys (local minima) in the signal by inverting and finding peaks
    """
    inverted = -signal
    peaks, properties = find_peaks(inverted, distance=distance, prominence=prominence)
    return peaks, properties


def resegment_single_file(df, signal_column='filteredMag', min_rep_duration_ms=500, max_rep_duration_ms=10000):
    """
    Re-segment reps based on valley-to-valley detection
    
    Returns:
    - df_resegmented: DataFrame with corrected 'rep' column
    - rep_info: List of dicts with rep boundary information
    """
    # Detect equipment and exercise type for adaptive parameters
    equipment_code = df['equipment_code'].iloc[0] if 'equipment_code' in df.columns else 0
    exercise_code = df['exercise_code'].iloc[0] if 'exercise_code' in df.columns else 0
    
    # Exercise-specific parameters
    # Weight Stack exercises (4=Lateral Pulldown, 5=Seated Leg Extension) have different patterns
    is_weight_stack = (equipment_code == 2) or (exercise_code in [4, 5])
    
    if is_weight_stack:
        signal_column = 'filteredMag'
        # Weight stack has smoother, lower amplitude signals - need lower prominence
        prominence_factor = 0.05  # 5% of range (lower for subtle variations)
        std_factor = 0.3
        min_prominence_floor = 0.05  # Lower floor for weight stack
        min_rep_duration_ms = max(min_rep_duration_ms, 1500)  # Weight stack reps are slower (1.5s min)
        max_rep_duration_ms = min(max_rep_duration_ms, 12000)  # Allow longer reps
    else:
        # Dumbbell/Barbell exercises (including Overhead Extension)
        prominence_factor = 0.1  # 10% of range
        std_factor = 0.5
        min_prominence_floor = 0.1  # Lower floor to catch valleys
    
    if signal_column not in df.columns:
        signal_column = 'filteredMag'
    
    signal = df[signal_column].values
    timestamps = df['timestamp_ms'].values
    
    # Smooth the signal - use stronger smoothing for weight stack to reduce noise
    if is_weight_stack:
        window_length = min(21, len(signal) if len(signal) % 2 == 1 else len(signal) - 1)
        if window_length >= 5:
            signal_smooth = savgol_filter(signal, window_length=window_length, polyorder=3)
        else:
            signal_smooth = signal
    else:
        if len(signal) > 11:
            signal_smooth = savgol_filter(signal, window_length=11, polyorder=3)
        else:
            signal_smooth = signal
    
    # Calculate adaptive parameters using exercise-specific factors
    signal_range = np.max(signal_smooth) - np.min(signal_smooth)
    signal_std = np.std(signal_smooth)
    min_prominence = max(signal_range * prominence_factor, signal_std * std_factor, min_prominence_floor)
    
    # Estimate minimum distance between valleys based on exercise type
    if len(timestamps) > 1:
        median_dt = np.median(np.diff(timestamps))
        sample_rate = 1000 / median_dt
        if is_weight_stack:
            min_distance = int(1.5 * sample_rate)  # Weight stack reps are slower
        else:
            min_distance = int(0.5 * sample_rate)
    else:
        min_distance = 5
    
    # Find all valleys
    valley_indices, _ = find_valleys(signal_smooth, distance=min_distance, prominence=min_prominence)
    
    # If too few valleys found, try with lower prominence (applies to ALL exercises)
    if len(valley_indices) < 2:
        for retry_factor in [0.5, 0.25, 0.1]:
            retry_prominence = min_prominence * retry_factor
            valley_indices, _ = find_valleys(signal_smooth, distance=min_distance, prominence=retry_prominence)
            if len(valley_indices) >= 2:
                min_prominence = retry_prominence
                break
        
        # If still not enough, try finding peaks instead
        if len(valley_indices) < 2:
            peak_indices, _ = find_peaks(signal_smooth, distance=min_distance, prominence=min_prominence * 0.5)
            if len(peak_indices) >= 2:
                valley_indices = peak_indices
    
    # Filter valleys by minimum rep duration
    valid_valleys = [0]
    
    for idx in valley_indices:
        last_valley_time = timestamps[valid_valleys[-1]]
        current_valley_time = timestamps[idx]
        duration_ms = current_valley_time - last_valley_time
        
        if min_rep_duration_ms <= duration_ms <= max_rep_duration_ms:
            valid_valleys.append(idx)
    
    # Fallback: if valley detection failed, use original rep boundaries refined with valleys
    original_reps = sorted([r for r in df['rep'].unique() if r > 0]) if 'rep' in df.columns else []
    if len(valid_valleys) <= 1 and len(original_reps) >= 1:
        valid_valleys = [0]
        for rep in original_reps:
            rep_data = df[df['rep'] == rep]
            if len(rep_data) == 0:
                continue
            
            rep_end_idx = rep_data.index[-1]
            rep_start_idx = rep_data.index[0]
            rep_duration = rep_end_idx - rep_start_idx
            search_window = max(int(rep_duration * 0.2), 10)
            
            search_start = max(0, rep_end_idx - search_window)
            search_end = min(len(signal_smooth), rep_end_idx + search_window)
            
            search_signal = signal_smooth[search_start:search_end]
            if len(search_signal) > 3:
                local_min_idx = np.argmin(search_signal)
                valley_idx = search_start + local_min_idx
                if valley_idx > valid_valleys[-1]:
                    valid_valleys.append(valley_idx)
        
        if valid_valleys[-1] < len(df) - 1:
            valid_valleys.append(len(df) - 1)
    
    # Create new rep labels
    df_resegmented = df.copy()
    if 'rep' in df_resegmented.columns:
        df_resegmented['rep_original'] = df_resegmented['rep'].copy()
    df_resegmented['rep'] = 0
    
    rep_info = []
    
    for i in range(len(valid_valleys) - 1):
        start_idx = valid_valleys[i]
        end_idx = valid_valleys[i + 1]
        rep_num = i + 1
        
        df_resegmented.loc[start_idx:end_idx-1, 'rep'] = rep_num
        
        start_time = timestamps[start_idx]
        end_time = timestamps[end_idx - 1]
        
        rep_signal = signal_smooth[start_idx:end_idx]
        peak_local_idx = np.argmax(rep_signal)
        peak_idx = start_idx + peak_local_idx
        
        rep_info.append({
            'rep': rep_num,
            'start_idx': start_idx,
            'end_idx': end_idx - 1,
            'start_time_ms': start_time,
            'end_time_ms': end_time,
            'duration_ms': end_time - start_time,
            'peak_idx': peak_idx
        })
    
    # Handle last segment
    if valid_valleys[-1] < len(df) - 1:
        start_idx = valid_valleys[-1]
        end_idx = len(df) - 1
        rep_num = len(valid_valleys)
        df_resegmented.loc[start_idx:end_idx, 'rep'] = rep_num
        
        rep_info.append({
            'rep': rep_num,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'start_time_ms': timestamps[start_idx],
            'end_time_ms': timestamps[end_idx],
            'duration_ms': timestamps[end_idx] - timestamps[start_idx],
            'peak_idx': start_idx + np.argmax(signal_smooth[start_idx:end_idx+1])
        })
    
    return df_resegmented, rep_info


def resegment_dataset(df, verbose=True):
    """
    Resegment all data grouped by source file
    
    Returns:
    - df_resegmented: Full resegmented DataFrame
    - resegment_summary: Summary of resegmentation
    """
    if verbose:
        print("\n🔄 Resegmenting dataset...")
    
    if 'source_file' not in df.columns:
        df['source_file'] = 'unknown'
    
    source_files = df['source_file'].unique()
    
    if verbose:
        print(f"  Processing {len(source_files)} unique source files...")
    
    all_resegmented = []
    resegment_summary = []
    
    for i, source in enumerate(source_files):
        df_source = df[df['source_file'] == source].copy()
        df_source = df_source.reset_index(drop=True)
        
        original_reps = df_source['rep'].nunique() if 'rep' in df_source.columns else 0
        
        try:
            df_reseg, rep_info = resegment_single_file(
                df_source,
                min_rep_duration_ms=800,
                max_rep_duration_ms=8000
            )
            
            new_reps = len(rep_info)
            
            resegment_summary.append({
                'source_file': source,
                'original_reps': original_reps,
                'new_reps': new_reps,
                'rep_change': new_reps - original_reps,
                'status': 'success'
            })
            
            all_resegmented.append(df_reseg)
            
        except Exception as e:
            resegment_summary.append({
                'source_file': source,
                'original_reps': original_reps,
                'new_reps': original_reps,
                'rep_change': 0,
                'status': f'error: {str(e)}'
            })
            all_resegmented.append(df_source)
        
        if verbose and (i + 1) % 20 == 0:
            print(f"    Processed {i + 1}/{len(source_files)} files...")
    
    df_resegmented = pd.concat(all_resegmented, ignore_index=True)
    
    if verbose:
        successful = sum(1 for s in resegment_summary if s['status'] == 'success')
        print(f"  ✓ Resegmented {successful}/{len(source_files)} files successfully")
    
    return df_resegmented, resegment_summary


# =============================================================================
# VISUALIZATIONS
# =============================================================================

def create_summary_visualizations(df, file_info, cleaning_report, output_folder, prefix="phase1"):
    """
    Create summary visualizations for the processed data
    """
    print("\n📊 Creating visualizations...")
    
    viz_folder = Path(output_folder)
    viz_folder.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    colors = ['#4CAF50', '#FF9800', '#f44336', '#2196F3', '#9C27B0', '#00BCD4']
    
    # 1. Dataset Overview
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('📊 Dataset Overview', fontsize=16, fontweight='bold', y=1.02)
    
    # Target distribution
    if 'target' in df.columns:
        target_counts = df['target'].value_counts().sort_index()
        target_labels = [f"{QUALITY_NAMES.get(i, f'Code {i}')}\n(n={target_counts.get(i, 0)})" 
                        for i in target_counts.index]
        axes[0, 0].pie(target_counts.values, labels=target_labels, colors=colors[:len(target_counts)],
                      autopct='%1.1f%%', startangle=90, explode=[0.02]*len(target_counts))
        axes[0, 0].set_title('Target Distribution', fontweight='bold')
    
    # Equipment distribution
    if 'equipment_code' in df.columns:
        equip_counts = df['equipment_code'].value_counts().sort_index()
        equip_names = {0: 'Dumbbell', 1: 'Barbell', 2: 'Weight Stack'}
        equip_labels = [equip_names.get(i, f'Code {i}') for i in equip_counts.index]
        bars = axes[0, 1].bar(equip_labels, equip_counts.values, color=colors[:len(equip_counts)])
        axes[0, 1].set_title('Samples by Equipment', fontweight='bold')
        axes[0, 1].set_ylabel('Number of Samples')
        for bar, count in zip(bars, equip_counts.values):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, 
                           f'{count:,}', ha='center', va='bottom', fontsize=10)
    
    # Exercise distribution
    if 'exercise_code' in df.columns:
        ex_counts = df['exercise_code'].value_counts().sort_index()
        ex_names = {0: 'Conc. Curls', 1: 'Overhead Ext.', 2: 'Bench Press', 
                   3: 'Back Squat', 4: 'Lat. Pulldown', 5: 'Leg Extension'}
        ex_labels = [ex_names.get(i, f'Code {i}') for i in ex_counts.index]
        bars = axes[1, 0].barh(ex_labels, ex_counts.values, color=colors[:len(ex_counts)])
        axes[1, 0].set_title('Samples by Exercise', fontweight='bold')
        axes[1, 0].set_xlabel('Number of Samples')
        for bar, count in zip(bars, ex_counts.values):
            axes[1, 0].text(count + 50, bar.get_y() + bar.get_height()/2, 
                           f'{count:,}', ha='left', va='center', fontsize=10)
    
    # Participant distribution
    if 'participant' in df.columns:
        part_counts = df['participant'].value_counts().sort_index()
        axes[1, 1].bar(part_counts.index.astype(str), part_counts.values, color='#2196F3', alpha=0.7)
        axes[1, 1].set_title('Samples by Participant', fontweight='bold')
        axes[1, 1].set_xlabel('Participant ID')
        axes[1, 1].set_ylabel('Number of Samples')
        axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(viz_folder / f'{prefix}_dataset_overview.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"  ✓ Saved: {prefix}_dataset_overview.png")
    
    # 2. Signal Statistics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('📈 Signal Statistics', fontsize=16, fontweight='bold', y=1.02)
    
    signal_cols = ['filteredMag', 'accelMag', 'filteredX', 'filteredY', 'filteredZ']
    available_cols = [col for col in signal_cols if col in df.columns]
    
    if available_cols:
        # Signal distributions
        for idx, col in enumerate(available_cols[:4]):
            row, col_idx = idx // 2, idx % 2
            axes[row, col_idx].hist(df[col].dropna(), bins=50, color=colors[idx], alpha=0.7, edgecolor='white')
            axes[row, col_idx].set_title(f'{col} Distribution', fontweight='bold')
            axes[row, col_idx].set_xlabel(col)
            axes[row, col_idx].set_ylabel('Frequency')
            
            # Add statistics
            mean_val = df[col].mean()
            std_val = df[col].std()
            axes[row, col_idx].axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
            axes[row, col_idx].legend()
    
    plt.tight_layout()
    plt.savefig(viz_folder / f'{prefix}_signal_statistics.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"  ✓ Saved: {prefix}_signal_statistics.png")
    
    # 3. Target vs Equipment/Exercise heatmap
    if 'target' in df.columns and 'exercise_code' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        cross_tab = pd.crosstab(df['exercise_code'], df['target'])
        ex_names = {0: 'Concentration Curls', 1: 'Overhead Extension', 2: 'Bench Press', 
                   3: 'Back Squat', 4: 'Lateral Pulldown', 5: 'Leg Extension'}
        cross_tab.index = [ex_names.get(i, f'Exercise {i}') for i in cross_tab.index]
        cross_tab.columns = [QUALITY_NAMES.get(i, f'Target {i}') for i in cross_tab.columns]
        
        sns.heatmap(cross_tab, annot=True, fmt='d', cmap='YlGnBu', ax=ax, 
                   cbar_kws={'label': 'Sample Count'})
        ax.set_title('Samples by Exercise and Target', fontweight='bold', fontsize=14)
        ax.set_xlabel('Target Class')
        ax.set_ylabel('Exercise')
        
        plt.tight_layout()
        plt.savefig(viz_folder / f'{prefix}_exercise_target_heatmap.png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        print(f"  ✓ Saved: {prefix}_exercise_target_heatmap.png")
    
    return viz_folder


def create_resegmentation_visualizations(df_original, df_resegmented, resegment_summary, output_folder):
    """
    Create visualizations for resegmentation results
    """
    print("\n📊 Creating resegmentation visualizations...")
    
    viz_folder = Path(output_folder)
    viz_folder.mkdir(parents=True, exist_ok=True)
    
    # 1. Resegmentation Summary
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('🔄 Resegmentation Summary', fontsize=16, fontweight='bold', y=1.05)
    
    summary_df = pd.DataFrame(resegment_summary)
    
    # Rep changes histogram
    if 'rep_change' in summary_df.columns:
        changes = summary_df['rep_change'].dropna()
        axes[0].hist(changes, bins=20, color='#2196F3', alpha=0.7, edgecolor='white')
        axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='No change')
        axes[0].set_title('Rep Count Changes per File', fontweight='bold')
        axes[0].set_xlabel('Change in Rep Count')
        axes[0].set_ylabel('Number of Files')
        axes[0].legend()
    
    # Success rate
    if 'status' in summary_df.columns:
        status_counts = summary_df['status'].apply(lambda x: 'Success' if x == 'success' else 'Error').value_counts()
        colors_pie = ['#4CAF50', '#f44336']
        axes[1].pie(status_counts.values, labels=status_counts.index, colors=colors_pie[:len(status_counts)],
                   autopct='%1.1f%%', startangle=90)
        axes[1].set_title('Resegmentation Success Rate', fontweight='bold')
    
    # Original vs New rep counts
    if 'original_reps' in summary_df.columns and 'new_reps' in summary_df.columns:
        axes[2].scatter(summary_df['original_reps'], summary_df['new_reps'], 
                       alpha=0.5, color='#9C27B0', s=50)
        max_val = max(summary_df['original_reps'].max(), summary_df['new_reps'].max())
        axes[2].plot([0, max_val], [0, max_val], 'r--', label='No change line')
        axes[2].set_title('Original vs New Rep Counts', fontweight='bold')
        axes[2].set_xlabel('Original Rep Count')
        axes[2].set_ylabel('New Rep Count')
        axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(viz_folder / 'phase2_resegmentation_summary.png', dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.close()
    print(f"  ✓ Saved: phase2_resegmentation_summary.png")
    
    return viz_folder


# =============================================================================
# REPORTS
# =============================================================================

def generate_summary_report(df, file_info, cleaning_report, resegment_summary, output_folder, folder_name):
    """
    Generate a comprehensive text summary report
    """
    report_path = Path(output_folder) / f'{folder_name}_preprocessing_report.txt'
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("        APPLIFT ML TRAINING - PREPROCESSING PIPELINE REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Generated: {timestamp}\n")
        f.write(f"Source Folder: {folder_name}\n")
        f.write("\n")
        
        # Phase 1 Summary
        f.write("-" * 70 + "\n")
        f.write("PHASE 1: DATA MERGING & CLEANING\n")
        f.write("-" * 70 + "\n\n")
        
        f.write(f"Files Processed: {len(file_info)}\n")
        f.write(f"Original Shape: {cleaning_report.get('original_shape', 'N/A')}\n")
        f.write(f"Final Shape: {cleaning_report.get('final_shape', 'N/A')}\n")
        f.write(f"Duplicates Removed: {cleaning_report.get('duplicates_removed', 0)}\n")
        f.write(f"Missing Values Fixed: {cleaning_report.get('missing_values_after', 0)} remaining\n")
        
        f.write("\n📁 Files Summary:\n")
        files_with_quality = sum(1 for fi in file_info if fi.get('had_quality_code', False))
        files_without_quality = len(file_info) - files_with_quality
        f.write(f"  - Files with quality_code column: {files_with_quality}\n")
        f.write(f"  - Files needing quality inference: {files_without_quality}\n")
        
        f.write("\n📊 Dataset Composition:\n")
        if 'target' in df.columns:
            target_counts = df['target'].value_counts().sort_index()
            for target, count in target_counts.items():
                name = QUALITY_NAMES.get(target, f'Unknown ({target})')
                pct = count / len(df) * 100
                f.write(f"  - Target {target} ({name}): {count:,} samples ({pct:.1f}%)\n")
        
        if 'equipment_code' in df.columns:
            f.write("\n🏋️ Equipment Distribution:\n")
            equip_names = {0: 'Dumbbell', 1: 'Barbell', 2: 'Weight Stack'}
            equip_counts = df['equipment_code'].value_counts().sort_index()
            for code, count in equip_counts.items():
                name = equip_names.get(code, f'Unknown ({code})')
                pct = count / len(df) * 100
                f.write(f"  - {name}: {count:,} samples ({pct:.1f}%)\n")
        
        if 'exercise_code' in df.columns:
            f.write("\n💪 Exercise Distribution:\n")
            ex_names = {0: 'Concentration Curls', 1: 'Overhead Extension', 2: 'Bench Press',
                       3: 'Back Squat', 4: 'Lateral Pulldown', 5: 'Seated Leg Extension'}
            ex_counts = df['exercise_code'].value_counts().sort_index()
            for code, count in ex_counts.items():
                name = ex_names.get(code, f'Unknown ({code})')
                pct = count / len(df) * 100
                f.write(f"  - {name}: {count:,} samples ({pct:.1f}%)\n")
        
        # Phase 2 Summary
        if resegment_summary:
            f.write("\n" + "-" * 70 + "\n")
            f.write("PHASE 2: RESEGMENTATION\n")
            f.write("-" * 70 + "\n\n")
            
            summary_df = pd.DataFrame(resegment_summary)
            successful = sum(1 for s in resegment_summary if s['status'] == 'success')
            
            f.write(f"Files Resegmented: {successful}/{len(resegment_summary)}\n")
            
            if 'rep_change' in summary_df.columns:
                avg_change = summary_df['rep_change'].mean()
                total_original = summary_df['original_reps'].sum()
                total_new = summary_df['new_reps'].sum()
                
                f.write(f"Total Original Reps: {total_original}\n")
                f.write(f"Total New Reps: {total_new}\n")
                f.write(f"Average Rep Change per File: {avg_change:.2f}\n")
        
        # Column Information
        f.write("\n" + "-" * 70 + "\n")
        f.write("FINAL DATASET COLUMNS\n")
        f.write("-" * 70 + "\n\n")
        
        for col in df.columns:
            dtype = df[col].dtype
            non_null = df[col].notna().sum()
            f.write(f"  {col}: {dtype} ({non_null:,} non-null)\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 70 + "\n")
    
    print(f"  ✓ Report saved: {report_path.name}")
    
    return report_path


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline():
    """
    Main function to run the complete preprocessing pipeline
    """
    print("\n" + "=" * 70)
    print("     🏋️ APPLIFT ML TRAINING - PREPROCESSING PIPELINE")
    print("=" * 70)
    print(f"\n📂 Project Root: {PROJECT_ROOT}")
    print(f"📁 Output Directory: {OUTPUT_DIR}")
    
    # =========================================================================
    # PHASE 1: Select folder and merge data
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 1: DATA MERGING & CLEANING")
    print("=" * 70)
    
    # Select folder
    print("\n📂 Opening folder selection dialog...")
    selected_folder = select_folder_ui()
    
    if not selected_folder:
        print("\n❌ No folder selected. Exiting.")
        return
    
    folder_name = Path(selected_folder).name
    print(f"\n✓ Selected folder: {folder_name}")
    print(f"  Full path: {selected_folder}")
    
    # Load and merge CSV files
    print("\n" + "-" * 50)
    print("Step 1.1: Loading and merging CSV files...")
    print("-" * 50)
    
    merged_df, file_info = load_and_merge_csv_files(selected_folder)
    
    if merged_df is None or len(merged_df) == 0:
        print("\n❌ No data loaded. Check if the folder contains CSV files.")
        return
    
    # Clean dataset
    print("\n" + "-" * 50)
    print("Step 1.2: Cleaning dataset...")
    print("-" * 50)
    
    cleaned_df, cleaning_report = clean_dataset(merged_df)
    
    # Save Phase 1 output
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Extract exercise and equipment names for filename
    exercise_name = "Unknown_Exercise"
    equipment_name = "Unknown_Equipment"
    
    if 'exercise_code' in cleaned_df.columns and len(cleaned_df) > 0:
        exercise_code = cleaned_df['exercise_code'].iloc[0]
        exercise_names = {0: 'Concentration_Curls', 1: 'Overhead_Extension', 2: 'Bench_Press',
                         3: 'Back_Squat', 4: 'Lateral_Pulldown', 5: 'Seated_Leg_Extension'}
        exercise_name = exercise_names.get(exercise_code, f'Exercise_{exercise_code}')
    
    if 'equipment_code' in cleaned_df.columns and len(cleaned_df) > 0:
        equipment_code = cleaned_df['equipment_code'].iloc[0]
        equipment_names = {0: 'Dumbbell', 1: 'Barbell', 2: 'Weight_Stack'}
        equipment_name = equipment_names.get(equipment_code, f'Equipment_{equipment_code}')
    
    phase1_filename = f'{equipment_name}_{exercise_name}_{folder_name}_merged_cleaned_{timestamp}.csv'
    phase1_path = MERGED_DIR / phase1_filename
    cleaned_df.to_csv(phase1_path, index=False)
    print(f"\n✓ Phase 1 output saved: {phase1_path.name}")
    
    # Create Phase 1 visualizations
    phase1_viz_folder = VISUALIZATIONS_DIR / 'phase1_merged'
    create_summary_visualizations(cleaned_df, file_info, cleaning_report, phase1_viz_folder, prefix="phase1")
    
    # =========================================================================
    # PHASE 2: Resegmentation
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 2: RESEGMENTATION")
    print("=" * 70)
    
    print("\n" + "-" * 50)
    print("Step 2.1: Resegmenting reps...")
    print("-" * 50)
    
    resegmented_df, resegment_summary = resegment_dataset(cleaned_df)
    
    # Save Phase 2 output
    phase2_filename = f'{equipment_name}_{exercise_name}_{folder_name}_resegmented_{timestamp}.csv'
    phase2_path = RESEGMENTED_DIR / phase2_filename
    resegmented_df.to_csv(phase2_path, index=False)
    print(f"\n✓ Phase 2 output saved: {phase2_path.name}")
    
    # Create Phase 2 visualizations
    phase2_viz_folder = VISUALIZATIONS_DIR / 'phase2_resegmented'
    create_resegmentation_visualizations(cleaned_df, resegmented_df, resegment_summary, phase2_viz_folder)
    
    # =========================================================================
    # Generate Summary Report
    # =========================================================================
    print("\n" + "=" * 70)
    print("GENERATING SUMMARY REPORT")
    print("=" * 70)
    
    generate_summary_report(resegmented_df, file_info, cleaning_report, resegment_summary, 
                           REPORTS_DIR, folder_name)
    
    # =========================================================================
    # Final Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("📋 PIPELINE COMPLETE - SUMMARY")
    print("=" * 70)
    
    print(f"\n📊 Dataset Statistics:")
    print(f"  • Files processed: {len(file_info)}")
    print(f"  • Total samples: {len(resegmented_df):,}")
    print(f"  • Features: {len(resegmented_df.columns)}")
    
    if 'target' in resegmented_df.columns:
        print(f"\n🎯 Target Distribution:")
        target_counts = resegmented_df['target'].value_counts().sort_index()
        for target, count in target_counts.items():
            name = QUALITY_NAMES.get(target, f'Code {target}')
            pct = count / len(resegmented_df) * 100
            print(f"  • {name}: {count:,} ({pct:.1f}%)")
    
    print(f"\n📁 Output Files:")
    print(f"  • Merged & Cleaned: {phase1_path}")
    print(f"  • Resegmented: {phase2_path}")
    print(f"  • Visualizations: {VISUALIZATIONS_DIR}")
    print(f"  • Reports: {REPORTS_DIR}")
    
    print("\n" + "=" * 70)
    print("✅ PREPROCESSING COMPLETE!")
    print("=" * 70 + "\n")
    
    return resegmented_df, file_info, cleaning_report, resegment_summary


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    run_pipeline()
