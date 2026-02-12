"""
AppLift ML Training - Model Tester
===================================
A straightforward script to test the trained Random Forest model by uploading 
a CSV file and classifying each rep.

Features:
- Load trained .pkl model
- Upload and process new CSV data
- Classify each rep with confidence scores
- Display simple results
- Export predictions to CSV

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

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR
OUTPUT_DIR = PROJECT_ROOT / 'output'
MODELS_DIR = OUTPUT_DIR / 'models'
PREDICTIONS_DIR = OUTPUT_DIR / 'predictions'
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

# Equipment and Exercise codes
EQUIPMENT_CODES = {
    0: 'Dumbbell',
    1: 'Barbell', 
    2: 'Weight Stack'
}

EXERCISE_CODES = {
    0: 'Concentration Curls',
    1: 'Overhead Extension',
    2: 'Bench Press',
    3: 'Back Squat',
    4: 'Lateral Pulldown',
    5: 'Seated Leg Extension'
}

# Quality names by exercise code (exercise-specific mistake types)
QUALITY_NAMES_BY_EXERCISE = {
    # Dumbbell exercises (0, 1)
    0: {0: 'Clean', 1: 'Uncontrolled Movement', 2: 'Abrupt Initiation'},
    1: {0: 'Clean', 1: 'Uncontrolled Movement', 2: 'Abrupt Initiation'},
    # Barbell exercises (2, 3)
    2: {0: 'Clean', 1: 'Uncontrolled Movement', 2: 'Inclination Asymmetry'},
    3: {0: 'Clean', 1: 'Uncontrolled Movement', 2: 'Inclination Asymmetry'},
    # Weight Stack exercises (4, 5)
    4: {0: 'Clean', 1: 'Pulling Too Fast', 2: 'Releasing Too Fast'},
    5: {0: 'Clean', 1: 'Pulling Too Fast', 2: 'Releasing Too Fast'},
}

# Default quality names (fallback)
QUALITY_NAMES = {
    0: 'Clean',
    1: 'Uncontrolled Movement',
    2: 'Abrupt Initiation'
}


def get_quality_names(exercise_code=None, df=None):
    """Get quality names based on exercise code or dataframe"""
    # If exercise_code provided directly
    if exercise_code is not None and exercise_code in QUALITY_NAMES_BY_EXERCISE:
        return QUALITY_NAMES_BY_EXERCISE[exercise_code]
    
    # Try to get from dataframe
    if df is not None and 'exercise_code' in df.columns:
        ex_code = df['exercise_code'].iloc[0]
        if ex_code in QUALITY_NAMES_BY_EXERCISE:
            return QUALITY_NAMES_BY_EXERCISE[ex_code]
    
    # Return default
    return QUALITY_NAMES


# =============================================================================
# FILE SELECTION DIALOGS
# =============================================================================

def select_model_file():
    """Open a file dialog to select a trained model (.pkl) file"""
    root = tk.Tk()
    root.withdraw()
    
    file_path = filedialog.askopenfilename(
        title="Select Trained Model (.pkl) File",
        initialdir=str(MODELS_DIR),
        filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
    )
    
    root.destroy()
    return file_path


def select_csv_file():
    """Open a file dialog to select a CSV file for prediction"""
    root = tk.Tk()
    root.withdraw()
    
    file_path = filedialog.askopenfilename(
        title="Select CSV File to Analyze",
        initialdir=str(PROJECT_ROOT),
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    
    root.destroy()
    return file_path


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model(model_path):
    """
    Load a trained model package from a .pkl file
    
    Returns:
    - model_package: Dictionary containing model, scaler, feature_names, etc.
    """
    print(f"\n📦 Loading model from: {Path(model_path).name}")
    
    model_package = joblib.load(model_path)
    
    print(f"  ✓ Model type: {model_package.get('model_type', 'Unknown')}")
    print(f"  ✓ Training date: {model_package.get('training_date', 'Unknown')}")
    print(f"  ✓ Number of features: {len(model_package.get('feature_names', []))}")
    
    # Display model metrics
    metrics = model_package.get('metrics', {})
    if metrics:
        print(f"\n📊 Model Performance (from training):")
        print(f"  • Test Accuracy: {metrics.get('test_accuracy', 'N/A'):.4f}" if isinstance(metrics.get('test_accuracy'), float) else f"  • Test Accuracy: N/A")
        print(f"  • CV Accuracy: {metrics.get('cv_accuracy_mean', 'N/A'):.4f} ± {metrics.get('cv_accuracy_std', 'N/A'):.4f}" if isinstance(metrics.get('cv_accuracy_mean'), float) else f"  • CV Accuracy: N/A")
    
    return model_package


# =============================================================================
# FEATURE COMPUTATION
# =============================================================================

def compute_rep_features(df, feature_names):
    """
    Compute aggregate features per rep to match the training features.
    Each rep becomes one sample for prediction.
    
    Parameters:
    - df: Raw DataFrame with sensor readings
    - feature_names: List of features expected by the model
    
    Returns:
    - features_df: DataFrame with one row per rep and computed features
    - rep_info: DataFrame with rep metadata (for displaying results)
    """
    print("\n📐 Computing rep-level features...")
    
    # Signal columns to compute features from
    signal_columns = ['filteredMag', 'filteredX', 'filteredY', 'filteredZ',
                     'accelMag', 'accelX', 'accelY', 'accelZ',
                     'gyroMag', 'gyroX', 'gyroY', 'gyroZ']
    signal_columns = [col for col in signal_columns if col in df.columns]
    
    print(f"  Signal columns found: {len(signal_columns)}")
    
    # Group by participant and rep (or just rep if no participant column)
    if 'participant' in df.columns and 'rep' in df.columns:
        group_cols = ['participant', 'rep']
    elif 'rep' in df.columns:
        group_cols = ['rep']
    else:
        # If no rep column, treat entire file as one rep
        df = df.copy()
        df['rep'] = 1
        group_cols = ['rep']
    
    # Filter out rep 0 (usually incomplete data)
    if 'rep' in df.columns:
        df = df[df['rep'] > 0].copy()
    
    if len(df) == 0:
        print("  ⚠️ No valid reps found (all reps are 0)")
        return None, None
    
    all_features = []
    rep_info_list = []
    
    grouped = df.groupby(group_cols)
    total_groups = len(grouped)
    print(f"  Found {total_groups} rep(s) to analyze...")
    
    for i, (group_key, group_df) in enumerate(grouped):
        features = {}
        rep_metadata = {}
        
        # Store rep info for display - handle different grouping scenarios
        if isinstance(group_key, tuple) and len(group_key) == 2:
            # Grouped by ['participant', 'rep']
            rep_metadata['participant'] = group_key[0]
            rep_metadata['rep'] = group_key[1]
        elif isinstance(group_key, tuple) and len(group_key) == 1:
            # Grouped by ['rep'] but still returns tuple
            rep_metadata['participant'] = df['participant'].iloc[0] if 'participant' in df.columns else 'Unknown'
            rep_metadata['rep'] = group_key[0]
        else:
            # Grouped by single column (returns scalar)
            rep_metadata['participant'] = df['participant'].iloc[0] if 'participant' in df.columns else 'Unknown'
            rep_metadata['rep'] = group_key
        
        # Get source file if available
        if 'source_file' in group_df.columns:
            rep_metadata['source_file'] = group_df['source_file'].iloc[0]
        
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
        rep_info_list.append(rep_metadata)
    
    features_df = pd.DataFrame(all_features)
    rep_info = pd.DataFrame(rep_info_list)
    
    print(f"  ✓ Computed {len(features_df.columns)} features for {len(features_df)} rep(s)")
    
    return features_df, rep_info


# =============================================================================
# PREDICTION
# =============================================================================

def predict_reps(model_package, features_df):
    """
    Use the trained model to predict classifications for each rep
    
    Parameters:
    - model_package: Dictionary containing model, scaler, feature_names
    - features_df: DataFrame with computed features
    
    Returns:
    - predictions: Array of predicted class labels
    - probabilities: Array of prediction probabilities
    """
    print("\n🔮 Making predictions...")
    
    model = model_package['model']
    scaler = model_package['scaler']
    feature_names = model_package['feature_names']
    # class_names loaded but not used here - used in display_results
    
    # Ensure we have all required features
    available_features = [col for col in feature_names if col in features_df.columns]
    missing_features = [col for col in feature_names if col not in features_df.columns]
    
    if missing_features:
        print(f"  ⚠️ Missing {len(missing_features)} features (will use 0 as placeholder):")
        print(f"     First 5: {missing_features[:5]}")
    
    # Create feature matrix with all required features
    X = pd.DataFrame(index=features_df.index)
    for col in feature_names:
        if col in features_df.columns:
            X[col] = features_df[col]
        else:
            X[col] = 0  # Placeholder for missing features
    
    # Handle missing values and infinities
    X = X.fillna(0)
    X = X.replace([np.inf, -np.inf], 0)
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Make predictions
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)
    
    print(f"  ✓ Predicted {len(predictions)} rep(s)")
    
    return predictions, probabilities


# =============================================================================
# RESULTS DISPLAY
# =============================================================================

def display_results(rep_info, predictions, probabilities, class_names):
    """
    Display prediction results in a formatted manner with detailed analysis
    
    Parameters:
    - rep_info: DataFrame with rep metadata
    - predictions: Array of predicted class labels
    - probabilities: Array of prediction probabilities
    - class_names: Dictionary mapping class indices to names
    """
    print("\n" + "=" * 70)
    print("📊 CLASSIFICATION RESULTS")
    print("=" * 70)
    
    results_data = []
    
    for i in range(len(predictions)):
        pred_class = predictions[i]
        pred_name = class_names.get(pred_class, f'Class {pred_class}')
        confidence = probabilities[i][pred_class] * 100
        
        # Get rep info
        participant = rep_info.iloc[i].get('participant', 'Unknown')
        rep_num = rep_info.iloc[i].get('rep', i+1)
        
        # Confidence indicator
        if confidence >= 80:
            conf_indicator = "🟢 HIGH"
        elif confidence >= 60:
            conf_indicator = "🟡 MEDIUM"
        else:
            conf_indicator = "🔴 LOW"
        
        print(f"\n  Rep {rep_num} (Participant {participant}):")
        print(f"    🎯 Prediction: {pred_name}")
        print(f"    📊 Confidence: {confidence:.1f}% {conf_indicator}")
        
        # Show all class probabilities
        print(f"    📈 Probabilities:")
        for j, prob in enumerate(probabilities[i]):
            class_label = class_names.get(j, f'Class {j}')
            bar_length = int(prob * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            print(f"       {class_label}: [{bar}] {prob*100:.1f}%")
        
        # Collect results for export
        results_data.append({
            'participant': participant,
            'rep': rep_num,
            'prediction': pred_name,
            'prediction_code': pred_class,
            'confidence': confidence,
            **{f'prob_{class_names.get(j, f"Class_{j}")}': prob*100 for j, prob in enumerate(probabilities[i])}
        })
    
    results_df = pd.DataFrame(results_data)
    
    # Summary statistics
    print("\n" + "-" * 70)
    print("📋 SUMMARY")
    print("-" * 70)
    
    prediction_counts = pd.Series(predictions).value_counts()
    total = len(predictions)
    
    print("🎯 Prediction Distribution:")
    for class_idx, count in prediction_counts.items():
        class_label = class_names.get(class_idx, f'Class {class_idx}')
        percentage = count / total * 100
        print(f"  {class_label}: {count} rep(s) ({percentage:.1f}%)")
    
    avg_confidence = np.mean([probabilities[i][predictions[i]] for i in range(len(predictions))]) * 100
    print(f"\n📊 Average Confidence: {avg_confidence:.1f}%")
    
    return results_df


def show_results_gui(results_df, class_names):
    """
    Display results in a GUI window with a table
    """
    root = tk.Tk()
    root.title("🏋️ AppLift - Rep Classification Results")
    root.geometry("900x600")
    root.configure(bg='#f5f5f5')
    
    # Header
    header_frame = tk.Frame(root, bg='#4CAF50', pady=15)
    header_frame.pack(fill=tk.X)
    
    header = tk.Label(header_frame, text="🏋️ Rep Classification Results", 
                      font=('Arial', 18, 'bold'), bg='#4CAF50', fg='white')
    header.pack()
    
    # Summary frame
    summary_frame = tk.Frame(root, bg='#e8f5e9', pady=10)
    summary_frame.pack(fill=tk.X, padx=20, pady=10)
    
    total_reps = len(results_df)
    avg_conf = results_df['confidence'].mean()
    
    summary_text = f"Total Reps: {total_reps} | Average Confidence: {avg_conf:.1f}%"
    summary_label = tk.Label(summary_frame, text=summary_text, 
                            font=('Arial', 12), bg='#e8f5e9', fg='#333')
    summary_label.pack()
    
    # Class distribution
    dist_text = "Distribution: "
    for pred_name in results_df['prediction'].unique():
        count = len(results_df[results_df['prediction'] == pred_name])
        dist_text += f"{pred_name}: {count} | "
    
    dist_label = tk.Label(summary_frame, text=dist_text.rstrip(' | '), 
                         font=('Arial', 10), bg='#e8f5e9', fg='#666')
    dist_label.pack()
    
    # Table frame
    table_frame = tk.Frame(root, bg='#f5f5f5')
    table_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
    
    # Create treeview (table)
    columns = ('Rep', 'Participant', 'Prediction', 'Confidence')
    tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=15)
    
    # Configure columns
    tree.heading('Rep', text='Rep #')
    tree.heading('Participant', text='Participant')
    tree.heading('Prediction', text='Classification')
    tree.heading('Confidence', text='Confidence')
    
    tree.column('Rep', width=80, anchor='center')
    tree.column('Participant', width=100, anchor='center')
    tree.column('Prediction', width=200, anchor='center')
    tree.column('Confidence', width=120, anchor='center')
    
    # Add scrollbar
    scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=tree.yview)
    tree.configure(yscrollcommand=scrollbar.set)
    
    # Configure tags for coloring based on prediction code (0, 1, 2)
    tree.tag_configure('quality_0', background='#c8e6c9')  # Green for Clean (code 0)
    tree.tag_configure('quality_1', background='#fff9c4')  # Yellow for code 1 (Uncontrolled/Pulling Too Fast)
    tree.tag_configure('quality_2', background='#ffcdd2')  # Red/Pink for code 2 (Abrupt/Inclination/Releasing)
    tree.tag_configure('high_conf', foreground='#2e7d32')
    tree.tag_configure('low_conf', foreground='#c62828')
    
    # Insert data
    for _, row in results_df.iterrows():
        conf_str = f"{row['confidence']:.1f}%"
        
        # Determine tag based on prediction code (0, 1, 2)
        pred_code = row.get('prediction_code', 0)
        tag = f'quality_{pred_code}'
        
        tree.insert('', tk.END, values=(
            row['rep'],
            row['participant'],
            row['prediction'],
            conf_str
        ), tags=(tag,))
    
    tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    # Button frame
    button_frame = tk.Frame(root, bg='#f5f5f5')
    button_frame.pack(fill=tk.X, padx=20, pady=15)
    
    def save_results():
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = PREDICTIONS_DIR / f'predictions_{timestamp}.csv'
        results_df.to_csv(save_path, index=False)
        messagebox.showinfo("Saved", f"Results saved to:\n{save_path}")
    
    tk.Button(button_frame, text="💾 Save Results to CSV", 
             command=save_results, font=('Arial', 11, 'bold'),
             bg='#2196F3', fg='white', padx=20, pady=8).pack(side=tk.LEFT, padx=5)
    
    tk.Button(button_frame, text="❌ Close", 
             command=root.destroy, font=('Arial', 11),
             bg='#f44336', fg='white', padx=20, pady=8).pack(side=tk.LEFT, padx=5)
    
    # Center window
    root.update_idletasks()
    x = (root.winfo_screenwidth() - root.winfo_width()) // 2
    y = (root.winfo_screenheight() - root.winfo_height()) // 2
    root.geometry(f"+{x}+{y}")
    
    root.mainloop()


# =============================================================================
# EXPORT RESULTS
# =============================================================================

def export_results(results_df, csv_filename):
    """
    Export prediction results to a CSV file
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_name = Path(csv_filename).stem
    output_path = PREDICTIONS_DIR / f'{base_name}_predictions_{timestamp}.csv'
    
    results_df.to_csv(output_path, index=False)
    
    print(f"\n💾 Results exported to: {output_path}")
    
    return output_path


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_model_tester():
    """
    Main function to run the model testing pipeline
    """
    print("\n" + "=" * 70)
    print("     🧪 APPLIFT MODEL TESTER")
    print("=" * 70)
    print(f"\n📂 Project Root: {PROJECT_ROOT}")
    print(f"📁 Models Directory: {MODELS_DIR}")
    print(f"📁 Predictions Directory: {PREDICTIONS_DIR}")
    
    # =========================================================================
    # STEP 1: Select Model File
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: SELECT MODEL")
    print("=" * 70)
    
    print("\n📂 Opening model selection dialog...")
    model_path = select_model_file()
    
    if not model_path:
        print("\n❌ No model selected. Exiting.")
        return
    
    print(f"\n✓ Selected model: {Path(model_path).name}")
    
    # Load model
    model_package = load_model(model_path)
    
    if model_package is None:
        print("\n❌ Failed to load model. Exiting.")
        return
    
    # =========================================================================
    # STEP 2: Select CSV File
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: SELECT CSV FILE TO ANALYZE")
    print("=" * 70)
    
    print("\n📂 Opening CSV file selection dialog...")
    csv_file = select_csv_file()
    
    if not csv_file:
        print("\n❌ No CSV file selected. Exiting.")
        return
    
    print(f"\n✓ Selected file: {Path(csv_file).name}")
    
    # Load CSV
    print("\n📖 Loading CSV data...")
    df = pd.read_csv(csv_file)
    print(f"  ✓ Loaded {len(df):,} rows × {len(df.columns)} columns")
    
    # Display data info
    print(f"\n📊 Data Overview:")
    print(f"  • Columns: {list(df.columns[:10])}{'...' if len(df.columns) > 10 else ''}")
    if 'rep' in df.columns:
        unique_reps = df['rep'].unique()
        print(f"  • Unique reps: {len(unique_reps)} (values: {sorted(unique_reps)[:10]})")
    if 'participant' in df.columns:
        print(f"  • Participants: {df['participant'].unique()}")
    
    # =========================================================================
    # STEP 3: Compute Features
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: COMPUTE FEATURES")
    print("=" * 70)
    
    feature_names = model_package['feature_names']
    features_df, rep_info = compute_rep_features(df, feature_names)
    
    if features_df is None or len(features_df) == 0:
        print("\n❌ No features computed. Check your data.")
        return
    
    # =========================================================================
    # STEP 4: Make Predictions
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 4: CLASSIFY REPS")
    print("=" * 70)
    
    predictions, probabilities = predict_reps(model_package, features_df)
    
    # =========================================================================
    # STEP 5: Display Results
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 5: RESULTS")
    print("=" * 70)
    
    # Get class names: prefer from model, fallback to exercise-specific from CSV
    class_names = model_package.get('class_names')
    if class_names is None:
        class_names = get_quality_names(df=df)
    results_df = display_results(rep_info, predictions, probabilities, class_names)
    
    # =========================================================================
    # STEP 6: Export Results
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 6: EXPORT RESULTS")
    print("=" * 70)
    
    output_path = export_results(results_df, csv_file)
    
    # =========================================================================
    # Final Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("✅ MODEL TESTING COMPLETE!")
    print("=" * 70)
    
    # Show GUI with results
    print("\n🖼️ Opening results window...")
    show_results_gui(results_df, class_names)
    
    return results_df


def run_quick_test(model_path=None, csv_path=None):
    """
    Quick test function for programmatic use (no GUI file dialogs)
    
    Parameters:
    - model_path: Path to the .pkl model file
    - csv_path: Path to the CSV file to analyze
    
    Returns:
    - results_df: DataFrame with predictions
    """
    if model_path is None or csv_path is None:
        print("Error: Both model_path and csv_path are required for quick test")
        return None
    
    print("\n" + "=" * 70)
    print("     🧪 QUICK MODEL TEST")
    print("=" * 70)
    
    # Load model
    model_package = load_model(model_path)
    
    # Load CSV
    print(f"\n📖 Loading: {Path(csv_path).name}")
    df = pd.read_csv(csv_path)
    print(f"  ✓ Loaded {len(df):,} rows")
    
    # Compute features
    feature_names = model_package['feature_names']
    features_df, rep_info = compute_rep_features(df, feature_names)
    
    if features_df is None or len(features_df) == 0:
        print("\n❌ No features computed.")
        return None
    
    # Predict
    predictions, probabilities = predict_reps(model_package, features_df)
    
    # Display results - prefer model class_names, fallback to exercise-specific
    class_names = model_package.get('class_names')
    if class_names is None:
        class_names = get_quality_names(df=df)
    results_df = display_results(rep_info, predictions, probabilities, class_names)
    
    # Export
    export_results(results_df, csv_path)
    
    print("\n✅ Quick test complete!")
    
    return results_df


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    run_model_tester()
