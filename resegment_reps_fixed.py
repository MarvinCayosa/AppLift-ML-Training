
"""
Re-segmentation Script for Exercise Rep Data
============================================
This script corrects rep boundaries by finding actual valley (lowest) points
and re-assigning rep numbers so each rep starts immediately after the 
previous rep's valley (lowest point).

The problem with the original segmentation:
- Reps were labeled when DETECTED (at peak), not when they STARTED (at valley)
- This causes gaps between rep segments

The fix:
- Find all valleys (local minima) in the filtered magnitude signal
- Each rep starts at a valley and ends at the next valley
- Re-assign rep numbers based on these corrected boundaries
"""

import pandas as pd
import numpy as np
from scipy.signal import find_peaks, savgol_filter
import matplotlib.pyplot as plt
import os
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, ttk
from pathlib import Path

# Get the script's directory and project root
SCRIPT_DIR = Path(__file__).parent.resolve()  # Points to "AppLift ML Training" folder
PROJECT_ROOT = SCRIPT_DIR  # Use "AppLift ML Training" as project root

# Paths for data and output - look in current folder structure  
DATA_DIR = SCRIPT_DIR  # "AppLift ML Training" directory
VIZ_DIR = SCRIPT_DIR / 'visualizations' / 'segmentation'  # AppLift ML Training/visualizations/segmentation
DATASETS_DIR = SCRIPT_DIR  # "AppLift ML Training" directory where CSV files are located

# Ensure directories exist
VIZ_DIR.mkdir(parents=True, exist_ok=True)


def select_from_datasets_ui():
    """
    Show a UI to browse and select from the datasets folder structure
    """
    root = tk.Tk()
    root.title("📊 Select Dataset for Resegmentation")
    root.geometry("600x500")
    root.configure(bg='#f0f0f0')
    
    selected_file = [None]  # Use list to allow modification in nested function
    
    # Header
    header = tk.Label(root, text="🔄 Rep Resegmentation - File Selector", 
                      font=('Arial', 16, 'bold'), bg='#f0f0f0', fg='#333')
    header.pack(pady=10)
    
    # Instructions
    instructions = tk.Label(root, text="Select a CSV file to resegment:",
                           font=('Arial', 10), bg='#f0f0f0', fg='#666')
    instructions.pack(pady=5)
    
    # Frame for treeview
    tree_frame = tk.Frame(root, bg='#f0f0f0')
    tree_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
    
    # Scrollbar
    scrollbar = ttk.Scrollbar(tree_frame)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    # Treeview for folder structure
    tree = ttk.Treeview(tree_frame, yscrollcommand=scrollbar.set)
    tree.pack(fill=tk.BOTH, expand=True)
    scrollbar.config(command=tree.yview)
    
    tree.heading('#0', text='Datasets', anchor='w')
    
    # Populate tree with datasets folder structure - use defined path constants
    datasets_dir = str(DATASETS_DIR)
    data_dir = str(DATA_DIR)
    
    def populate_tree(parent, path):
        """Recursively populate tree with folder contents"""
        try:
            items = sorted(os.listdir(path))
            for item in items:
                item_path = os.path.join(path, item)
                if os.path.isdir(item_path):
                    # Folder - add with folder icon
                    node = tree.insert(parent, 'end', text=f"📁 {item}", open=False, values=(item_path,))
                    populate_tree(node, item_path)
                elif item.endswith('.csv'):
                    # CSV file - add with file icon
                    tree.insert(parent, 'end', text=f"📄 {item}", values=(item_path,))
        except PermissionError:
            pass
    
    # Add datasets folder
    if os.path.exists(datasets_dir):
        datasets_node = tree.insert('', 'end', text='📁 datasets', open=True, values=(datasets_dir,))
        populate_tree(datasets_node, datasets_dir)
    
    # Add data folder
    if os.path.exists(data_dir):
        data_node = tree.insert('', 'end', text='📁 data', open=True, values=(data_dir,))
        populate_tree(data_node, data_dir)
    
    # Selected file label
    selected_label = tk.Label(root, text="Selected: None", 
                              font=('Arial', 10), bg='#f0f0f0', fg='#007bff')
    selected_label.pack(pady=5)
    
    def on_select(event):
        """Handle tree selection"""
        selection = tree.selection()
        if selection:
            item = tree.item(selection[0])
            values = item.get('values', [])
            if values:
                path = values[0]
                if path.endswith('.csv'):
                    selected_file[0] = path
                    filename = os.path.basename(path)
                    selected_label.config(text=f"Selected: {filename}", fg='#28a745')
    
    def on_double_click(event):
        """Handle double-click to select and confirm"""
        selection = tree.selection()
        if selection:
            item = tree.item(selection[0])
            values = item.get('values', [])
            if values:
                path = values[0]
                if path.endswith('.csv'):
                    selected_file[0] = path
                    root.destroy()
    
    tree.bind('<<TreeviewSelect>>', on_select)
    tree.bind('<Double-1>', on_double_click)
    
    # Button frame
    btn_frame = tk.Frame(root, bg='#f0f0f0')
    btn_frame.pack(pady=10)
    
    def browse_file():
        """Open file browser"""
        file_path = filedialog.askopenfilename(
            title="Select CSV Dataset",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if file_path:
            selected_file[0] = file_path
            root.destroy()
    
    def confirm_selection():
        """Confirm the current selection"""
        if selected_file[0]:
            root.destroy()
        else:
            selected_label.config(text="⚠️ Please select a CSV file first!", fg='red')
    
    def cancel():
        """Cancel and exit"""
        selected_file[0] = None
        root.destroy()
    
    browse_btn = tk.Button(btn_frame, text="📂 Browse...", command=browse_file,
                          font=('Arial', 10), bg='#6c757d', fg='white', 
                          padx=15, pady=5, cursor='hand2')
    browse_btn.pack(side=tk.LEFT, padx=5)
    
    confirm_btn = tk.Button(btn_frame, text="✅ Resegment Selected", command=confirm_selection,
                           font=('Arial', 10, 'bold'), bg='#28a745', fg='white',
                           padx=15, pady=5, cursor='hand2')
    confirm_btn.pack(side=tk.LEFT, padx=5)
    
    cancel_btn = tk.Button(btn_frame, text="❌ Cancel", command=cancel,
                          font=('Arial', 10), bg='#dc3545', fg='white',
                          padx=15, pady=5, cursor='hand2')
    cancel_btn.pack(side=tk.LEFT, padx=5)
    
    # Center the window
    root.update_idletasks()
    x = (root.winfo_screenwidth() - root.winfo_width()) // 2
    y = (root.winfo_screenheight() - root.winfo_height()) // 2
    root.geometry(f"+{x}+{y}")
    
    root.mainloop()
    
    return selected_file[0]


def find_valleys(signal, distance=15, prominence=0.3):
    """
    Find valleys (local minima) in the signal by inverting and finding peaks
    
    Parameters:
    - signal: The signal array to analyze
    - distance: Minimum distance between valleys (in samples)
    - prominence: Minimum prominence of valleys
    
    Returns:
    - valley_indices: Array of indices where valleys occur
    - valley_properties: Properties of the valleys
    """
    # Invert signal to find minima as peaks
    inverted = -signal
    peaks, properties = find_peaks(inverted, distance=distance, prominence=prominence)
    return peaks, properties


def resegment_reps(df, signal_column='filteredMag', min_rep_duration_ms=500, max_rep_duration_ms=10000):
    """
    Re-segment reps based on valley-to-valley detection
    
    Each rep is defined as: valley → peak → next valley
    The rep starts at the valley and ends at the next valley
    
    Parameters:
    - df: DataFrame with sensor data
    - signal_column: Column to use for valley detection (default: 'filteredMag')
                     NOTE: Always use magnitude for weight stack exercises, NOT raw Z values
    - min_rep_duration_ms: Minimum rep duration in milliseconds
    - max_rep_duration_ms: Maximum rep duration in milliseconds
    
    Returns:
    - df_resegmented: DataFrame with corrected 'rep' column
    - rep_info: List of dicts with rep boundary information
    """
    # Detect equipment type and ensure proper signal column
    # For weight stack exercises (equipment_code == 2), always use magnitude
    if 'equipment_code' in df.columns:
        equipment_code = df['equipment_code'].iloc[0]
        if equipment_code == 2:  # Weight Stack
            signal_column = 'filteredMag'
            print(f"  Detected Weight Stack exercise - using filteredMag (magnitude) for segmentation")
    
    # Verify signal column exists
    if signal_column not in df.columns:
        print(f"  ⚠️ Warning: '{signal_column}' not found, falling back to 'filteredMag'")
        signal_column = 'filteredMag'
    
    print(f"  Using signal column: '{signal_column}'")
    
    signal = df[signal_column].values
    timestamps = df['timestamp_ms'].values
    
    # Smooth the signal slightly to reduce noise-induced false valleys
    if len(signal) > 11:
        signal_smooth = savgol_filter(signal, window_length=11, polyorder=3)
    else:
        signal_smooth = signal
    
    # Calculate adaptive parameters based on signal characteristics
    signal_range = np.max(signal_smooth) - np.min(signal_smooth)
    signal_std = np.std(signal_smooth)
    
    # Find valleys with adaptive prominence (at least 10% of range or 1 std)
    min_prominence = max(signal_range * 0.1, signal_std * 0.5, 0.3)
    
    # Estimate minimum distance between valleys (assume ~2-5 second reps at ~10-20Hz)
    # Use median time step to calculate sample rate
    if len(timestamps) > 1:
        median_dt = np.median(np.diff(timestamps))
        sample_rate = 1000 / median_dt  # Hz
        min_distance = int(0.5 * sample_rate)  # Minimum 0.5 seconds between valleys
    else:
        min_distance = 5
    
    print(f"Signal analysis:")
    print(f"  Range: {signal_range:.3f}")
    print(f"  Std: {signal_std:.3f}")
    print(f"  Min prominence for valleys: {min_prominence:.3f}")
    print(f"  Min distance between valleys: {min_distance} samples")
    
    # Find all valleys
    valley_indices, valley_props = find_valleys(signal_smooth, distance=min_distance, prominence=min_prominence)
    
    print(f"\nFound {len(valley_indices)} potential valleys")
    
    # Filter valleys by minimum rep duration
    valid_valleys = [0]  # Always start from the beginning
    
    for i, idx in enumerate(valley_indices):
        # Check if this valley is far enough from the last valid valley
        last_valley_time = timestamps[valid_valleys[-1]]
        current_valley_time = timestamps[idx]
        duration_ms = current_valley_time - last_valley_time
        
        if duration_ms >= min_rep_duration_ms and duration_ms <= max_rep_duration_ms:
            valid_valleys.append(idx)
    
    # Add the last index if not already included
    if valid_valleys[-1] != len(df) - 1:
        # Check if there's a valid rep between last valley and end
        last_valley_time = timestamps[valid_valleys[-1]]
        end_time = timestamps[-1]
        if end_time - last_valley_time >= min_rep_duration_ms:
            # Find the actual valley closest to the end
            remaining_signal = signal_smooth[valid_valleys[-1]:]
            if len(remaining_signal) > min_distance:
                local_valleys, _ = find_valleys(remaining_signal, distance=min_distance//2, prominence=min_prominence*0.5)
                if len(local_valleys) > 0:
                    valid_valleys.append(valid_valleys[-1] + local_valleys[-1])
    
    print(f"Valid valley boundaries: {len(valid_valleys)}")
    print(f"Valley indices: {valid_valleys}")
    print(f"Valley timestamps: {[timestamps[v] for v in valid_valleys]}")
    
    # Create new rep labels based on valley boundaries
    df_resegmented = df.copy()
    df_resegmented['rep_original'] = df_resegmented['rep'].copy()  # Keep original for comparison
    df_resegmented['rep'] = 0
    
    rep_info = []
    
    for i in range(len(valid_valleys) - 1):
        start_idx = valid_valleys[i]
        end_idx = valid_valleys[i + 1]
        rep_num = i + 1
        
        # Assign rep number to all samples in this range
        df_resegmented.loc[start_idx:end_idx-1, 'rep'] = rep_num
        
        # Store rep info
        start_time = timestamps[start_idx]
        end_time = timestamps[end_idx - 1]
        
        # Find peak within this rep
        rep_signal = signal_smooth[start_idx:end_idx]
        peak_local_idx = np.argmax(rep_signal)
        peak_idx = start_idx + peak_local_idx
        peak_time = timestamps[peak_idx]
        peak_value = signal[peak_idx]
        valley_value = signal[start_idx]
        
        rep_info.append({
            'rep': rep_num,
            'start_idx': start_idx,
            'end_idx': end_idx - 1,
            'next_rep_start_idx': end_idx,
            'start_time_ms': start_time,
            'end_time_ms': end_time,
            'duration_ms': end_time - start_time,
            'peak_idx': peak_idx,
            'peak_time_ms': peak_time,
            'peak_value': peak_value,
            'valley_value': valley_value,
            'amplitude': peak_value - valley_value
        })
    
    # Handle the last segment (from last valley to end)
    if valid_valleys[-1] < len(df) - 1:
        start_idx = valid_valleys[-1]
        end_idx = len(df) - 1
        rep_num = len(valid_valleys)
        
        df_resegmented.loc[start_idx:end_idx, 'rep'] = rep_num
        
        start_time = timestamps[start_idx]
        end_time = timestamps[end_idx]
        
        rep_signal = signal_smooth[start_idx:end_idx+1]
        peak_local_idx = np.argmax(rep_signal)
        peak_idx = start_idx + peak_local_idx
        peak_time = timestamps[peak_idx]
        peak_value = signal[peak_idx]
        valley_value = signal[start_idx]
        
        rep_info.append({
            'rep': rep_num,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'start_time_ms': start_time,
            'end_time_ms': end_time,
            'duration_ms': end_time - start_time,
            'peak_idx': peak_idx,
            'peak_time_ms': peak_time,
            'peak_value': peak_value,
            'valley_value': valley_value,
            'amplitude': peak_value - valley_value
        })
    
    return df_resegmented, rep_info


def visualize_resegmentation(df_original, df_resegmented, rep_info, signal_column='filteredMag', output_path=None):
    """
    Create visualization comparing original and resegmented reps
    
    Parameters:
    - df_original: Original DataFrame
    - df_resegmented: Resegmented DataFrame
    - rep_info: List of rep information dictionaries
    - signal_column: The signal column used for segmentation (default: 'filteredMag')
    - output_path: Path to save the visualization
    """
    if output_path is None:
        output_path = str(VIZ_DIR / 'resegmentation_comparison.png')
    
    # Determine label based on signal column
    signal_label = 'Filtered Magnitude' if 'Mag' in signal_column else signal_column
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    fig.suptitle('Rep Resegmentation Comparison', fontsize=16, fontweight='bold')
    
    timestamps = df_original['timestamp_ms'].values
    signal = df_original[signal_column].values
    
    # Get sorted unique reps for both original and resegmented
    original_reps = sorted(df_original['rep'].unique())
    
    # Plot 1: Original segmentation (showing gaps)
    axes[0].plot(timestamps, signal, 'b-', linewidth=1, alpha=0.7, label=signal_label)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(original_reps)))
    for i, rep in enumerate(original_reps):
        rep_data = df_original[df_original['rep'] == rep]
        axes[0].axvspan(rep_data['timestamp_ms'].min(), rep_data['timestamp_ms'].max(),
                       alpha=0.3, color=colors[i], label=f'Rep {rep}')
    
    axes[0].set_xlabel('Time (ms)')
    axes[0].set_ylabel(signal_label)
    axes[0].set_title('ORIGINAL Segmentation (gaps visible between reps)')
    axes[0].legend(loc='upper right', ncol=min(len(original_reps), 6))
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Resegmented with continuous boundaries
    axes[1].plot(timestamps, signal, 'b-', linewidth=1, alpha=0.7, label=signal_label)
    
    new_reps = sorted([r for r in df_resegmented['rep'].unique() if r > 0])
    colors = plt.cm.tab10(np.linspace(0, 1, len(new_reps)))
    
    for i, rep in enumerate(new_reps):
        rep_data = df_resegmented[df_resegmented['rep'] == rep]
        start_time = rep_data['timestamp_ms'].min()
        
        if i < len(new_reps) - 1:
            next_rep = new_reps[i + 1]
            next_rep_data = df_resegmented[df_resegmented['rep'] == next_rep]
            end_time = next_rep_data['timestamp_ms'].min()
        else:
            end_time = rep_data['timestamp_ms'].max()
        
        axes[1].axvspan(start_time, end_time, alpha=0.3, color=colors[i], label=f'Rep {rep}')
    
    # Mark valleys (rep boundaries)
    for info in rep_info:
        axes[1].axvline(x=info['start_time_ms'], color='red', linestyle='--', alpha=0.7, linewidth=1)
        axes[1].plot(info['start_time_ms'], info['valley_value'], 'rv', markersize=10)
        axes[1].plot(info['peak_time_ms'], info['peak_value'], 'g^', markersize=10)
    
    axes[1].set_xlabel('Time (ms)')
    axes[1].set_ylabel(signal_label)
    axes[1].set_title('RESEGMENTED (continuous valley-to-valley boundaries - NO GAPS)')
    axes[1].legend(loc='upper right', ncol=min(len(new_reps), 6))
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Overlay comparison
    axes[2].plot(timestamps, signal, 'b-', linewidth=1.5, label='Signal')
    
    for rep in original_reps:
        rep_data = df_original[df_original['rep'] == rep]
        axes[2].axvline(x=rep_data['timestamp_ms'].min(), color='orange', linestyle='-', 
                       alpha=0.8, linewidth=2, label='Original boundary' if rep == sorted(original_reps)[0] else '')
    
    for info in rep_info:
        axes[2].axvline(x=info['start_time_ms'], color='green', linestyle='--', 
                       alpha=0.8, linewidth=2, label='Corrected boundary' if info == rep_info[0] else '')
        axes[2].plot(info['start_time_ms'], info['valley_value'], 'gv', markersize=12, zorder=5)
    
    axes[2].set_xlabel('Time (ms)')
    axes[2].set_ylabel(signal_label)
    axes[2].set_title(f'Boundary Comparison (Orange=Original, Green=Corrected at valleys) - Using {signal_label}')
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Comparison visualization saved to '{output_path}'")
    plt.show()
    
    return fig


def select_participant_and_session_ui(df):
    """
    Show a UI to select a participant number and session/set from the dataset
    
    Parameters:
    - df: DataFrame containing the data
    
    Returns:
    - dict with selected_participant, selected_session, or None if cancelled
    """
    root = tk.Tk()
    root.title("👤 Select Participant & Session - Multiple Sessions Supported")
    root.geometry("850x650")  # Increased width to accommodate quality columns
    root.configure(bg='#f5f5f5')
    
    selection = [None]
    
    # Header
    header_frame = tk.Frame(root, bg='#2196F3', pady=15)
    header_frame.pack(fill=tk.X)
    
    header = tk.Label(header_frame, text="👤 Participant & Session Selector", 
                      font=('Arial', 18, 'bold'), bg='#2196F3', fg='white')
    header.pack()
    
    subtitle = tk.Label(header_frame, text="Select a participant and session to visualize\n(After viewing, you'll automatically return here for another selection)",
                       font=('Arial', 11), bg='#2196F3', fg='white')
    subtitle.pack()
    
    # Get participant and session statistics with quality distribution
    if 'participant' in df.columns and 'source_file' in df.columns:
        # Check for quality column (could be 'quality_code', 'target', or 'quality')
        quality_column = None
        for col in ['quality_code', 'target', 'quality']:
            if col in df.columns:
                quality_column = col
                break
        
        # Quality labels mapping (based on common exercise quality classification)
        quality_labels = {
            0: 'Clean',
            1: 'Uncontrolled Movement', 
            2: 'Abrupt Initiation'
        }
        
        # Group by participant and source_file to get sessions
        session_stats = []
        
        for participant_id in sorted(df['participant'].dropna().unique()):
            p_data = df[df['participant'] == participant_id]
            sessions = p_data['source_file'].unique()
            
            for session_file in sorted(sessions):
                session_data = p_data[p_data['source_file'] == session_file]
                reps = session_data['rep'].nunique()
                samples = len(session_data)
                duration = (session_data['timestamp_ms'].max() - session_data['timestamp_ms'].min()) / 1000 if len(session_data) > 1 else 0
                
                # Calculate quality distribution if quality column exists
                quality_dist = {}
                if quality_column and quality_column in session_data.columns:
                    # Group by rep and get the quality for each rep (take mode/most common)
                    rep_qualities = []
                    for rep_num in session_data['rep'].unique():
                        if rep_num > 0:  # Skip rep 0
                            rep_data = session_data[session_data['rep'] == rep_num]
                            if len(rep_data) > 0:
                                # Get most common quality for this rep
                                quality_mode = rep_data[quality_column].mode()
                                if len(quality_mode) > 0:
                                    rep_qualities.append(quality_mode.iloc[0])
                    
                    # Count quality distribution
                    for quality_code in [0, 1, 2]:
                        count = rep_qualities.count(quality_code)
                        quality_dist[quality_code] = count
                else:
                    quality_dist = {0: 0, 1: 0, 2: 0}
                
                # Extract session identifier from filename
                session_name = session_file.replace('.csv', '').split('_')[-2:]  # Get last 2 parts
                session_display = '_'.join(session_name)
                
                session_stats.append({
                    'participant_id': int(participant_id),
                    'session_file': session_file,
                    'session_display': session_display,
                    'reps': reps,
                    'samples': samples,
                    'duration': duration,
                    'quality_dist': quality_dist
                })
    else:
        session_stats = []
        quality_column = None
        quality_labels = {}
    
    # Instructions with quality distribution summary
    total_participants = len(set(s['participant_id'] for s in session_stats))
    
    instructions_text = f"Found {len(session_stats)} sessions across {total_participants} participants.\n"
    
    # Add quality distribution summary if available
    if quality_column and session_stats:
        total_quality_dist = {0: 0, 1: 0, 2: 0}
        for stat in session_stats:
            for quality_code, count in stat['quality_dist'].items():
                total_quality_dist[quality_code] += count
        
        total_reps = sum(total_quality_dist.values())
        if total_reps > 0:
            instructions_text += f"\n📊 Overall Quality Distribution ({total_reps} reps):\n"
            for quality_code, count in total_quality_dist.items():
                label = quality_labels.get(quality_code, f'Quality {quality_code}')
                percentage = (count / total_reps) * 100
                instructions_text += f"• {label}: {count} reps ({percentage:.1f}%)\n"
        
        instructions_text += "\nSelect a specific participant session to visualize:"
    else:
        instructions_text += "Select a specific participant session to visualize:"
    
    instructions = tk.Label(root, 
                           text=instructions_text,
                           font=('Arial', 10), bg='#f5f5f5', fg='#666', justify='left')
    instructions.pack(pady=15)
    
    # Session list frame
    list_frame = tk.LabelFrame(root, text="📋 Available Participant Sessions", 
                              font=('Arial', 12, 'bold'), bg='#f5f5f5', fg='#333')
    list_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
    
    # Create treeview for session list with quality distribution
    if quality_column:
        columns = ('Participant', 'Session', 'Reps', 'Clean', 'Uncontrolled', 'Abrupt', 'Duration (s)')
    else:
        columns = ('Participant', 'Session', 'Reps', 'Samples', 'Duration (s)')
    
    tree = ttk.Treeview(list_frame, columns=columns, show='headings', height=12)
    
    # Configure columns
    tree.heading('Participant', text='Participant')
    tree.heading('Session', text='Session/Set')
    tree.heading('Reps', text='Reps')
    
    if quality_column:
        tree.heading('Clean', text='Clean')
        tree.heading('Uncontrolled', text='Uncontrolled')
        tree.heading('Abrupt', text='Abrupt')
        tree.heading('Duration (s)', text='Duration (s)')
        
        tree.column('Participant', width=80, anchor='center')
        tree.column('Session', width=180, anchor='center')
        tree.column('Reps', width=50, anchor='center')
        tree.column('Clean', width=60, anchor='center')
        tree.column('Uncontrolled', width=85, anchor='center')
        tree.column('Abrupt', width=60, anchor='center')
        tree.column('Duration (s)', width=80, anchor='center')
    else:
        tree.heading('Samples', text='Samples')
        tree.heading('Duration (s)', text='Duration (s)')
        
        tree.column('Participant', width=80, anchor='center')
        tree.column('Session', width=200, anchor='center')
        tree.column('Reps', width=60, anchor='center')
        tree.column('Samples', width=80, anchor='center')
        tree.column('Duration (s)', width=100, anchor='center')
    
    # Add data to tree grouped by participant
    current_participant = None
    for stat in session_stats:
        participant_display = f"P{stat['participant_id']:03d}"
        
        # Add participant grouping (visual separation)
        if current_participant != stat['participant_id']:
            current_participant = stat['participant_id']
            # Insert a separator-like entry with appropriate number of columns
            if quality_column:
                separator_values = (participant_display, "=" * 15, "", "", "", "", "")
            else:
                separator_values = (participant_display, "=" * 20, "", "", "")
            
            tree.insert('', tk.END, values=separator_values, tags=('separator',))
        
        # Insert session data with or without quality distribution
        if quality_column:
            # Include quality distribution columns
            tree.insert('', tk.END, values=(
                participant_display,
                stat['session_display'],
                stat['reps'],
                stat['quality_dist'][0],  # Clean
                stat['quality_dist'][1],  # Uncontrolled
                stat['quality_dist'][2],  # Abrupt
                f"{stat['duration']:.1f}"
            ), tags=(f"p{stat['participant_id']}", f"s{stat['session_file']}"))
        else:
            # Original format without quality distribution
            tree.insert('', tk.END, values=(
                participant_display,
                stat['session_display'],
                stat['reps'],
                f"{stat['samples']:,}",
                f"{stat['duration']:.1f}"
            ), tags=(f"p{stat['participant_id']}", f"s{stat['session_file']}"))
    
    # Configure tags for visual styling
    tree.tag_configure('separator', background='#e0e0e0', foreground='#666')
    
    tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Scrollbar
    scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=tree.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    tree.configure(yscrollcommand=scrollbar.set)
    
    # Selection label
    selected_label = tk.Label(root, text="Selected: None", 
                              font=('Arial', 10, 'bold'), bg='#f5f5f5', fg='#666')
    selected_label.pack(pady=5)
    
    def on_select(event):
        selection_item = tree.selection()
        if selection_item:
            item = tree.item(selection_item[0])
            tags = item.get('tags', [])
            
            # Skip separator rows
            if 'separator' in tags:
                return
            
            # Extract participant and session from tags
            participant_tag = [t for t in tags if t.startswith('p')]
            session_tag = [t for t in tags if t.startswith('s')]
            
            if participant_tag and session_tag:
                participant_id = int(participant_tag[0][1:])  # Remove 'p' prefix
                session_file = session_tag[0][1:]  # Remove 's' prefix
                
                selection[0] = {
                    'participant_id': participant_id,
                    'session_file': session_file,
                    'session_display': item['values'][1]
                }
                
                selected_label.config(text=f"✓ Selected: P{participant_id:03d} - {item['values'][1]}", fg='#4CAF50')
    
    def on_double_click(event):
        selection_item = tree.selection()
        if selection_item:
            item = tree.item(selection_item[0])
            tags = item.get('tags', [])
            
            if 'separator' not in tags:
                participant_tag = [t for t in tags if t.startswith('p')]
                session_tag = [t for t in tags if t.startswith('s')]
                
                if participant_tag and session_tag:
                    participant_id = int(participant_tag[0][1:])
                    session_file = session_tag[0][1:]
                    
                    selection[0] = {
                        'participant_id': participant_id,
                        'session_file': session_file,
                        'session_display': item['values'][1]
                    }
                    root.destroy()
    
    tree.bind('<<TreeviewSelect>>', on_select)
    tree.bind('<Double-1>', on_double_click)
    
    # Button frame
    btn_frame = tk.Frame(root, bg='#f5f5f5')
    btn_frame.pack(pady=15)
    
    def confirm_selection():
        if selection[0] is not None:
            root.destroy()
        else:
            selected_label.config(text="⚠️ Please select a participant session first!", fg='red')
    
    def cancel():
        selection[0] = None
        root.destroy()
    
    confirm_btn = tk.Button(btn_frame, text="✅ Visualize Selected Session", command=confirm_selection,
                           font=('Arial', 11, 'bold'), bg='#4CAF50', fg='white',
                           padx=20, pady=8, cursor='hand2', relief='flat')
    confirm_btn.pack(side=tk.LEFT, padx=5)
    
    exit_btn = tk.Button(btn_frame, text="🚪 Exit Analysis", command=cancel,
                        font=('Arial', 11), bg='#FF5722', fg='white',
                        padx=20, pady=8, cursor='hand2', relief='flat')
    exit_btn.pack(side=tk.LEFT, padx=5)
    
    cancel_btn = tk.Button(btn_frame, text="❌ Cancel", command=cancel,
                          font=('Arial', 11), bg='#f44336', fg='white',
                          padx=20, pady=8, cursor='hand2', relief='flat')
    cancel_btn.pack(side=tk.LEFT, padx=5)
    
    # Center window
    root.update_idletasks()
    x = (root.winfo_screenwidth() - root.winfo_width()) // 2
    y = (root.winfo_screenheight() - root.winfo_height()) // 2
    root.geometry(f"+{x}+{y}")
    
    root.mainloop()
    
    return selection[0]


def visualize_participant_session_reps(df, participant_id, session_file, signal_column='filteredMag', output_path=None):
    """
    Create visualization of all reps for a specific participant session
    
    Parameters:
    - df: DataFrame with sensor data
    - participant_id: ID of the participant to visualize
    - session_file: Specific session file to visualize
    - signal_column: Column to use for visualization (default: 'filteredMag')
    - output_path: Path to save the visualization
    """
    # Filter data for selected participant and session
    if 'participant' in df.columns and 'source_file' in df.columns:
        session_data = df[(df['participant'] == participant_id) & (df['source_file'] == session_file)].copy()
    else:
        print("⚠️ No participant or source_file column found, using all data")
        session_data = df.copy()
    
    if len(session_data) == 0:
        print(f"❌ No data found for participant {participant_id}, session {session_file}")
        return None
    
    # Extract session display name
    session_display = session_file.replace('.csv', '').split('_')[-2:]
    session_name = '_'.join(session_display)
    
    print(f"📊 Visualizing Participant {participant_id}, Session '{session_name}': {len(session_data)} samples")
    
    # Verify signal column exists
    if signal_column not in session_data.columns:
        signal_column = 'filteredMag'
    
    timestamps = session_data['timestamp_ms'].values
    signal = session_data[signal_column].values
    reps = sorted([r for r in session_data['rep'].unique() if r > 0])  # Exclude rep 0
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    fig.suptitle(f'Participant {participant_id} - Session "{session_name}" - Rep Analysis', fontsize=16, fontweight='bold')
    
    # Determine label based on signal column
    signal_label = 'Filtered Magnitude' if 'Mag' in signal_column else signal_column
    
    # Colors for reps (use distinct colors for 1-10 reps)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    # Plot 1: Complete session signal with rep boundaries
    axes[0].plot(timestamps, signal, 'b-', linewidth=1, alpha=0.7, label=signal_label)
    
    for i, rep in enumerate(reps):
        rep_data = session_data[session_data['rep'] == rep]
        if len(rep_data) == 0:
            continue
        
        start_time = rep_data['timestamp_ms'].min()
        end_time = rep_data['timestamp_ms'].max()
        
        # Add colored background for each rep
        color_idx = (rep - 1) % 10  # Map rep 1-10 to color indices 0-9
        axes[0].axvspan(start_time, end_time, alpha=0.3, color=colors[color_idx], 
                       label=f'Rep {rep}')
        
        # Mark rep boundaries
        axes[0].axvline(start_time, color=colors[color_idx], linestyle='--', alpha=0.8)
        
        # Add rep number annotation
        mid_time = (start_time + end_time) / 2
        max_signal = signal.max()
        axes[0].text(mid_time, max_signal * 0.9, f'{rep}', 
                    ha='center', va='center', fontweight='bold', fontsize=12,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[color_idx], alpha=0.8))
    
    axes[0].set_xlabel('Time (ms)')
    axes[0].set_ylabel(signal_label)
    axes[0].set_title(f'Complete Session - {len(reps)} Reps (Session: {session_name})')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)
    
    # Plot 2: Individual reps normalized and overlaid
    axes[1].set_xlabel('Normalized Time (0-1)')
    axes[1].set_ylabel(signal_label)
    axes[1].set_title('All Reps Overlaid (Time-Normalized for Pattern Comparison)')
    axes[1].grid(True, alpha=0.3)
    
    rep_signals = []
    for rep in reps:
        rep_data = session_data[session_data['rep'] == rep]
        if len(rep_data) < 3:  # Skip very short reps
            continue
        
        rep_signal = rep_data[signal_column].values
        rep_timestamps = rep_data['timestamp_ms'].values
        
        # Normalize time to 0-1
        normalized_time = (rep_timestamps - rep_timestamps.min()) / (rep_timestamps.max() - rep_timestamps.min())
        
        # Store for statistics
        rep_signals.append(rep_signal)
        
        # Plot with transparency
        color_idx = (rep - 1) % 10
        axes[1].plot(normalized_time, rep_signal, color=colors[color_idx], 
                    alpha=0.7, linewidth=2.5, label=f'Rep {rep}')
    
    if rep_signals:
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)
    
    # Plot 3: Rep statistics
    if len(reps) > 0:
        rep_stats = []
        for rep in reps:
            rep_data = session_data[session_data['rep'] == rep]
            if len(rep_data) == 0:
                continue
            
            duration = (rep_data['timestamp_ms'].max() - rep_data['timestamp_ms'].min()) / 1000
            max_val = rep_data[signal_column].max()
            min_val = rep_data[signal_column].min()
            amplitude = max_val - min_val
            mean_val = rep_data[signal_column].mean()
            
            rep_stats.append({
                'rep': rep,
                'duration': duration,
                'amplitude': amplitude,
                'max': max_val,
                'min': min_val,
                'mean': mean_val
            })
        
        if rep_stats:
            rep_nums = [s['rep'] for s in rep_stats]
            durations = [s['duration'] for s in rep_stats]
            amplitudes = [s['amplitude'] for s in rep_stats]
            
            # Duration bars
            bars1 = axes[2].bar([r - 0.2 for r in rep_nums], durations, width=0.35, 
                              color='skyblue', alpha=0.8, label='Duration (s)')
            
            # Amplitude bars (secondary y-axis)
            ax2 = axes[2].twinx()
            bars2 = ax2.bar([r + 0.2 for r in rep_nums], amplitudes, width=0.35, 
                           color='orange', alpha=0.8, label='Amplitude')
            
            axes[2].set_xlabel('Rep Number')
            axes[2].set_ylabel('Duration (seconds)', color='blue')
            ax2.set_ylabel('Signal Amplitude', color='orange')
            axes[2].set_title(f'Rep Performance Metrics (Session: {session_name})')
            axes[2].set_xticks(rep_nums)
            
            # Add value labels on bars
            for bar, duration in zip(bars1, durations):
                axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                           f'{duration:.1f}s', ha='center', va='bottom', fontsize=9)
            
            for bar, amplitude in zip(bars2, amplitudes):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                        f'{amplitude:.1f}', ha='center', va='bottom', fontsize=9)
            
            # Combined legend
            lines1, labels1 = axes[2].get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            axes[2].legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save visualization
    if output_path is None:
        session_clean = session_name.replace(' ', '_').replace('-', '_')
        graphs_dir = VIZ_DIR / 'participant_analysis'
        graphs_dir.mkdir(parents=True, exist_ok=True)
        output_path = graphs_dir / f'participant_{participant_id}_session_{session_clean}_reps.png'
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Session visualization saved to '{output_path}'")
    plt.show()
    
    # Print summary statistics
    print(f"\n📈 PARTICIPANT {participant_id} - SESSION '{session_name}' SUMMARY:")
    print("=" * 60)
    print(f"Session file: {session_file}")
    print(f"Total samples: {len(session_data):,}")
    print(f"Total reps: {len(reps)}")
    print(f"Session duration: {(timestamps.max() - timestamps.min()) / 1000:.1f} seconds")
    
    if rep_stats:
        print(f"\n📊 Rep-by-Rep Statistics:")
        for stat in rep_stats:
            print(f"  Rep {stat['rep']}: {stat['duration']:.1f}s, amplitude: {stat['amplitude']:.2f}, "
                  f"mean: {stat['mean']:.2f}")
        
        avg_duration = np.mean([s['duration'] for s in rep_stats])
        avg_amplitude = np.mean([s['amplitude'] for s in rep_stats])
        avg_mean = np.mean([s['mean'] for s in rep_stats])
        
        print(f"\n📋 Session Averages:")
        print(f"  Duration: {avg_duration:.1f}s")
        print(f"  Amplitude: {avg_amplitude:.2f}")
        print(f"  Mean signal: {avg_mean:.2f}")
        
        # Performance consistency analysis
        duration_std = np.std([s['duration'] for s in rep_stats])
        amplitude_std = np.std([s['amplitude'] for s in rep_stats])
        
        print(f"\n🎯 Consistency Metrics:")
        print(f"  Duration variability: ±{duration_std:.2f}s")
        print(f"  Amplitude variability: ±{amplitude_std:.2f}")
        
        consistency_score = 100 - min(100, (duration_std/avg_duration + amplitude_std/avg_amplitude) * 50)
        print(f"  Consistency score: {consistency_score:.1f}% (higher = more consistent)")
    
    return fig
    """
    Create visualization of all reps for a specific participant
    
    Parameters:
    - df: DataFrame with sensor data
    - participant_id: ID of the participant to visualize
    - signal_column: Column to use for visualization (default: 'filteredMag')
    - output_path: Path to save the visualization
    """
    # Filter data for selected participant
    if 'participant' in df.columns:
        participant_data = df[df['participant'] == participant_id].copy()
    else:
        print("⚠️ No participant column found, using all data")
        participant_data = df.copy()
    
    if len(participant_data) == 0:
        print(f"❌ No data found for participant {participant_id}")
        return None
    
    print(f"📊 Visualizing participant {participant_id}: {len(participant_data)} samples")
    
    # Verify signal column exists
    if signal_column not in participant_data.columns:
        signal_column = 'filteredMag'
    
    timestamps = participant_data['timestamp_ms'].values
    signal = participant_data[signal_column].values
    reps = sorted(participant_data['rep'].unique())
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    fig.suptitle(f'Participant {participant_id} - All Reps Visualization', fontsize=16, fontweight='bold')
    
    # Determine label based on signal column
    signal_label = 'Filtered Magnitude' if 'Mag' in signal_column else signal_column
    
    # Colors for reps
    colors = plt.cm.tab20(np.linspace(0, 1, len(reps)))
    
    # Plot 1: Complete signal with rep boundaries
    axes[0].plot(timestamps, signal, 'b-', linewidth=1, alpha=0.7, label=signal_label)
    
    for i, rep in enumerate(reps):
        if rep == 0:  # Skip rep 0 (usually background)
            continue
        rep_data = participant_data[participant_data['rep'] == rep]
        if len(rep_data) == 0:
            continue
        
        start_time = rep_data['timestamp_ms'].min()
        end_time = rep_data['timestamp_ms'].max()
        
        # Add colored background for each rep
        axes[0].axvspan(start_time, end_time, alpha=0.3, color=colors[i % len(colors)], 
                       label=f'Rep {rep}')
        
        # Mark rep boundaries
        axes[0].axvline(start_time, color=colors[i % len(colors)], linestyle='--', alpha=0.8)
        
        # Add rep number annotation
        mid_time = (start_time + end_time) / 2
        max_signal = signal.max()
        axes[0].text(mid_time, max_signal * 0.9, f'R{rep}', 
                    ha='center', va='center', fontweight='bold', 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i % len(colors)], alpha=0.7))
    
    axes[0].set_xlabel('Time (ms)')
    axes[0].set_ylabel(signal_label)
    axes[0].set_title(f'Complete Exercise Session - {len(reps)-1} Reps')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)
    
    # Plot 2: Individual reps normalized and overlaid
    axes[1].set_xlabel('Normalized Time (0-1)')
    axes[1].set_ylabel(signal_label)
    axes[1].set_title('All Reps Overlaid (Time-Normalized)')
    axes[1].grid(True, alpha=0.3)
    
    rep_signals = []
    for i, rep in enumerate(reps):
        if rep == 0:
            continue
        rep_data = participant_data[participant_data['rep'] == rep]
        if len(rep_data) < 3:  # Skip very short reps
            continue
        
        rep_signal = rep_data[signal_column].values
        rep_timestamps = rep_data['timestamp_ms'].values
        
        # Normalize time to 0-1
        normalized_time = (rep_timestamps - rep_timestamps.min()) / (rep_timestamps.max() - rep_timestamps.min())
        
        # Store for statistics
        rep_signals.append(rep_signal)
        
        # Plot with transparency
        axes[1].plot(normalized_time, rep_signal, color=colors[i % len(colors)], 
                    alpha=0.6, linewidth=2, label=f'Rep {rep}')
    
    if rep_signals:
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)
    
    # Plot 3: Rep statistics
    if len(reps) > 1:
        rep_stats = []
        for rep in reps:
            if rep == 0:
                continue
            rep_data = participant_data[participant_data['rep'] == rep]
            if len(rep_data) == 0:
                continue
            
            duration = (rep_data['timestamp_ms'].max() - rep_data['timestamp_ms'].min()) / 1000
            max_val = rep_data[signal_column].max()
            min_val = rep_data[signal_column].min()
            amplitude = max_val - min_val
            
            rep_stats.append({
                'rep': rep,
                'duration': duration,
                'amplitude': amplitude,
                'max': max_val,
                'min': min_val
            })
        
        if rep_stats:
            rep_nums = [s['rep'] for s in rep_stats]
            durations = [s['duration'] for s in rep_stats]
            amplitudes = [s['amplitude'] for s in rep_stats]
            
            # Duration plot
            bars1 = axes[2].bar([r - 0.2 for r in rep_nums], durations, width=0.4, 
                              color='skyblue', alpha=0.7, label='Duration (s)')
            
            # Amplitude plot (secondary y-axis)
            ax2 = axes[2].twinx()
            bars2 = ax2.bar([r + 0.2 for r in rep_nums], amplitudes, width=0.4, 
                           color='orange', alpha=0.7, label='Amplitude')
            
            axes[2].set_xlabel('Rep Number')
            axes[2].set_ylabel('Duration (seconds)', color='blue')
            ax2.set_ylabel('Signal Amplitude', color='orange')
            axes[2].set_title('Rep Duration and Signal Amplitude')
            
            # Add value labels on bars
            for bar, duration in zip(bars1, durations):
                axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                           f'{duration:.1f}s', ha='center', va='bottom', fontsize=9)
            
            for bar, amplitude in zip(bars2, amplitudes):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                        f'{amplitude:.1f}', ha='center', va='bottom', fontsize=9)
            
            # Combined legend
            lines1, labels1 = axes[2].get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            axes[2].legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save visualization
    if output_path is None:
        input_name = f"participant_{participant_id}"
        graphs_dir = VIZ_DIR / 'participant_analysis'
        graphs_dir.mkdir(parents=True, exist_ok=True)
        output_path = graphs_dir / f'{input_name}_all_reps_analysis.png'
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Participant visualization saved to '{output_path}'")
    plt.show()
    
    # Print summary statistics
    print(f"\n📈 PARTICIPANT {participant_id} SUMMARY:")
    print("=" * 50)
    print(f"Total samples: {len(participant_data):,}")
    print(f"Total reps: {len([r for r in reps if r > 0])}")
    print(f"Session duration: {(timestamps.max() - timestamps.min()) / 1000:.1f} seconds")
    
    if rep_stats:
        print(f"\n📊 Rep Statistics:")
        for stat in rep_stats:
            print(f"  Rep {stat['rep']}: {stat['duration']:.1f}s, amplitude: {stat['amplitude']:.2f}")
        
        avg_duration = np.mean([s['duration'] for s in rep_stats])
        avg_amplitude = np.mean([s['amplitude'] for s in rep_stats])
        print(f"\n📋 Averages:")
        print(f"  Duration: {avg_duration:.1f}s")
        print(f"  Amplitude: {avg_amplitude:.2f}")
    
    return fig


def main_with_participant_selection():
    """Main function with participant and session selection workflow"""
    print("=" * 60)
    print("REP VISUALIZATION - PARTICIPANT & SESSION ANALYSIS")
    print("=" * 60)
    print("\nThis script allows you to visualize all reps for a specific participant session.")
    print("You can analyze their exercise patterns, rep consistency, and performance.\n")
    
    # Step 1: Select CSV file (only once)
    print("📂 Step 1: Select CSV file...")
    input_file = select_from_datasets_ui()
    
    if input_file is None:
        print("\n❌ No file selected. Exiting.")
        return
    
    print(f"\n✓ Selected file: {input_file}")
    
    if not os.path.exists(input_file):
        print(f"❌ Error: Input file '{input_file}' not found!")
        return
    
    # Load data (only once)
    df = pd.read_csv(input_file)
    print(f"✓ Loaded {len(df)} samples from dataset")
    
    # Check if required columns exist
    required_columns = ['participant', 'source_file']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"⚠️ Missing columns for session selection: {missing_columns}")
        print("Available columns:", list(df.columns))
        print("Proceeding with basic analysis...")
        # Run once without session selection
        signal_col = 'filteredMag'
        if 'equipment_code' in df.columns:
            equipment_code = df['equipment_code'].iloc[0]
            if equipment_code == 2:  # Weight Stack
                signal_col = 'filteredMag'
        print("⚠️ Cannot create session-specific visualization due to missing columns")
        return
    
    participants = df['participant'].unique()
    participants = sorted([p for p in participants if pd.notna(p)])
    sessions = df['source_file'].nunique()
    print(f"✓ Found {len(participants)} participants with {sessions} total sessions")
    
    # Determine signal column (prefer filteredMag for weight stack)
    signal_col = 'filteredMag'
    if 'equipment_code' in df.columns:
        equipment_code = df['equipment_code'].iloc[0]
        if equipment_code == 2:  # Weight Stack
            signal_col = 'filteredMag'
    
    # Loop for selecting different participants/sessions
    while True:
        try:
            print("\n" + "="*60)
            print("SESSION SELECTOR")
            print("="*60)
            
            # Step 2: Select participant and session
            print("\n👤 Select participant and session...")
            print("   💡 Tip: Close visualization windows to return here automatically")
            print("   💡 Use Ctrl+C anytime to exit")
            
            selection = select_participant_and_session_ui(df)
            
            if selection is None:
                print("\n👋 Exiting participant analysis. Thank you!")
                return
            
            participant_id = selection['participant_id']
            session_file = selection['session_file']
            session_display = selection['session_display']
            
            print(f"\n✓ Selected participant {participant_id}, session '{session_display}'")
            
            # Step 3: Create visualization
            print("\n📊 Creating participant session visualization...")
            print("-" * 40)
            
            # Create visualization
            fig = visualize_participant_session_reps(df, participant_id, session_file, signal_column=signal_col)
            
            if fig is not None:
                print("\n✅ Participant session analysis complete!")
                print("\n🔄 Returning to participant/session selection...")
                print("   Close the visualization window to continue, or press Ctrl+C to exit.")
            else:
                print("\n❌ Failed to create visualization.")
                print("\n🔄 Returning to participant/session selection...")
                print("   You can try selecting another participant/session.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Exiting participant analysis (Ctrl+C pressed). Thank you!")
            return
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            print("🔄 Returning to participant/session selection...")
            continue


def select_mode_ui():
    """
    Show a UI to select between resegmentation mode and participant visualization mode
    """
    root = tk.Tk()
    root.title("🎯 Select Analysis Mode")
    root.geometry("500x300")
    root.configure(bg='#f5f5f5')
    
    selected_mode = [None]
    
    # Header
    header_frame = tk.Frame(root, bg='#2196F3', pady=20)
    header_frame.pack(fill=tk.X)
    
    header = tk.Label(header_frame, text="🎯 Rep Analysis Tool", 
                      font=('Arial', 20, 'bold'), bg='#2196F3', fg='white')
    header.pack()
    
    subtitle = tk.Label(header_frame, text="Choose your analysis mode",
                       font=('Arial', 12), bg='#2196F3', fg='white')
    subtitle.pack()
    
    # Mode selection frame
    mode_frame = tk.Frame(root, bg='#f5f5f5')
    mode_frame.pack(expand=True, fill=tk.BOTH, padx=40, pady=40)
    
    # Resegmentation mode
    reseg_frame = tk.LabelFrame(mode_frame, text="🔄 Resegmentation Mode", 
                               font=('Arial', 12, 'bold'), bg='#f5f5f5', fg='#333', padx=20, pady=15)
    reseg_frame.pack(fill=tk.X, pady=(0, 20))
    
    reseg_desc = tk.Label(reseg_frame, 
                         text="• Fix rep boundaries using valley detection\n"
                              "• Generate before/after comparison\n"
                              "• Save corrected dataset and visualizations",
                         font=('Arial', 10), bg='#f5f5f5', fg='#666', justify='left')
    reseg_desc.pack(anchor='w')
    
    reseg_btn = tk.Button(reseg_frame, text="🔄 Start Resegmentation", 
                         command=lambda: [selected_mode.__setitem__(0, 'resegment'), root.destroy()],
                         font=('Arial', 11, 'bold'), bg='#4CAF50', fg='white',
                         padx=20, pady=8, cursor='hand2', relief='flat')
    reseg_btn.pack(anchor='w', pady=(10, 0))
    
    # Participant visualization mode
    viz_frame = tk.LabelFrame(mode_frame, text="👤 Participant Analysis Mode", 
                             font=('Arial', 12, 'bold'), bg='#f5f5f5', fg='#333', padx=20, pady=15)
    viz_frame.pack(fill=tk.X)
    
    viz_desc = tk.Label(viz_frame, 
                       text="• Select a specific participant to analyze\n"
                            "• Visualize all their reps and patterns\n"
                            "• Compare rep consistency and performance",
                       font=('Arial', 10), bg='#f5f5f5', fg='#666', justify='left')
    viz_desc.pack(anchor='w')
    
    viz_btn = tk.Button(viz_frame, text="👤 Analyze Participant", 
                       command=lambda: [selected_mode.__setitem__(0, 'participant'), root.destroy()],
                       font=('Arial', 11, 'bold'), bg='#FF9800', fg='white',
                       padx=20, pady=8, cursor='hand2', relief='flat')
    viz_btn.pack(anchor='w', pady=(10, 0))
    
    # Cancel button
    cancel_btn = tk.Button(root, text="❌ Cancel", 
                          command=lambda: [selected_mode.__setitem__(0, None), root.destroy()],
                          font=('Arial', 11), bg='#757575', fg='white',
                          padx=25, pady=8, cursor='hand2', relief='flat')
    cancel_btn.pack(pady=15)
    
    # Center window
    root.update_idletasks()
    x = (root.winfo_screenwidth() - root.winfo_width()) // 2
    y = (root.winfo_screenheight() - root.winfo_height()) // 2
    root.geometry(f"+{x}+{y}")
    
    root.mainloop()
    
    return selected_mode[0]


def main():
    """Main function with mode selection"""
    print("=" * 60)
    print("REP ANALYSIS TOOL")
    print("=" * 60)
    print("\nChoose your analysis mode:")
    print("1. 🔄 Resegmentation - Fix rep boundaries using valley detection")
    print("2. 👤 Participant Analysis - Visualize specific participant's reps")
    
    # Show mode selection UI
    print("\n🎯 Opening mode selection dialog...")
    mode = select_mode_ui()
    
    if mode is None:
        print("\n❌ No mode selected. Exiting.")
        return
    
    print(f"\n✓ Selected mode: {mode}")
    
    if mode == 'resegment':
        # Run original resegmentation workflow
        main_resegment()
    elif mode == 'participant':
        # Run participant analysis workflow
        main_with_participant_selection()
    else:
        print("❌ Invalid mode selected.")


def main_resegment():
    """Original resegmentation main function"""
    print("=" * 60)
    print("REP RESEGMENTATION SCRIPT")
    print("=" * 60)
    print("\nThis script corrects rep boundaries by finding actual valley points")
    print("and ensuring each rep starts immediately after the previous rep's valley.\n")
    
    # Show file selection UI
    print("📂 Opening file selection dialog...")
    input_file = select_from_datasets_ui()
    
    if input_file is None:
        print("\n❌ No file selected. Exiting.")
        return None, None
    
    print(f"\n✓ Selected file: {input_file}")
    
    if not os.path.exists(input_file):
        print(f"❌ Error: Input file '{input_file}' not found!")
        return None, None
    
    df_original = pd.read_csv(input_file)
    print(f"✓ Loaded {len(df_original)} samples from '{input_file}'")
    print(f"  Original reps: {sorted(df_original['rep'].unique())}")
    
    # Check equipment type
    if 'equipment_code' in df_original.columns:
        equipment_code = df_original['equipment_code'].iloc[0]
        equipment_names = {0: 'Dumbbell', 1: 'Barbell', 2: 'Weight Stack'}
        equipment_name = equipment_names.get(equipment_code, 'Unknown')
        print(f"  Equipment: {equipment_name} (code: {equipment_code})")
        if equipment_code == 2:
            print(f"  ⚠️  Weight Stack detected - will use MAGNITUDE (filteredMag), NOT raw Z values")
    
    # Perform resegmentation
    print("\n" + "-" * 40)
    print("Performing valley-based resegmentation...")
    print("-" * 40)
    
    df_resegmented, rep_info = resegment_reps(
        df_original, 
        signal_column='filteredMag',
        min_rep_duration_ms=800,
        max_rep_duration_ms=8000
    )
    
    # Print rep info
    print("\n" + "=" * 60)
    print("RESEGMENTED REP INFORMATION")
    print("=" * 60)
    
    for info in rep_info:
        print(f"\nRep {info['rep']}:")
        print(f"  Time: {info['start_time_ms']:.0f}ms - {info['end_time_ms']:.0f}ms")
        print(f"  Duration: {info['duration_ms']:.0f}ms ({info['duration_ms']/1000:.2f}s)")
        print(f"  Peak at: {info['peak_time_ms']:.0f}ms (value: {info['peak_value']:.2f})")
        print(f"  Valley value: {info['valley_value']:.2f}")
        print(f"  Amplitude: {info['amplitude']:.2f}")
    
    # Determine output paths based on input file location
    input_dir = os.path.dirname(input_file)
    input_basename = os.path.basename(input_file)
    input_name = os.path.splitext(input_basename)[0]
    
    # Save resegmented data in same folder as input
    output_file = os.path.join(input_dir, f'{input_name}_resegmented.csv')
    
    df_output = df_resegmented.copy()
    df_output.to_csv(output_file, index=False)
    print(f"\n✓ Resegmented data saved to '{output_file}'")
    
    # Also save rep info
    rep_info_file = os.path.join(input_dir, f'{input_name}_rep_boundaries.csv')
    rep_info_df = pd.DataFrame(rep_info)
    rep_info_df.to_csv(rep_info_file, index=False)
    print(f"✓ Rep boundary info saved to '{rep_info_file}'")
    
    # Create visualization
    print("\n" + "-" * 40)
    print("Creating visualization...")
    print("-" * 40)
    
    # Determine which signal column was used
    signal_col = 'filteredMag'  # Default to magnitude (especially for weight stack)
    if 'equipment_code' in df_resegmented.columns:
        if df_resegmented['equipment_code'].iloc[0] == 2:
            signal_col = 'filteredMag'  # Force magnitude for weight stack
    
    # Create graphs folder in the same directory as input file
    graphs_dir = os.path.join(input_dir, 'graphs')
    os.makedirs(graphs_dir, exist_ok=True)
    print(f"📊 Graphs will be saved to: '{graphs_dir}'")
    
    viz_output_path = os.path.join(graphs_dir, f'{input_name}_resegmentation_comparison.png')
    visualize_resegmentation(df_original, df_resegmented, rep_info, 
                            signal_column=signal_col, output_path=viz_output_path)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Original reps: {len(df_original['rep'].unique())}")
    print(f"Resegmented reps: {len(rep_info)}")
    print(f"Total samples: {len(df_original)}")
    
    # Check for continuity
    print("\n📊 Continuity Analysis:")
    print("  Checking that rep boundaries are continuous (no time gaps)")
    
    all_continuous = True
    total_gap_ms = 0
    for i in range(len(rep_info) - 1):
        current_end_time = rep_info[i]['end_time_ms']
        next_start_time = rep_info[i+1]['start_time_ms']
        time_gap = next_start_time - current_end_time
        
        current_end_idx = rep_info[i]['end_idx']
        next_start_idx = rep_info[i+1]['start_idx']
        sample_gap = next_start_idx - current_end_idx
        
        is_continuous = (sample_gap == 1)
        status = "✓" if is_continuous else "❌"
        if not is_continuous:
            all_continuous = False
        total_gap_ms += time_gap
        
        print(f"  {status} Rep {rep_info[i]['rep']} → Rep {rep_info[i+1]['rep']}: "
              f"sample gap={sample_gap}, time gap={time_gap:.0f}ms")
    
    if all_continuous:
        print(f"\n  ✅ All reps are sample-continuous!")
        print(f"  ℹ️  Time between consecutive samples: ~{total_gap_ms/(len(rep_info)-1):.0f}ms (normal sampling interval)")
    else:
        print(f"\n  ⚠️ Some gaps detected - review segmentation parameters")
    
    return df_resegmented, rep_info


if __name__ == "__main__":
    main()
