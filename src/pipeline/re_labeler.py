"""
AppLift ML Training - Rep Labeler v2
=====================================
A visual tool to view exercise reps, relabel quality codes, and edit data.

Features:
- Auto-detect equipment/exercise from folder path and filename
- Dynamic quality labels based on exercise type
- Visualize individual reps with sensor signals
- Select and delete portions of data from graph
- Change quality labels with dropdown selection
- Save modified data back to CSV

LABEL DEFINITIONS:
==================

DUMBBELL (Equipment Code: 0)
├── Concentration Curls (Exercise Code: 0)
│   ├── 0: Clean
│   ├── 1: Uncontrolled
│   └── 2: Abrupt
└── Overhead Extension (Exercise Code: 1)
    ├── 0: Clean
    ├── 1: Uncontrolled
    └── 2: Abrupt

BARBELL (Equipment Code: 1)
├── Bench Press (Exercise Code: 2)
│   ├── 0: Clean
│   ├── 1: Uncontrolled
│   └── 2: Inclination Asymmetry
└── Back Squat (Exercise Code: 3)
    ├── 0: Clean
    ├── 1: Uncontrolled
    └── 2: Inclination Asymmetry

WEIGHT STACK (Equipment Code: 2)
├── Lateral Pulldown (Exercise Code: 4)
│   ├── 0: Clean
│   ├── 1: Pulling Too Fast
│   └── 2: Releasing Too Fast
└── Seated Leg Extension (Exercise Code: 5)
    ├── 0: Clean
    ├── 1: Pulling Too Fast
    └── 2: Releasing Too Fast

Author: AppLift ML Training Pipeline
"""

import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from datetime import datetime
import copy
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector
import warnings
import re
import os

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION - COMPLETE LABEL DEFINITIONS
# =============================================================================

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR

# Equipment codes and names
EQUIPMENT_CODES = {
    0: 'Dumbbell',
    1: 'Barbell',
    2: 'Weight Stack'
}

EQUIPMENT_NAME_TO_CODE = {
    'dumbbell': 0,
    'barbell': 1,
    'weight_stack': 2,
    'weight stack': 2,
    'weightstack': 2
}

# Exercise codes and names
EXERCISE_CODES = {
    0: 'Concentration Curls',
    1: 'Overhead Extension',
    2: 'Bench Press',
    3: 'Back Squat',
    4: 'Lateral Pulldown',
    5: 'Seated Leg Extension'
}

EXERCISE_NAME_TO_CODE = {
    'concentration_curls': 0,
    'concentration curls': 0,
    'concentrationcurls': 0,
    'overhead_extension': 1,
    'overhead extension': 1,
    'overheadextension': 1,
    'bench_press': 2,
    'bench press': 2,
    'benchpress': 2,
    'back_squat': 3,
    'back squat': 3,
    'backsquat': 3,
    'lateral_pulldown': 4,
    'lateral pulldown': 4,
    'lateralpulldown': 4,
    'lat_pulldown': 4,
    'seated_leg_extension': 5,
    'seated leg extension': 5,
    'seatedlegextension': 5,
    'leg_extension': 5
}

# Quality labels per exercise type
# Key: exercise_code, Value: {quality_code: label_name}
QUALITY_LABELS_BY_EXERCISE = {
    # Dumbbell exercises
    0: {0: 'Clean', 1: 'Uncontrolled', 2: 'Abrupt'},  # Concentration Curls
    1: {0: 'Clean', 1: 'Uncontrolled', 2: 'Abrupt'},  # Overhead Extension
    # Barbell exercises
    2: {0: 'Clean', 1: 'Uncontrolled', 2: 'Inclination Asymmetry'},  # Bench Press
    3: {0: 'Clean', 1: 'Uncontrolled', 2: 'Inclination Asymmetry'},  # Back Squat
    # Weight Stack exercises
    4: {0: 'Clean', 1: 'Pulling Too Fast', 2: 'Releasing Too Fast'},  # Lateral Pulldown
    5: {0: 'Clean', 1: 'Pulling Too Fast', 2: 'Releasing Too Fast'},  # Seated Leg Extension
}

# Default quality labels (fallback)
DEFAULT_QUALITY_LABELS = {
    0: 'Clean',
    1: 'Error Type 1',
    2: 'Error Type 2'
}

# Colors for quality labels
QUALITY_COLORS = {
    0: '#4CAF50',  # Green for Clean
    1: '#FF9800',  # Orange for Error Type 1
    2: '#f44336'   # Red for Error Type 2
}

# Signal columns for visualization
SIGNAL_COLUMNS = {
    'Filtered Magnitude': 'filteredMag',
    'Filtered X': 'filteredX',
    'Filtered Y': 'filteredY',
    'Filtered Z': 'filteredZ',
    'Accel Magnitude': 'accelMag',
    'Accel X': 'accelX',
    'Accel Y': 'accelY',
    'Accel Z': 'accelZ',
    'Gyro X': 'gyroX',
    'Gyro Y': 'gyroY',
    'Gyro Z': 'gyroZ',
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def detect_equipment_exercise_from_path(file_path):
    """
    Auto-detect equipment and exercise from file path and filename.
    
    Expected folder structures:
    - .../Dumbbell/Concentration_Curls/Clean/filename.csv
    - .../Weight_Stack/Lateral_Pulldown/Clean/filename.csv
    - .../Data/Weight_Stack/Lateral_Pulldown/Clean/P001_Weight_Stack_Lateral_Pulldown_Clean_0_06.csv
    
    Returns:
    - equipment_code: int or None
    - exercise_code: int or None
    - quality_from_path: int or None (detected from folder name like 'Clean', 'Uncontrolled')
    - detected_info: dict with detection details
    """
    path = Path(file_path)
    path_str = str(path).lower().replace('\\', '/').replace('_', ' ')
    filename = path.stem.lower().replace('_', ' ')
    
    detected_info = {
        'equipment_name': None,
        'exercise_name': None,
        'quality_from_folder': None,
        'participant': None,
        'filename': path.name,
        'full_path': str(path)
    }
    
    equipment_code = None
    exercise_code = None
    quality_from_path = None
    
    # Detect equipment from path
    for eq_name, eq_code in EQUIPMENT_NAME_TO_CODE.items():
        if eq_name in path_str or eq_name in filename:
            equipment_code = eq_code
            detected_info['equipment_name'] = EQUIPMENT_CODES[eq_code]
            break
    
    # Detect exercise from path
    for ex_name, ex_code in EXERCISE_NAME_TO_CODE.items():
        if ex_name in path_str or ex_name in filename:
            exercise_code = ex_code
            detected_info['exercise_name'] = EXERCISE_CODES[ex_code]
            break
    
    # Detect quality from folder name
    path_parts = [p.lower() for p in path.parts]
    quality_folders = {
        'clean': 0,
        'uncontrolled': 1,
        'uncontrolled movement': 1,
        'abrupt': 2,
        'abrupt initiation': 2,
        'inclination asymmetry': 2,
        'pulling too fast': 1,
        'releasing too fast': 2
    }
    
    for part in path_parts:
        part_clean = part.replace('_', ' ')
        for folder_name, q_code in quality_folders.items():
            if folder_name in part_clean:
                quality_from_path = q_code
                detected_info['quality_from_folder'] = folder_name.title()
                break
    
    # Try to detect participant from filename (P001, P002, etc.)
    participant_match = re.search(r'p(\d+)', filename)
    if participant_match:
        detected_info['participant'] = int(participant_match.group(1))
    
    return equipment_code, exercise_code, quality_from_path, detected_info


def get_quality_labels_for_exercise(exercise_code):
    """Get the appropriate quality labels for a given exercise"""
    if exercise_code is not None and exercise_code in QUALITY_LABELS_BY_EXERCISE:
        return QUALITY_LABELS_BY_EXERCISE[exercise_code]
    return DEFAULT_QUALITY_LABELS


def get_quality_column_name(df):
    """
    Determine which column contains quality/target values.
    Returns the column name ('quality_code' or 'target') or None if neither exists.
    """
    if 'quality_code' in df.columns:
        return 'quality_code'
    elif 'target' in df.columns:
        return 'target'
    return None


def get_quality_value(df_or_series, default=0):
    """
    Get quality value from a dataframe or series, checking both 'quality_code' and 'target' columns.
    """
    if isinstance(df_or_series, pd.DataFrame):
        if 'quality_code' in df_or_series.columns:
            return int(df_or_series['quality_code'].iloc[0])
        elif 'target' in df_or_series.columns:
            return int(df_or_series['target'].iloc[0])
    elif isinstance(df_or_series, pd.Series):
        if 'quality_code' in df_or_series.index:
            return int(df_or_series['quality_code'])
        elif 'target' in df_or_series.index:
            return int(df_or_series['target'])
    return default


# =============================================================================
# MAIN APPLICATION CLASS
# =============================================================================

class RepLabelerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🏋️ AppLift Rep Labeler v2 - Visual Quality Assessment")
        self.root.geometry("1500x950")
        self.root.configure(bg='#f5f5f5')
        
        # Data storage
        self.df = None
        self.original_df = None
        self.current_file = None
        self.reps_data = {}
        self.sessions_data = {}  # Track sessions for hierarchical display
        self.selected_rep = None
        self.selected_session = None  # Currently selected session filter
        self.changes_made = {}
        self.deleted_ranges = {}  # Track deleted ranges per rep
        
        # Auto-detected info
        self.equipment_code = None
        self.exercise_code = None
        self.detected_info = {}
        self.current_quality_labels = DEFAULT_QUALITY_LABELS
        
        # Selected signals for plotting
        self.selected_signals = ['filteredMag', 'accelMag']
        
        # Selection state for deletion
        self.selection_start = None
        self.selection_end = None
        self.span_selector = None
        
        # Rep boundaries for full dataset view click detection
        self.rep_boundaries = []
        
        # Rep boundary editing - track splits and renumbering
        self.rep_splits = {}  # {rep_id: split_timestamp_ms} - splits to apply
        self.rep_renumber_map = {}  # {old_rep_id: new_rep_num} - renumbering map
        
        # Single-step undo stack (stores snapshots before each edit action)
        self.undo_stack = []
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the main UI layout"""
        
        # Main container
        main_container = tk.Frame(self.root, bg='#f5f5f5')
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # =====================================================================
        # TOP FRAME - File Controls & Detection Info
        # =====================================================================
        top_frame = tk.Frame(main_container, bg='#f5f5f5')
        top_frame.pack(fill=tk.X, pady=(0, 10))
        
        # File controls
        file_frame = tk.LabelFrame(top_frame, text="📁 File Controls", 
                                   font=('Arial', 11, 'bold'), bg='#f5f5f5')
        file_frame.pack(side=tk.LEFT, padx=(0, 10), fill=tk.Y)
        
        tk.Button(file_frame, text="📂 Load CSV", command=self.load_csv,
                 font=('Arial', 10, 'bold'), bg='#2196F3', fg='white',
                 padx=15, pady=5, cursor='hand2').pack(side=tk.LEFT, padx=5, pady=5)
        
        tk.Button(file_frame, text="💾 Save Changes", command=self.save_changes,
                 font=('Arial', 10, 'bold'), bg='#4CAF50', fg='white',
                 padx=15, pady=5, cursor='hand2').pack(side=tk.LEFT, padx=5, pady=5)
        
        tk.Button(file_frame, text="💾 Save As...", command=self.save_as,
                 font=('Arial', 10, 'bold'), bg='#FF9800', fg='white',
                 padx=15, pady=5, cursor='hand2').pack(side=tk.LEFT, padx=5, pady=5)
        
        tk.Button(file_frame, text="↩️ Undo Last", command=self.undo_last_action,
                 font=('Arial', 10), bg='#9E9E9E', fg='white',
                 padx=15, pady=5, cursor='hand2').pack(side=tk.LEFT, padx=5, pady=5)
        
        # Detection info frame
        detection_frame = tk.LabelFrame(top_frame, text="🔍 Auto-Detected Info (from folder/filename)", 
                                       font=('Arial', 11, 'bold'), bg='#e3f2fd')
        detection_frame.pack(side=tk.LEFT, padx=10, fill=tk.BOTH, expand=True)
        
        self.detection_var = tk.StringVar(value="Load a file to auto-detect equipment/exercise")
        detection_label = tk.Label(detection_frame, textvariable=self.detection_var,
                                  font=('Consolas', 10), bg='#e3f2fd', fg='#1565c0',
                                  justify='left', anchor='w')
        detection_label.pack(fill=tk.X, padx=10, pady=5)
        
        # =====================================================================
        # QUALITY LABELS FRAME (Dynamic based on exercise)
        # =====================================================================
        self.labels_frame = tk.LabelFrame(main_container, text="📋 Quality Labels for Current Exercise", 
                                         font=('Arial', 11, 'bold'), bg='#f5f5f5')
        self.labels_frame.pack(fill=tk.X, pady=(0, 10))
        
        labels_row = tk.Frame(self.labels_frame, bg='#f5f5f5')
        labels_row.pack(fill=tk.X, padx=10, pady=5)
        
        self.labels_container = tk.Frame(labels_row, bg='#f5f5f5')
        self.labels_container.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Top boundary tools (moved from bottom panel)
        top_boundary_frame = tk.Frame(labels_row, bg='#e8f5e9', relief=tk.GROOVE, borderwidth=1)
        top_boundary_frame.pack(side=tk.RIGHT, padx=(10, 0))
        
        tk.Label(top_boundary_frame, text="Boundary Tools:", font=('Arial', 9, 'bold'),
                bg='#e8f5e9', fg='#2e7d32').pack(side=tk.LEFT, padx=(8, 6))
        
        tk.Button(top_boundary_frame, text="Split",
                 command=self.split_rep_at_selection,
                 font=('Arial', 8, 'bold'), bg='#7B1FA2', fg='white',
                 padx=7, pady=2, cursor='hand2').pack(side=tk.LEFT, padx=2, pady=3)
        
        tk.Button(top_boundary_frame, text="Merge",
                 command=self.merge_selected_rep,
                 font=('Arial', 8, 'bold'), bg='#455A64', fg='white',
                 padx=7, pady=2, cursor='hand2').pack(side=tk.LEFT, padx=2, pady=3)
        
        tk.Label(top_boundary_frame, text="Rep#:", font=('Arial', 8), bg='#e8f5e9').pack(side=tk.LEFT, padx=(8, 3))
        
        self.assign_rep_var = tk.StringVar(value="")
        self.assign_rep_entry = tk.Entry(top_boundary_frame, textvariable=self.assign_rep_var,
                                        width=5, font=('Arial', 9))
        self.assign_rep_entry.pack(side=tk.LEFT, padx=2)
        
        tk.Button(top_boundary_frame, text="Assign",
                 command=self.assign_selection_to_rep,
                 font=('Arial', 8), bg='#00796B', fg='white',
                 padx=7, pady=2, cursor='hand2').pack(side=tk.LEFT, padx=(2, 8), pady=3)
        
        self.update_quality_labels_display()
        
        # =====================================================================
        # MIDDLE FRAME - Rep List + Visualization
        # =====================================================================
        middle_frame = tk.Frame(main_container, bg='#f5f5f5')
        middle_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Rep list
        left_panel = tk.Frame(middle_frame, bg='#f5f5f5', width=350)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)
        
        # Rep list header
        rep_header = tk.LabelFrame(left_panel, text="📊 Reps Overview (Ctrl+Click for multi-select)", 
                                  font=('Arial', 11, 'bold'), bg='#f5f5f5')
        rep_header.pack(fill=tk.X, pady=(0, 5))
        
        # Session selector (Participant + Set)
        session_frame = tk.Frame(rep_header, bg='#f5f5f5')
        session_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Label(session_frame, text="Session:", font=('Arial', 9, 'bold'), bg='#f5f5f5').pack(side=tk.LEFT)
        
        self.session_var = tk.StringVar(value="All Sessions")
        self.session_combo = ttk.Combobox(session_frame, textvariable=self.session_var,
                                         values=["All Sessions"], state='readonly', width=30)
        self.session_combo.pack(side=tk.LEFT, padx=5)
        self.session_combo.bind('<<ComboboxSelected>>', self.on_session_select)
        
        # Filter controls
        filter_frame = tk.Frame(rep_header, bg='#f5f5f5')
        filter_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Label(filter_frame, text="Filter:", font=('Arial', 9), bg='#f5f5f5').pack(side=tk.LEFT)
        
        self.filter_var = tk.StringVar(value="All")
        filter_options = ["All", "Clean", "Type 1 Error", "Type 2 Error", "Changed", "Has Deletions"]
        self.filter_combo = ttk.Combobox(filter_frame, textvariable=self.filter_var,
                                        values=filter_options, state='readonly', width=14)
        self.filter_combo.pack(side=tk.LEFT, padx=5)
        self.filter_combo.bind('<<ComboboxSelected>>', self.apply_filter)
        
        # Summary label
        self.summary_var = tk.StringVar(value="")
        summary_label = tk.Label(rep_header, textvariable=self.summary_var,
                                font=('Arial', 9), bg='#f5f5f5', fg='#666')
        summary_label.pack(pady=2)
        
        # Rep listbox with scrollbar
        list_frame = tk.Frame(left_panel, bg='#f5f5f5')
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        self.rep_listbox = tk.Listbox(list_frame, font=('Consolas', 9),
                                     selectmode=tk.EXTENDED, bg='white',
                                     activestyle='dotbox', height=20)
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, 
                                 command=self.rep_listbox.yview)
        self.rep_listbox.configure(yscrollcommand=scrollbar.set)
        
        self.rep_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.rep_listbox.bind('<<ListboxSelect>>', self.on_rep_select)
        
        # Right panel - Visualization + Labeling
        right_panel = tk.Frame(middle_frame, bg='#f5f5f5')
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Visualization frame
        viz_frame = tk.LabelFrame(right_panel, text="📈 Visualization | Multi-select (Ctrl+Click): Bulk label change | Single select: View/Edit | Drag in Single View: Delete region", 
                                 font=('Arial', 11, 'bold'), bg='#f5f5f5')
        viz_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # View mode and signal selection
        view_signal_frame = tk.Frame(viz_frame, bg='#f5f5f5')
        view_signal_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # View mode toggle (NEW)
        tk.Label(view_signal_frame, text="View:", 
                font=('Arial', 9, 'bold'), bg='#f5f5f5').pack(side=tk.LEFT)
        
        self.view_mode_var = tk.StringVar(value="Full Dataset")
        view_modes = ["Full Dataset", "Single Rep"]
        for mode in view_modes:
            rb = tk.Radiobutton(view_signal_frame, text=mode, variable=self.view_mode_var,
                               value=mode, bg='#f5f5f5', font=('Arial', 9),
                               command=self.toggle_view_mode)
            rb.pack(side=tk.LEFT, padx=3)
        
        ttk.Separator(view_signal_frame, orient='vertical').pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        # Signal selection
        tk.Label(view_signal_frame, text="Signals:", 
                font=('Arial', 9, 'bold'), bg='#f5f5f5').pack(side=tk.LEFT)
        
        self.signal_vars = {}
        for display_name, col_name in SIGNAL_COLUMNS.items():
            var = tk.BooleanVar(value=(col_name in self.selected_signals))
            self.signal_vars[col_name] = var
            cb = tk.Checkbutton(view_signal_frame, text=display_name, variable=var,
                               bg='#f5f5f5', font=('Arial', 8),
                               command=self.update_signal_selection)
            cb.pack(side=tk.LEFT, padx=3)
        
        # Matplotlib figure
        self.fig = Figure(figsize=(10, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Connect click event for full dataset view
        self.canvas.mpl_connect('button_press_event', self.on_canvas_click)
        
        # Toolbar frame with delete button
        toolbar_container = tk.Frame(viz_frame, bg='#f5f5f5')
        toolbar_container.pack(fill=tk.X)
        
        toolbar_frame = tk.Frame(toolbar_container, bg='#f5f5f5')
        toolbar_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        toolbar.update()
        
        # Selection/Delete controls
        delete_frame = tk.Frame(toolbar_container, bg='#f5f5f5')
        delete_frame.pack(side=tk.RIGHT, padx=10)
        
        self.selection_var = tk.StringVar(value="No selection")
        selection_label = tk.Label(delete_frame, textvariable=self.selection_var,
                                  font=('Arial', 9), bg='#fff3e0', fg='#e65100',
                                  padx=10, pady=2)
        selection_label.pack(side=tk.LEFT, padx=5)
        
        tk.Button(delete_frame, text="🗑️ Delete Selected", command=self.delete_selection,
                 font=('Arial', 9, 'bold'), bg='#f44336', fg='white',
                 padx=10, pady=2, cursor='hand2').pack(side=tk.LEFT, padx=2)
        
        tk.Button(delete_frame, text="↩️ Undo Rep Deletions", command=self.undo_rep_deletions,
                 font=('Arial', 9), bg='#9E9E9E', fg='white',
                 padx=10, pady=2, cursor='hand2').pack(side=tk.LEFT, padx=2)
        
        tk.Button(delete_frame, text="🔄 Clear Selection", command=self.clear_selection,
                 font=('Arial', 9), bg='#607D8B', fg='white',
                 padx=10, pady=2, cursor='hand2').pack(side=tk.LEFT, padx=2)
        
        # =====================================================================
        # BOTTOM FRAME - Quality Relabeler (TOP PRIORITY)
        # =====================================================================
        bottom_frame = tk.Frame(right_panel, bg='#f5f5f5')
        bottom_frame.pack(fill=tk.X)
        
        # Row 1: Quality/Target Relabeling - NOW AT THE TOP
        quality_frame = tk.Frame(bottom_frame, bg='#fff3cd', relief=tk.RIDGE, borderwidth=2)
        quality_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Title for quality section
        tk.Label(quality_frame, text="🎯 QUALITY RELABELER", 
                font=('Arial', 11, 'bold'), bg='#fff3cd', fg='#856404').pack(pady=(5, 5))
        
        # Current label display
        current_label_frame = tk.Frame(quality_frame, bg='#fff3cd')
        current_label_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(current_label_frame, text="Current Label:", font=('Arial', 10, 'bold'),
                bg='#fff3cd').pack(side=tk.LEFT, padx=(0, 5))
        
        self.current_label_var = tk.StringVar(value="(Select a rep)")
        self.current_label_display = tk.Label(current_label_frame, textvariable=self.current_label_var,
                                             font=('Arial', 12, 'bold'), bg='#fff3cd',
                                             fg='#333', relief=tk.SUNKEN, padx=10, pady=3)
        self.current_label_display.pack(side=tk.LEFT, padx=(0, 10))
        
        # Rep info (compact) - moved here
        self.rep_info_var = tk.StringVar(value="")
        tk.Label(current_label_frame, textvariable=self.rep_info_var,
                font=('Arial', 9, 'italic'), bg='#fff3cd', fg='#666').pack(side=tk.LEFT, padx=10)
        
        # Controls row: Column 1 = quality labels, Column 2 = equipment/exercise relabeler
        controls_frame = tk.Frame(quality_frame, bg='#fff3cd')
        controls_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        controls_frame.grid_columnconfigure(0, weight=1)
        controls_frame.grid_columnconfigure(1, weight=0)
        
        # Column 1 (left): quality selection buttons
        buttons_frame = tk.Frame(controls_frame, bg='#fff3cd')
        buttons_frame.grid(row=0, column=0, sticky='w')
        
        tk.Label(buttons_frame, text="Change Quality to:", font=('Arial', 11, 'bold'),
                bg='#fff3cd', fg='#856404').pack(side=tk.LEFT, padx=(0, 10))
        
        self.quality_btn_frame = tk.Frame(buttons_frame, bg='#fff3cd')
        self.quality_btn_frame.pack(side=tk.LEFT)
        
        self.quality_buttons = {}
        self.create_quality_buttons()
        
        # Column 2 (right): equipment and exercise relabeler
        meta_row = tk.Frame(controls_frame, bg='#fff3cd')
        meta_row.grid(row=0, column=1, sticky='e', padx=(15, 0))
        
        tk.Label(meta_row, text="Equipment:", font=('Arial', 9),
                bg='#fff3cd').pack(side=tk.LEFT, padx=(0, 3))
        
        self.equipment_change_var = tk.StringVar()
        self.equipment_dropdown = ttk.Combobox(meta_row, textvariable=self.equipment_change_var,
                                               values=[f"{k}: {v}" for k, v in EQUIPMENT_CODES.items()],
                                               width=13, state='readonly')
        self.equipment_dropdown.pack(side=tk.LEFT, padx=(0, 8))
        self.equipment_dropdown.bind('<<ComboboxSelected>>', self.on_equipment_change_selected)
        
        tk.Label(meta_row, text="Exercise:", font=('Arial', 9),
                bg='#fff3cd').pack(side=tk.LEFT, padx=(0, 3))
        
        self.exercise_change_var = tk.StringVar()
        self.exercise_dropdown = ttk.Combobox(meta_row, textvariable=self.exercise_change_var,
                                              values=[f"{k}: {v}" for k, v in EXERCISE_CODES.items()],
                                              width=16, state='readonly')
        self.exercise_dropdown.pack(side=tk.LEFT, padx=(0, 8))
        
        tk.Button(meta_row, text="Apply", 
                 command=self.apply_metadata_change,
                 font=('Arial', 9), bg='#2196F3', fg='white',
                 padx=6, pady=2, cursor='hand2').pack(side=tk.LEFT)
        
        self.split_info_var = tk.StringVar(value="")
        tk.Label(quality_frame, textvariable=self.split_info_var,
                font=('Arial', 9, 'italic'), bg='#fff3cd', fg='#1b5e20',
                anchor='w').pack(fill=tk.X, padx=12, pady=(0, 6))
        
        # Changes counter
        self.changes_var = tk.StringVar(value="No changes made")
        changes_label = tk.Label(bottom_frame, textvariable=self.changes_var,
                                font=('Arial', 10, 'italic'), bg='#f5f5f5', fg='#666')
        changes_label.pack(pady=5)
    
    def on_equipment_change_selected(self, event=None):
        """Update exercise dropdown based on selected equipment"""
        equipment_str = self.equipment_change_var.get()
        if not equipment_str:
            return
        
        equipment_code = int(equipment_str.split(":")[0])
        
        # Filter exercises for this equipment
        if equipment_code == 0:  # Dumbbell
            exercises = {0: 'Concentration Curls', 1: 'Overhead Extension'}
        elif equipment_code == 1:  # Barbell
            exercises = {2: 'Bench Press', 3: 'Back Squat'}
        elif equipment_code == 2:  # Weight Stack
            exercises = {4: 'Lateral Pulldown', 5: 'Seated Leg Extension'}
        else:
            exercises = EXERCISE_CODES
        
        self.exercise_dropdown['values'] = [f"{k}: {v}" for k, v in exercises.items()]
        
        # Auto-select first exercise
        if exercises:
            first_key = list(exercises.keys())[0]
            self.exercise_change_var.set(f"{first_key}: {exercises[first_key]}")
    
    def apply_metadata_change(self):
        """Apply equipment and exercise changes to selected reps"""
        selections = self.rep_listbox.curselection()
        if not selections:
            messagebox.showwarning("No Selection", "Please select one or more reps first")
            return
        
        equipment_str = self.equipment_change_var.get()
        exercise_str = self.exercise_change_var.get()
        
        if not equipment_str or not exercise_str:
            messagebox.showwarning("No Selection", "Please select equipment and exercise")
            return
        
        new_equipment = int(equipment_str.split(":")[0])
        new_exercise = int(exercise_str.split(":")[0])
        
        # Extract rep IDs
        selected_rep_ids = []
        for idx in selections:
            display_text = self.rep_listbox.get(idx)
            rep_id = display_text.split(" | ")[0].strip()
            if rep_id in self.reps_data:
                selected_rep_ids.append(rep_id)
        
        if not selected_rep_ids:
            messagebox.showwarning("No Valid Selection", "No valid reps selected")
            return
        
        # Confirm
        equip_name = EQUIPMENT_CODES.get(new_equipment, 'Unknown')
        exer_name = EXERCISE_CODES.get(new_exercise, 'Unknown')
        
        if not messagebox.askyesno("Confirm Metadata Change",
                                   f"Change {len(selected_rep_ids)} rep(s) to:\n\n"
                                   f"Equipment: {equip_name}\n"
                                   f"Exercise: {exer_name}\n\n"
                                   f"This will update the equipment_code and exercise_code columns."):
            return
        
        will_change = any(
            self.reps_data[rep_id].get('equipment_code', self.equipment_code) != new_equipment or
            self.reps_data[rep_id].get('exercise_code', self.exercise_code) != new_exercise
            for rep_id in selected_rep_ids
        )
        if not will_change:
            messagebox.showinfo("No Change", "Selected reps already have this equipment/exercise")
            return
        
        self._push_undo_state("Metadata change")
        
        # Apply changes to dataframe
        for rep_id in selected_rep_ids:
            rep_info = self.reps_data[rep_id]
            
            # Update in dataframe
            participant = rep_info['participant']
            rep_num = rep_info['rep']
            
            mask = (self.df['participant'] == participant) & (self.df['rep'] == rep_num)
            self.df.loc[mask, 'equipment_code'] = new_equipment
            self.df.loc[mask, 'exercise_code'] = new_exercise
            
            # Update in reps_data
            rep_info['equipment_code'] = new_equipment
            rep_info['exercise_code'] = new_exercise
            
            # Mark as changed (if not already marked for label change)
            if rep_id not in self.changes_made:
                self.changes_made[rep_id] = rep_info['quality_code']  # Mark changed
        
        # Update UI
        self.update_changes_counter()
        
        # Refresh visualization
        if len(selected_rep_ids) == 1:
            self.visualize_rep(selected_rep_ids[0])
            self.update_rep_info(selected_rep_ids[0])
        else:
            self.visualize_full_dataset()
            self.update_multi_rep_info(selected_rep_ids)
        
        messagebox.showinfo("Success", 
                           f"Updated {len(selected_rep_ids)} rep(s) to:\n"
                           f"Equipment: {equip_name}\n"
                           f"Exercise: {exer_name}")

    def update_quality_labels_display(self):
        """Update the quality labels legend display based on current exercise"""
        # Clear existing labels
        for widget in self.labels_container.winfo_children():
            widget.destroy()
        
        # Add labels for current exercise
        for code, label in self.current_quality_labels.items():
            color = QUALITY_COLORS[code]
            lbl = tk.Label(self.labels_container, text=f"● {code}: {label}",
                          font=('Arial', 11, 'bold'), bg='#f5f5f5', fg=color)
            lbl.pack(side=tk.LEFT, padx=15, pady=5)
    
    def create_quality_buttons(self):
        """Create/update quality selection buttons based on current exercise"""
        # Clear existing buttons
        for widget in self.quality_btn_frame.winfo_children():
            widget.destroy()
        
        self.quality_buttons = {}
        
        for code, label in self.current_quality_labels.items():
            color = QUALITY_COLORS[code]
            btn = tk.Button(self.quality_btn_frame, text=f"{code}: {label}",
                          font=('Arial', 11, 'bold'), bg=color, fg='white',
                          padx=12, pady=8, cursor='hand2', relief=tk.RAISED,
                          borderwidth=3, activebackground=color,
                          command=lambda c=code: self.change_label(c))
            btn.pack(side=tk.LEFT, padx=6, pady=5)
            self.quality_buttons[code] = btn
    
    def _capture_undo_state(self):
        """Capture the current mutable state for single-step undo."""
        if self.df is None:
            return None
        
        return {
            'df': self.df.copy(deep=True),
            'changes_made': copy.deepcopy(self.changes_made),
            'deleted_ranges': copy.deepcopy(self.deleted_ranges),
            'rep_splits': copy.deepcopy(self.rep_splits),
            'selected_rep': self.selected_rep,
            'selected_session': self.selected_session,
            'view_mode': self.view_mode_var.get(),
            'filter': self.filter_var.get(),
            'split_info': self.split_info_var.get()
        }
    
    def _push_undo_state(self, action_name):
        """Push current state to the undo stack before an edit action."""
        state = self._capture_undo_state()
        if state is None:
            return
        
        self.undo_stack.append({'action': action_name, 'state': state})
        if len(self.undo_stack) > 50:
            self.undo_stack.pop(0)
    
    def _session_display_text(self, session_key):
        """Build combobox text for a given session key."""
        if session_key not in self.sessions_data:
            return "All Sessions"
        
        session_info = self.sessions_data[session_key]
        participant = session_info.get('participant', 1)
        session_name = session_info.get('session', 'Unknown')
        reps_count = len(session_info.get('reps', []))
        
        if len(session_name) > 25:
            session_name = session_name[:22] + "..."
        
        return f"P{participant:03d} | {session_name} ({reps_count} reps)"
    
    def _restore_undo_state(self, state):
        """Restore app state from an undo snapshot."""
        self.df = state['df'].copy(deep=True)
        self.changes_made = copy.deepcopy(state['changes_made'])
        self.deleted_ranges = copy.deepcopy(state['deleted_ranges'])
        self.rep_splits = copy.deepcopy(state['rep_splits'])
        self.selection_start = None
        self.selection_end = None
        
        self.parse_reps()
        self.update_session_selector()
        
        filter_value = state.get('filter', "All")
        if filter_value in self.filter_combo['values']:
            self.filter_var.set(filter_value)
        
        selected_session = state.get('selected_session')
        if selected_session in self.sessions_data:
            self.selected_session = selected_session
            self.session_var.set(self._session_display_text(selected_session))
        else:
            self.selected_session = None
            self.session_var.set("All Sessions")
        
        self.update_rep_list()
        self.update_changes_counter()
        self.split_info_var.set(state.get('split_info', ""))
        
        selected_rep = state.get('selected_rep')
        self.selected_rep = selected_rep if selected_rep in self.reps_data else None
        
        view_mode = state.get('view_mode', "Full Dataset")
        if view_mode == "Single Rep" and not self.selected_rep:
            view_mode = "Full Dataset"
        
        self.view_mode_var.set(view_mode)
        if self.view_mode_var.get() == "Single Rep" and self.selected_rep:
            self.visualize_rep(self.selected_rep)
            self.update_rep_info(self.selected_rep)
            self.reselect_rep(self.selected_rep)
        else:
            self.visualize_full_dataset()
        
        self.clear_selection()
    
    def undo_last_action(self):
        """Undo only the most recent editing action."""
        if not self.undo_stack:
            messagebox.showinfo("No Undo", "No previous action to undo")
            return
        
        last_entry = self.undo_stack.pop()
        self._restore_undo_state(last_entry['state'])
        messagebox.showinfo("Undone", f"Reverted last action: {last_entry['action']}")
    
    def load_csv(self):
        """Load a CSV file"""
        file_path = filedialog.askopenfilename(
            title="Select CSV File with Rep Data",
            initialdir=str(PROJECT_ROOT),
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            self.df = pd.read_csv(file_path)
            self.original_df = self.df.copy()
            self.current_file = file_path
            self.changes_made = {}
            self.deleted_ranges = {}
            self.rep_splits = {}
            self.undo_stack = []
            self.split_info_var.set("")
            
            # Auto-detect equipment and exercise from path
            self.equipment_code, self.exercise_code, quality_from_path, self.detected_info = \
                detect_equipment_exercise_from_path(file_path)
            
            # Update quality labels based on detected exercise
            self.current_quality_labels = get_quality_labels_for_exercise(self.exercise_code)
            self.update_quality_labels_display()
            self.create_quality_buttons()
            
            # Update detection info display
            self.update_detection_display()
            
            # Parse reps
            self.parse_reps()
            
            # Update session selector
            self.update_session_selector()
            
            # Update UI
            self.update_rep_list()
            self.update_changes_counter()
            
            # Show full dataset view by default
            self.view_mode_var.set("Full Dataset")
            self.visualize_full_dataset()
            
            # Build info message
            sessions_count = len(self.sessions_data)
            participants = len(set(s.get('participant', 1) for s in self.sessions_data.values()))
            
            messagebox.showinfo("Success", 
                              f"Loaded {len(self.reps_data)} reps in {sessions_count} session(s) from {participants} participant(s)\n\n"
                              f"📁 File: {self.detected_info.get('filename', 'Unknown')}\n"
                              f"📦 Equipment: {self.detected_info.get('equipment_name', 'Unknown')}\n"
                              f"🏋️ Exercise: {self.detected_info.get('exercise_name', 'Unknown')}\n"
                              f"📂 Quality from folder: {self.detected_info.get('quality_from_folder', 'Unknown')}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file:\n{e}")
    
    def update_detection_display(self):
        """Update the detection info display"""
        equipment = self.detected_info.get('equipment_name', 'Unknown')
        exercise = self.detected_info.get('exercise_name', 'Unknown')
        quality_folder = self.detected_info.get('quality_from_folder', 'Unknown')
        participant = self.detected_info.get('participant', 'Unknown')
        filename = self.detected_info.get('filename', 'Unknown')
        
        info_text = f"📄 File: {filename}\n"
        info_text += f"📦 Equipment: {equipment} (Code: {self.equipment_code}) | "
        info_text += f"🏋️ Exercise: {exercise} (Code: {self.exercise_code})\n"
        info_text += f"📂 Folder Quality: {quality_folder} | "
        info_text += f"👤 Participant: P{participant:03d}" if isinstance(participant, int) else f"👤 Participant: {participant}"
        
        self.detection_var.set(info_text)
    
    def parse_reps(self):
        """Parse the dataframe into individual reps, grouped by participant and session/set"""
        self.reps_data = {}
        self.sessions_data = {}  # Track sessions for hierarchical display
        
        if self.df is None:
            return
        
        # Determine grouping columns
        has_participant = 'participant' in self.df.columns
        has_source_file = 'source_file' in self.df.columns
        has_rep = 'rep' in self.df.columns
        
        if not has_rep:
            # Treat entire file as one rep
            quality = get_quality_value(self.df, default=0)
            self.reps_data['S1_R1'] = {
                'data': self.df.copy(),
                'participant': 1,
                'session': 'Set_1',
                'session_idx': 1,
                'rep': 1,
                'quality_code': quality,
                'equipment_code': self.df['equipment_code'].iloc[0] if 'equipment_code' in self.df.columns else self.equipment_code,
                'exercise_code': self.df['exercise_code'].iloc[0] if 'exercise_code' in self.df.columns else self.exercise_code,
                'indices': self.df.index.tolist()
            }
            self.sessions_data['S1'] = {'participant': 1, 'session': 'Set_1', 'reps': ['S1_R1']}
            return
        
        # Build grouping structure: participant -> source_file (session) -> rep
        if has_participant and has_source_file:
            # Full structure: participant + source_file + rep
            session_idx = 0
            
            for participant_id in sorted(self.df['participant'].dropna().unique()):
                p_data = self.df[self.df['participant'] == participant_id]
                sessions = sorted(p_data['source_file'].unique())
                
                for source_file in sessions:
                    session_idx += 1
                    session_data = p_data[p_data['source_file'] == source_file]
                    
                    # Extract session name from filename
                    session_name = Path(source_file).stem if source_file else f"Set_{session_idx}"
                    session_key = f"P{int(participant_id)}_S{session_idx}"
                    
                    self.sessions_data[session_key] = {
                        'participant': int(participant_id),
                        'session': session_name,
                        'session_idx': session_idx,
                        'source_file': source_file,
                        'reps': []
                    }
                    
                    # Get reps within this session
                    for rep_num in sorted(session_data['rep'].unique()):
                        rep_data = session_data[session_data['rep'] == rep_num]
                        rep_id = f"P{int(participant_id)}_S{session_idx}_R{int(rep_num)}"
                        
                        quality = get_quality_value(rep_data, default=0)
                        
                        self.reps_data[rep_id] = {
                            'data': rep_data.copy(),
                            'participant': int(participant_id),
                            'session': session_name,
                            'session_key': session_key,
                            'session_idx': session_idx,
                            'source_file': source_file,
                            'rep': int(rep_num),
                            'quality_code': quality,
                            'equipment_code': rep_data['equipment_code'].iloc[0] if 'equipment_code' in rep_data.columns else self.equipment_code,
                            'exercise_code': rep_data['exercise_code'].iloc[0] if 'exercise_code' in rep_data.columns else self.exercise_code,
                            'indices': rep_data.index.tolist()
                        }
                        
                        self.sessions_data[session_key]['reps'].append(rep_id)
        
        elif has_participant:
            # Only participant + rep (no source_file)
            for participant_id in sorted(self.df['participant'].dropna().unique()):
                p_data = self.df[self.df['participant'] == participant_id]
                session_key = f"P{int(participant_id)}_S1"
                
                self.sessions_data[session_key] = {
                    'participant': int(participant_id),
                    'session': 'Default',
                    'session_idx': 1,
                    'reps': []
                }
                
                for rep_num in sorted(p_data['rep'].unique()):
                    rep_data = p_data[p_data['rep'] == rep_num]
                    rep_id = f"P{int(participant_id)}_S1_R{int(rep_num)}"
                    
                    quality = get_quality_value(rep_data, default=0)
                    
                    self.reps_data[rep_id] = {
                        'data': rep_data.copy(),
                        'participant': int(participant_id),
                        'session': 'Default',
                        'session_key': session_key,
                        'session_idx': 1,
                        'rep': int(rep_num),
                        'quality_code': quality,
                        'equipment_code': rep_data['equipment_code'].iloc[0] if 'equipment_code' in rep_data.columns else self.equipment_code,
                        'exercise_code': rep_data['exercise_code'].iloc[0] if 'exercise_code' in rep_data.columns else self.exercise_code,
                        'indices': rep_data.index.tolist()
                    }
                    
                    self.sessions_data[session_key]['reps'].append(rep_id)
        
        else:
            # Only rep column - no participant grouping
            session_key = "S1"
            self.sessions_data[session_key] = {
                'participant': 1,
                'session': 'Default',
                'session_idx': 1,
                'reps': []
            }
            
            for rep_num in sorted(self.df['rep'].unique()):
                rep_data = self.df[self.df['rep'] == rep_num]
                rep_id = f"S1_R{int(rep_num)}"
                
                quality = get_quality_value(rep_data, default=0)
                
                self.reps_data[rep_id] = {
                    'data': rep_data.copy(),
                    'participant': 1,
                    'session': 'Default',
                    'session_key': session_key,
                    'session_idx': 1,
                    'rep': int(rep_num),
                    'quality_code': quality,
                    'equipment_code': rep_data['equipment_code'].iloc[0] if 'equipment_code' in rep_data.columns else self.equipment_code,
                    'exercise_code': rep_data['exercise_code'].iloc[0] if 'exercise_code' in rep_data.columns else self.exercise_code,
                    'indices': rep_data.index.tolist()
                }
                
                self.sessions_data[session_key]['reps'].append(rep_id)
    
    def update_session_selector(self):
        """Update the session selector combo box with available sessions"""
        session_options = ["All Sessions"]
        
        for session_key, session_info in sorted(self.sessions_data.items()):
            participant = session_info.get('participant', 1)
            session_name = session_info.get('session', 'Unknown')
            reps_count = len(session_info.get('reps', []))
            
            # Truncate session name if too long
            if len(session_name) > 25:
                session_name = session_name[:22] + "..."
            
            display_text = f"P{participant:03d} | {session_name} ({reps_count} reps)"
            session_options.append(display_text)
        
        self.session_combo['values'] = session_options
        self.session_var.set("All Sessions")
        self.selected_session = None
    
    def on_session_select(self, event=None):
        """Handle session selection from combo box"""
        selection = self.session_var.get()
        
        if selection == "All Sessions":
            self.selected_session = None
        else:
            # Extract session key from selection
            # Format: "P001 | session_name (X reps)"
            try:
                # Find matching session
                for session_key, session_info in self.sessions_data.items():
                    participant = session_info.get('participant', 1)
                    session_name = session_info.get('session', 'Unknown')
                    if len(session_name) > 25:
                        session_name = session_name[:22] + "..."
                    reps_count = len(session_info.get('reps', []))
                    
                    expected_display = f"P{participant:03d} | {session_name} ({reps_count} reps)"
                    if selection == expected_display:
                        self.selected_session = session_key
                        break
            except:
                self.selected_session = None
        
        # Update rep list and visualization
        self.update_rep_list()
        
        # Update visualization to show only selected session
        if self.view_mode_var.get() == "Full Dataset":
            self.visualize_full_dataset()
    
    def update_rep_list(self):
        """Update the rep listbox"""
        self.rep_listbox.delete(0, tk.END)
        
        if not self.reps_data:
            return
        
        # Count by quality
        quality_counts = {0: 0, 1: 0, 2: 0}
        
        filter_type = self.filter_var.get()
        
        for rep_id, rep_info in sorted(self.reps_data.items()):
            # Skip reps that are marked as completely deleted
            if rep_id in self.changes_made and self.changes_made[rep_id] == 'DELETED':
                continue
            
            # Filter by selected session
            if self.selected_session is not None:
                rep_session_key = rep_info.get('session_key', '')
                if rep_session_key != self.selected_session:
                    continue
            
            quality = rep_info['quality_code']
            
            # Check if this rep has been changed (but not deleted)
            if rep_id in self.changes_made and self.changes_made[rep_id] != 'DELETED':
                quality = self.changes_made[rep_id]
            
            quality_counts[quality] = quality_counts.get(quality, 0) + 1
            
            # Apply filter
            if filter_type == "Clean" and quality != 0:
                continue
            elif filter_type == "Type 1 Error" and quality != 1:
                continue
            elif filter_type == "Type 2 Error" and quality != 2:
                continue
            elif filter_type == "Changed" and rep_id not in self.changes_made:
                continue
            elif filter_type == "Has Deletions" and rep_id not in self.deleted_ranges:
                continue
            
            # Format display - use actual rep_id for proper lookup
            quality_label = self.current_quality_labels.get(quality, f"Unknown ({quality})")
            changed_marker = " ✏️" if rep_id in self.changes_made else ""
            deleted_marker = " 🗑️" if rep_id in self.deleted_ranges else ""
            
            # Add session info if viewing all sessions (to distinguish reps across sessions)
            if self.selected_session is None and 'source_file' in rep_info:
                # Show abbreviated session name for context
                session_name = rep_info.get('session', '')
                # Extract meaningful part (e.g., "...04.csv" from long name)
                if session_name:
                    session_abbrev = Path(session_name).stem[-8:] if len(session_name) > 8 else session_name
                    display_text = f"{rep_id:<15} | {session_abbrev[:8]:<8} | {quality_label[:12]:<12}{changed_marker}{deleted_marker}"
                else:
                    display_text = f"{rep_id:<15} | {quality_label[:12]:<12}{changed_marker}{deleted_marker}"
            else:
                # Single session view - simpler format
                display_text = f"{rep_id:<15} | {quality_label[:12]:<12}{changed_marker}{deleted_marker}"
            
            self.rep_listbox.insert(tk.END, display_text)
            
            # Color code by quality
            idx = self.rep_listbox.size() - 1
            color = QUALITY_COLORS.get(quality, '#333')
            self.rep_listbox.itemconfig(idx, fg=color)
        
        # Update summary
        total = sum(quality_counts.values())
        label_0 = self.current_quality_labels.get(0, "Clean")
        label_1 = self.current_quality_labels.get(1, "Type 1")
        label_2 = self.current_quality_labels.get(2, "Type 2")
        
        summary = f"Total: {total} | {label_0}: {quality_counts[0]} | {label_1}: {quality_counts[1]} | {label_2}: {quality_counts[2]}"
        if self.changes_made:
            summary += f" | Changed: {len(self.changes_made)}"
        if self.deleted_ranges:
            summary += f" | Deletions: {len(self.deleted_ranges)}"
        self.summary_var.set(summary)
    
    def apply_filter(self, event=None):
        """Apply the selected filter"""
        self.update_rep_list()
    
    def on_rep_select(self, event):
        """Handle rep selection from listbox"""
        selections = self.rep_listbox.curselection()
        if not selections:
            return
        
        if len(selections) == 1:
            # Single selection - show single rep view
            display_text = self.rep_listbox.get(selections[0])
            rep_id = display_text.split(" | ")[0].strip()
            
            if rep_id not in self.reps_data:
                return
            
            self.selected_rep = rep_id
            
            # Switch to single rep view when selecting a single rep
            self.view_mode_var.set("Single Rep")
            self.visualize_rep(rep_id)
            self.update_rep_info(rep_id)
            self.clear_selection()
        else:
            # Multiple selection - show multi-rep info
            selected_rep_ids = []
            for idx in selections:
                display_text = self.rep_listbox.get(idx)
                rep_id = display_text.split(" | ")[0].strip()
                if rep_id in self.reps_data:
                    selected_rep_ids.append(rep_id)
            
            self.selected_rep = None  # Clear single selection
            self.update_multi_rep_info(selected_rep_ids)
            
            # Switch to full dataset view for multiple selections
            self.view_mode_var.set("Full Dataset")
            self.visualize_full_dataset()
    
    def visualize_rep(self, rep_id):
        """Visualize the selected rep"""
        if rep_id not in self.reps_data:
            return
        
        rep_info = self.reps_data[rep_id]
        rep_data = rep_info['data'].copy()
        
        # Apply any deletions for visualization (show what will be kept)
        deleted_data = None
        if rep_id in self.deleted_ranges:
            # Store deleted data for visual indicator
            deleted_mask = pd.Series([False] * len(rep_data), index=rep_data.index)
            for start_time, end_time in self.deleted_ranges[rep_id]:
                if 'timestamp_ms' in rep_data.columns:
                    mask = (rep_data['timestamp_ms'] >= start_time) & (rep_data['timestamp_ms'] <= end_time)
                    deleted_mask = deleted_mask | mask
            deleted_data = rep_data[deleted_mask].copy()
            rep_data = rep_data[~deleted_mask]
        
        # Get quality (check for changes)
        quality = self.changes_made.get(rep_id, rep_info['quality_code'])
        quality_label = self.current_quality_labels.get(quality, f"Unknown ({quality})")
        quality_color = QUALITY_COLORS.get(quality, '#333')
        
        # Clear and setup plot
        self.ax.clear()
        
        # Get time column
        if 'timestamp_ms' in rep_info['data'].columns:
            base_time = rep_info['data']['timestamp_ms'].iloc[0]
            xlabel = "Time (seconds)"
            
            if len(rep_data) > 0:
                time_col = (rep_data['timestamp_ms'].values - base_time) / 1000
            else:
                time_col = np.array([])
            
            if deleted_data is not None and len(deleted_data) > 0:
                deleted_time = (deleted_data['timestamp_ms'].values - base_time) / 1000
        else:
            time_col = np.arange(len(rep_data))
            xlabel = "Sample Index"
        
        # Plot selected signals
        if len(rep_data) > 0:
            colors = plt.cm.tab10(np.linspace(0, 1, len(self.selected_signals)))
            
            for i, signal in enumerate(self.selected_signals):
                if signal in rep_data.columns:
                    self.ax.plot(time_col, rep_data[signal].values, 
                               label=signal, linewidth=1.5, color=colors[i])
            
            # Plot deleted regions in gray (strikethrough effect)
            if deleted_data is not None and len(deleted_data) > 0:
                for i, signal in enumerate(self.selected_signals):
                    if signal in deleted_data.columns:
                        self.ax.plot(deleted_time, deleted_data[signal].values, 
                                   linewidth=1.5, color='gray', alpha=0.5, linestyle='--')
        
        # Styling
        self.ax.set_xlabel(xlabel, fontsize=10)
        self.ax.set_ylabel("Signal Value", fontsize=10)
        
        # Title with quality indicator
        equipment = EQUIPMENT_CODES.get(rep_info.get('equipment_code', self.equipment_code), 'Unknown')
        exercise = EXERCISE_CODES.get(rep_info.get('exercise_code', self.exercise_code), 'Unknown')
        
        title = f"{rep_id} | {equipment} - {exercise}\n"
        title += f"Quality: {quality_label}"
        if rep_id in self.changes_made:
            title += " (LABEL MODIFIED)"
        if rep_id in self.deleted_ranges:
            title += f" | 🗑️ {len(self.deleted_ranges[rep_id])} deletion(s) - gray=deleted"
        
        self.ax.set_title(title, fontsize=12, fontweight='bold', color=quality_color)
        
        if len(rep_data) > 0:
            self.ax.legend(loc='upper right', fontsize=8)
        self.ax.grid(True, alpha=0.3)
        
        self.fig.tight_layout()
        self.canvas.draw()
        
        # Setup span selector for deletion
        self.setup_span_selector()
    
    def setup_span_selector(self):
        """Setup the span selector for selecting regions to delete"""
        # Disconnect and clear any existing span selector
        if self.span_selector:
            self.span_selector.set_visible(False)
            self.span_selector.disconnect_events()
            self.span_selector = None
        
        # Reset selection state
        self.selection_start = None
        self.selection_end = None
        self.selection_var.set("No selection")
        
        self.span_selector = SpanSelector(
            self.ax, self.on_select_span, 'horizontal',
            useblit=True,
            props=dict(alpha=0.3, facecolor='red'),
            interactive=True,
            drag_from_anywhere=True
        )
    
    def on_select_span(self, xmin, xmax):
        """Handle span selection"""
        self.selection_start = xmin
        self.selection_end = xmax
        self.selection_var.set(f"Selected: {xmin:.3f}s - {xmax:.3f}s")
    
    def delete_selection(self):
        """Delete the selected region from the current rep"""
        if self.selected_rep is None:
            messagebox.showwarning("No Rep Selected", "Please select a rep first")
            return
        
        if self.selection_start is None or self.selection_end is None:
            messagebox.showwarning("No Selection", "Please select a region on the graph first\n(Click and drag horizontally)")
            return
        
        rep_id = self.selected_rep
        rep_info = self.reps_data[rep_id]
        rep_data = rep_info['data']
        
        # Convert selection to timestamp_ms
        if 'timestamp_ms' in rep_data.columns:
            base_time = rep_data['timestamp_ms'].iloc[0]
            start_ms = base_time + (self.selection_start * 1000)
            end_ms = base_time + (self.selection_end * 1000)
            
            # Count how many samples will be deleted
            mask = (rep_data['timestamp_ms'] >= start_ms) & (rep_data['timestamp_ms'] <= end_ms)
            samples_to_delete = mask.sum()
            
            if samples_to_delete == 0:
                messagebox.showinfo("No Data", "No data points in selected region")
                return
            
            if not messagebox.askyesno("Confirm Delete", 
                                       f"Delete {samples_to_delete} samples from {self.selection_start:.3f}s to {self.selection_end:.3f}s?"):
                return
            
            self._push_undo_state("Delete selection")
            
            # Store deletion range
            if rep_id not in self.deleted_ranges:
                self.deleted_ranges[rep_id] = []
            self.deleted_ranges[rep_id].append((start_ms, end_ms))
            
            # Check if entire rep is now deleted (empty)
            remaining_samples = self.get_remaining_samples_count(rep_id)
            
            if remaining_samples == 0:
                # Entire rep is deleted - remove it from the list
                if messagebox.askyesno("Remove Rep", 
                                       f"All samples in {rep_id} have been deleted.\n\nRemove this rep entirely?"):
                    # Mark rep for complete removal
                    if rep_id not in self.changes_made:
                        self.changes_made[rep_id] = 'DELETED'
                    else:
                        self.changes_made[rep_id] = 'DELETED'
                    
                    # Clear selection and update UI
                    self.clear_selection()
                    self.selected_rep = None
                    self.update_rep_list()
                    self.update_changes_counter()
                    
                    # Switch to full dataset view or clear the single rep view
                    self.view_mode_var.set("Full Dataset")
                    if self.df is not None:
                        self.visualize_full_dataset()
                    
                    self.rep_info_var.set("Rep removed. Select another rep from the list.")
                    self.current_label_var.set("-")
                    
                    messagebox.showinfo("Rep Removed", f"{rep_id} has been removed from the dataset.")
                    return
            
            # Clear selection and redraw
            self.clear_selection()
            self.visualize_rep(rep_id)
            self.update_rep_list()
            self.update_changes_counter()
            
            messagebox.showinfo("Deleted", f"Marked {samples_to_delete} samples for deletion\n(Shown in gray dashed line)\nRemaining: {remaining_samples} samples")
        else:
            messagebox.showwarning("Error", "No timestamp column found in data")
    
    def get_remaining_samples_count(self, rep_id):
        """Get the count of samples remaining after deletions for a rep"""
        if rep_id not in self.reps_data:
            return 0
        
        rep_data = self.reps_data[rep_id]['data']
        
        if rep_id not in self.deleted_ranges or not self.deleted_ranges[rep_id]:
            return len(rep_data)
        
        if 'timestamp_ms' not in rep_data.columns:
            return len(rep_data)
        
        # Calculate remaining samples after all deletions
        deleted_mask = pd.Series([False] * len(rep_data), index=rep_data.index)
        for start_time, end_time in self.deleted_ranges[rep_id]:
            mask = (rep_data['timestamp_ms'] >= start_time) & (rep_data['timestamp_ms'] <= end_time)
            deleted_mask = deleted_mask | mask
        
        remaining = (~deleted_mask).sum()
        return remaining
    
    def clear_selection(self):
        """Clear the current selection and reset span selector visual"""
        self.selection_start = None
        self.selection_end = None
        self.selection_var.set("No selection")
        
        # Reset span selector to clear the visual selection
        if self.span_selector:
            self.span_selector.set_visible(False)
            self.span_selector.disconnect_events()
            self.span_selector = None
        
        # Recreate span selector if in single rep view
        if self.view_mode_var.get() == "Single Rep" and self.selected_rep:
            self.setup_span_selector()
        
        self.canvas.draw_idle()
    
    def undo_rep_deletions(self):
        """Undo deletions for the current rep"""
        if self.selected_rep is None:
            messagebox.showwarning("No Rep Selected", "Please select a rep first")
            return
        
        if self.selected_rep not in self.deleted_ranges:
            messagebox.showinfo("No Deletions", "No deletions to undo for this rep")
            return
        
        self._push_undo_state("Undo rep deletions")
        del self.deleted_ranges[self.selected_rep]
        
        self.visualize_rep(self.selected_rep)
        self.update_rep_list()
        self.update_changes_counter()
        
        messagebox.showinfo("Undone", "Deletions for this rep have been undone")
    
    def update_rep_info(self, rep_id):
        """Update the rep info display"""
        if rep_id not in self.reps_data:
            return
        
        rep_info = self.reps_data[rep_id]
        
        # Get current quality (check for changes)
        original_quality = rep_info['quality_code']
        current_quality = self.changes_made.get(rep_id, original_quality)
        
        equipment = EQUIPMENT_CODES.get(rep_info.get('equipment_code', self.equipment_code), 'Unknown')
        exercise = EXERCISE_CODES.get(rep_info.get('exercise_code', self.exercise_code), 'Unknown')
        
        # Compact info for the label editor
        info_text = f"{rep_id} | P{rep_info['participant']:03d} | {len(rep_info['data'])} samples"
        
        if rep_id in self.deleted_ranges:
            total_deletions = len(self.deleted_ranges[rep_id])
            info_text += f" | 🗑️{total_deletions}"
        
        self.rep_info_var.set(info_text)
        
        # Update current label display
        quality_label = self.current_quality_labels.get(current_quality, f"Unknown ({current_quality})")
        self.current_label_var.set(f"{current_quality}: {quality_label}")
        self.current_label_display.configure(fg=QUALITY_COLORS.get(current_quality, '#333'), bg='white')
    
    def update_multi_rep_info(self, rep_ids):
        """Update the rep info display for multiple selected reps"""
        if not rep_ids:
            self.rep_info_var.set("")
            self.current_label_var.set("(Select a rep)")
            return
        
        # Collect info about selected reps
        qualities = {}
        
        for rep_id in rep_ids:
            if rep_id not in self.reps_data:
                continue
                
            rep_info = self.reps_data[rep_id]
            
            # Get current quality
            current_quality = self.changes_made.get(rep_id, rep_info['quality_code'])
            if current_quality != 'DELETED':
                qualities[current_quality] = qualities.get(current_quality, 0) + 1
        
        # Compact info
        info_text = f"{len(rep_ids)} reps selected"
        
        self.rep_info_var.set(info_text)
        
        # Show mixed quality in label display
        if len(qualities) == 1:
            single_quality = list(qualities.keys())[0]
            quality_label = self.current_quality_labels.get(single_quality, f"Unknown ({single_quality})")
            self.current_label_var.set(f"{single_quality}: {quality_label}")
            self.current_label_display.configure(fg=QUALITY_COLORS.get(single_quality, '#333'), bg='white')
        else:
            self.current_label_var.set("Mixed Qualities")
            self.current_label_display.configure(fg='#666', bg='white')
    
    def update_signal_selection(self):
        """Update which signals to plot"""
        self.selected_signals = [col for col, var in self.signal_vars.items() if var.get()]
        
        if not self.selected_signals:
            self.selected_signals = ['filteredMag']
            self.signal_vars['filteredMag'].set(True)
        
        # Refresh the appropriate view
        if self.view_mode_var.get() == "Full Dataset" and self.df is not None:
            self.visualize_full_dataset()
        elif self.selected_rep:
            self.visualize_rep(self.selected_rep)
    
    def toggle_view_mode(self):
        """Toggle between full dataset view and single rep view"""
        mode = self.view_mode_var.get()
        
        if mode == "Full Dataset":
            if self.df is not None:
                self.visualize_full_dataset()
            else:
                self.ax.clear()
                self.ax.set_title("Load a CSV file to view the full dataset")
                self.canvas.draw()
        else:  # Single Rep
            if self.selected_rep:
                self.visualize_rep(self.selected_rep)
            else:
                self.ax.clear()
                self.ax.set_title("Select a rep from the list to visualize")
                self.canvas.draw()
    
    def visualize_full_dataset(self):
        """Visualize the entire dataset with all reps color-coded by quality"""
        if self.df is None or self.df.empty:
            return
        
        self.ax.clear()
        
        # Determine which reps to show based on selected session
        reps_to_show = {}
        for rep_id, rep_info in self.reps_data.items():
            # Skip deleted reps
            if rep_id in self.changes_made and self.changes_made[rep_id] == 'DELETED':
                continue
            
            # Filter by selected session
            if self.selected_session is not None:
                rep_session_key = rep_info.get('session_key', '')
                if rep_session_key != self.selected_session:
                    continue
            
            reps_to_show[rep_id] = rep_info
        
        if not reps_to_show:
            self.ax.set_title("No reps to display for selected session")
            self.canvas.draw()
            return
        
        # Get time base for the displayed reps
        first_rep = list(reps_to_show.values())[0]
        if 'timestamp_ms' in first_rep['data'].columns:
            base_time = first_rep['data']['timestamp_ms'].iloc[0]
            xlabel = "Time (seconds)"
        else:
            base_time = 0
            xlabel = "Sample Index"
        
        # Collect rep boundaries for annotation
        rep_boundaries = []
        rep_idx = 0
        
        # Plot each rep separately with quality-coded colors
        for rep_id, rep_info in sorted(reps_to_show.items()):
            rep_data = rep_info['data'].copy()
            
            # Get quality (check for changes, but not if deleted)
            quality = self.changes_made.get(rep_id, rep_info['quality_code'])
            if quality == 'DELETED':
                continue
            
            quality_color = QUALITY_COLORS.get(quality, '#333')
            quality_label = self.current_quality_labels.get(quality, f"Q{quality})")
            
            # Calculate time values
            if 'timestamp_ms' in rep_data.columns:
                time_col = (rep_data['timestamp_ms'].values - base_time) / 1000
            else:
                time_col = np.arange(len(rep_data)) + rep_idx * len(rep_data)
            
            # Store rep boundary for annotation
            if len(time_col) > 0:
                # Get compact label: just R# if session selected, otherwise P#S#R#
                if self.selected_session:
                    rep_label = f"R{rep_info.get('rep', rep_idx+1)}"
                else:
                    rep_label = f"P{rep_info.get('participant', 1)}S{rep_info.get('session_idx', 1)}R{rep_info.get('rep', rep_idx+1)}"
                
                rep_boundaries.append({
                    'rep_id': rep_id,
                    'rep_label': rep_label,
                    'start': time_col[0],
                    'end': time_col[-1],
                    'mid': (time_col[0] + time_col[-1]) / 2,
                    'quality': quality,
                    'color': quality_color
                })
            
            # Plot selected signals for this rep
            for signal in self.selected_signals:
                if signal in rep_data.columns:
                    self.ax.plot(time_col, rep_data[signal].values,
                               color=quality_color, linewidth=1.2, alpha=0.8)
            
            rep_idx += 1
        
        # Add vertical lines between reps and rep labels at bottom
        y_min, y_max = self.ax.get_ylim()
        label_y = y_min - (y_max - y_min) * 0.05
        
        for i, boundary in enumerate(rep_boundaries):
            # Add vertical separator at rep start (except first)
            if i > 0:
                self.ax.axvline(x=boundary['start'], color='gray', linestyle='--', 
                               alpha=0.3, linewidth=0.8)
            
            # Add rep label at top
            self.ax.annotate(boundary['rep_label'],
                           xy=(boundary['mid'], y_max),
                           fontsize=7, ha='center', va='bottom',
                           color=boundary['color'], fontweight='bold',
                           alpha=0.7)
        
        # Create legend for quality types
        legend_elements = []
        for code, label in self.current_quality_labels.items():
            color = QUALITY_COLORS.get(code, '#333')
            legend_elements.append(plt.Line2D([0], [0], color=color, linewidth=3,
                                             label=f"{code}: {label}"))
        
        self.ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
        
        # Styling
        self.ax.set_xlabel(xlabel, fontsize=10)
        self.ax.set_ylabel("Signal Value", fontsize=10)
        
        equipment = EQUIPMENT_CODES.get(self.equipment_code, 'Unknown')
        exercise = EXERCISE_CODES.get(self.exercise_code, 'Unknown')
        
        # Build title based on session selection
        if self.selected_session and self.selected_session in self.sessions_data:
            session_info = self.sessions_data[self.selected_session]
            participant = session_info.get('participant', 1)
            session_name = session_info.get('session', 'Unknown')
            if len(session_name) > 30:
                session_name = session_name[:27] + "..."
            title = f"📊 P{participant:03d} | {session_name}\n"
            title += f"Reps: {len(reps_to_show)} | {equipment} - {exercise}"
        else:
            title = f"📊 All Sessions | {equipment} - {exercise}\n"
            title += f"Sessions: {len(self.sessions_data)} | Total Reps: {len(reps_to_show)}"
        
        self.ax.set_title(title, fontsize=11, fontweight='bold')
        self.ax.grid(True, alpha=0.3)
        
        self.fig.tight_layout()
        self.canvas.draw()
        
        # Store rep boundaries for click detection
        self.rep_boundaries = rep_boundaries
    
    def on_canvas_click(self, event):
        """Handle clicks on the canvas - select rep from full dataset view"""
        # Only process in full dataset view mode
        if self.view_mode_var.get() != "Full Dataset":
            return
        
        # Check if click is within axes
        if event.inaxes != self.ax or event.xdata is None:
            return
        
        # Find which rep was clicked based on x coordinate (time)
        clicked_time = event.xdata
        
        for boundary in self.rep_boundaries:
            if boundary['start'] <= clicked_time <= boundary['end']:
                rep_id = boundary['rep_id']
                
                # Select the rep in listbox
                for i in range(self.rep_listbox.size()):
                    if self.rep_listbox.get(i).startswith(rep_id):
                        self.rep_listbox.selection_clear(0, tk.END)
                        self.rep_listbox.selection_set(i)
                        self.rep_listbox.see(i)
                        break
                
                # Update selected rep and show single rep view
                self.selected_rep = rep_id
                self.view_mode_var.set("Single Rep")
                self.visualize_rep(rep_id)
                self.update_rep_info(rep_id)
                self.clear_selection()
                break

    # =========================================================================
    # REP BOUNDARY EDITING METHODS
    # =========================================================================
    
    def split_rep_at_selection(self):
        """Split the current rep at the selected time point into two reps"""
        if self.selected_rep is None:
            messagebox.showwarning("No Rep Selected", "Please select a rep first")
            return
        
        if self.selection_start is None or self.selection_end is None:
            messagebox.showwarning("No Selection", 
                "Please select a split point on the graph first.\n"
                "(Click and drag horizontally to select the split region)")
            return
        
        rep_id = self.selected_rep
        if rep_id not in self.reps_data:
            messagebox.showwarning("Invalid Rep", f"Rep {rep_id} not found")
            return
        
        rep_info = self.reps_data[rep_id]
        rep_data = rep_info['data']
        
        if 'timestamp_ms' not in rep_data.columns:
            messagebox.showwarning("Error", "No timestamp column found - cannot split")
            return
        
        # Use the midpoint of selection as split point
        split_time_seconds = (self.selection_start + self.selection_end) / 2
        base_time = rep_data['timestamp_ms'].iloc[0]
        split_timestamp_ms = base_time + (split_time_seconds * 1000)
        
        # Count samples before and after split
        samples_before = (rep_data['timestamp_ms'] < split_timestamp_ms).sum()
        samples_after = (rep_data['timestamp_ms'] >= split_timestamp_ms).sum()
        
        if samples_before < 5 or samples_after < 5:
            messagebox.showwarning("Invalid Split", 
                f"Split would create reps with too few samples.\n"
                f"Before: {samples_before}, After: {samples_after}\n"
                f"Each part needs at least 5 samples.")
            return
        
        # Confirm the split
        current_rep_num = rep_info['rep']
        if not messagebox.askyesno("Confirm Split", 
            f"Split Rep {current_rep_num} at {split_time_seconds:.3f}s?\n\n"
            f"Part 1: {samples_before} samples (stays as Rep {current_rep_num})\n"
            f"Part 2: {samples_after} samples (becomes Rep {current_rep_num + 1})\n\n"
            f"All subsequent reps will be renumbered (+1)."):
            return
        
        self._push_undo_state("Split rep")
        
        # Perform the split
        self._perform_rep_split(rep_id, split_timestamp_ms, samples_before, samples_after)
        
        # Clear selection and update UI
        self.clear_selection()
        self.update_rep_list()
        self.update_changes_counter()
        
        # Visualize the first part of the split
        self.visualize_rep(rep_id)
        self.update_rep_info(rep_id)
        
        self.split_info_var.set(f"✂️ Split Rep {current_rep_num} → {current_rep_num} + {current_rep_num + 1}")
        
        messagebox.showinfo("Split Complete", 
            f"Rep {current_rep_num} split successfully!\n\n"
            f"• Rep {current_rep_num}: {samples_before} samples\n"
            f"• Rep {current_rep_num + 1}: {samples_after} samples\n"
            f"• Subsequent reps renumbered")
    
    def _perform_rep_split(self, rep_id, split_timestamp_ms, samples_before, samples_after):
        """Internal method to perform the actual rep split"""
        rep_info = self.reps_data[rep_id]
        rep_data = rep_info['data']
        original_rep_num = rep_info['rep']
        participant = rep_info['participant']
        session_key = rep_info.get('session_key', 'S1')
        session_idx = rep_info.get('session_idx', 1)
        
        # Split the data
        mask_part1 = rep_data['timestamp_ms'] < split_timestamp_ms
        mask_part2 = rep_data['timestamp_ms'] >= split_timestamp_ms
        
        data_part1 = rep_data[mask_part1].copy()
        data_part2 = rep_data[mask_part2].copy()
        
        indices_part1 = data_part1.index.tolist()
        indices_part2 = data_part2.index.tolist()
        
        # Step 1: Renumber all subsequent reps in the dataframe (+1)
        # Find all reps in the same session with rep number > current
        for other_rep_id, other_info in list(self.reps_data.items()):
            if other_info.get('session_key') == session_key and other_info['rep'] > original_rep_num:
                old_rep_num = other_info['rep']
                new_rep_num = old_rep_num + 1
                
                # Update dataframe
                self.df.loc[other_info['indices'], 'rep'] = new_rep_num
                
                # Update reps_data
                other_info['rep'] = new_rep_num
                other_info['data']['rep'] = new_rep_num
                
                # Create new rep_id and transfer data
                new_rep_id = f"P{participant}_S{session_idx}_R{new_rep_num}"
                old_rep_id = other_rep_id
                
                if old_rep_id != new_rep_id:
                    # Transfer to new key
                    self.reps_data[new_rep_id] = other_info
                    
                    # Transfer any changes/deletions made to this rep
                    if old_rep_id in self.changes_made:
                        self.changes_made[new_rep_id] = self.changes_made.pop(old_rep_id)
                    if old_rep_id in self.deleted_ranges:
                        self.deleted_ranges[new_rep_id] = self.deleted_ranges.pop(old_rep_id)
                    
                    # Update session reps list
                    if session_key in self.sessions_data:
                        reps_list = self.sessions_data[session_key].get('reps', [])
                        if old_rep_id in reps_list:
                            idx = reps_list.index(old_rep_id)
                            reps_list[idx] = new_rep_id
                    
                    # Delete old key (but not if it's the same as new)
                    if old_rep_id in self.reps_data and old_rep_id != new_rep_id:
                        del self.reps_data[old_rep_id]
        
        # Step 2: Update the original rep (part 1) - keep same rep_id
        self.df.loc[indices_part1, 'rep'] = original_rep_num
        rep_info['data'] = data_part1
        rep_info['indices'] = indices_part1
        
        # Step 3: Create the new rep (part 2) with rep number = original + 1
        new_rep_num = original_rep_num + 1
        new_rep_id = f"P{participant}_S{session_idx}_R{new_rep_num}"
        
        # Update dataframe for part 2
        self.df.loc[indices_part2, 'rep'] = new_rep_num
        data_part2['rep'] = new_rep_num
        
        # Create new rep entry
        self.reps_data[new_rep_id] = {
            'data': data_part2,
            'participant': participant,
            'session': rep_info.get('session', 'Default'),
            'session_key': session_key,
            'session_idx': session_idx,
            'source_file': rep_info.get('source_file'),
            'rep': new_rep_num,
            'quality_code': rep_info['quality_code'],  # Inherit quality from original
            'equipment_code': rep_info.get('equipment_code', self.equipment_code),
            'exercise_code': rep_info.get('exercise_code', self.exercise_code),
            'indices': indices_part2
        }
        
        # Add to session reps list
        if session_key in self.sessions_data:
            reps_list = self.sessions_data[session_key].get('reps', [])
            # Find position to insert (after original rep)
            if rep_id in reps_list:
                idx = reps_list.index(rep_id)
                reps_list.insert(idx + 1, new_rep_id)
            else:
                reps_list.append(new_rep_id)
        
        # Track the split for potential undo and for save
        if rep_id not in self.rep_splits:
            self.rep_splits[rep_id] = []
        self.rep_splits[rep_id].append({
            'split_timestamp_ms': split_timestamp_ms,
            'new_rep_id': new_rep_id,
            'original_rep_num': original_rep_num
        })
        
        # Update selected rep if needed
        if self.selected_rep == rep_id:
            pass  # Keep the same selection (part 1)
    
    def assign_selection_to_rep(self):
        """Assign the selected region to a specific rep number"""
        if self.selected_rep is None:
            messagebox.showwarning("No Rep Selected", "Please select a rep first")
            return
        
        if self.selection_start is None or self.selection_end is None:
            messagebox.showwarning("No Selection", 
                "Please select a region on the graph first.\n"
                "(Click and drag horizontally)")
            return
        
        target_rep_str = self.assign_rep_var.get().strip()
        if not target_rep_str:
            messagebox.showwarning("No Target Rep", "Please enter a target rep number")
            return
        
        try:
            target_rep_num = int(target_rep_str)
        except ValueError:
            messagebox.showwarning("Invalid Number", "Please enter a valid rep number")
            return
        
        rep_id = self.selected_rep
        if rep_id not in self.reps_data:
            messagebox.showwarning("Invalid Rep", f"Rep {rep_id} not found")
            return
        
        rep_info = self.reps_data[rep_id]
        rep_data = rep_info['data']
        current_rep_num = rep_info['rep']
        
        if target_rep_num == current_rep_num:
            messagebox.showinfo("No Change", "Selection is already in this rep")
            return
        
        if 'timestamp_ms' not in rep_data.columns:
            messagebox.showwarning("Error", "No timestamp column found")
            return
        
        # Convert selection to timestamp_ms
        base_time = rep_data['timestamp_ms'].iloc[0]
        start_ms = base_time + (self.selection_start * 1000)
        end_ms = base_time + (self.selection_end * 1000)
        
        # Find samples in selection
        mask = (rep_data['timestamp_ms'] >= start_ms) & (rep_data['timestamp_ms'] <= end_ms)
        samples_to_move = mask.sum()
        
        if samples_to_move == 0:
            messagebox.showinfo("No Data", "No data points in selected region")
            return
        
        # Confirm
        if not messagebox.askyesno("Confirm Assignment", 
            f"Move {samples_to_move} samples from Rep {current_rep_num} to Rep {target_rep_num}?"):
            return
        
        self._push_undo_state("Assign selection to rep")
        
        # Get indices to move
        selected_data = rep_data[mask]
        indices_to_move = selected_data.index.tolist()
        
        # Update the rep number in dataframe
        self.df.loc[indices_to_move, 'rep'] = target_rep_num
        
        # Find or create target rep
        participant = rep_info['participant']
        session_idx = rep_info.get('session_idx', 1)
        session_key = rep_info.get('session_key', 'S1')
        target_rep_id = f"P{participant}_S{session_idx}_R{target_rep_num}"
        
        if target_rep_id in self.reps_data:
            # Append to existing rep
            target_info = self.reps_data[target_rep_id]
            target_data = pd.concat([target_info['data'], selected_data]).sort_values('timestamp_ms')
            target_info['data'] = target_data.reset_index(drop=True)
            target_info['indices'] = target_info['indices'] + indices_to_move
        else:
            # Create new rep
            new_data = selected_data.copy()
            new_data['rep'] = target_rep_num
            self.reps_data[target_rep_id] = {
                'data': new_data,
                'participant': participant,
                'session': rep_info.get('session', 'Default'),
                'session_key': session_key,
                'session_idx': session_idx,
                'source_file': rep_info.get('source_file'),
                'rep': target_rep_num,
                'quality_code': rep_info['quality_code'],
                'equipment_code': rep_info.get('equipment_code', self.equipment_code),
                'exercise_code': rep_info.get('exercise_code', self.exercise_code),
                'indices': indices_to_move
            }
            
            # Add to session
            if session_key in self.sessions_data:
                reps_list = self.sessions_data[session_key].get('reps', [])
                if target_rep_id not in reps_list:
                    reps_list.append(target_rep_id)
                    reps_list.sort()
        
        # Remove from current rep
        remaining_data = rep_data[~mask]
        rep_info['data'] = remaining_data.reset_index(drop=True)
        rep_info['indices'] = [i for i in rep_info['indices'] if i not in indices_to_move]
        
        # If current rep is now empty, mark for deletion
        if len(rep_info['data']) == 0:
            self.changes_made[rep_id] = 'DELETED'
            if session_key in self.sessions_data:
                reps_list = self.sessions_data[session_key].get('reps', [])
                if rep_id in reps_list:
                    reps_list.remove(rep_id)
        
        # Clear and update UI
        self.clear_selection()
        self.update_rep_list()
        self.update_changes_counter()
        
        # Visualize target rep
        if target_rep_id in self.reps_data:
            self.selected_rep = target_rep_id
            self.visualize_rep(target_rep_id)
            self.update_rep_info(target_rep_id)
            self.reselect_rep(target_rep_id)
        
        self.split_info_var.set(f"📌 Moved {samples_to_move} samples to Rep {target_rep_num}")
        
        messagebox.showinfo("Assignment Complete", 
            f"Moved {samples_to_move} samples to Rep {target_rep_num}")
    
    def merge_selected_rep(self):
        """Merge the currently selected rep into a target rep number."""
        if self.selected_rep is None:
            messagebox.showwarning("No Rep Selected", "Please select a rep first")
            return
        
        source_rep_id = self.selected_rep
        if source_rep_id not in self.reps_data:
            messagebox.showwarning("Invalid Rep", f"Rep {source_rep_id} not found")
            return
        
        target_rep_str = self.assign_rep_var.get().strip()
        if not target_rep_str:
            messagebox.showwarning("No Target Rep", "Enter a target rep number in Rep# first")
            return
        
        try:
            target_rep_num = int(target_rep_str)
        except ValueError:
            messagebox.showwarning("Invalid Number", "Please enter a valid rep number")
            return
        
        source_info = self.reps_data[source_rep_id]
        source_rep_num = source_info['rep']
        if target_rep_num == source_rep_num:
            messagebox.showinfo("No Change", "Source and target reps are the same")
            return
        
        participant = source_info['participant']
        session_idx = source_info.get('session_idx', 1)
        session_key = source_info.get('session_key', 'S1')
        target_rep_id = f"P{participant}_S{session_idx}_R{target_rep_num}"
        
        if target_rep_id not in self.reps_data:
            messagebox.showwarning(
                "Target Missing",
                f"Rep {target_rep_num} does not exist in this session.\nUse Split/Assign first to create it."
            )
            return
        
        target_info = self.reps_data[target_rep_id]
        if target_info.get('session_key') != session_key:
            messagebox.showwarning("Invalid Target", "Target rep must be in the same session")
            return
        
        source_count = len(source_info.get('data', []))
        target_count = len(target_info.get('data', []))
        
        if not messagebox.askyesno(
            "Confirm Merge",
            f"Merge Rep {source_rep_num} into Rep {target_rep_num}?\n\n"
            f"Rep {target_rep_num}: {target_count} samples\n"
            f"Rep {source_rep_num}: {source_count} samples\n\n"
            f"After merge, Rep {source_rep_num} will be removed."
        ):
            return
        
        self._push_undo_state("Merge reps")
        
        source_indices = list(source_info.get('indices', []))
        self.df.loc[source_indices, 'rep'] = target_rep_num
        
        source_data = source_info['data'].copy()
        target_data = target_info['data'].copy()
        merged_data = pd.concat([target_data, source_data], ignore_index=True)
        if 'timestamp_ms' in merged_data.columns:
            merged_data = merged_data.sort_values('timestamp_ms')
        merged_data = merged_data.reset_index(drop=True)
        
        target_info['data'] = merged_data
        target_info['indices'] = sorted(set(target_info.get('indices', []) + source_indices))
        
        # Move pending deletion marks from source rep to target rep
        if source_rep_id in self.deleted_ranges:
            self.deleted_ranges.setdefault(target_rep_id, []).extend(self.deleted_ranges.pop(source_rep_id))
        
        # Source rep is now empty and marked as deleted
        source_info['data'] = source_info['data'].iloc[0:0].copy()
        source_info['indices'] = []
        self.changes_made.pop(source_rep_id, None)
        self.changes_made[source_rep_id] = 'DELETED'
        
        if session_key in self.sessions_data:
            reps_list = self.sessions_data[session_key].get('reps', [])
            if source_rep_id in reps_list:
                reps_list.remove(source_rep_id)
            if target_rep_id not in reps_list:
                reps_list.append(target_rep_id)
            reps_list.sort()
        
        self.selected_rep = target_rep_id
        self.clear_selection()
        self.update_rep_list()
        self.update_changes_counter()
        self.visualize_rep(target_rep_id)
        self.update_rep_info(target_rep_id)
        self.reselect_rep(target_rep_id)
        
        self.split_info_var.set(f"Merged Rep {source_rep_num} into Rep {target_rep_num}")
        messagebox.showinfo(
            "Merge Complete",
            f"Merged Rep {source_rep_num} into Rep {target_rep_num}.\n"
            f"New sample count: {len(target_info['data'])}"
        )
    def undo_rep_splits(self):
        """Undo the last boundary edit action."""
        self.undo_last_action()

    def change_label(self, new_quality):
        """Change the label of the selected rep(s)"""
        # Get currently selected reps
        selections = self.rep_listbox.curselection()
        if not selections:
            messagebox.showwarning("No Selection", "Please select one or more reps first")
            return
        
        # Extract rep IDs from selections
        selected_rep_ids = []
        for idx in selections:
            display_text = self.rep_listbox.get(idx)
            rep_id = display_text.split(" | ")[0].strip()
            if rep_id in self.reps_data:
                selected_rep_ids.append(rep_id)
        
        if not selected_rep_ids:
            messagebox.showwarning("No Valid Selection", "No valid reps selected")
            return
        
        # Confirm for multiple reps
        if len(selected_rep_ids) > 1:
            quality_label = self.current_quality_labels.get(new_quality, f"Quality {new_quality}")
            if not messagebox.askyesno("Confirm Multiple Label Change",
                                       f"Change label to '{quality_label}' for {len(selected_rep_ids)} reps?"):
                return
        
        # Apply changes to all selected reps
        changes_made = 0
        undo_captured = False
        for rep_id in selected_rep_ids:
            rep_info = self.reps_data[rep_id]
            original_quality = rep_info['quality_code']
            
            if new_quality == original_quality and rep_id not in self.changes_made:
                continue  # No change needed
            elif new_quality == original_quality and rep_id in self.changes_made:
                if not undo_captured:
                    self._push_undo_state("Change label")
                    undo_captured = True
                del self.changes_made[rep_id]  # Revert to original
                changes_made += 1
            else:
                if not undo_captured:
                    self._push_undo_state("Change label")
                    undo_captured = True
                self.changes_made[rep_id] = new_quality  # Apply new label
                changes_made += 1
        
        if changes_made == 0:
            messagebox.showinfo("No Change", "Selected reps already have that label")
            return
        
        # Update UI
        self.update_rep_list()
        self.update_changes_counter()
        
        # Update visualization and info based on selection type
        if len(selected_rep_ids) == 1:
            # Single selection - show single rep view
            rep_id = selected_rep_ids[0]
            self.selected_rep = rep_id
            self.view_mode_var.set("Single Rep")
            self.visualize_rep(rep_id)
            self.update_rep_info(rep_id)
            self.reselect_rep(rep_id)
        else:
            # Multiple selection - show full dataset view and multi-rep info
            self.selected_rep = None
            self.view_mode_var.set("Full Dataset")
            self.visualize_full_dataset()
            self.update_multi_rep_info(selected_rep_ids)
            self.reselect_multiple_reps(selected_rep_ids)
        
        # Show success message
        if changes_made > 0:
            quality_label = self.current_quality_labels.get(new_quality, f"Quality {new_quality}")
            messagebox.showinfo("Label Changed", f"Updated {changes_made} rep(s) to '{quality_label}'")
    
    def reselect_multiple_reps(self, rep_ids):
        """Re-select multiple reps in the listbox after update"""
        self.rep_listbox.selection_clear(0, tk.END)
        
        for i in range(self.rep_listbox.size()):
            display_text = self.rep_listbox.get(i)
            rep_id = display_text.split(" | ")[0].strip()
            if rep_id in rep_ids:
                self.rep_listbox.selection_set(i)
    
    def reselect_rep(self, rep_id):
        """Re-select a rep in the listbox after update"""
        for i in range(self.rep_listbox.size()):
            if self.rep_listbox.get(i).startswith(rep_id):
                self.rep_listbox.selection_clear(0, tk.END)
                self.rep_listbox.selection_set(i)
                self.rep_listbox.see(i)
                break
    
    def update_changes_counter(self):
        """Update the changes counter display"""
        changes = []
        
        # Count label changes vs deleted reps
        label_changes = sum(1 for v in self.changes_made.values() if v != 'DELETED')
        deleted_reps = sum(1 for v in self.changes_made.values() if v == 'DELETED')
        
        if label_changes > 0:
            changes.append(f"{label_changes} label change(s)")
        
        if deleted_reps > 0:
            changes.append(f"{deleted_reps} rep(s) removed")
        
        if self.deleted_ranges:
            # Only count ranges for non-deleted reps
            active_deletions = {k: v for k, v in self.deleted_ranges.items() 
                               if k not in self.changes_made or self.changes_made[k] != 'DELETED'}
            if active_deletions:
                total_deletions = sum(len(ranges) for ranges in active_deletions.values())
                changes.append(f"{total_deletions} deletion(s) in {len(active_deletions)} rep(s)")
        
        # Count rep splits
        if self.rep_splits:
            total_splits = sum(len(splits) for splits in self.rep_splits.values())
            changes.append(f"{total_splits} rep split(s)")
        
        if changes:
            self.changes_var.set(f"📝 Pending: {', '.join(changes)}")
        else:
            self.changes_var.set("No changes made")
    
    def save_changes(self):
        """Save changes to the original file"""
        if not self.changes_made and not self.deleted_ranges and not self.rep_splits:
            messagebox.showinfo("No Changes", "No changes to save")
            return
        
        if not self.current_file:
            self.save_as()
            return
        
        changes_summary = []
        if self.changes_made:
            changes_summary.append(f"{len(self.changes_made)} label change(s)")
        if self.deleted_ranges:
            changes_summary.append(f"deletions in {len(self.deleted_ranges)} rep(s)")
        if self.rep_splits:
            total_splits = sum(len(splits) for splits in self.rep_splits.values())
            changes_summary.append(f"{total_splits} rep split(s)")
        
        if not messagebox.askyesno("Confirm Save", 
                                   f"Save changes to:\n{self.current_file}?\n\nChanges: {', '.join(changes_summary)}"):
            return
        
        self.apply_all_changes()
        
        try:
            self.df.to_csv(self.current_file, index=False)
            
            self.original_df = self.df.copy()
            
            for rep_id, new_quality in self.changes_made.items():
                if rep_id in self.reps_data and new_quality != 'DELETED':
                    self.reps_data[rep_id]['quality_code'] = new_quality
            
            # Re-parse reps to get updated data
            self.parse_reps()
            self.update_session_selector()
            
            saved_label_changes = len(self.changes_made)
            saved_deletions = len(self.deleted_ranges)
            saved_splits = sum(len(splits) for splits in self.rep_splits.values())
            
            self.changes_made = {}
            self.deleted_ranges = {}
            self.rep_splits = {}
            self.undo_stack = []
            self.split_info_var.set("")
            
            self.update_rep_list()
            self.update_changes_counter()
            
            if self.selected_rep and self.selected_rep in self.reps_data:
                self.visualize_rep(self.selected_rep)
                self.update_rep_info(self.selected_rep)
            else:
                self.view_mode_var.set("Full Dataset")
                self.visualize_full_dataset()
            
            messagebox.showinfo("Success", f"Saved!\n• {saved_label_changes} label change(s)\n• {saved_deletions} rep(s) with deletions\n• {saved_splits} rep split(s)")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save:\n{e}")
    
    def save_as(self):
        """Save changes to a new file"""
        if self.df is None:
            messagebox.showwarning("No Data", "No data to save")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save CSV As",
            initialdir=str(PROJECT_ROOT),
            defaultextension=".csv",
            initialfile=f"{Path(self.current_file).stem}_relabeled.csv" if self.current_file else "relabeled_data.csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        self.apply_all_changes()
        
        try:
            self.df.to_csv(file_path, index=False)
            
            self.current_file = file_path
            self.original_df = self.df.copy()
            
            for rep_id, new_quality in self.changes_made.items():
                if rep_id in self.reps_data and new_quality != 'DELETED':
                    self.reps_data[rep_id]['quality_code'] = new_quality
            
            self.parse_reps()
            self.update_session_selector()
            
            saved_label_changes = len(self.changes_made)
            saved_deletions = len(self.deleted_ranges)
            saved_splits = sum(len(splits) for splits in self.rep_splits.values())
            
            self.changes_made = {}
            self.deleted_ranges = {}
            self.rep_splits = {}
            self.undo_stack = []
            self.split_info_var.set("")
            
            self.update_rep_list()
            self.update_changes_counter()
            
            if self.selected_rep and self.selected_rep in self.reps_data:
                self.visualize_rep(self.selected_rep)
                self.update_rep_info(self.selected_rep)
            else:
                self.view_mode_var.set("Full Dataset")
                self.visualize_full_dataset()
            
            messagebox.showinfo("Success", f"Saved to: {Path(file_path).name}\n• {saved_label_changes} label change(s)\n• {saved_deletions} rep(s) with deletions\n• {saved_splits} rep split(s)")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save:\n{e}")
    
    def apply_all_changes(self):
        """Apply all pending changes to the dataframe"""
        # Collect all indices to delete (from fully deleted reps and partial deletions)
        indices_to_delete = []
        
        # Determine which column to use for quality
        quality_col = get_quality_column_name(self.df)
        if quality_col is None:
            # Create quality_code column if neither exists
            quality_col = 'quality_code'
            self.df['quality_code'] = 0
        
        # Apply label changes (skip deleted reps)
        for rep_id, new_quality in self.changes_made.items():
            if new_quality == 'DELETED':
                # Remove entire rep
                if rep_id in self.reps_data:
                    indices_to_delete.extend(self.reps_data[rep_id]['indices'])
            elif rep_id in self.reps_data:
                indices = self.reps_data[rep_id]['indices']
                self.df.loc[indices, quality_col] = new_quality
        
        # Apply partial deletions (only for non-deleted reps)
        for rep_id, ranges in self.deleted_ranges.items():
            # Skip if rep is already fully deleted
            if rep_id in self.changes_made and self.changes_made[rep_id] == 'DELETED':
                continue
            
            if rep_id in self.reps_data:
                rep_indices = self.reps_data[rep_id]['indices']
                rep_data = self.df.loc[rep_indices]
                
                for start_ms, end_ms in ranges:
                    if 'timestamp_ms' in rep_data.columns:
                        mask = (rep_data['timestamp_ms'] >= start_ms) & (rep_data['timestamp_ms'] <= end_ms)
                        indices_to_delete.extend(rep_data[mask].index.tolist())
        
        if indices_to_delete:
            self.df = self.df.drop(indices_to_delete).reset_index(drop=True)
    
    def undo_all(self):
        """Backward-compatible alias for single-step undo."""
        self.undo_last_action()


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    root = tk.Tk()
    app = RepLabelerApp(root)
    
    # Center window
    root.update_idletasks()
    x = (root.winfo_screenwidth() - root.winfo_width()) // 2
    y = (root.winfo_screenheight() - root.winfo_height()) // 2
    root.geometry(f"+{x}+{y}")
    
    root.mainloop()


if __name__ == "__main__":
    main()
