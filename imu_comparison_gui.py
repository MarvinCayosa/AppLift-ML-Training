"""
IMU Comparison GUI Tool
A graphical interface for selecting and analyzing IMU sensor comparison data
Includes quaternion, orientation, accelerometer, and gyroscope analysis
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import threading
import sys
import os
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.style as style

# ====================================================================
# CONFIGURABLE AXIS MAPPING - EDIT HERE TO TEST DIFFERENT COMBINATIONS
# ====================================================================
AXIS_MAPPING_CONFIG = {
    'X': {'esp32_axis': 'Y', 'invert': True},   # Phone X ↔ ESP32 Y (inverted)
    'Y': {'esp32_axis': 'Z', 'invert': False},  # Phone Y ↔ ESP32 Z
    'Z': {'esp32_axis': 'X', 'invert': True}    # Phone Z ↔ ESP32 X (inverted)
}

# Alternative mappings to try (uncomment to test):
# Option 1: Original mapping
# AXIS_MAPPING_CONFIG = {
#     'X': {'esp32_axis': 'Y', 'invert': True},   # Phone X ↔ ESP32 Y (inverted)
#     'Y': {'esp32_axis': 'Z', 'invert': False},  # Phone Y ↔ ESP32 Z
#     'Z': {'esp32_axis': 'X', 'invert': False}   # Phone Z ↔ ESP32 X
# }

# Option 2: Direct mapping
# AXIS_MAPPING_CONFIG = {
#     'X': {'esp32_axis': 'X', 'invert': False},  # Phone X ↔ ESP32 X
#     'Y': {'esp32_axis': 'Y', 'invert': False},  # Phone Y ↔ ESP32 Y
#     'Z': {'esp32_axis': 'Z', 'invert': False}   # Phone Z ↔ ESP32 Z
# }

# Option 3: Custom test (modify as needed)
# AXIS_MAPPING_CONFIG = {
#     'X': {'esp32_axis': 'Z', 'invert': True},   # Phone X ↔ ESP32 Z (inverted)
#     'Y': {'esp32_axis': 'X', 'invert': False},  # Phone Y ↔ ESP32 X
#     'Z': {'esp32_axis': 'Y', 'invert': True}    # Phone Z ↔ ESP32 Y (inverted)
# }

# Import the analysis functions from the main script
sys.path.append(os.path.dirname(__file__))

class IMUComparisonGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("IMU Sensor Comparison Tool")
        self.root.geometry("900x700")
        
        # Set matplotlib backend to avoid display issues
        plt.switch_backend('Agg')
        
        # Data storage
        self.df = None
        self.csv_path = None
        self.output_dir = None
        
        self.create_widgets()
        
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(3, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="🔄 IMU Sensor Comparison Tool", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # File selection frame
        file_frame = ttk.LabelFrame(main_frame, text="📁 CSV File Selection", padding="10")
        file_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        file_frame.columnconfigure(1, weight=1)
        
        # File path display
        ttk.Label(file_frame, text="Selected File:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.file_var = tk.StringVar(value="No file selected")
        self.file_label = ttk.Label(file_frame, textvariable=self.file_var, foreground="gray")
        self.file_label.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        
        # Browse button
        self.browse_btn = ttk.Button(file_frame, text="Browse CSV File", command=self.browse_file)
        self.browse_btn.grid(row=0, column=2, padx=(10, 0))
        
        # Analysis options frame
        options_frame = ttk.LabelFrame(main_frame, text="📊 Analysis Options", padding="10")
        options_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Checkboxes for analysis types
        self.generate_report = tk.BooleanVar(value=True)
        self.generate_quaternion = tk.BooleanVar(value=True)
        self.generate_orientation = tk.BooleanVar(value=True)
        self.generate_accel = tk.BooleanVar(value=True)
        self.generate_gyro = tk.BooleanVar(value=True)
        self.generate_histograms = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(options_frame, text="📄 Comprehensive Report", 
                       variable=self.generate_report).grid(row=0, column=0, sticky=tk.W, padx=(0, 20))
        ttk.Checkbutton(options_frame, text="🔄 Quaternion Analysis", 
                       variable=self.generate_quaternion).grid(row=0, column=1, sticky=tk.W, padx=(0, 20))
        ttk.Checkbutton(options_frame, text="📐 Orientation Comparison", 
                       variable=self.generate_orientation).grid(row=0, column=2, sticky=tk.W)
        
        ttk.Checkbutton(options_frame, text="📊 Accelerometer Analysis", 
                       variable=self.generate_accel).grid(row=1, column=0, sticky=tk.W, padx=(0, 20))
        ttk.Checkbutton(options_frame, text="🌀 Gyroscope Analysis", 
                       variable=self.generate_gyro).grid(row=1, column=1, sticky=tk.W, padx=(0, 20))
        ttk.Checkbutton(options_frame, text="📈 Distribution Histograms", 
                       variable=self.generate_histograms).grid(row=1, column=2, sticky=tk.W)
        
        # Control buttons frame
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        control_frame.columnconfigure(1, weight=1)
        
        # Analyze button
        self.analyze_btn = ttk.Button(control_frame, text="🚀 Run Analysis", 
                                     command=self.run_analysis, state='disabled')
        self.analyze_btn.grid(row=0, column=0, padx=(0, 10))
        
        # Progress bar
        self.progress = ttk.Progressbar(control_frame, mode='indeterminate')
        self.progress.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        
        # Open results button
        self.results_btn = ttk.Button(control_frame, text="📁 Open Results Folder", 
                                     command=self.open_results, state='disabled')
        self.results_btn.grid(row=0, column=2)
        
        # Output text area
        output_frame = ttk.LabelFrame(main_frame, text="📋 Analysis Output", padding="10")
        output_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        output_frame.columnconfigure(0, weight=1)
        output_frame.rowconfigure(0, weight=1)
        
        self.output_text = scrolledtext.ScrolledText(output_frame, height=15, width=80)
        self.output_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Add some initial text
        self.output_text.insert(tk.END, "Welcome to IMU Sensor Comparison Tool!\n")
        self.output_text.insert(tk.END, "Please select a CSV file containing IMU comparison data to begin.\n\n")
        self.output_text.insert(tk.END, "Expected CSV columns:\n")
        self.output_text.insert(tk.END, "- timestamp_ms, phone_roll/pitch/yaw, esp32_roll/pitch/yaw\n")
        self.output_text.insert(tk.END, "- phone_accelX/Y/Z, esp32_accelX/Y/Z\n")
        self.output_text.insert(tk.END, "- phone_gyroX/Y/Z_rad, esp32_gyroX/Y/Z_deg\n")
        self.output_text.insert(tk.END, "- phone_qw/qx/qy/qz, esp32_qw/qx/qy/qz\n")
        self.output_text.insert(tk.END, "- diff_quat_angle_deg\n\n")
        
    def browse_file(self):
        """Open file dialog to select CSV file"""
        file_types = [
            ('CSV files', '*.csv'),
            ('All files', '*.*')
        ]
        
        filename = filedialog.askopenfilename(
            title="Select IMU Comparison CSV File",
            filetypes=file_types,
            initialdir=os.getcwd()
        )
        
        if filename:
            self.csv_path = filename
            self.file_var.set(os.path.basename(filename))
            self.file_label.configure(foreground="black")
            self.analyze_btn.configure(state='normal')
            
            # Try to preview the CSV
            try:
                preview_df = pd.read_csv(filename, nrows=5)
                self.output_text.delete(1.0, tk.END)
                self.output_text.insert(tk.END, f"✅ File loaded successfully: {os.path.basename(filename)}\n\n")
                self.output_text.insert(tk.END, f"Preview (first 5 rows):\n")
                self.output_text.insert(tk.END, f"Columns found: {len(preview_df.columns)}\n")
                self.output_text.insert(tk.END, f"Sample columns: {', '.join(preview_df.columns[:10])}\n")
                if len(preview_df.columns) > 10:
                    self.output_text.insert(tk.END, f"... and {len(preview_df.columns) - 10} more columns\n")
                self.output_text.insert(tk.END, f"\nTotal rows in file: {len(pd.read_csv(filename))}\n\n")
                self.output_text.insert(tk.END, "Ready for analysis! Click 'Run Analysis' to begin.\n")
            except Exception as e:
                self.output_text.delete(1.0, tk.END)
                self.output_text.insert(tk.END, f"⚠️ Warning: Could not preview file: {str(e)}\n")
                self.output_text.insert(tk.END, "File selected, but please verify it's a valid CSV with IMU data.\n")
    
    def log_output(self, message):
        """Add message to output text area"""
        self.output_text.insert(tk.END, message + "\n")
        self.output_text.see(tk.END)
        self.root.update_idletasks()
    
    def run_analysis(self):
        """Run the IMU analysis in a separate thread"""
        if not self.csv_path:
            messagebox.showerror("Error", "Please select a CSV file first!")
            return
        
        # Start progress bar
        self.progress.start(10)
        self.analyze_btn.configure(state='disabled')
        
        # Run analysis in thread to avoid GUI freezing
        analysis_thread = threading.Thread(target=self.perform_analysis)
        analysis_thread.daemon = True
        analysis_thread.start()
    
    def perform_analysis(self):
        """Perform the actual analysis"""
        try:
            self.log_output("🔄 Starting IMU comparison analysis...")
            
            # Load data
            self.log_output("📂 Loading CSV data...")
            self.df = self.load_comparison_data(self.csv_path)
            
            if self.df is None:
                return
            
            # Set up output directory
            base_name = Path(self.csv_path).stem
            self.output_dir = Path("visualizations") / "imu_comparison_plots" / f"{base_name}_analysis"
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            self.log_output(f"📁 Output directory: {self.output_dir}")
            
            # Generate reports and visualizations based on selected options
            if self.generate_report.get():
                self.log_output("📄 Generating comprehensive report...")
                report_path = self.output_dir / "comprehensive_sensor_report.txt"
                self.generate_comprehensive_report(self.df, str(report_path))
                
            visualization_count = 0
            
            if self.generate_quaternion.get() and self.has_quaternion_data():
                self.log_output("🔄 Creating quaternion visualizations...")
                self.plot_quaternion_comparison(self.df, str(self.output_dir / "quaternion_comparison.png"))
                self.plot_quaternion_analysis(self.df, str(self.output_dir / "quaternion_analysis.png"))
                visualization_count += 2
                
            if self.generate_orientation.get():
                self.log_output("📐 Creating orientation comparison...")
                self.plot_orientation_comparison(self.df, str(self.output_dir / "orientation_comparison.png"))
                visualization_count += 1
                
            if self.generate_accel.get() and self.has_accelerometer_data():
                self.log_output("📊 Creating accelerometer comparison...")
                self.plot_accelerometer_comparison(self.df, str(self.output_dir / "accelerometer_comparison.png"))
                visualization_count += 1
                
            if self.generate_gyro.get() and self.has_gyroscope_data():
                self.log_output("🌀 Creating gyroscope comparison...")
                self.plot_gyroscope_comparison(self.df, str(self.output_dir / "gyroscope_comparison.png"))
                visualization_count += 1
                
            if self.generate_histograms.get():
                self.log_output("📈 Creating distribution histograms...")
                self.plot_difference_histograms(self.df, str(self.output_dir / "difference_histograms.png"))
                visualization_count += 1
            
            # Analysis complete
            self.log_output(f"\n✅ Analysis Complete!")
            self.log_output(f"📊 Generated {visualization_count} visualizations")
            self.log_output(f"📁 All files saved to: {self.output_dir}")
            
            # Enable results button
            self.root.after(0, lambda: self.results_btn.configure(state='normal'))
            
        except Exception as e:
            self.log_output(f"❌ Error during analysis: {str(e)}")
            messagebox.showerror("Analysis Error", f"An error occurred: {str(e)}")
        
        finally:
            # Stop progress bar and re-enable button
            self.root.after(0, lambda: self.progress.stop())
            self.root.after(0, lambda: self.analyze_btn.configure(state='normal'))
    
    def has_quaternion_data(self):
        """Check if quaternion data is available"""
        required_cols = ['phone_qw', 'phone_qx', 'phone_qy', 'phone_qz', 
                        'esp32_qw', 'esp32_qx', 'esp32_qy', 'esp32_qz']
        return all(col in self.df.columns for col in required_cols)
    
    def has_accelerometer_data(self):
        """Check if accelerometer data is available"""
        required_cols = ['phone_accelX', 'phone_accelY', 'phone_accelZ',
                        'esp32_accelX', 'esp32_accelY', 'esp32_accelZ']
        return all(col in self.df.columns for col in required_cols)
    
    def has_gyroscope_data(self):
        """Check if gyroscope data is available"""
        required_cols = ['phone_gyroX_rad', 'phone_gyroY_rad', 'phone_gyroZ_rad',
                        'esp32_gyroX_deg', 'esp32_gyroY_deg', 'esp32_gyroZ_deg']
        return all(col in self.df.columns for col in required_cols)
    
    def open_results(self):
        """Open the results folder in file explorer"""
        if self.output_dir and self.output_dir.exists():
            if sys.platform == "win32":
                os.startfile(str(self.output_dir))
            elif sys.platform == "darwin":
                os.system(f"open '{self.output_dir}'")
            else:
                os.system(f"xdg-open '{self.output_dir}'")
        else:
            messagebox.showwarning("Warning", "Results folder not found!")

    # Analysis functions (taken from the main script)
    def load_comparison_data(self, csv_path: str) -> pd.DataFrame:
        """Load IMU comparison CSV data including quaternion values"""
        try:
            df = pd.read_csv(csv_path)
            
            # Convert timestamp to seconds from start
            if 'timestamp_ms' in df.columns:
                df['time_sec'] = (df['timestamp_ms'] - df['timestamp_ms'].iloc[0]) / 1000
            
            self.log_output(f"✅ Loaded {len(df)} samples")
            
            # Custom axis remapping for ESP32 vs Phone gyroscopes
            self.log_output("📍 Applying custom axis mapping:")
            for phone_axis, config in AXIS_MAPPING_CONFIG.items():
                esp32_axis = config['esp32_axis']
                invert_str = " (inverted)" if config['invert'] else ""
                self.log_output(f"  Phone {phone_axis} ↔ ESP32 {esp32_axis}{invert_str}")
            
            axis_mapping = AXIS_MAPPING_CONFIG
            
            for phone_axis in ['X', 'Y', 'Z']:
                phone_rad_col = f'phone_gyro{phone_axis}_rad'
                esp32_axis = axis_mapping[phone_axis]['esp32_axis']
                invert = axis_mapping[phone_axis]['invert']
                esp32_deg_col = f'esp32_gyro{esp32_axis}_deg'
                
                # Use phone rad/s as-is
                if phone_rad_col in df.columns:
                    df[f'phone_gyro{phone_axis}'] = df[phone_rad_col]
                
                # Map ESP32 axis and interpolate
                if esp32_deg_col in df.columns:
                    df[f'esp32_gyro{phone_axis}_raw'] = df[esp32_deg_col]
                    df[f'esp32_gyro{phone_axis}_temp'] = df[esp32_deg_col].interpolate(method='linear', limit_direction='both')
                    
                    # Apply inversion if needed
                    if invert:
                        df[f'esp32_gyro{phone_axis}'] = -df[f'esp32_gyro{phone_axis}_temp']
                        self.log_output(f"📍 Mapped {esp32_deg_col} → esp32_gyro{phone_axis} (INVERTED)")
                    else:
                        df[f'esp32_gyro{phone_axis}'] = df[f'esp32_gyro{phone_axis}_temp']
                        self.log_output(f"📍 Mapped {esp32_deg_col} → esp32_gyro{phone_axis}")
                
                # Calculate differences with remapped data
                if f'phone_gyro{phone_axis}' in df.columns and f'esp32_gyro{phone_axis}' in df.columns:
                    # Convert ESP32 deg/s to rad/s for difference calculation
                    esp32_rad = np.radians(df[f'esp32_gyro{phone_axis}'])
                    df[f'diff_gyro{phone_axis}'] = esp32_rad - df[f'phone_gyro{phone_axis}']
            
            # Store axis mapping info for reporting
            df.attrs['axis_mapping'] = axis_mapping
            self.log_output(f"✅ Custom axis remapping applied successfully.")
            
            # Interpolate ESP32 orientation data
            for axis in ['roll', 'pitch', 'yaw']:
                esp32_col = f'esp32_{axis}'
                if esp32_col in df.columns:
                    df[f'{esp32_col}_raw'] = df[esp32_col]
                    df[esp32_col] = df[esp32_col].interpolate(method='linear', limit_direction='both')
            
            # Interpolate ESP32 accelerometer data
            for axis in ['X', 'Y', 'Z']:
                esp32_col = f'esp32_accel{axis}'
                if esp32_col in df.columns:
                    df[f'{esp32_col}_raw'] = df[esp32_col]
                    df[esp32_col] = df[esp32_col].interpolate(method='linear', limit_direction='both')
            
            # Interpolate ESP32 quaternion data
            for quat in ['qw', 'qx', 'qy', 'qz']:
                esp32_col = f'esp32_{quat}'
                if esp32_col in df.columns:
                    df[f'{esp32_col}_raw'] = df[esp32_col]
                    df[esp32_col] = df[esp32_col].interpolate(method='linear', limit_direction='both')
            
            # Interpolate quaternion norm if available
            if 'esp32_quatNorm' in df.columns:
                df['esp32_quatNorm_raw'] = df['esp32_quatNorm']
                df['esp32_quatNorm'] = df['esp32_quatNorm'].interpolate(method='linear', limit_direction='both')
            
            self.log_output(f"📊 Data processing complete. Columns: {len(df.columns)}")
            return df
            
        except Exception as e:
            self.log_output(f"❌ Error loading CSV: {str(e)}")
            messagebox.showerror("Load Error", f"Could not load CSV file: {str(e)}")
            return None

    def plot_quaternion_comparison(self, df: pd.DataFrame, save_path: str = None):
        """Plot quaternion components comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Quaternion Components Comparison: Phone vs ESP32', fontsize=14, fontweight='bold')
        
        time = df['time_sec'] if 'time_sec' in df.columns else df.index
        
        components = ['qw', 'qx', 'qy', 'qz']
        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        
        for comp, (row, col) in zip(components, positions):
            phone_col = f'phone_{comp}'
            esp32_col = f'esp32_{comp}'
            
            axes[row, col].plot(time, df[phone_col], label='Phone', color='#2196f3', alpha=0.8, linewidth=1.5)
            axes[row, col].plot(time, df[esp32_col], label='ESP32', color='#f44336', alpha=0.8, linewidth=1.5)
            axes[row, col].set_ylabel(f'{comp.upper()}')
            axes[row, col].set_title(f'Quaternion {comp.upper()} Component')
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)
            
            # Add range information
            phone_range = df[phone_col].max() - df[phone_col].min()
            esp32_range = df[esp32_col].max() - df[esp32_col].min()
            axes[row, col].text(0.02, 0.98, f'Phone range: {phone_range:.4f}\nESP32 range: {esp32_range:.4f}', 
                               transform=axes[row, col].transAxes, fontsize=8, 
                               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 1].set_xlabel('Time (s)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
    
    def plot_quaternion_analysis(self, df: pd.DataFrame, save_path: str = None):
        """Plot quaternion analysis including angle differences and norms"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Quaternion Analysis: Angle Differences and Norms', fontsize=14, fontweight='bold')
        
        time = df['time_sec'] if 'time_sec' in df.columns else df.index
        
        # Quaternion angle difference
        if 'diff_quat_angle_deg' in df.columns:
            axes[0, 0].plot(time, df['diff_quat_angle_deg'], color='#ff9800', linewidth=1.5)
            axes[0, 0].axhline(y=0, color='green', linestyle='--', alpha=0.5)
            axes[0, 0].axhline(y=5, color='orange', linestyle=':', alpha=0.5)
            axes[0, 0].axhline(y=10, color='red', linestyle=':', alpha=0.5)
            axes[0, 0].fill_between(time, 0, 5, alpha=0.1, color='green')
            axes[0, 0].fill_between(time, 5, 10, alpha=0.1, color='orange')
            axes[0, 0].set_ylabel('Angle Difference (°)')
            axes[0, 0].set_title(f'Quaternion Angle Difference\n(Mean: {df["diff_quat_angle_deg"].mean():.2f}°, Std: {df["diff_quat_angle_deg"].std():.2f}°)')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Calculate quaternion norms for both devices
        phone_norm = np.sqrt(df['phone_qw']**2 + df['phone_qx']**2 + df['phone_qy']**2 + df['phone_qz']**2)
        axes[0, 1].plot(time, phone_norm, label='Phone Norm', color='#2196f3', alpha=0.8)
        if 'esp32_quatNorm' in df.columns:
            axes[0, 1].plot(time, df['esp32_quatNorm'], label='ESP32 Norm', color='#f44336', alpha=0.8)
        else:
            esp32_norm = np.sqrt(df['esp32_qw']**2 + df['esp32_qx']**2 + df['esp32_qy']**2 + df['esp32_qz']**2)
            axes[0, 1].plot(time, esp32_norm, label='ESP32 Norm (calc)', color='#f44336', alpha=0.8)
        axes[0, 1].axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Ideal (1.0)')
        axes[0, 1].set_ylabel('Quaternion Norm')
        axes[0, 1].set_title('Quaternion Normalization')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Quaternion difference histogram
        if 'diff_quat_angle_deg' in df.columns:
            data = df['diff_quat_angle_deg'].dropna()
            axes[1, 0].hist(data, bins=50, color='#ff9800', alpha=0.7, edgecolor='white')
            axes[1, 0].axvline(x=data.mean(), color='red', linestyle='-', linewidth=2, label=f'Mean ({data.mean():.2f}°)')
            axes[1, 0].axvline(x=5, color='orange', linestyle=':', alpha=0.7, label='5° threshold')
            axes[1, 0].axvline(x=10, color='red', linestyle=':', alpha=0.7, label='10° threshold')
            axes[1, 0].set_xlabel('Angle Difference (°)')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].set_title('Distribution of Quaternion Angle Differences')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Quaternion component differences
        quat_diffs = []
        for comp in ['qw', 'qx', 'qy', 'qz']:
            phone_col = f'phone_{comp}'
            esp32_col = f'esp32_{comp}'
            if phone_col in df.columns and esp32_col in df.columns:
                diff = df[esp32_col] - df[phone_col]
                quat_diffs.append((comp, diff))
        
        if quat_diffs:
            for i, (comp, diff) in enumerate(quat_diffs):
                axes[1, 1].plot(time, diff, label=f'Δ{comp.upper()}', alpha=0.7)
            axes[1, 1].axhline(y=0, color='green', linestyle='--', alpha=0.5)
            axes[1, 1].set_xlabel('Time (s)')
            axes[1, 1].set_ylabel('Component Difference')
            axes[1, 1].set_title('Quaternion Component Differences')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

    def plot_orientation_comparison(self, df: pd.DataFrame, save_path: str = None):
        """Plot orientation (Roll, Pitch, Yaw) comparison"""
        fig, axes = plt.subplots(3, 2, figsize=(14, 10))
        fig.suptitle('Orientation Comparison: Phone vs ESP32', fontsize=14, fontweight='bold')
        
        time = df['time_sec'] if 'time_sec' in df.columns else df.index
        
        # Roll
        axes[0, 0].plot(time, df['phone_roll'], label='Phone', color='#2196f3', alpha=0.8)
        axes[0, 0].plot(time, df['esp32_roll'], label='ESP32', color='#f44336', alpha=0.8)
        axes[0, 0].set_ylabel('Roll (°)')
        axes[0, 0].set_title('Roll Comparison')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(time, df['diff_roll'], color='#ff9800', linewidth=1)
        axes[0, 1].axhline(y=0, color='green', linestyle='--', alpha=0.5)
        axes[0, 1].axhline(y=5, color='orange', linestyle=':', alpha=0.5)
        axes[0, 1].axhline(y=-5, color='orange', linestyle=':', alpha=0.5)
        axes[0, 1].fill_between(time, -5, 5, alpha=0.1, color='green')
        axes[0, 1].set_ylabel('Δ Roll (°)')
        axes[0, 1].set_title(f'Roll Difference (Mean: {df["diff_roll"].mean():.2f}°, Std: {df["diff_roll"].std():.2f}°)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Pitch
        axes[1, 0].plot(time, df['phone_pitch'], label='Phone', color='#2196f3', alpha=0.8)
        axes[1, 0].plot(time, df['esp32_pitch'], label='ESP32', color='#f44336', alpha=0.8)
        axes[1, 0].set_ylabel('Pitch (°)')
        axes[1, 0].set_title('Pitch Comparison')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(time, df['diff_pitch'], color='#ff9800', linewidth=1)
        axes[1, 1].axhline(y=0, color='green', linestyle='--', alpha=0.5)
        axes[1, 1].axhline(y=5, color='orange', linestyle=':', alpha=0.5)
        axes[1, 1].axhline(y=-5, color='orange', linestyle=':', alpha=0.5)
        axes[1, 1].fill_between(time, -5, 5, alpha=0.1, color='green')
        axes[1, 1].set_ylabel('Δ Pitch (°)')
        axes[1, 1].set_title(f'Pitch Difference (Mean: {df["diff_pitch"].mean():.2f}°, Std: {df["diff_pitch"].std():.2f}°)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Yaw
        axes[2, 0].plot(time, df['phone_yaw'], label='Phone', color='#2196f3', alpha=0.8)
        axes[2, 0].plot(time, df['esp32_yaw'], label='ESP32', color='#f44336', alpha=0.8)
        axes[2, 0].set_ylabel('Yaw (°)')
        axes[2, 0].set_xlabel('Time (s)')
        axes[2, 0].set_title('Yaw Comparison')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        axes[2, 1].plot(time, df['diff_yaw'], color='#ff9800', linewidth=1)
        axes[2, 1].axhline(y=0, color='green', linestyle='--', alpha=0.5)
        axes[2, 1].axhline(y=5, color='orange', linestyle=':', alpha=0.5)
        axes[2, 1].axhline(y=-5, color='orange', linestyle=':', alpha=0.5)
        axes[2, 1].fill_between(time, -5, 5, alpha=0.1, color='green')
        axes[2, 1].set_ylabel('Δ Yaw (°)')
        axes[2, 1].set_xlabel('Time (s)')
        axes[2, 1].set_title(f'Yaw Difference (Mean: {df["diff_yaw"].mean():.2f}°, Std: {df["diff_yaw"].std():.2f}°)')
        axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
    
    def plot_accelerometer_comparison(self, df: pd.DataFrame, save_path: str = None):
        """Plot accelerometer comparison"""
        fig, axes = plt.subplots(3, 2, figsize=(14, 10))
        fig.suptitle('Accelerometer Comparison: Phone vs ESP32', fontsize=14, fontweight='bold')
        
        time = df['time_sec'] if 'time_sec' in df.columns else df.index
        
        for i, axis in enumerate(['X', 'Y', 'Z']):
            phone_col = f'phone_accel{axis}'
            esp32_col = f'esp32_accel{axis}'
            diff_col = f'diff_accel{axis}'
            
            # Comparison plot
            axes[i, 0].plot(time, df[phone_col], label='Phone', color='#2196f3', alpha=0.8)
            axes[i, 0].plot(time, df[esp32_col], label='ESP32', color='#f44336', alpha=0.8)
            axes[i, 0].set_ylabel(f'Accel {axis} (m/s²)')
            axes[i, 0].set_title(f'Accelerometer {axis} Comparison')
            axes[i, 0].legend()
            axes[i, 0].grid(True, alpha=0.3)
            
            # Difference plot
            if diff_col in df.columns:
                axes[i, 1].plot(time, df[diff_col], color='#ff9800', linewidth=1)
                axes[i, 1].axhline(y=0, color='green', linestyle='--', alpha=0.5)
                axes[i, 1].axhline(y=1.0, color='orange', linestyle=':', alpha=0.5)
                axes[i, 1].axhline(y=-1.0, color='orange', linestyle=':', alpha=0.5)
                axes[i, 1].fill_between(time, -1.0, 1.0, alpha=0.1, color='green')
                axes[i, 1].set_ylabel(f'Δ Accel {axis} (m/s²)')
                axes[i, 1].set_title(f'Accel {axis} Difference (Mean: {df[diff_col].mean():.2f} m/s², Std: {df[diff_col].std():.2f} m/s²)')
                axes[i, 1].grid(True, alpha=0.3)
        
        axes[2, 0].set_xlabel('Time (s)')
        axes[2, 1].set_xlabel('Time (s)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

    def plot_gyroscope_comparison(self, df: pd.DataFrame, save_path: str = None):
        """Plot gyroscope comparison with inversion detection"""
        fig, axes = plt.subplots(3, 2, figsize=(14, 10))
        
        # Check if any axes were inverted and update title
        title_suffix = ""
        if hasattr(df, 'attrs') and 'axis_inversions' in df.attrs:
            inverted_axes = [axis for axis, inverted in df.attrs['axis_inversions'].items() if inverted]
            if inverted_axes:
                title_suffix = f" (Inverted: {', '.join(inverted_axes)})"
        
        fig.suptitle(f'Gyroscope Comparison: Phone (rad/s) vs ESP32 (deg/s){title_suffix}', fontsize=14, fontweight='bold')
        
        time = df['time_sec'] if 'time_sec' in df.columns else df.index
        
        for i, axis in enumerate(['X', 'Y', 'Z']):
            phone_col = f'phone_gyro{axis}'
            esp32_col = f'esp32_gyro{axis}'
            diff_col = f'diff_gyro{axis}'
            
            if phone_col not in df.columns or esp32_col not in df.columns:
                continue
                
            # Comparison plot - dual y-axis since different units
            ax1 = axes[i, 0]
            ax2 = ax1.twinx()
            
            # Phone (rad/s) - left y-axis
            line1 = ax1.plot(time, df[phone_col], label='Phone (rad/s)', color='#2196f3', alpha=0.8, linewidth=1.5)
            ax1.set_ylabel(f'Phone Gyro {axis} (rad/s)', color='#2196f3')
            ax1.tick_params(axis='y', labelcolor='#2196f3')
            
            # ESP32 (deg/s) - right y-axis
            line2 = ax2.plot(time, df[esp32_col], label='ESP32 (deg/s)', color='#f44336', alpha=0.8, linewidth=1.5)
            ax2.set_ylabel(f'ESP32 Gyro {axis} (deg/s)', color='#f44336')
            ax2.tick_params(axis='y', labelcolor='#f44336')
            
            # Add inversion indicator to comparison plots
            title_suffix = ""
            if hasattr(df, 'attrs') and 'axis_inversions' in df.attrs:
                if df.attrs['axis_inversions'].get(axis, False):
                    title_suffix = " [INVERTED]"
            
            ax1.set_title(f'Gyroscope {axis} Comparison{title_suffix}')
            ax1.grid(True, alpha=0.3)
            
            # Combine legends
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper left')
            
            # Difference plot
            if diff_col in df.columns:
                axes[i, 1].plot(time, df[diff_col], color='#ff9800', linewidth=1)
                axes[i, 1].axhline(y=0, color='green', linestyle='--', alpha=0.5)
                axes[i, 1].axhline(y=0.087, color='orange', linestyle=':', alpha=0.5)
                axes[i, 1].axhline(y=-0.087, color='orange', linestyle=':', alpha=0.5)
                axes[i, 1].fill_between(time, -0.087, 0.087, alpha=0.1, color='green')
                axes[i, 1].set_ylabel(f'Δ Gyro {axis} (rad/s)')
                axes[i, 1].set_title(f'Gyro {axis} Difference (Mean: {df[diff_col].mean():.4f} rad/s)')
                axes[i, 1].grid(True, alpha=0.3)
        
        axes[2, 0].set_xlabel('Time (s)')
        axes[2, 1].set_xlabel('Time (s)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

    def plot_difference_histograms(self, df: pd.DataFrame, save_path: str = None):
        """Plot histograms of all differences"""
        fig, axes = plt.subplots(4, 3, figsize=(14, 12))
        fig.suptitle('Distribution of Differences (ESP32 - Phone)', fontsize=14, fontweight='bold')
        
        # Orientation differences
        diff_cols = [
            ('diff_roll', 'Roll (°)', 5, axes[0, 0]),
            ('diff_pitch', 'Pitch (°)', 5, axes[0, 1]),
            ('diff_yaw', 'Yaw (°)', 5, axes[0, 2]),
        ]
        
        # Add other sensor differences if available
        if 'diff_accelX' in df.columns:
            diff_cols.extend([
                ('diff_accelX', 'Accel X (m/s²)', 1.0, axes[1, 0]),
                ('diff_accelY', 'Accel Y (m/s²)', 1.0, axes[1, 1]),
                ('diff_accelZ', 'Accel Z (m/s²)', 1.0, axes[1, 2]),
            ])
            
        if 'diff_gyroX' in df.columns:
            diff_cols.extend([
                ('diff_gyroX', 'Gyro X (rad/s)', 0.087, axes[2, 0]),
                ('diff_gyroY', 'Gyro Y (rad/s)', 0.087, axes[2, 1]),
                ('diff_gyroZ', 'Gyro Z (rad/s)', 0.087, axes[2, 2]),
            ])
        
        # Quaternion differences
        if 'diff_quat_angle_deg' in df.columns:
            diff_cols.append(('diff_quat_angle_deg', 'Quat Angle (°)', 10.0, axes[3, 0]))
        
        # Hide unused subplots
        total_plots = len(diff_cols)
        total_subplots = 12
        for i in range(total_plots, total_subplots):
            row = i // 3
            col = i % 3
            if row < 4:
                axes[row, col].set_visible(False)
        
        for col, label, threshold, ax in diff_cols:
            if col in df.columns:
                data = df[col].dropna()
                
                if len(data) == 0:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                    continue
                
                # Calculate statistics
                mean = data.mean()
                std = data.std()
                within_threshold = (data.abs() < threshold).sum() / len(data) * 100
                
                # Plot histogram
                ax.hist(data, bins=min(50, len(data)//2), color='#ff9800', alpha=0.7, edgecolor='white')
                ax.axvline(x=0, color='green', linestyle='--', linewidth=2, label='Ideal (0)')
                ax.axvline(x=mean, color='red', linestyle='-', linewidth=2, label=f'Mean ({mean:.2f})')
                ax.axvline(x=threshold, color='orange', linestyle=':', alpha=0.7)
                ax.axvline(x=-threshold, color='orange', linestyle=':', alpha=0.7)
                
                ax.set_xlabel(f'Δ {label}')
                ax.set_ylabel('Count')
                ax.set_title(f'{label}\nMean: {mean:.2f}, Std: {std:.2f}\n{within_threshold:.1f}% within ±{threshold}')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

    def generate_comprehensive_report(self, df: pd.DataFrame, save_path: str = None):
        """Generate comprehensive sensor comparison report"""
        summary_lines = []
        
        summary_lines.append("=" * 80)
        summary_lines.append("COMPREHENSIVE IMU SENSOR COMPARISON REPORT")
        summary_lines.append("ESP32 vs Smartphone Sensor Analysis with Quaternion Data")
        summary_lines.append("=" * 80)
        summary_lines.append("")
        summary_lines.append(f"📊 Dataset Information:")
        summary_lines.append(f"   Total Samples: {len(df):,}")
        if 'time_sec' in df.columns:
            duration = df['time_sec'].max()
            sampling_rate = len(df) / duration if duration > 0 else 0
            summary_lines.append(f"   Duration: {duration:.1f} seconds")
            summary_lines.append(f"   Average Sampling Rate: {sampling_rate:.1f} Hz")
        
        if 'timestamp_ms' in df.columns:
            summary_lines.append(f"   Timestamp: {pd.to_datetime(df['timestamp_ms'].iloc[0], unit='ms').strftime('%Y-%m-%d %H:%M:%S')} to {pd.to_datetime(df['timestamp_ms'].iloc[-1], unit='ms').strftime('%Y-%m-%d %H:%M:%S')}")
        summary_lines.append("")

        # Add analysis sections for each sensor type...
        
        # Quaternion Analysis
        if 'diff_quat_angle_deg' in df.columns:
            summary_lines.append("🔄 QUATERNION ANALYSIS:")
            summary_lines.append("-" * 40)
            quat_angle_data = df['diff_quat_angle_deg'].dropna()
            quat_mean = quat_angle_data.mean()
            quat_std = quat_angle_data.std()
            quat_abs_mean = quat_angle_data.abs().mean()
            quat_max = quat_angle_data.abs().max()
            within_5deg = (quat_angle_data.abs() < 5).sum() / len(quat_angle_data) * 100
            within_10deg = (quat_angle_data.abs() < 10).sum() / len(quat_angle_data) * 100
            
            summary_lines.append(f"  Quaternion Angle Difference:")
            summary_lines.append(f"    Mean: {quat_mean:7.2f}° | Std: {quat_std:6.2f}° | Abs Mean: {quat_abs_mean:6.2f}° | Max: {quat_max:6.2f}°")
            summary_lines.append(f"    Within ±5°:  {within_5deg:5.1f}%")
            summary_lines.append(f"    Within ±10°: {within_10deg:5.1f}%")
            summary_lines.append("")
        
        # Save report
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(summary_lines))

def main():
    # Create the main window
    root = tk.Tk()
    
    # Create the application
    app = IMUComparisonGUI(root)
    
    # Start the GUI event loop
    root.mainloop()

if __name__ == "__main__":
    main()