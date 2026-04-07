"""
Interactive Axis Mapping GUI
Allows easy experimentation with different gyroscope axis mappings
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import numpy as np
from pathlib import Path
import subprocess
import sys
import os

class AxisMappingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("IMU Axis Mapping Configuration")
        self.root.geometry("800x600")
        
        # Data storage
        self.df = None
        self.csv_path = None
        
        # Current mapping configuration
        self.mapping_config = {
            'X': {'esp32_axis': 'Y', 'invert': True},
            'Y': {'esp32_axis': 'Z', 'invert': False},
            'Z': {'esp32_axis': 'X', 'invert': True}
        }
        
        self.create_widgets()
        
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(4, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="IMU Gyroscope Axis Mapping Configuration", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # File selection
        file_frame = ttk.LabelFrame(main_frame, text="1. Select CSV File", padding="10")
        file_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        file_frame.columnconfigure(1, weight=1)
        
        ttk.Label(file_frame, text="CSV File:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.file_var = tk.StringVar()
        self.file_entry = ttk.Entry(file_frame, textvariable=self.file_var, state="readonly")
        self.file_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        
        ttk.Button(file_frame, text="Browse", command=self.browse_file).grid(row=0, column=2)
        
        # Axis mapping configuration
        mapping_frame = ttk.LabelFrame(main_frame, text="2. Configure Axis Mapping", padding="10")
        mapping_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Headers
        ttk.Label(mapping_frame, text="Phone Axis", font=('Arial', 10, 'bold')).grid(row=0, column=0, padx=10, pady=5)
        ttk.Label(mapping_frame, text="↔", font=('Arial', 12, 'bold')).grid(row=0, column=1, padx=10, pady=5)
        ttk.Label(mapping_frame, text="ESP32 Axis", font=('Arial', 10, 'bold')).grid(row=0, column=2, padx=10, pady=5)
        ttk.Label(mapping_frame, text="Invert", font=('Arial', 10, 'bold')).grid(row=0, column=3, padx=10, pady=5)
        ttk.Label(mapping_frame, text="Correlation", font=('Arial', 10, 'bold')).grid(row=0, column=4, padx=10, pady=5)
        
        # Axis mapping controls
        self.axis_combos = {}
        self.invert_vars = {}
        self.corr_labels = {}
        
        for i, phone_axis in enumerate(['X', 'Y', 'Z']):
            row = i + 1
            
            # Phone axis label
            ttk.Label(mapping_frame, text=f"Phone {phone_axis}", 
                     font=('Arial', 10, 'bold')).grid(row=row, column=0, padx=10, pady=5)
            
            # Arrow
            ttk.Label(mapping_frame, text="↔").grid(row=row, column=1, padx=10, pady=5)
            
            # ESP32 axis dropdown
            self.axis_combos[phone_axis] = ttk.Combobox(mapping_frame, values=['X', 'Y', 'Z'], 
                                                       state="readonly", width=8)
            self.axis_combos[phone_axis].set(self.mapping_config[phone_axis]['esp32_axis'])
            self.axis_combos[phone_axis].grid(row=row, column=2, padx=10, pady=5)
            self.axis_combos[phone_axis].bind('<<ComboboxSelected>>', self.on_mapping_change)
            
            # Invert checkbox
            self.invert_vars[phone_axis] = tk.BooleanVar(value=self.mapping_config[phone_axis]['invert'])
            invert_check = ttk.Checkbutton(mapping_frame, variable=self.invert_vars[phone_axis], 
                                         command=self.on_mapping_change)
            invert_check.grid(row=row, column=3, padx=10, pady=5)
            
            # Correlation display
            self.corr_labels[phone_axis] = ttk.Label(mapping_frame, text="---", 
                                                    font=('Arial', 10))
            self.corr_labels[phone_axis].grid(row=row, column=4, padx=10, pady=5)
        
        # Test button
        test_frame = ttk.LabelFrame(main_frame, text="3. Test Current Mapping", padding="10")
        test_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        test_buttons_frame = ttk.Frame(test_frame)
        test_buttons_frame.grid(row=0, column=0)
        
        ttk.Button(test_buttons_frame, text="🧪 Test Correlations", 
                  command=self.test_correlations).grid(row=0, column=0, padx=(0, 10))
        
        ttk.Button(test_buttons_frame, text="🚀 Run Analysis", 
                  command=self.run_analysis).grid(row=0, column=1, padx=(0, 10))
        
        ttk.Button(test_buttons_frame, text="🔄 Show All Combinations", 
                  command=self.show_all_combinations).grid(row=0, column=2)
        
        # Results area
        results_frame = ttk.LabelFrame(main_frame, text="Results", padding="10")
        results_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
        # Text area with scrollbar
        text_frame = ttk.Frame(results_frame)
        text_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)
        
        self.results_text = tk.Text(text_frame, height=15, wrap=tk.WORD, 
                                   font=('Consolas', 9))
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready - Select a CSV file to begin")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
    def browse_file(self):
        """Browse for CSV file"""
        filetypes = [
            ('CSV files', '*.csv'),
            ('All files', '*.*')
        ]
        
        filename = filedialog.askopenfilename(
            title="Select IMU Comparison CSV File",
            filetypes=filetypes,
            initialdir=str(Path.cwd())
        )
        
        if filename:
            self.csv_path = filename
            self.file_var.set(filename)
            self.load_data()
            
    def load_data(self):
        """Load CSV data"""
        try:
            self.df = pd.read_csv(self.csv_path)
            self.log_output(f"✅ Loaded {len(self.df)} samples from {Path(self.csv_path).name}")
            
            # Check required columns
            required_cols = ['phone_gyroX_rad', 'phone_gyroY_rad', 'phone_gyroZ_rad',
                           'esp32_gyroX_deg', 'esp32_gyroY_deg', 'esp32_gyroZ_deg']
            missing_cols = [col for col in required_cols if col not in self.df.columns]
            
            if missing_cols:
                self.log_output(f"⚠️  Missing columns: {', '.join(missing_cols)}")
            else:
                self.log_output("✅ All required gyroscope columns found")
                self.test_correlations()
                
            self.status_var.set("Data loaded - Ready to test axis mappings")
            
        except Exception as e:
            self.log_output(f"❌ Error loading file: {str(e)}")
            self.status_var.set("Error loading file")
            
    def on_mapping_change(self, event=None):
        """Called when mapping configuration changes"""
        # Update internal config
        for phone_axis in ['X', 'Y', 'Z']:
            self.mapping_config[phone_axis]['esp32_axis'] = self.axis_combos[phone_axis].get()
            self.mapping_config[phone_axis]['invert'] = self.invert_vars[phone_axis].get()
        
        # Test correlations if data is loaded
        if self.df is not None:
            self.test_correlations()
            
    def test_correlations(self):
        """Test correlations for current mapping"""
        if self.df is None:
            return
            
        try:
            for phone_axis in ['X', 'Y', 'Z']:
                phone_col = f'phone_gyro{phone_axis}_rad'
                esp32_axis = self.mapping_config[phone_axis]['esp32_axis']
                esp32_col = f'esp32_gyro{esp32_axis}_deg'
                invert = self.mapping_config[phone_axis]['invert']
                
                if phone_col in self.df.columns and esp32_col in self.df.columns:
                    # Get data
                    phone_data = self.df[phone_col].dropna()
                    esp32_data = self.df[esp32_col].interpolate(method='linear', limit_direction='both').dropna()
                    esp32_rad = np.radians(esp32_data)
                    
                    if invert:
                        esp32_rad = -esp32_rad
                    
                    # Calculate correlation
                    if len(phone_data) == len(esp32_rad):
                        corr = np.corrcoef(phone_data, esp32_rad)[0, 1]
                        
                        # Update correlation display
                        corr_text = f"{corr:.3f}"
                        if abs(corr) > 0.9:
                            corr_text += " ⭐"
                        elif abs(corr) > 0.7:
                            corr_text += " ✓"
                        elif abs(corr) < 0.3:
                            corr_text += " ⚠️"
                            
                        self.corr_labels[phone_axis].config(text=corr_text)
                        
            self.status_var.set("Correlations updated")
            
        except Exception as e:
            self.log_output(f"❌ Error testing correlations: {str(e)}")
            
    def show_all_combinations(self):
        """Show all possible axis combinations and their correlations"""
        if self.df is None:
            messagebox.showwarning("No Data", "Please load a CSV file first")
            return
            
        self.log_output("\n🔍 TESTING ALL AXIS COMBINATIONS...")
        self.log_output("=" * 70)
        
        results = []
        
        for phone_axis in ['X', 'Y', 'Z']:
            phone_col = f'phone_gyro{phone_axis}_rad'
            if phone_col not in self.df.columns:
                continue
                
            phone_data = self.df[phone_col].dropna()
            
            for esp32_axis in ['X', 'Y', 'Z']:
                esp32_col = f'esp32_gyro{esp32_axis}_deg'
                if esp32_col not in self.df.columns:
                    continue
                    
                esp32_data = self.df[esp32_col].interpolate(method='linear', limit_direction='both').dropna()
                esp32_rad = np.radians(esp32_data)
                
                # Test both normal and inverted
                if len(phone_data) == len(esp32_rad):
                    corr_normal = np.corrcoef(phone_data, esp32_rad)[0, 1]
                    corr_inverted = np.corrcoef(phone_data, -esp32_rad)[0, 1]
                    
                    results.append((phone_axis, esp32_axis, False, corr_normal))
                    results.append((phone_axis, esp32_axis, True, corr_inverted))
        
        # Sort by absolute correlation (best first)
        results.sort(key=lambda x: abs(x[3]), reverse=True)
        
        self.log_output("📊 CORRELATION RESULTS (sorted by strength):")
        self.log_output("-" * 70)
        self.log_output("Phone → ESP32   │ Invert │ Correlation   │ Rating")
        self.log_output("-" * 70)
        
        for phone_axis, esp32_axis, invert, corr in results:
            invert_str = "Yes" if invert else "No "
            abs_corr = abs(corr)
            if abs_corr > 0.9:
                rating = "⭐ Excellent"
            elif abs_corr > 0.7:
                rating = "✓ Good     "
            elif abs_corr > 0.5:
                rating = "~ Fair     "
            else:
                rating = "⚠️ Poor     "
            
            self.log_output(f"  {phone_axis}   →   {esp32_axis}    │   {invert_str}   │   {corr:7.3f}   │ {rating}")
        
        self.log_output("-" * 70)
        self.log_output("\n💡 TIP: Double-click on a result to apply that mapping!")
        
    def run_analysis(self):
        """Run full analysis with current mapping"""
        if self.df is None:
            messagebox.showwarning("No Data", "Please load a CSV file first")
            return
            
        try:
            # Update the main script's mapping config
            self.update_script_config()
            
            # Run the analysis
            self.log_output(f"\n🚀 Running analysis with current mapping...")
            self.status_var.set("Running analysis...")
            
            # Build command
            python_exe = sys.executable
            script_path = Path(__file__).parent / "visualize_imu_comparison.py"
            
            cmd = [python_exe, str(script_path), self.csv_path]
            
            # Run subprocess
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(Path(__file__).parent))
            
            if result.returncode == 0:
                self.log_output("✅ Analysis completed successfully!")
                self.log_output("Check the visualizations folder for results")
            else:
                self.log_output(f"❌ Analysis failed:")
                self.log_output(result.stderr)
                
            self.status_var.set("Analysis complete")
            
        except Exception as e:
            self.log_output(f"❌ Error running analysis: {str(e)}")
            self.status_var.set("Error running analysis")
            
    def update_script_config(self):
        """Update the main script's axis mapping configuration"""
        try:
            script_path = Path(__file__).parent / "visualize_imu_comparison.py"
            
            # Read current script
            with open(script_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Build new config string
            new_config = "AXIS_MAPPING_CONFIG = {\n"
            for phone_axis in ['X', 'Y', 'Z']:
                esp32_axis = self.mapping_config[phone_axis]['esp32_axis']
                invert = self.mapping_config[phone_axis]['invert']
                comment = f"Phone {phone_axis} ↔ ESP32 {esp32_axis}"
                if invert:
                    comment += " (inverted)"
                new_config += f"    '{phone_axis}': {{'esp32_axis': '{esp32_axis}', 'invert': {invert}}},   # {comment}\\n"
            new_config += "}"
            
            # Replace in content
            import re
            pattern = r'AXIS_MAPPING_CONFIG = \{[^}]*\}'
            content = re.sub(pattern, new_config, content, flags=re.DOTALL)
            
            # Write back
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(content)
                
            self.log_output("📝 Updated script configuration")
            
        except Exception as e:
            self.log_output(f"❌ Error updating script: {str(e)}")
            
    def log_output(self, message):
        """Add message to results text area"""
        self.results_text.insert(tk.END, message + "\n")
        self.results_text.see(tk.END)
        self.root.update_idletasks()

def main():
    root = tk.Tk()
    app = AxisMappingGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()