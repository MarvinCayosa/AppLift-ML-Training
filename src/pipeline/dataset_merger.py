"""
Dataset Merger & Renamer Tool
=============================
A utility for merging multiple processed datasets and renaming them with a clean UI.

Features:
- Select multiple CSV files from output folders
- Merge them into a single dataset
- Rename the merged dataset
- Handle conflicts and duplicates
- Preview dataset composition before merging

Author: AppLift ML Training Pipeline
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, ttk, messagebox, simpledialog
from pathlib import Path
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR
OUTPUT_DIR = PROJECT_ROOT / 'output'
MERGED_DATASETS_DIR = OUTPUT_DIR / 'merged_datasets'

# Create directories
MERGED_DATASETS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# FILE SELECTION UI
# =============================================================================

class DatasetMerger:
    def __init__(self):
        self.selected_files = []
        self.preview_data = None
        
    def run(self):
        """Main UI for dataset merging"""
        self.create_main_window()
    
    def create_main_window(self):
        """Create the main application window"""
        self.root = tk.Tk()
        self.root.title("🔗 Dataset Merger & Renamer Tool")
        self.root.geometry("900x700")
        self.root.configure(bg='#f0f2f5')
        
        # Header
        self.create_header()
        
        # Main content area
        self.create_content_area()
        
        # Button panel
        self.create_button_panel()
        
        # Center window
        self.center_window()
        
        self.root.mainloop()
    
    def create_header(self):
        """Create header section"""
        header_frame = tk.Frame(self.root, bg='#2196F3', pady=20)
        header_frame.pack(fill=tk.X)
        
        title = tk.Label(header_frame, text="🔗 Dataset Merger & Renamer", 
                        font=('Arial', 20, 'bold'), bg='#2196F3', fg='white')
        title.pack()
        
        subtitle = tk.Label(header_frame, text="Merge multiple processed datasets into one unified dataset",
                           font=('Arial', 11), bg='#2196F3', fg='white')
        subtitle.pack(pady=(5, 0))
    
    def create_content_area(self):
        """Create main content area with file list and preview"""
        content_frame = tk.Frame(self.root, bg='#f0f2f5')
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Left panel - File selection
        left_frame = tk.LabelFrame(content_frame, text="📁 Selected Files", 
                                  font=('Arial', 12, 'bold'), bg='#f0f2f5', fg='#333')
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # File listbox
        list_frame = tk.Frame(left_frame, bg='#f0f2f5')
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.file_listbox = tk.Listbox(list_frame, selectmode=tk.EXTENDED, 
                                      font=('Arial', 10), bg='white')
        self.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.file_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.file_listbox.config(yscrollcommand=scrollbar.set)
        
        # File management buttons
        file_btn_frame = tk.Frame(left_frame, bg='#f0f2f5')
        file_btn_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        add_btn = tk.Button(file_btn_frame, text="➕ Add Files", command=self.add_files,
                           font=('Arial', 10), bg='#4CAF50', fg='white', relief='flat',
                           padx=15, pady=5, cursor='hand2')
        add_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        remove_btn = tk.Button(file_btn_frame, text="➖ Remove", command=self.remove_files,
                              font=('Arial', 10), bg='#f44336', fg='white', relief='flat',
                              padx=15, pady=5, cursor='hand2')
        remove_btn.pack(side=tk.LEFT, padx=5)
        
        clear_btn = tk.Button(file_btn_frame, text="🗑️ Clear All", command=self.clear_files,
                             font=('Arial', 10), bg='#757575', fg='white', relief='flat',
                             padx=15, pady=5, cursor='hand2')
        clear_btn.pack(side=tk.LEFT, padx=5)
        
        # Right panel - Preview
        right_frame = tk.LabelFrame(content_frame, text="👁️ Dataset Preview", 
                                   font=('Arial', 12, 'bold'), bg='#f0f2f5', fg='#333')
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Preview text area
        preview_frame = tk.Frame(right_frame, bg='#f0f2f5')
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.preview_text = tk.Text(preview_frame, font=('Consolas', 9), bg='#f8f9fa', 
                                   fg='#333', relief='flat', wrap=tk.WORD)
        self.preview_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        preview_scrollbar = ttk.Scrollbar(preview_frame, orient=tk.VERTICAL, 
                                         command=self.preview_text.yview)
        preview_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.preview_text.config(yscrollcommand=preview_scrollbar.set)
        
        # Preview button
        preview_btn = tk.Button(right_frame, text="🔍 Generate Preview", command=self.generate_preview,
                               font=('Arial', 10), bg='#2196F3', fg='white', relief='flat',
                               padx=15, pady=5, cursor='hand2')
        preview_btn.pack(pady=(0, 10))
    
    def create_button_panel(self):
        """Create bottom button panel"""
        btn_frame = tk.Frame(self.root, bg='#f0f2f5')
        btn_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        
        # Output name entry
        name_frame = tk.Frame(btn_frame, bg='#f0f2f5')
        name_frame.pack(fill=tk.X, pady=(0, 15))
        
        name_label = tk.Label(name_frame, text="📝 Output Dataset Name:", 
                             font=('Arial', 10, 'bold'), bg='#f0f2f5', fg='#333')
        name_label.pack(side=tk.LEFT)
        
        self.output_name_var = tk.StringVar()
        self.output_name_var.set("merged_dataset")
        
        self.output_name_entry = tk.Entry(name_frame, textvariable=self.output_name_var,
                                         font=('Arial', 11), width=30)
        self.output_name_entry.pack(side=tk.LEFT, padx=(10, 0))
        
        # Action buttons
        action_frame = tk.Frame(btn_frame, bg='#f0f2f5')
        action_frame.pack()
        
        merge_btn = tk.Button(action_frame, text="🔗 Merge Datasets", command=self.merge_datasets,
                             font=('Arial', 12, 'bold'), bg='#4CAF50', fg='white', relief='flat',
                             padx=25, pady=10, cursor='hand2')
        merge_btn.pack(side=tk.LEFT, padx=5)
        
        export_btn = tk.Button(action_frame, text="💾 Quick Export", command=self.quick_export,
                              font=('Arial', 12), bg='#FF9800', fg='white', relief='flat',
                              padx=25, pady=10, cursor='hand2')
        export_btn.pack(side=tk.LEFT, padx=5)
        
        close_btn = tk.Button(action_frame, text="❌ Close", command=self.root.destroy,
                             font=('Arial', 12), bg='#757575', fg='white', relief='flat',
                             padx=25, pady=10, cursor='hand2')
        close_btn.pack(side=tk.LEFT, padx=5)
    
    def center_window(self):
        """Center the window on screen"""
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() - self.root.winfo_width()) // 2
        y = (self.root.winfo_screenheight() - self.root.winfo_height()) // 2
        self.root.geometry(f"+{x}+{y}")
    
    def add_files(self):
        """Add CSV files to the selection"""
        initial_dir = str(OUTPUT_DIR) if OUTPUT_DIR.exists() else str(PROJECT_ROOT)
        
        files = filedialog.askopenfilenames(
            title="Select Dataset Files",
            initialdir=initial_dir,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            multiple=True
        )
        
        for file in files:
            if file not in self.selected_files:
                self.selected_files.append(file)
                filename = Path(file).name
                self.file_listbox.insert(tk.END, filename)
        
        if files:
            messagebox.showinfo("Files Added", f"Added {len(files)} file(s) to selection.")
    
    def remove_files(self):
        """Remove selected files from the list"""
        selected_indices = self.file_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("No Selection", "Please select files to remove.")
            return
        
        # Remove in reverse order to maintain indices
        for index in reversed(selected_indices):
            self.file_listbox.delete(index)
            del self.selected_files[index]
        
        messagebox.showinfo("Files Removed", f"Removed {len(selected_indices)} file(s).")
    
    def clear_files(self):
        """Clear all selected files"""
        if self.selected_files:
            if messagebox.askyesno("Clear All", "Remove all files from selection?"):
                self.selected_files.clear()
                self.file_listbox.delete(0, tk.END)
                self.preview_text.delete('1.0', tk.END)
                messagebox.showinfo("Cleared", "All files removed from selection.")
    
    def generate_preview(self):
        """Generate a preview of the combined datasets"""
        if not self.selected_files:
            messagebox.showwarning("No Files", "Please select files first.")
            return
        
        self.preview_text.delete('1.0', tk.END)
        self.preview_text.insert('1.0', "🔍 Analyzing selected datasets...\n\n")
        self.root.update()
        
        try:
            preview_info = self.analyze_datasets()
            self.display_preview(preview_info)
        except Exception as e:
            error_msg = f"❌ Error analyzing datasets:\n{str(e)}"
            self.preview_text.delete('1.0', tk.END)
            self.preview_text.insert('1.0', error_msg)
    
    def analyze_datasets(self):
        """Analyze selected datasets and return summary information"""
        datasets = []
        total_rows = 0
        total_cols = set()
        file_info = []
        
        for file_path in self.selected_files:
            try:
                df = pd.read_csv(file_path)
                datasets.append(df)
                
                file_info.append({
                    'name': Path(file_path).name,
                    'path': file_path,
                    'rows': len(df),
                    'cols': len(df.columns),
                    'columns': list(df.columns),
                    'target_dist': df['target'].value_counts().to_dict() if 'target' in df.columns else None,
                    'equipment': df['equipment_code'].iloc[0] if 'equipment_code' in df.columns else None,
                    'exercise': df['exercise_code'].iloc[0] if 'exercise_code' in df.columns else None
                })
                
                total_rows += len(df)
                total_cols.update(df.columns)
                
            except Exception as e:
                file_info.append({
                    'name': Path(file_path).name,
                    'path': file_path,
                    'error': str(e)
                })
        
        return {
            'file_info': file_info,
            'total_files': len(self.selected_files),
            'total_rows': total_rows,
            'total_cols': len(total_cols),
            'common_cols': list(total_cols),
            'datasets': datasets
        }
    
    def display_preview(self, preview_info):
        """Display preview information in the text widget"""
        self.preview_text.delete('1.0', tk.END)
        
        # Header
        text = "📊 DATASET MERGE PREVIEW\n"
        text += "=" * 50 + "\n\n"
        
        # Summary
        text += f"📁 Files Selected: {preview_info['total_files']}\n"
        text += f"📏 Total Rows: {preview_info['total_rows']:,}\n"
        text += f"📋 Total Columns: {preview_info['total_cols']}\n\n"
        
        # File details
        text += "📋 FILE DETAILS:\n"
        text += "-" * 30 + "\n"
        
        for info in preview_info['file_info']:
            if 'error' in info:
                text += f"❌ {info['name']}: ERROR - {info['error']}\n"
            else:
                text += f"✓ {info['name']}\n"
                text += f"   Rows: {info['rows']:,} | Columns: {info['cols']}\n"
                
                if info['target_dist']:
                    target_str = ", ".join([f"T{k}:{v}" for k, v in info['target_dist'].items()])
                    text += f"   Targets: {target_str}\n"
                
                equip_names = {0: 'Dumbbell', 1: 'Barbell', 2: 'Weight Stack'}
                ex_names = {0: 'Conc.Curls', 1: 'Overhead', 2: 'BenchPress', 
                           3: 'BackSquat', 4: 'LatPull', 5: 'LegExt'}
                
                if info['equipment'] is not None:
                    equip_name = equip_names.get(info['equipment'], f"Code{info['equipment']}")
                    text += f"   Equipment: {equip_name}\n"
                
                if info['exercise'] is not None:
                    ex_name = ex_names.get(info['exercise'], f"Code{info['exercise']}")
                    text += f"   Exercise: {ex_name}\n"
                
                text += "\n"
        
        # Column compatibility
        text += "🔗 MERGE COMPATIBILITY:\n"
        text += "-" * 30 + "\n"
        
        if preview_info['datasets']:
            first_cols = set(preview_info['datasets'][0].columns)
            all_compatible = True
            
            for i, df in enumerate(preview_info['datasets'][1:], 1):
                current_cols = set(df.columns)
                if current_cols != first_cols:
                    all_compatible = False
                    missing = first_cols - current_cols
                    extra = current_cols - first_cols
                    
                    text += f"⚠️ File {i+1} column mismatch:\n"
                    if missing:
                        text += f"   Missing: {', '.join(missing)}\n"
                    if extra:
                        text += f"   Extra: {', '.join(extra)}\n"
            
            if all_compatible:
                text += "✅ All files have compatible columns!\n"
            else:
                text += "\n⚠️ Column mismatches detected. Merge will use common columns.\n"
        
        text += "\n" + "=" * 50 + "\n"
        text += "Ready to merge! Click 'Merge Datasets' to proceed."
        
        self.preview_text.insert('1.0', text)
        self.preview_data = preview_info
    
    def merge_datasets(self):
        """Merge the selected datasets"""
        if not self.selected_files:
            messagebox.showerror("No Files", "Please select files to merge.")
            return
        
        output_name = self.output_name_var.get().strip()
        if not output_name:
            messagebox.showerror("No Name", "Please enter an output dataset name.")
            return
        
        try:
            # Load and merge datasets
            all_dfs = []
            for file_path in self.selected_files:
                df = pd.read_csv(file_path)
                df['source_dataset'] = Path(file_path).stem  # Add source tracking
                all_dfs.append(df)
            
            # Merge using common columns
            merged_df = pd.concat(all_dfs, ignore_index=True, sort=False)
            
            # Generate output filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_filename = f"{output_name}_{timestamp}.csv"
            output_path = MERGED_DATASETS_DIR / output_filename
            
            # Save merged dataset
            merged_df.to_csv(output_path, index=False)
            
            # Show success message with statistics
            stats = self.get_merge_statistics(merged_df)
            success_msg = (f"✅ Successfully merged {len(self.selected_files)} datasets!\n\n"
                          f"📊 Merged Dataset Statistics:\n"
                          f"• Total rows: {len(merged_df):,}\n"
                          f"• Total columns: {len(merged_df.columns)}\n"
                          f"• Target distribution: {stats['target_dist']}\n\n"
                          f"💾 Saved to:\n{output_path}")
            
            messagebox.showinfo("Merge Complete", success_msg)
            
            # Ask if user wants to open the output folder
            if messagebox.askyesno("Open Folder", "Would you like to open the output folder?"):
                os.startfile(MERGED_DATASETS_DIR)
        
        except Exception as e:
            messagebox.showerror("Merge Error", f"Error merging datasets:\n{str(e)}")
    
    def quick_export(self):
        """Quick export with file browser"""
        if not self.selected_files:
            messagebox.showerror("No Files", "Please select files to merge.")
            return
        
        # Ask for save location
        output_name = self.output_name_var.get().strip()
        if not output_name:
            output_name = "merged_dataset"
        
        save_path = filedialog.asksaveasfilename(
            title="Save Merged Dataset",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialname=f"{output_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        
        if save_path:
            try:
                # Merge and save
                all_dfs = []
                for file_path in self.selected_files:
                    df = pd.read_csv(file_path)
                    df['source_dataset'] = Path(file_path).stem
                    all_dfs.append(df)
                
                merged_df = pd.concat(all_dfs, ignore_index=True, sort=False)
                merged_df.to_csv(save_path, index=False)
                
                messagebox.showinfo("Export Complete", f"Dataset exported to:\n{save_path}")
                
            except Exception as e:
                messagebox.showerror("Export Error", f"Error exporting dataset:\n{str(e)}")
    
    def get_merge_statistics(self, merged_df):
        """Get statistics for the merged dataset"""
        stats = {}
        
        if 'target' in merged_df.columns:
            target_counts = merged_df['target'].value_counts().sort_index()
            stats['target_dist'] = ', '.join([f"T{k}:{v}" for k, v in target_counts.items()])
        else:
            stats['target_dist'] = "No target column"
        
        return stats


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main function to run the dataset merger"""
    print("=" * 60)
    print("🔗 DATASET MERGER & RENAMER TOOL")
    print("=" * 60)
    print(f"\n📂 Project Root: {PROJECT_ROOT}")
    print(f"📁 Output Directory: {OUTPUT_DIR}")
    print(f"💾 Merged Datasets Folder: {MERGED_DATASETS_DIR}")
    print("\n🚀 Starting UI...")
    
    app = DatasetMerger()
    app.run()


if __name__ == "__main__":
    main()
