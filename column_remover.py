#!/usr/bin/env python3
"""
Column Remover Tool
==================
Remove specified columns from CSV files (Participant and Quality Code columns).
Useful for creating anonymized datasets or removing target variables for prediction.
"""

import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
import os

# Project root for file operations
PROJECT_ROOT = Path(__file__).parent

class ColumnRemoverApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Column Remover Tool")
        self.root.geometry("700x500")
        self.root.configure(bg='#f0f0f0')
        
        # Data
        self.df = None
        self.current_file = None
        self.columns_to_remove = ['participant', 'quality_code', 'target']  # Default columns
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the user interface"""
        # Title
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=60)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)
        
        tk.Label(title_frame, text="🗑️ CSV Column Remover", 
                font=('Arial', 16, 'bold'), bg='#2c3e50', fg='white').pack(pady=15)
        
        # Main content frame
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # File selection frame
        file_frame = tk.LabelFrame(main_frame, text="File Selection", 
                                  font=('Arial', 12, 'bold'), bg='#f0f0f0')
        file_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.file_var = tk.StringVar(value="No file selected")
        tk.Label(file_frame, textvariable=self.file_var, 
                font=('Arial', 10), bg='#f0f0f0', wraplength=500).pack(pady=10)
        
        tk.Button(file_frame, text="📁 Select CSV File", command=self.select_file,
                 font=('Arial', 11), bg='#3498db', fg='white', 
                 padx=20, pady=8, cursor='hand2').pack(pady=5)
        
        # Columns frame
        cols_frame = tk.LabelFrame(main_frame, text="Columns to Remove", 
                                  font=('Arial', 12, 'bold'), bg='#f0f0f0')
        cols_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        # Left side: Available columns
        left_frame = tk.Frame(cols_frame, bg='#f0f0f0')
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 5), pady=10)
        
        tk.Label(left_frame, text="Available Columns:", font=('Arial', 11, 'bold'),
                bg='#f0f0f0').pack(anchor=tk.W)
        
        self.available_listbox = tk.Listbox(left_frame, selectmode=tk.MULTIPLE,
                                          font=('Arial', 10), height=8)
        self.available_listbox.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        
        # Control buttons frame
        control_frame = tk.Frame(cols_frame, bg='#f0f0f0')
        control_frame.pack(side=tk.LEFT, padx=10, pady=40)
        
        tk.Button(control_frame, text="➡️\nAdd", command=self.add_columns,
                 font=('Arial', 10), bg='#27ae60', fg='white', 
                 padx=15, pady=10, cursor='hand2').pack(pady=5)
        
        tk.Button(control_frame, text="⬅️\nRemove", command=self.remove_columns,
                 font=('Arial', 10), bg='#e74c3c', fg='white', 
                 padx=15, pady=10, cursor='hand2').pack(pady=5)
        
        tk.Button(control_frame, text="🔄\nReset", command=self.reset_columns,
                 font=('Arial', 10), bg='#f39c12', fg='white', 
                 padx=15, pady=10, cursor='hand2').pack(pady=5)
        
        # Right side: Columns to remove
        right_frame = tk.Frame(cols_frame, bg='#f0f0f0')
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 10), pady=10)
        
        tk.Label(right_frame, text="Columns to Remove:", font=('Arial', 11, 'bold'),
                bg='#f0f0f0', fg='#e74c3c').pack(anchor=tk.W)
        
        self.remove_listbox = tk.Listbox(right_frame, selectmode=tk.MULTIPLE,
                                       font=('Arial', 10), height=8)
        self.remove_listbox.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        
        # Action buttons
        action_frame = tk.Frame(main_frame, bg='#f0f0f0')
        action_frame.pack(fill=tk.X)
        
        tk.Button(action_frame, text="🔍 Preview Changes", command=self.preview_changes,
                 font=('Arial', 11), bg='#9b59b6', fg='white', 
                 padx=20, pady=8, cursor='hand2').pack(side=tk.LEFT, padx=(0, 10))
        
        tk.Button(action_frame, text="💾 Save Cleaned CSV", command=self.save_cleaned_csv,
                 font=('Arial', 11, 'bold'), bg='#27ae60', fg='white', 
                 padx=20, pady=8, cursor='hand2').pack(side=tk.LEFT)
        
        # Status
        self.status_var = tk.StringVar(value="Select a CSV file to start")
        status_label = tk.Label(main_frame, textvariable=self.status_var,
                               font=('Arial', 10, 'italic'), bg='#f0f0f0', fg='#7f8c8d')
        status_label.pack(pady=(10, 0))
        
        # Initialize with default columns
        self.update_remove_listbox()
    
    def select_file(self):
        """Select a CSV file to process"""
        file_path = filedialog.askopenfilename(
            title="Select CSV File",
            initialdir=str(PROJECT_ROOT),
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            self.df = pd.read_csv(file_path)
            self.current_file = file_path
            
            # Update UI
            file_name = Path(file_path).name
            self.file_var.set(f"📄 {file_name} ({len(self.df)} rows, {len(self.df.columns)} columns)")
            
            # Update available columns
            self.update_available_columns()
            
            self.status_var.set(f"Loaded {len(self.df)} rows with {len(self.df.columns)} columns")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV file:\n{e}")
            self.status_var.set("Error loading file")
    
    def update_available_columns(self):
        """Update the available columns listbox"""
        if self.df is None:
            return
        
        self.available_listbox.delete(0, tk.END)
        
        # Show columns not in the removal list
        for col in self.df.columns:
            if col not in self.columns_to_remove:
                self.available_listbox.insert(tk.END, col)
    
    def update_remove_listbox(self):
        """Update the columns to remove listbox"""
        self.remove_listbox.delete(0, tk.END)
        
        for col in self.columns_to_remove:
            self.remove_listbox.insert(tk.END, col)
    
    def add_columns(self):
        """Add selected columns to removal list"""
        selections = self.available_listbox.curselection()
        if not selections:
            messagebox.showwarning("No Selection", "Please select columns to add to removal list")
            return
        
        # Get selected column names
        selected_cols = [self.available_listbox.get(i) for i in selections]
        
        # Add to removal list if not already there
        for col in selected_cols:
            if col not in self.columns_to_remove:
                self.columns_to_remove.append(col)
        
        # Update both listboxes
        self.update_available_columns()
        self.update_remove_listbox()
        
        self.status_var.set(f"Added {len(selected_cols)} column(s) to removal list")
    
    def remove_columns(self):
        """Remove selected columns from removal list"""
        selections = self.remove_listbox.curselection()
        if not selections:
            messagebox.showwarning("No Selection", "Please select columns to remove from removal list")
            return
        
        # Get selected column names (in reverse order to avoid index issues)
        selected_cols = [self.remove_listbox.get(i) for i in reversed(selections)]
        
        # Remove from removal list
        for col in selected_cols:
            if col in self.columns_to_remove:
                self.columns_to_remove.remove(col)
        
        # Update both listboxes
        self.update_available_columns()
        self.update_remove_listbox()
        
        self.status_var.set(f"Removed {len(selected_cols)} column(s) from removal list")
    
    def reset_columns(self):
        """Reset to default columns to remove"""
        self.columns_to_remove = ['participant', 'quality_code', 'target']
        
        # Update both listboxes
        self.update_available_columns()
        self.update_remove_listbox()
        
        self.status_var.set("Reset to default columns (participant, quality_code, target)")
    
    def preview_changes(self):
        """Preview the changes that will be made"""
        if self.df is None:
            messagebox.showwarning("No File", "Please select a CSV file first")
            return
        
        # Find which columns will actually be removed (exist in the dataframe)
        existing_cols_to_remove = [col for col in self.columns_to_remove if col in self.df.columns]
        remaining_cols = [col for col in self.df.columns if col not in self.columns_to_remove]
        
        # Create preview window
        preview_window = tk.Toplevel(self.root)
        preview_window.title("Preview Changes")
        preview_window.geometry("600x400")
        preview_window.configure(bg='#f0f0f0')
        
        # Preview content
        tk.Label(preview_window, text="📋 Preview of Changes", 
                font=('Arial', 14, 'bold'), bg='#f0f0f0').pack(pady=10)
        
        # Current vs New info
        info_frame = tk.Frame(preview_window, bg='#f0f0f0')
        info_frame.pack(fill=tk.X, padx=20, pady=10)
        
        tk.Label(info_frame, text=f"Current: {len(self.df.columns)} columns, {len(self.df)} rows",
                font=('Arial', 11), bg='#f0f0f0').pack(anchor=tk.W)
        
        tk.Label(info_frame, text=f"After removal: {len(remaining_cols)} columns, {len(self.df)} rows",
                font=('Arial', 11), bg='#f0f0f0', fg='#27ae60').pack(anchor=tk.W)
        
        # Columns to be removed
        if existing_cols_to_remove:
            tk.Label(preview_window, text="Columns to be removed:", 
                    font=('Arial', 11, 'bold'), bg='#f0f0f0', fg='#e74c3c').pack(anchor=tk.W, padx=20, pady=(10, 5))
            
            removed_text = tk.Text(preview_window, height=4, font=('Arial', 10))
            removed_text.pack(fill=tk.X, padx=20, pady=(0, 10))
            removed_text.insert(tk.END, ", ".join(existing_cols_to_remove))
            removed_text.config(state=tk.DISABLED)
        
        # Remaining columns
        tk.Label(preview_window, text="Remaining columns:", 
                font=('Arial', 11, 'bold'), bg='#f0f0f0', fg='#27ae60').pack(anchor=tk.W, padx=20, pady=(10, 5))
        
        remaining_text = tk.Text(preview_window, height=8, font=('Arial', 10))
        remaining_text.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 10))
        remaining_text.insert(tk.END, ", ".join(remaining_cols))
        remaining_text.config(state=tk.DISABLED)
        
        # Buttons
        btn_frame = tk.Frame(preview_window, bg='#f0f0f0')
        btn_frame.pack(fill=tk.X, padx=20, pady=10)
        
        tk.Button(btn_frame, text="✅ Proceed with Save", 
                 command=lambda: [preview_window.destroy(), self.save_cleaned_csv()],
                 font=('Arial', 11), bg='#27ae60', fg='white', 
                 padx=15, pady=5, cursor='hand2').pack(side=tk.LEFT, padx=(0, 10))
        
        tk.Button(btn_frame, text="❌ Cancel", command=preview_window.destroy,
                 font=('Arial', 11), bg='#95a5a6', fg='white', 
                 padx=15, pady=5, cursor='hand2').pack(side=tk.LEFT)
    
    def save_cleaned_csv(self):
        """Save the cleaned CSV with specified columns removed"""
        if self.df is None:
            messagebox.showwarning("No File", "Please select a CSV file first")
            return
        
        # Find which columns will actually be removed
        existing_cols_to_remove = [col for col in self.columns_to_remove if col in self.df.columns]
        
        if not existing_cols_to_remove:
            messagebox.showinfo("No Changes", "No specified columns found in the dataset to remove")
            return
        
        # Create cleaned dataframe
        cleaned_df = self.df.drop(columns=existing_cols_to_remove)
        
        # Get save path
        default_name = f"{Path(self.current_file).stem}_cleaned.csv" if self.current_file else "cleaned_data.csv"
        
        save_path = filedialog.asksaveasfilename(
            title="Save Cleaned CSV",
            initialdir=str(PROJECT_ROOT),
            defaultextension=".csv",
            initialfile=default_name,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if not save_path:
            return
        
        try:
            cleaned_df.to_csv(save_path, index=False)
            
            # Success message
            messagebox.showinfo("Success", 
                               f"✅ Cleaned CSV saved successfully!\n\n"
                               f"📄 File: {Path(save_path).name}\n"
                               f"📊 Rows: {len(cleaned_df)}\n"
                               f"📋 Columns: {len(cleaned_df.columns)} (removed {len(existing_cols_to_remove)})\n"
                               f"🗑️ Removed: {', '.join(existing_cols_to_remove)}")
            
            self.status_var.set(f"Saved cleaned CSV with {len(cleaned_df.columns)} columns")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save cleaned CSV:\n{e}")
            self.status_var.set("Error saving file")

def main():
    """Main function to run the Column Remover application"""
    root = tk.Tk()
    app = ColumnRemoverApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
