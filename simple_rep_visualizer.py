import tkinter as tk
from pathlib import Path
from tkinter import filedialog

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PREFERRED_SIGNAL_COLUMNS = [
    "filteredMag",
    "accelMag",
    "filteredX",
    "accelX",
    "gyroX",
    "roll",
]


def pick_csv_file() -> str:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    file_path = filedialog.askopenfilename(
        title="Select a CSV file",
        filetypes=[("CSV files", "*.csv")],
        initialdir="Data",
    )
    root.destroy()
    return file_path


def infer_time_axis(df: pd.DataFrame):
    if "timestamp_ms" in df.columns:
        time_values = (df["timestamp_ms"].values - df["timestamp_ms"].values[0]) / 1000.0
        return time_values, "Time (s)"

    return np.arange(len(df)), "Sample Index"


def pick_signal_column(df: pd.DataFrame) -> str:
    for col in PREFERRED_SIGNAL_COLUMNS:
        if col in df.columns:
            return col

    numeric_columns = [c for c in df.select_dtypes(include=[np.number]).columns if c != "timestamp_ms"]
    if not numeric_columns:
        raise ValueError("No numeric signal column found in this CSV.")

    return numeric_columns[0]


def main():
    file_path = pick_csv_file()
    if not file_path:
        print("No file selected.")
        return

    df = pd.read_csv(file_path)
    signal_col = pick_signal_column(df)
    x_values, x_label = infer_time_axis(df)

    plt.figure(figsize=(12, 5))
    plt.plot(x_values, df[signal_col].values, color="#1f77b4", linewidth=1.5)
    plt.title(f"Waveform: {Path(file_path).name}", fontsize=11)
    plt.xlabel(x_label, fontsize=9)
    plt.ylabel(signal_col, fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()