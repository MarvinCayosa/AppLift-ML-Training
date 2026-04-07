import tkinter as tk
from pathlib import Path
from tkinter import filedialog

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CLASS_STYLES = {
    0: ("Clean", "#1f77b4"),
    1: ("Mistake 1", "#f1c40f"),
    2: ("Mistake 2", "#e74c3c"),
}

FOLDER_KEYWORDS = {
    0: ["clean"],
    1: ["uncontrolled movement", "uncontrolled_movement", "pull fast", "pulling too fast", "pulling_too_fast"],
    2: [
        "inclination asymmetry",
        "inclination assymetry",
        "inclination_asymmetry",
        "inclination_assymetry",
        "abrupt initiation",
        "abrupt intitiation",
        "abrupt_initiation",
    ],
}


PREFERRED_SIGNAL_COLUMNS = [
    "filteredMag",
    "accelMag",
    "filteredX",
    "accelX",
    "gyroX",
    "roll",
]


TITLE_FONT_SIZE = 14
LABEL_FONT_SIZE = 12
TICK_FONT_SIZE = 11
SUPTITLE_FONT_SIZE = 16
ERROR_FONT_SIZE = 11


def pick_csv_files() -> list[str]:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    file_paths = []
    current_dir = "Data"

    # Pick files one-by-one so files can come from different folders.
    for idx in range(3):
        file_path = filedialog.askopenfilename(
            title=f"Select CSV file {idx + 1} of up to 3 (Cancel to finish)",
            filetypes=[("CSV files", "*.csv")],
            initialdir=current_dir,
        )
        if not file_path:
            break

        file_paths.append(file_path)
        current_dir = str(Path(file_path).parent)

    root.destroy()
    return file_paths


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


def extract_data_context(file_path: str) -> str:
    path_obj = Path(file_path).resolve()
    lower_parts = [part.lower() for part in path_obj.parts]
    if "data" not in lower_parts:
        return "Outside Data folder"

    data_index = lower_parts.index("data")
    relative_parts = path_obj.parts[data_index + 1 : -1]
    if not relative_parts:
        return "Data"

    return " / ".join(relative_parts)


def infer_class_code(df: pd.DataFrame, file_path: str):
    if "quality_code" in df.columns and not df["quality_code"].dropna().empty:
        try:
            quality_code = int(df["quality_code"].iloc[0])
            if quality_code in CLASS_STYLES:
                return quality_code, "quality_code column"
        except (TypeError, ValueError):
            pass

    joined_path = str(Path(file_path).resolve()).lower()
    for class_code, keywords in FOLDER_KEYWORDS.items():
        if any(keyword in joined_path for keyword in keywords):
            return class_code, "folder/file name"

    return None, "not detected"


def main():
    file_paths = pick_csv_files()
    if not file_paths:
        print("No files selected.")
        return

    fig, axes = plt.subplots(len(file_paths), 1, figsize=(12, 3.8 * len(file_paths)), squeeze=False)
    axes = axes.flatten()

    for ax, file_path in zip(axes, file_paths):
        try:
            df = pd.read_csv(file_path)
            signal_col = pick_signal_column(df)
            x_values, x_label = infer_time_axis(df)
            class_code, _ = infer_class_code(df, file_path)
            class_label, color = CLASS_STYLES.get(class_code, ("Unknown", "#7f8c8d"))

            ax.plot(x_values, df[signal_col].values, color=color, linewidth=1.5)
            ax.set_title(f"{Path(file_path).name} - {class_label}", fontsize=TITLE_FONT_SIZE)
            ax.set_xlabel(x_label, fontsize=LABEL_FONT_SIZE)
            ax.set_ylabel(signal_col, fontsize=LABEL_FONT_SIZE)
            ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
            ax.grid(True, linestyle="--", alpha=0.5)
        except Exception as exc:
            ax.text(
                0.5,
                0.5,
                f"Failed to load:\n{Path(file_path).name}\n{exc}",
                ha="center",
                va="center",
                fontsize=ERROR_FONT_SIZE,
            )
            ax.set_axis_off()

    fig.suptitle("Waveform Comparison", fontsize=SUPTITLE_FONT_SIZE, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


if __name__ == "__main__":
    main()