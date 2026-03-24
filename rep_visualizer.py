import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from pathlib import Path

# ── Thesis colour palette ──────────────────────────────────────────────────────
CLASSIFICATION_COLORS = {
    0: ('#2ecc71', '#27ae60', 'Clean'),
    1: ('#e74c3c', '#c0392b', 'Uncontrolled Movement'),
    2: ('#f39c12', '#d68910', 'Inclination Asymmetry'),
}
EXERCISE_MAP  = {3: 'Back Squats', 4: 'Front Squats', 5: 'Bench Press'}
EQUIPMENT_MAP = {1: 'Barbell', 2: 'Dumbbell', 3: 'Kettlebell'}

PLOT_BG   = '#fafafa'
SPINE_COL = '#cccccc'
GRID_COL  = '#e0e0e0'

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
})


class RepVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Exercise Session Waveform Visualizer")
        self.root.geometry("1400x820")
        self.root.configure(bg='#f0f0f0')

        self.loaded_files = []   # list of dicts
        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill=tk.X)

        ttk.Button(top, text="+ Add CSV File(s)", command=self.load_files).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Clear All",         command=self.clear_all).pack(side=tk.LEFT, padx=4)

        ttk.Separator(top, orient='vertical').pack(side=tk.LEFT, fill=tk.Y, padx=10)

        ttk.Label(top, text="Signal:").pack(side=tk.LEFT)
        self.signal_var = tk.StringVar(value="Filtered Magnitude")
        self.signal_combo = ttk.Combobox(
            top, textvariable=self.signal_var, state='readonly', width=20,
            values=["Filtered Magnitude", "Raw Magnitude", "Filtered X/Y/Z", "Raw X/Y/Z",
                    "Gyroscope X/Y/Z", "Roll / Pitch / Yaw"])
        self.signal_combo.pack(side=tk.LEFT, padx=4)
        self.signal_combo.bind('<<ComboboxSelected>>', lambda _: self.refresh())

        ttk.Separator(top, orient='vertical').pack(side=tk.LEFT, fill=tk.Y, padx=10)

        self.shade_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="Shade reps", variable=self.shade_var,
                        command=self.refresh).pack(side=tk.LEFT, padx=4)

        self.boundary_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="Rep boundaries", variable=self.boundary_var,
                        command=self.refresh).pack(side=tk.LEFT, padx=4)

        # File list
        list_frame = ttk.Frame(self.root, padding=(10, 0))
        list_frame.pack(fill=tk.X)
        ttk.Label(list_frame, text="Loaded files:").pack(side=tk.LEFT)
        self.file_list_label = ttk.Label(list_frame, text="None", foreground='gray')
        self.file_list_label.pack(side=tk.LEFT, padx=6)

        # Canvas
        canvas_frame = ttk.Frame(self.root)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)

        self.fig = plt.figure(figsize=(14, 7))
        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ── File loading ──────────────────────────────────────────────────────────
    def load_files(self):
        paths = filedialog.askopenfilenames(
            title="Select CSV file(s)",
            filetypes=[("CSV files", "*.csv")],
            initialdir="Data"
        )
        for p in paths:
            try:
                df = pd.read_csv(p)
                self.loaded_files.append({'path': p, 'name': Path(p).name, 'df': df})
            except Exception as e:
                messagebox.showerror("Load error", str(e))

        names = [f['name'] for f in self.loaded_files]
        self.file_list_label.config(
            text=', '.join(names) if names else 'None',
            foreground='black' if names else 'gray'
        )
        self.refresh()

    def clear_all(self):
        self.loaded_files.clear()
        self.file_list_label.config(text='None', foreground='gray')
        self.fig.clear()
        self.canvas.draw()

    # ── Plotting ──────────────────────────────────────────────────────────────
    def refresh(self):
        if not self.loaded_files:
            return
        self.fig.clear()

        n = len(self.loaded_files)
        axes = self.fig.subplots(n, 1, squeeze=False)

        for row, file_info in enumerate(self.loaded_files):
            ax = axes[row][0]
            self._plot_session(ax, file_info)

        self.fig.tight_layout(rect=[0, 0, 1, 0.97])
        self.canvas.draw()

    def _plot_session(self, ax, file_info):
        df   = file_info['df']
        name = file_info['name']

        # ── metadata ──────────────────────────────────────────────────────────
        quality_code  = int(df['quality_code'].iloc[0])
        exercise_code = int(df['exercise_code'].iloc[0])
        participant   = int(df['participant'].iloc[0])

        color_main, color_dark, quality_label = CLASSIFICATION_COLORS.get(
            quality_code, ('#3498db', '#2980b9', f'Code {quality_code}'))
        exercise_label  = EXERCISE_MAP.get(exercise_code, f'Exercise {exercise_code}')

        # ── time axis (seconds from 0) ─────────────────────────────────────
        t = (df['timestamp_ms'].values - df['timestamp_ms'].values[0]) / 1000.0

        # ── choose signal ─────────────────────────────────────────────────
        sig = self.signal_var.get()
        if sig == "Filtered Magnitude":
            signals = [('filteredMag', 'Filtered Magnitude', color_main)]
            ylabel  = 'Magnitude (m/s²)'
        elif sig == "Raw Magnitude":
            signals = [('accelMag', 'Raw Magnitude', color_main)]
            ylabel  = 'Magnitude (m/s²)'
        elif sig == "Filtered X/Y/Z":
            signals = [('filteredX','X','#e74c3c'),('filteredY','Y','#2ecc71'),('filteredZ','Z','#3498db')]
            ylabel  = 'Filtered Accel (m/s²)'
        elif sig == "Raw X/Y/Z":
            signals = [('accelX','X','#e74c3c'),('accelY','Y','#2ecc71'),('accelZ','Z','#3498db')]
            ylabel  = 'Raw Accel (m/s²)'
        elif sig == "Gyroscope X/Y/Z":
            signals = [('gyroX','X','#e74c3c'),('gyroY','Y','#2ecc71'),('gyroZ','Z','#3498db')]
            ylabel  = 'Angular Velocity (rad/s)'
        else:  # Roll/Pitch/Yaw
            signals = [('roll','Roll','#e74c3c'),('pitch','Pitch','#2ecc71'),('yaw','Yaw','#3498db')]
            ylabel  = 'Angle (°)'

        # ── rep shading & boundaries ───────────────────────────────────────
        reps = df['rep'].values
        unique_reps = sorted(df['rep'].unique())

        if self.shade_var.get():
            for i, rep in enumerate(unique_reps):
                mask = reps == rep
                rep_t = t[mask]
                if len(rep_t) == 0:
                    continue
                shade = '#e8f5e9' if i % 2 == 0 else '#f3f3f3'
                ax.axvspan(rep_t[0], rep_t[-1], alpha=0.5, color=shade, zorder=0)

        if self.boundary_var.get():
            for rep in unique_reps[1:]:
                mask = reps == rep
                rep_t = t[mask]
                if len(rep_t):
                    ax.axvline(rep_t[0], color=SPINE_COL, linewidth=0.8,
                               linestyle='--', zorder=1)

        # ── rep number labels ──────────────────────────────────────────────
        for rep in unique_reps:
            mask = reps == rep
            rep_t = t[mask]
            if len(rep_t):
                mid = (rep_t[0] + rep_t[-1]) / 2
                ax.text(mid, ax.get_ylim()[1] if ax.get_ylim()[1] != 1.0 else 0,
                        f'R{rep}', ha='center', va='bottom',
                        fontsize=7, color='#888888', zorder=5)

        # ── plot signals ───────────────────────────────────────────────────
        for col, label, c in signals:
            if col in df.columns:
                ax.plot(t, df[col].values, color=c, linewidth=1.4,
                        label=label, alpha=0.9, zorder=3)

        # ── rep number labels (after plot so y-range is set) ──────────────
        ymax = ax.get_ylim()[1]
        for rep in unique_reps:
            mask = reps == rep
            rep_t = t[mask]
            if len(rep_t):
                mid = (rep_t[0] + rep_t[-1]) / 2
                ax.text(mid, ymax * 0.97, f'R{rep}', ha='center', va='top',
                        fontsize=7, color='#999999', zorder=5)

        # ── styling ────────────────────────────────────────────────────────
        ax.set_facecolor(PLOT_BG)
        ax.set_ylabel(ylabel, fontsize=10, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.grid(True, color=GRID_COL, linewidth=0.6, zorder=0)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['left', 'bottom']:
            ax.spines[spine].set_color(SPINE_COL)
        ax.set_xlim(t[0], t[-1])

        # legend
        if len(signals) > 1:
            ax.legend(loc='upper right', framealpha=0.7)

        # title with classification badge
        title = (f'P{participant:03d}  ·  {exercise_label}  ·  '
                 f'{len(unique_reps)} reps  ·  {name}')
        ax.set_title(title, fontsize=11, fontweight='bold', loc='left', pad=6)

        # classification badge (right side of title)
        ax.annotate(
            f'  {quality_label}  ',
            xy=(1, 1), xycoords='axes fraction',
            xytext=(-4, 4), textcoords='offset points',
            ha='right', va='bottom', fontsize=9, fontweight='bold',
            color='white',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=color_main,
                      edgecolor=color_dark, linewidth=1.2)
        )

        # overall figure title
        self.fig.suptitle(
            f'Session Waveform  ·  {self.signal_var.get()}',
            fontsize=13, fontweight='bold', color='#2c3e50'
        )


def main():
    root = tk.Tk()
    RepVisualizer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
