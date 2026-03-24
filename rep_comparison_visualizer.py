import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

CLASSIFICATION_COLORS = {
    0: ('#2ecc71', '#27ae60', 'Clean'),
    1: ('#e74c3c', '#c0392b', 'Uncontrolled Movement'),
    2: ('#f39c12', '#d68910', 'Inclination Asymmetry'),
}
EXERCISE_MAP  = {3: 'Back Squats', 4: 'Front Squats', 5: 'Bench Press'}
PLOT_BG   = '#fafafa'
GRID_COL  = '#e0e0e0'
SPINE_COL = '#cccccc'

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
})

SLOTS = [
    ('Clean',                  0),
    ('Uncontrolled Movement',  1),
    ('Inclination Asymmetry',  2),
]


class ComparisonVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Classification Waveform Comparison")
        self.root.geometry("1500x860")
        self.root.configure(bg='#f0f0f0')

        self.slot_data = {name: None for name, _ in SLOTS}
        self._build_ui()

    def _build_ui(self):
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill=tk.X)

        # One load button per classification slot
        for label, code in SLOTS:
            color, _, _ = CLASSIFICATION_COLORS[code]
            frame = ttk.LabelFrame(top, text=label, padding=6)
            frame.pack(side=tk.LEFT, padx=8)
            self._make_slot_widget(frame, label)

        ttk.Separator(top, orient='vertical').pack(side=tk.LEFT, fill=tk.Y, padx=10)

        ttk.Label(top, text="Signal:").pack(side=tk.LEFT)
        self.signal_var = tk.StringVar(value="Filtered Magnitude")
        ttk.Combobox(
            top, textvariable=self.signal_var, state='readonly', width=20,
            values=["Filtered Magnitude", "Raw Magnitude",
                    "Filtered X/Y/Z", "Raw X/Y/Z",
                    "Gyroscope X/Y/Z", "Roll / Pitch / Yaw"]
        ).pack(side=tk.LEFT, padx=4)
        self.signal_var.trace_add('write', lambda *_: self.refresh())

        self.shade_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="Shade reps", variable=self.shade_var,
                        command=self.refresh).pack(side=tk.LEFT, padx=8)

        # Canvas
        canvas_frame = ttk.Frame(self.root)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)
        self.fig, self.axes = plt.subplots(3, 1, figsize=(15, 8), sharex=False)
        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._draw_empty()

    def _make_slot_widget(self, frame, label):
        lbl = ttk.Label(frame, text="No file loaded", foreground='gray', width=28)
        lbl.pack()
        setattr(self, f'lbl_{label}', lbl)
        ttk.Button(frame, text="Load CSV",
                   command=lambda l=label: self.load_slot(l)).pack(pady=2)

    def load_slot(self, label):
        path = filedialog.askopenfilename(
            title=f"Load CSV for {label}",
            filetypes=[("CSV files", "*.csv")],
            initialdir="Data"
        )
        if not path:
            return
        try:
            df = pd.read_csv(path)
            self.slot_data[label] = {'df': df, 'name': Path(path).name}
            getattr(self, f'lbl_{label}').config(
                text=Path(path).name, foreground='black')
            self.refresh()
        except Exception as e:
            messagebox.showerror("Load error", str(e))

    def _draw_empty(self):
        for ax, (label, code) in zip(self.axes, SLOTS):
            ax.clear()
            color, _, _ = CLASSIFICATION_COLORS[code]
            ax.set_facecolor(PLOT_BG)
            ax.text(0.5, 0.5, f'Load a CSV for\n"{label}"',
                    ha='center', va='center', fontsize=11,
                    color='#aaaaaa', transform=ax.transAxes)
            ax.set_title(label, fontsize=11, fontweight='bold', color=color, loc='left')
            for sp in ['top', 'right']:
                ax.spines[sp].set_visible(False)
        self.fig.tight_layout(rect=[0, 0, 1, 0.96])
        self.canvas.draw()

    def refresh(self):
        sig = self.signal_var.get()

        for ax, (label, code) in zip(self.axes, SLOTS):
            ax.clear()
            color_main, color_dark, quality_label = CLASSIFICATION_COLORS[code]

            if self.slot_data[label] is None:
                ax.set_facecolor(PLOT_BG)
                ax.text(0.5, 0.5, f'Load a CSV for\n"{label}"',
                        ha='center', va='center', fontsize=11,
                        color='#aaaaaa', transform=ax.transAxes)
                ax.set_title(label, fontsize=11, fontweight='bold',
                             color=color_main, loc='left')
                for sp in ['top', 'right']:
                    ax.spines[sp].set_visible(False)
                continue

            df   = self.slot_data[label]['df']
            name = self.slot_data[label]['name']

            participant   = int(df['participant'].iloc[0])
            exercise_code = int(df['exercise_code'].iloc[0])
            exercise_label = EXERCISE_MAP.get(exercise_code, f'Ex {exercise_code}')

            t = (df['timestamp_ms'].values - df['timestamp_ms'].values[0]) / 1000.0
            reps = df['rep'].values
            unique_reps = sorted(df['rep'].unique())

            # ── choose signals ────────────────────────────────────────────
            if sig == "Filtered Magnitude":
                signals = [('filteredMag', 'Filtered Mag', color_main)]
                ylabel  = 'Magnitude (m/s²)'
            elif sig == "Raw Magnitude":
                signals = [('accelMag', 'Raw Mag', color_main)]
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
            else:
                signals = [('roll','Roll','#e74c3c'),('pitch','Pitch','#2ecc71'),('yaw','Yaw','#3498db')]
                ylabel  = 'Angle (°)'

            # ── rep shading ───────────────────────────────────────────────
            if self.shade_var.get():
                for i, rep in enumerate(unique_reps):
                    mask  = reps == rep
                    rep_t = t[mask]
                    if len(rep_t) == 0:
                        continue
                    shade = '#f0faf0' if i % 2 == 0 else '#f8f8f8'
                    ax.axvspan(rep_t[0], rep_t[-1], alpha=0.6, color=shade, zorder=0)
                    ax.axvline(rep_t[0], color=SPINE_COL, linewidth=0.7,
                               linestyle='--', zorder=1)

            # ── plot ──────────────────────────────────────────────────────
            for col, lbl, c in signals:
                if col in df.columns:
                    ax.plot(t, df[col].values, color=c, linewidth=1.5,
                            label=lbl, alpha=0.9, zorder=3)

            # ── rep labels ────────────────────────────────────────────────
            ymax = ax.get_ylim()[1]
            for rep in unique_reps:
                mask  = reps == rep
                rep_t = t[mask]
                if len(rep_t):
                    mid = (rep_t[0] + rep_t[-1]) / 2
                    ax.text(mid, ymax * 0.97, f'R{rep}',
                            ha='center', va='top', fontsize=7,
                            color='#999999', zorder=5)

            # ── styling ───────────────────────────────────────────────────
            ax.set_facecolor(PLOT_BG)
            ax.set_ylabel(ylabel, fontsize=10, fontweight='bold')
            ax.grid(True, color=GRID_COL, linewidth=0.6, zorder=0)
            for sp in ['top', 'right']:
                ax.spines[sp].set_visible(False)
            for sp in ['left', 'bottom']:
                ax.spines[sp].set_color(SPINE_COL)
            ax.set_xlim(t[0], t[-1])

            if len(signals) > 1:
                ax.legend(loc='upper right', framealpha=0.7, fontsize=8)

            title = (f'P{participant:03d}  ·  {exercise_label}  ·  '
                     f'{len(unique_reps)} reps  ·  {name}')
            ax.set_title(title, fontsize=10, fontweight='bold', loc='left', pad=5)

            ax.annotate(
                f'  {quality_label}  ',
                xy=(1, 1), xycoords='axes fraction',
                xytext=(-4, 4), textcoords='offset points',
                ha='right', va='bottom', fontsize=9, fontweight='bold',
                color='white',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=color_main,
                          edgecolor=color_dark, linewidth=1.2)
            )

        self.axes[-1].set_xlabel('Time (s)', fontsize=10)
        self.fig.suptitle(
            f'Classification Waveform Comparison  ·  {sig}',
            fontsize=13, fontweight='bold', color='#2c3e50'
        )
        self.fig.tight_layout(rect=[0, 0, 1, 0.96])
        self.canvas.draw()


def main():
    root = tk.Tk()
    ComparisonVisualizer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
