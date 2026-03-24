import matplotlib.pyplot as plt
import numpy as np

exercises = [
    "Lateral Pulldown",
    "Seated Leg Extension",
    "Concentration Curls",
    "Overhead Extension",
    "Back Squats",
    "Flat Bench Press"
]

training = [0.9565, 0.9429, 0.9472, 0.9704, 0.9531, 0.9559]
testing = [0.9172, 0.8588, 0.9037, 0.9176, 0.8812, 0.8727]
field = [0.8654, 0.8000, 0.7963, 0.7000, 0.6500, 0.5926]

# Compute gap
gap = [t - f for t, f in zip(training, field)]

x = np.arange(len(exercises))
width = 0.25

def add_percentage_labels(ax, bars, offset=0.01):
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + offset,
            f"{height * 100:.1f}%",
            ha='center',
            va='bottom',
            fontsize=8
        )

fig, ax = plt.subplots(figsize=(12, 6))

bars_training = ax.bar(x - width, training, width, label='Training Accuracy')
bars_testing = ax.bar(x, testing, width, label='Testing Accuracy')
bars_field = ax.bar(x + width, field, width, label='Field Testing Accuracy')

ax.set_xticks(x)
ax.set_xticklabels(exercises, rotation=0, ha='center', fontsize=8)
ax.set_ylabel('Accuracy', fontsize=9)
ax.set_xlabel('Exercise', fontsize=9)
ax.set_title('Comparison of Training, Testing, and Field Testing Accuracy', fontsize=11)
ax.set_ylim(0, 1.08)
ax.tick_params(axis='y', labelsize=8)

ax.legend(
    fontsize=8,
    loc='upper center',
    bbox_to_anchor=(0.5, 1.15),
    ncol=3,
    frameon=False
)
ax.grid(axis='y', linestyle='--', alpha=0.5)

add_percentage_labels(ax, bars_training)
add_percentage_labels(ax, bars_testing)
add_percentage_labels(ax, bars_field)

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.show()

# ---- GAP PLOT ----
fig, ax = plt.subplots(figsize=(10, 5))

bars_gap = ax.bar(x, gap)

ax.set_xticks(x)
ax.set_xticklabels(exercises, rotation=0, ha='center', fontsize=8)
ax.set_ylabel('Generalization Gap (Training - Field)', fontsize=9)
ax.set_xlabel('Exercise', fontsize=9)
ax.set_title('Generalization Gap Across Exercises', fontsize=11)
ax.tick_params(axis='y', labelsize=8)
ax.set_ylim(0, max(gap) + 0.08)

ax.grid(axis='y', linestyle='--', alpha=0.5)

add_percentage_labels(ax, bars_gap, offset=0.005)

plt.tight_layout()
plt.show()