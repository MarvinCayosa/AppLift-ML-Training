import matplotlib.pyplot as plt
import numpy as np

# Data from Table 7
exercises = [
    "Lateral Pulldown",
    "Seated Leg Extension",
    "Concentration Curls",
    "Overhead Extension",
    "Back Squats",
    "Flat Bench Press"
]

accuracy = [0.8654, 0.8000, 0.7963, 0.7000, 0.6500, 0.5926]
macro_precision = [0.7989, 0.7895, 0.6737, 0.5652, 0.5625, 0.3455]
macro_recall = [0.6111, 0.8480, 0.6113, 0.4899, 0.6490, 0.5123]
macro_f1 = [0.5982, 0.7961, 0.6126, 0.4923, 0.5517, 0.4126]

# Bar positions
x = np.arange(len(exercises))
width = 0.2
colors = ["#1F5A99", "#6EA83A", "#F2B233", "#D9534F"]

# Create figure
plt.figure(figsize=(12, 6))

# Bars
plt.bar(x - 1.5*width, accuracy, width, label='Accuracy', color=colors[0])
plt.bar(x - 0.5*width, macro_precision, width, label='Macro Precision', color=colors[1])
plt.bar(x + 0.5*width, macro_recall, width, label='Macro Recall', color=colors[2])
plt.bar(x + 1.5*width, macro_f1, width, label='Macro F1 Score', color=colors[3])

# Labels and title
plt.ylabel('Performance Score', fontsize=9)
plt.xlabel('Exercise', fontsize=9)
plt.title('Field Testing Repetition Classification Performance of AppLift', fontsize=11)
plt.xticks(x, exercises, rotation=0, ha='center', fontsize=8)
plt.yticks(fontsize=8)
plt.ylim(0, 1.0)

# Legend
plt.legend(fontsize=8)

# Grid for cleaner thesis-style look
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Adjust layout
plt.tight_layout()

# Save figure
plt.savefig("table7_grouped_bar_chart.png", dpi=300, bbox_inches='tight')

# Show plot
plt.show()