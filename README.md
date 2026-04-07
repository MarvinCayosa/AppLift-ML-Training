# AppLift ML Training - Exercise Quality Classification

## 📁 Project Structure

This repository contains a complete machine learning pipeline for classifying exercise execution quality using sensor data. The codebase has been organized into logical modules for easy navigation and maintenance.

```
AppLift-ML-Training/
├── 📚 docs/                          # All documentation
│   ├── README.md                     # Original project documentation
│   ├── ML_PIPELINE_MASTER_README.md  # Complete pipeline guide (START HERE!)
│   ├── COMPARISON_GUIDE.md           # Original vs Fixed version comparison
│   ├── BACK_SQUATS_FALSE_POSITIVE_FIX.md
│   ├── ENCODED_LABELS_README.md
│   └── VISUALIZER_README.md
│
├── 💻 src/                           # Source code
│   ├── pipeline/                     # Data preprocessing pipeline
│   │   ├── preprocessing_pipeline.py     # Stage 1: Data cleaning & merging
│   │   ├── resegment_reps_fixed.py      # Stage 2: Rep boundary correction
│   │   ├── re_labeler.py                # Stage 3: Quality label correction
│   │   └── dataset_merger.py            # Merge multiple datasets
│   │
│   ├── models/                       # Model training scripts
│   │   ├── rf_classifier_realistic.py   # ⭐ Random Forest (Fixed version)
│   │   ├── xgb_classifier_realistic.py  # ⭐ XGBoost Classifier
│   │   ├── svm_classifier_realistic.py  # ⭐ SVM Classifier
│   │   ├── rf_classifier_fixed.py       # Alternative fixed version
│   │   ├── rf_classifier.py             # ⚠️ Legacy (has data leakage)
│   │   ├── rf_classifier_copy.py        # ⚠️ Legacy (has data leakage)
│   │   └── model_tester.py              # Test trained models
│   │
│   ├── visualization/                # Visualization tools
│   │   ├── rep_visualizer.py            # Visualize individual reps
│   │   ├── rep_comparison_visualizer.py # Compare clean vs error reps
│   │   ├── simple_rep_visualizer.py     # Simplified rep viewer
│   │   ├── performance_visualizer.py    # Model performance plots
│   │   ├── imu_comparison_gui.py        # IMU sensor comparison tool
│   │   └── visualize_imu_comparison.py  # IMU visualization
│   │
│   ├── analysis/                     # Analysis & testing tools
│   │   ├── fix_false_positives.py       # False positive analysis
│   │   ├── test_fp_reduction.py         # Validate FP reduction
│   │   ├── test_enhanced_report.py      # Generate detailed reports
│   │   ├── model_comparison.py          # ⭐ Compare RF vs XGBoost vs SVM
│   │   └── probe_codes.py               # Code exploration utility
│   │
│   └── utils/                        # Utility scripts
│       ├── axis_mapping_gui.py          # Sensor axis mapping tool
│       ├── column_remover.py            # CSV column removal utility
│       └── fix_unicode.py               # Unicode encoding fixer
│
├── 📊 Data/                          # Raw sensor data
│   ├── Barbell/
│   ├── Dumbbell/
│   └── Weight_Stack/
│
├── 📈 outputs/                       # Training outputs & results
│   ├── output/                          # Original version outputs
│   ├── output_fixed/                    # Fixed version outputs
│   ├── output_realistic_random_forest/  # Realistic RF outputs
│   ├── visualizations/                  # Generated plots
│   ├── visualizations_chapter4/         # Chapter 4 figures
│   └── table7_grouped_bar_chart.png
│
└── 🧪 tests/                         # Test data & experiments
    ├── test_data/                       # Test datasets
    ├── calibration_test/                # IMU calibration tests
    ├── cloud_latency_test.csv
    └── imu_comparison_2026-03-12T20-20-16.csv
```

---

## 🚀 Quick Start

### For New Users

1. **Read the documentation**:
   ```bash
   # Start with the master pipeline guide
   docs/ML_PIPELINE_MASTER_README.md
   ```

2. **Preprocess your data**:
   ```bash
   python src/pipeline/preprocessing_pipeline.py
   ```

3. **Resegment reps** (recommended):
   ```bash
   python src/pipeline/resegment_reps_fixed.py
   ```

4. **Train model**:
   ```bash
   python src/models/rf_classifier_realistic.py
   ```

### For Experienced Users

**Full pipeline in one go**:
```bash
# 1. Preprocess
python src/pipeline/preprocessing_pipeline.py

# 2. Resegment
python src/pipeline/resegment_reps_fixed.py

# 3. Train Random Forest
python src/models/rf_classifier_realistic.py

# 4. Train XGBoost (for comparison)
python src/models/xgb_classifier_realistic.py

# 5. Train SVM (for comparison)
python src/models/svm_classifier_realistic.py

# 6. Compare models
python src/analysis/model_comparison.py
```

---

## 📖 Documentation Guide

| Document | Purpose | When to Read |
|----------|---------|--------------|
| **ML_PIPELINE_MASTER_README.md** | Complete pipeline guide | Start here! |
| **README.md** (docs/) | Original project overview | For background |
| **COMPARISON_GUIDE.md** | Original vs Fixed comparison | Understanding fixes |
| **BACK_SQUATS_FALSE_POSITIVE_FIX.md** | FP reduction strategy | Reducing false alarms |
| **VISUALIZER_README.md** | Visualization tools guide | Using viz tools |

---

## ⚠️ Important Notes

### Which Model Training Script to Use?

**✅ USE**: 
- `src/models/rf_classifier_realistic.py` - Random Forest (proven, reliable)
- `src/models/xgb_classifier_realistic.py` - XGBoost (often better performance)
- `src/models/svm_classifier_realistic.py` - SVM (good for smaller datasets)

**❌ AVOID**: 
- `rf_classifier.py` - Has data leakage
- `rf_classifier_copy.py` - Has data leakage
- `rf_classifier_fixed.py` - Use realistic version instead

### Model Comparison

After training both models, compare their performance:
```bash
python src/analysis/model_comparison.py
```

This generates:
- Side-by-side performance metrics
- Visualization of differences
- Recommendation on which model to use

### Output Directories

- **outputs/realistic/** - Random Forest model outputs
- **outputs/xgboost/** - XGBoost model outputs
- **outputs/svm/** - SVM model outputs
- **outputs/comparison/** - Model comparison results
- **outputs/output_fixed/** - Legacy fixed version outputs
- **outputs/output/** - Legacy original outputs (has data leakage)

---

## 🔧 Installation

### Prerequisites
```bash
Python 3.8+
pip or conda
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Required Packages
- scikit-learn >= 1.0.0
- pandas >= 1.3.0
- numpy >= 1.21.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0
- imbalanced-learn >= 0.8.0

---

## 📊 Pipeline Stages

### Stage 1: Preprocessing
**Script**: `src/pipeline/preprocessing_pipeline.py`
- Merge multiple CSV files
- Clean data quality issues
- Handle missing values
- Remove outliers

### Stage 2: Resegmentation
**Script**: `src/pipeline/resegment_reps_fixed.py`
- Fix rep boundaries using valley detection
- Ensure continuous rep segments
- Exercise-specific duration constraints

### Stage 3: Relabeling
**Script**: `src/pipeline/re_labeler.py`
- Visual rep inspection
- Quality label correction
- Rep boundary editing
- Metadata updates

### Stage 4-7: Feature Engineering & Training
**Script**: `src/models/rf_classifier_realistic.py`
- Extract statistical features per rep
- Dimensionality reduction
- Class imbalance handling
- Random Forest training
- Model evaluation & export

---

## 🎯 Common Tasks

### Visualize Reps
```bash
# Single rep viewer
python src/visualization/rep_visualizer.py

# Compare clean vs error reps
python src/visualization/rep_comparison_visualizer.py
```

### Analyze False Positives
```bash
# Analyze FP patterns
python src/analysis/fix_false_positives.py

# Test FP reduction
python src/analysis/test_fp_reduction.py
```

### Test Trained Model
```bash
python src/models/model_tester.py
```

### Utility Tasks
```bash
# Remove unwanted columns
python src/utils/column_remover.py

# Map sensor axes
python src/utils/axis_mapping_gui.py
```

---

## 🐛 Troubleshooting

### Import Errors After Reorganization

If you get import errors, update your Python path:

```python
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))
```

Or run from project root:
```bash
cd /path/to/AppLift-ML-Training
python src/models/rf_classifier_realistic.py
```

### Low Model Performance

1. Check you're using `rf_classifier_realistic.py`
2. Ensure data is resegmented
3. Review data quality in preprocessing
4. Check class imbalance settings
5. See troubleshooting in `docs/ML_PIPELINE_MASTER_README.md`

---

## 📝 Version History

### v2.1 (March 2026) - Current
- ✅ Reorganized codebase into logical modules
- ✅ Created comprehensive master documentation
- ✅ Separated docs, src, outputs, and tests
- ✅ Improved project navigation

### v2.0 (March 2026)
- Fixed data leakage in cross-validation
- Fixed SMOTE application
- Added false positive reduction
- Created master pipeline documentation

### v1.0 (February 2026)
- Initial pipeline implementation

---

## 📧 Support

For questions or issues:
1. Check `docs/ML_PIPELINE_MASTER_README.md`
2. Review troubleshooting section
3. Check existing documentation in `docs/`

---

## 📄 License

Academic and research use only. Commercial use requires permission.

---

**Last Updated**: March 28, 2026  
**Version**: 2.1  
**Status**: Production Ready
