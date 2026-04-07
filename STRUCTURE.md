# Codebase Structure Documentation

## 📋 Overview

This document explains the reorganized codebase structure and the rationale behind each organizational decision.

---

## 🗂️ Directory Structure

### `/docs` - Documentation
**Purpose**: Centralized location for all project documentation

**Contents**:
- `ML_PIPELINE_MASTER_README.md` - **Primary documentation** - Complete pipeline guide
- `README.md` - Original project overview and background
- `COMPARISON_GUIDE.md` - Explains differences between buggy and fixed versions
- `BACK_SQUATS_FALSE_POSITIVE_FIX.md` - False positive reduction strategy
- `ENCODED_LABELS_README.md` - Label encoding system documentation
- `VISUALIZER_README.md` - Visualization tools usage guide

**Why separate docs folder?**
- Keeps documentation organized and easy to find
- Separates documentation from code
- Makes it easy to generate documentation sites
- Clear entry point for new users

---

### `/src` - Source Code
**Purpose**: All executable Python code organized by functionality

#### `/src/pipeline` - Data Processing Pipeline
**Purpose**: Scripts that transform raw data into ML-ready datasets

**Files**:
- `preprocessing_pipeline.py` - **Stage 1**: Data cleaning, merging, outlier removal
- `resegment_reps_fixed.py` - **Stage 2**: Rep boundary correction using valley detection
- `re_labeler.py` - **Stage 3**: Interactive quality label correction tool
- `dataset_merger.py` - Utility to merge multiple datasets

**Workflow**: Raw CSV → Cleaned → Resegmented → Relabeled → Ready for training

#### `/src/models` - Model Training
**Purpose**: Machine learning model training and testing scripts

**Files**:
- `rf_classifier_realistic.py` - ⭐ **PRIMARY SCRIPT** - Fixed version with proper validation
- `rf_classifier_fixed.py` - Alternative fixed version
- `rf_classifier.py` - ⚠️ Legacy (has data leakage issues)
- `rf_classifier_copy.py` - ⚠️ Legacy backup (has data leakage issues)
- `model_tester.py` - Test and evaluate trained models

**Why keep legacy files?**
- Historical reference
- Comparison purposes
- Understanding what was fixed
- Reproducibility of old results

**Which to use?** Always use `rf_classifier_realistic.py`

#### `/src/visualization` - Visualization Tools
**Purpose**: Tools for visualizing data, reps, and model performance

**Files**:
- `rep_visualizer.py` - View individual repetitions
- `rep_comparison_visualizer.py` - Compare clean vs error reps side-by-side
- `simple_rep_visualizer.py` - Simplified rep viewer
- `performance_visualizer.py` - Model performance plots
- `imu_comparison_gui.py` - Interactive IMU sensor comparison
- `visualize_imu_comparison.py` - IMU visualization script

**Use cases**:
- Data quality inspection
- Understanding error patterns
- Model debugging
- Report generation

#### `/src/analysis` - Analysis & Testing
**Purpose**: Scripts for analyzing model behavior and testing improvements

**Files**:
- `fix_false_positives.py` - Analyze and fix false positive patterns
- `test_fp_reduction.py` - Validate false positive reduction strategies
- `test_enhanced_report.py` - Generate comprehensive classification reports
- `probe_codes.py` - Code exploration and debugging utility

**When to use**:
- Model performance issues
- False positive problems
- Detailed analysis needs
- Testing improvements

#### `/src/utils` - Utility Scripts
**Purpose**: Helper scripts and tools that don't fit other categories

**Files**:
- `axis_mapping_gui.py` - Interactive sensor axis mapping tool
- `column_remover.py` - Remove unwanted columns from CSV files
- `fix_unicode.py` - Fix Unicode encoding issues in data files

**Characteristics**:
- General-purpose utilities
- Can be used across multiple stages
- Not part of main pipeline flow

---

### `/Data` - Raw Sensor Data
**Purpose**: Original, unmodified sensor data organized by equipment and exercise type

**Structure**:
```
Data/
├── Barbell/
│   ├── Back_Squats/
│   │   ├── Clean/
│   │   ├── Uncontrolled Movement/
│   │   └── Inclination Asymmetry/
│   └── Bench_Press/
├── Dumbbell/
│   ├── Concentration_Curls/
│   └── Overhead_Extension/
└── Weight_Stack/
    ├── Lateral_Pulldown/
    └── Seated_Leg_Extension/
```

**Important**: Never modify files in this directory directly. Always work with copies.

---

### `/outputs` - Training Outputs & Results
**Purpose**: Consolidated location for all generated outputs

**Subdirectories**:
- `realistic/` - ⭐ Random Forest model outputs (current)
- `xgboost/` - ⭐ XGBoost model outputs (current)
- `comparison/` - ⭐ Model comparison results (current)
- `output_fixed/` - Legacy fixed version outputs
- `output/` - Legacy original outputs (has data leakage)
- `visualizations/` - Generated plots and figures
- `visualizations_chapter4/` - Specific chapter figures

**Typical contents**:
```
realistic/
├── models/
│   └── [Exercise_Name]/
│       ├── model.pkl
│       ├── scaler.pkl
│       ├── feature_names.pkl
│       ├── model_metadata.json
│       └── classification_report_*.txt
└── visualizations/
    ├── confusion_matrix.png
    ├── feature_importance.png
    └── roc_curves.png
```

---

### `/tests` - Test Data & Experiments
**Purpose**: Test datasets, calibration data, and experimental results

**Contents**:
- `test_data/` - Test datasets for validation (formerly "Test/")
- `calibration_test/` - IMU calibration experiments (formerly "Calibration Test/")
- `cloud_latency_test.csv` - Cloud processing latency measurements
- `imu_comparison_*.csv` - IMU sensor comparison data

**Why separate from Data/?**
- Test data is different from training data
- Experimental/temporary nature
- Calibration and validation purposes
- Not part of main dataset

---

## 🔄 Migration Guide

### Old Structure → New Structure

| Old Location | New Location | Reason |
|--------------|--------------|--------|
| `*.md` (root) | `docs/*.md` | Centralize documentation |
| `rf_classifier_*.py` | `src/models/` | Group model training |
| `resegment_*.py` | `src/pipeline/` | Pipeline stage |
| `re_labeler.py` | `src/pipeline/` | Pipeline stage |
| `*_visualizer.py` | `src/visualization/` | Group viz tools |
| `fix_false_positives.py` | `src/analysis/` | Analysis tool |
| `test_*.py` | `src/analysis/` | Testing tools |
| `axis_mapping_gui.py` | `src/utils/` | Utility tool |
| `column_remover.py` | `src/utils/` | Utility tool |
| `Test/` | `tests/test_data/` | Clearer naming |
| `Calibration Test/` | `tests/calibration_test/` | Clearer naming |
| `output*/` | `outputs/output*/` | Consolidate outputs |
| `visualizations/` | `outputs/visualizations/` | Group with outputs |

---

## 🎯 Design Principles

### 1. Separation of Concerns
- Documentation separate from code
- Code organized by functionality
- Data separate from outputs
- Tests separate from production data

### 2. Discoverability
- Clear folder names
- Logical grouping
- README in root
- Documentation index

### 3. Maintainability
- Related files together
- Clear naming conventions
- Minimal nesting
- Consistent structure

### 4. Scalability
- Easy to add new scripts
- Clear where new files go
- Modular organization
- No circular dependencies

---

## 📝 Naming Conventions

### Folders
- **Lowercase with underscores**: `test_data`, `output_fixed`
- **Descriptive names**: `visualization` not `viz`
- **Plural for collections**: `docs`, `outputs`, `tests`

### Files
- **Snake_case**: `rf_classifier_realistic.py`
- **Descriptive**: `fix_false_positives.py` not `fix_fp.py`
- **Consistent suffixes**: `*_visualizer.py`, `*_pipeline.py`

---

## 🚀 Running Scripts After Reorganization

### Option 1: Run from project root (Recommended)
```bash
cd /path/to/AppLift-ML-Training
python src/models/rf_classifier_realistic.py
python src/pipeline/preprocessing_pipeline.py
```

### Option 2: Add src to Python path
```python
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))
```

### Option 3: Install as package (Advanced)
```bash
pip install -e .
```

---

## 🔍 Finding Files

### "Where is...?"

**Pipeline scripts?** → `src/pipeline/`  
**Model training?** → `src/models/rf_classifier_realistic.py`  
**Visualization tools?** → `src/visualization/`  
**Documentation?** → `docs/ML_PIPELINE_MASTER_README.md`  
**Test data?** → `tests/test_data/`  
**Trained models?** → `outputs/output_fixed/models/`  
**Plots and figures?** → `outputs/visualizations/`  

---

## ✅ Benefits of New Structure

### Before (Flat Structure)
❌ 30+ files in root directory  
❌ Hard to find specific files  
❌ Documentation mixed with code  
❌ Unclear which scripts to use  
❌ No clear organization  

### After (Organized Structure)
✅ Clear separation of concerns  
✅ Easy to navigate  
✅ Documentation centralized  
✅ Clear entry points  
✅ Scalable and maintainable  
✅ Professional structure  

---

## 🔮 Future Additions

### Where to add new files?

**New preprocessing step?** → `src/pipeline/`  
**New model architecture?** → `src/models/`  
**New visualization?** → `src/visualization/`  
**New analysis tool?** → `src/analysis/`  
**New utility?** → `src/utils/`  
**New documentation?** → `docs/`  
**New test data?** → `tests/`  

---

## 📊 Structure Statistics

- **Total folders**: 11 main directories
- **Documentation files**: 6 markdown files
- **Pipeline scripts**: 4 files
- **Model scripts**: 5 files
- **Visualization tools**: 6 files
- **Analysis tools**: 4 files
- **Utility scripts**: 3 files

---

**Document Version**: 1.0  
**Last Updated**: March 28, 2026  
**Status**: Complete
