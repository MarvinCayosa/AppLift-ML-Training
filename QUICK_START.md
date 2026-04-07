# Quick Start Guide

## 🚀 Get Started in 5 Minutes

This guide will get you up and running with the AppLift ML Training pipeline quickly.

---

## 📋 Prerequisites

- Python 3.8 or higher
- pip package manager
- 4GB+ RAM
- Windows/Linux/macOS

---

## ⚡ Installation

### 1. Clone or Navigate to Project
```bash
cd /path/to/AppLift-ML-Training
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

**Required packages**:
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- imbalanced-learn

---

## 🎯 Your First Model Training

### Option A: Use Existing Preprocessed Data

If you already have cleaned and labeled data:

```bash
python src/models/rf_classifier_realistic.py
```

**What happens**:
1. File selector opens → Choose your CSV
2. Feature selector opens → Select features (or use defaults)
3. Configure dimensionality reduction → Choose Method 4 (recommended)
4. Configure class imbalance → Choose auto-detect
5. Training starts → Wait 5-15 minutes
6. Results saved to `outputs/output_fixed/`

### Option B: Full Pipeline from Raw Data

If starting with raw sensor data:

**Step 1: Preprocess**
```bash
python src/pipeline/preprocessing_pipeline.py
```
- Select your data folder
- Review cleaning report
- Output: `*_cleaned.csv`

**Step 2: Resegment (Recommended)**
```bash
python src/pipeline/resegment_reps_fixed.py
```
- Select cleaned CSV
- Review visualization
- Output: `*_resegmented.csv`

**Step 3: Train Model**
```bash
python src/models/rf_classifier_realistic.py
```
- Select resegmented CSV
- Follow prompts
- Get trained model!

---

## 📊 View Your Results

### Check Model Performance
```bash
# Navigate to outputs
cd outputs/output_fixed/models/[Exercise_Name]/

# View classification report
cat classification_report_*.txt
```

### View Visualizations
```bash
cd outputs/output_fixed/models/[Exercise_Name]/visualizations/

# Files generated:
# - confusion_matrix.png
# - feature_importance.png
# - roc_curves.png
# - precision_recall_curves.png
```

---

## 🔍 Common Tasks

### Visualize Your Data

**View individual reps**:
```bash
python src/visualization/rep_visualizer.py
```

**Compare clean vs error reps**:
```bash
python src/visualization/rep_comparison_visualizer.py
```

### Fix Quality Labels

If you need to correct labels:
```bash
python src/pipeline/re_labeler.py
```
- Interactive GUI opens
- Review reps one by one
- Correct labels
- Save changes

### Test Trained Model

```bash
python src/models/model_tester.py
```
- Load your trained model
- Test on new data
- View predictions

---

## 📖 Learn More

### Essential Documentation

1. **Start Here**: `docs/ML_PIPELINE_MASTER_README.md`
   - Complete pipeline guide
   - All 7 stages explained
   - Best practices

2. **Project Structure**: `STRUCTURE.md`
   - Where everything is
   - How to navigate
   - File organization

3. **Quick Reference**: `README.md`
   - Project overview
   - Common tasks
   - Troubleshooting

---

## ⚠️ Important Notes

### Which Script to Use?

**✅ ALWAYS USE**: `src/models/rf_classifier_realistic.py`

**❌ NEVER USE**:
- `src/models/rf_classifier.py` (has data leakage)
- `src/models/rf_classifier_copy.py` (has data leakage)

### Where Are My Outputs?

**Use this folder**: `outputs/output_fixed/`

**Avoid**: `outputs/output/` (from buggy version)

---

## 🐛 Troubleshooting

### Import Errors

If you get import errors after reorganization:

```python
# Add this at the top of your script
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))
```

Or run from project root:
```bash
cd /path/to/AppLift-ML-Training
python src/models/rf_classifier_realistic.py
```

### Low Model Performance

1. ✅ Check you're using `rf_classifier_realistic.py`
2. ✅ Ensure data is resegmented
3. ✅ Review preprocessing quality
4. ✅ Check class imbalance settings
5. ✅ See `docs/ML_PIPELINE_MASTER_README.md` troubleshooting

### File Not Found

Make sure you're running from project root:
```bash
pwd  # Should show: .../AppLift-ML-Training
ls   # Should show: src/, docs/, Data/, etc.
```

---

## 🎓 Next Steps

### After Your First Model

1. **Evaluate Performance**
   - Check confusion matrix
   - Review feature importance
   - Analyze per-class metrics

2. **Improve Model**
   - Collect more data if needed
   - Try different dimensionality reduction
   - Adjust hyperparameters
   - Fix false positives (see `docs/BACK_SQUATS_FALSE_POSITIVE_FIX.md`)

3. **Deploy**
   - Export model artifacts
   - Test on new data
   - Monitor performance

### Explore Advanced Features

- **Hyperparameter tuning**: Grid search or random search
- **Threshold optimization**: Reduce false positives
- **Cross-validation**: Assess generalization
- **Feature engineering**: Add custom features

---

## 📞 Need Help?

### Documentation Resources

1. **Pipeline Guide**: `docs/ML_PIPELINE_MASTER_README.md`
2. **Structure Guide**: `STRUCTURE.md`
3. **Comparison Guide**: `docs/COMPARISON_GUIDE.md`
4. **Visualizer Guide**: `docs/VISUALIZER_README.md`

### Common Questions

**Q: How long does training take?**  
A: 5-30 minutes depending on dataset size and hyperparameter search

**Q: How much data do I need?**  
A: Minimum 50 reps per class, recommended 200+ per class

**Q: Can I use GPU?**  
A: Random Forest doesn't use GPU, but runs efficiently on CPU

**Q: What if my CV score is low?**  
A: Scores of 0.70-0.85 are normal. Below 0.60 indicates issues.

---

## ✅ Checklist

Before you start:
- [ ] Python 3.8+ installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Data in `Data/` folder or ready to load
- [ ] Read this quick start guide

After training:
- [ ] Check classification report
- [ ] Review confusion matrix
- [ ] Verify CV score > 0.60
- [ ] Check generalization gap < 0.20
- [ ] Review feature importance

---

## 🎉 You're Ready!

You now have everything you need to:
- ✅ Preprocess sensor data
- ✅ Train Random Forest models
- ✅ Evaluate performance
- ✅ Visualize results
- ✅ Deploy models

**Start with**: `python src/models/rf_classifier_realistic.py`

Good luck! 🚀

---

**Version**: 1.0  
**Last Updated**: March 28, 2026  
**Status**: Ready to Use
