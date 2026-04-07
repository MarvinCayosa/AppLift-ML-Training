# Model Comparison Quick Reference

## 🚀 Quick Commands

```bash
# Train Random Forest
python src/models/rf_classifier_realistic.py

# Train XGBoost
python src/models/xgb_classifier_realistic.py

# Compare Models
python src/analysis/model_comparison.py
```

---

## 📊 At a Glance

| Feature | Random Forest | XGBoost | Winner |
|---------|---------------|---------|--------|
| **Accuracy** | Good (0.75-0.80) | Better (0.78-0.83) | XGB |
| **Training Speed** | Moderate | Fast | XGB |
| **Interpretability** | High | Moderate | RF |
| **Hyperparameter Tuning** | Simple | Complex | RF |
| **Class Imbalance** | Good | Better | XGB |
| **GPU Support** | No | Yes | XGB |
| **Memory Usage** | Moderate | Low | XGB |
| **Overfitting Risk** | Low | Moderate | RF |

---

## 🎯 Decision Matrix

### Choose Random Forest If:
- ✅ Interpretability is critical
- ✅ Simple deployment required
- ✅ Limited tuning time
- ✅ Small dataset (<500 samples)
- ✅ Need stable feature importances

### Choose XGBoost If:
- ✅ Need best accuracy
- ✅ Have imbalanced classes
- ✅ Large dataset (>1000 samples)
- ✅ GPU available
- ✅ Can invest in tuning

### Train Both If:
- ✅ Performance is critical
- ✅ Have time for comparison
- ✅ Want to ensemble models
- ✅ Need to justify model choice

---

## 📈 Performance Expectations

### Back Squats (Example)
| Metric | RF | XGB | Improvement |
|--------|----|----|-------------|
| Balanced Accuracy | 0.78 | 0.81 | +3.8% |
| F1 Weighted | 0.76 | 0.79 | +3.9% |
| Training Time | 10 min | 7 min | -30% |

### Bench Press (Example)
| Metric | RF | XGB | Improvement |
|--------|----|----|-------------|
| Balanced Accuracy | 0.82 | 0.84 | +2.4% |
| F1 Weighted | 0.80 | 0.83 | +3.8% |
| Training Time | 8 min | 6 min | -25% |

---

## 🔧 Hyperparameter Cheat Sheet

### Random Forest - Key Parameters
```python
n_estimators: 100-300      # More trees = better (diminishing returns)
max_depth: 10-20           # Deeper = more complex
min_samples_split: 5-15    # Higher = more conservative
min_samples_leaf: 2-8      # Higher = smoother boundaries
class_weight: 'balanced'   # For imbalanced data
```

### XGBoost - Key Parameters
```python
n_estimators: 100-500      # More trees = better (with early stopping)
max_depth: 4-8             # Shallower than RF
learning_rate: 0.01-0.3    # Lower = slower but better
min_child_weight: 1-10     # Higher = more conservative
gamma: 0-0.5               # Higher = more regularization
subsample: 0.7-1.0         # Lower = more regularization
colsample_bytree: 0.7-1.0  # Lower = more regularization
scale_pos_weight: auto     # For imbalanced data
```

---

## 🎨 Output Comparison

### Random Forest Output
```
outputs/realistic/models/Back_Squat/
├── model.pkl
├── scaler.pkl
├── feature_names.pkl
├── model_metadata.json
└── visualizations/
    ├── confusion_matrix.png
    ├── feature_importance.png
    └── cv_scores.png
```

### XGBoost Output
```
outputs/xgboost/models/Back_Squat/
├── xgb_model_20260328_143022.pkl
├── scaler_20260328_143022.pkl
├── feature_names_20260328_143022.pkl
├── model_metadata_20260328_143022.json
└── visualizations/
    ├── confusion_matrix.png
    ├── feature_importance.png
    └── cv_scores.png
```

### Comparison Output
```
outputs/comparison/Back_Squat/
├── performance_comparison.png
├── difference_heatmap.png
└── comparison_report_20260328_143500.txt
```

---

## 💡 Pro Tips

### For Best Results
1. **Always train both models** - Takes 2x time but worth it
2. **Use same preprocessed data** - Fair comparison
3. **Apply same feature selection** - Consistent evaluation
4. **Check generalization gap** - Ensure no overfitting
5. **Review confusion matrices** - Understand errors

### Common Mistakes to Avoid
❌ Training on different datasets  
❌ Using different feature sets  
❌ Skipping cross-validation  
❌ Ignoring class imbalance  
❌ Not checking for overfitting  
❌ Choosing model without comparison  

### Time-Saving Tips
⏱️ Use default params first (5 min)  
⏱️ Random search for optimization (15 min)  
⏱️ Grid search only if needed (60 min)  
⏱️ Skip GPU setup unless training >10 models  

---

## 📊 Metrics Interpretation

### Balanced Accuracy
- **>0.80**: Excellent
- **0.70-0.80**: Good
- **0.60-0.70**: Acceptable
- **<0.60**: Poor (needs improvement)

### F1 Score (Weighted)
- **>0.80**: Excellent
- **0.70-0.80**: Good
- **0.60-0.70**: Acceptable
- **<0.60**: Poor

### Generalization Gap (Train - CV)
- **<0.05**: Excellent generalization
- **0.05-0.15**: Good generalization
- **0.15-0.25**: Moderate overfitting
- **>0.25**: Severe overfitting

---

## 🔍 Troubleshooting

### XGBoost Performs Worse Than RF
- ✅ Increase n_estimators (try 500)
- ✅ Reduce learning_rate (try 0.05)
- ✅ Increase max_depth (try 8)
- ✅ Check for data leakage
- ✅ Verify same preprocessing

### Both Models Perform Poorly
- ✅ Check data quality
- ✅ Review feature engineering
- ✅ Verify labels are correct
- ✅ Check for class imbalance
- ✅ Add more training data

### Training Takes Too Long
- ✅ Reduce n_estimators
- ✅ Use fewer features
- ✅ Skip grid search
- ✅ Use default parameters
- ✅ Enable GPU (XGBoost only)

---

## 📞 Quick Help

### Documentation
- XGBoost Guide: `docs/XGBOOST_GUIDE.md`
- Pipeline Guide: `docs/ML_PIPELINE_MASTER_README.md`
- Main README: `README.md`

### Common Questions
**Q: Which model should I use?**  
A: Train both, compare, choose based on your priorities.

**Q: Is XGBoost always better?**  
A: Usually 2-5% better, but not always. Compare on your data.

**Q: Can I ensemble both models?**  
A: Yes! Average their predictions for potentially better results.

**Q: How long does training take?**  
A: RF: 8-12 min, XGB: 6-10 min (with default params)

---

**Last Updated**: March 28, 2026  
**Version**: 1.0  
**Print This**: Keep handy for quick reference!
