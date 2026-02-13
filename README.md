# AppLift ML Training - Exercise Quality Classification

A machine learning system for detecting and classifying exercise quality from sensor data collected during gym workouts.

## 📋 Project Overview

This project uses sensor data (accelerometer and gyroscope) to:
1. **Detect exercise repetitions (reps)** using valley detection algorithms
2. **Classify exercise quality** using Random Forest classifiers
3. **Provide real-time feedback** on exercise form

## 🏋️ Exercise Classification System

### Equipment Types (equipment_code)

| Code | Equipment Type | Description |
|------|---------------|-------------|
| `0` | **Dumbbell** | Free weights - unilateral exercises |
| `1` | **Barbell** | Free weights - bilateral exercises |
| `2` | **Weight Stack** | Cable machine exercises |

---

### Exercise Types (exercise_code)

#### 🔹 Dumbbell Exercises (equipment_code = 0)

| Code | Exercise Name | Description |
|------|--------------|-------------|
| `0` | **Concentration Curls** | Seated bicep curls with elbow supported |
| `1` | **Overhead Extension** | Tricep extension overhead |

#### 🔹 Barbell Exercises (equipment_code = 1)

| Code | Exercise Name | Description |
|------|--------------|-------------|
| `2` | **Bench Press** | Upper body compound press movement |
| `3` | **Back Squat** | Lower body compound squat movement |

#### 🔹 Weight Stack Exercises (equipment_code = 2)

| Code | Exercise Name | Description |
|------|--------------|-------------|
| `4` | **Lateral Pulldown** | Upper body pulling movement (lat pulldown) |
| `5` | **Seated Leg Extension** | Lower body knee extension |

---

## 🎯 Quality Classification System (target/quality_code)

### Dumbbell Exercises (Concentration Curls, Overhead Extension)

| Code | Quality Label | Description |
|------|--------------|-------------|
| `0` | **Clean** | Proper form with controlled movement |
| `1` | **Uncontrolled Movement** | Loss of control during execution |
| `2` | **Abrupt Initiation** | Sudden/jerky start of movement |

### Barbell Exercises (Bench Press, Back Squat)

| Code | Quality Label | Description |
|------|--------------|-------------|
| `0` | **Clean** | Proper form with controlled movement |
| `1` | **Uncontrolled Movement** | Loss of control during execution |
| `2` | **Inclination Asymmetry** | Uneven bar path or asymmetric lifting |

### Weight Stack Exercises (Lateral Pulldown, Seated Leg Extension)

| Code | Quality Label | Description |
|------|--------------|-------------|
| `0` | **Clean** | Proper form with controlled movement |
| `1` | **Pulling Too Fast** | Excessive speed during concentric phase |
| `2` | **Releasing Too Fast** | Excessive speed during eccentric phase |

---

## 📁 Project Structure

```
AppLift ML Training/
├── README.md                          # This file
├── dataset_merger.py                  # Merges individual CSV files by exercise type
├── preprocessing_pipeline.py          # Data preprocessing and rep segmentation
├── resegment_reps_fixed.py           # Rep boundary correction using valley detection
├── rf_classifier.py                  # Random Forest classifier training
│
├── Dumbbell/                         # Raw sensor data
│   ├── Concentration_Curls/
│   │   ├── Abrupt Initiation/
│   │   ├── Clean/
│   │   └── Uncontrolled Movement/
│   └── Overhead_Extension/
│       ├── Abrupt Intitiation/
│       ├── Clean/
│       └── Uncontrolled Movement/
│
├── output/                           # Processed data and models
│   ├── merged_datasets/              # Exercise-specific merged datasets
│   │   ├── CONCENTRATION_CURLS_*.csv
│   │   ├── OVERHEAD_EXTENSION_*.csv
│   │   └── LAT_PULLDOWN_*.csv
│   ├── models/                       # Trained ML models
│   │   ├── rf_classifier_*.pkl
│   │   ├── classification_report_*.txt
│   │   └── feature_importance_*.csv
│   ├── phase1_merged/                # Initial merged data
│   ├── phase2_resegmented/           # Re-segmented rep boundaries
│   └── reports/                      # Processing reports
│
└── visualizations/                   # Analysis visualizations
    └── segmentation/
        └── participant_analysis/
```

---

## 🚀 Usage

### 1. Dataset Merging

Combine individual participant CSV files by exercise type:

```bash
python dataset_merger.py
```

**Output:** Merged datasets in `output/merged_datasets/`
- Example: `CONCENTRATION_CURLS_20260212_135130.csv`

### 2. Rep Resegmentation

Correct rep boundaries using valley detection:

```bash
python resegment_reps_fixed.py
```

**Features:**
- Detects exercise type from filename
- Applies exercise-specific parameters:
  - **Dumbbell/Barbell:** Standard parameters (prominence_factor=0.1, min_duration=800ms)
  - **Weight Stack:** Specialized parameters (prominence_factor=0.05, min_duration=1500ms, stronger smoothing)
- Processes each source file separately for accurate segmentation
- Preserves quality labels from `target` column

**Console Output:**
```
📋 Detected exercise: OVERHEAD_EXTENSION
📊 Quality distribution in data:
   - Clean: 1234 samples
   - Uncontrolled: 2345 samples
   - Abrupt Initiation: 3456 samples
ℹ️  Note: Quality labels are preserved from the 'target' column
✓ Using OVERHEAD_EXTENSION parameters
```

### 3. Model Training

Train Random Forest classifier with hyperparameter optimization:

```bash
python rf_classifier.py
```

**Features:**
- Grid Search with cross-validation
- Class imbalance handling with `class_weight='balanced'`
- Feature importance analysis
- Confusion matrix and classification reports

---

## 📊 Data Format

### Input CSV Structure

| Column | Type | Description |
|--------|------|-------------|
| `participant` | int | Participant ID (e.g., 1, 2, 3...) |
| `rep` | int | Repetition number within session |
| `equipment_code` | int | Equipment type (0=Dumbbell, 1=Barbell, 2=Weight Stack) |
| `exercise_code` | int | Exercise type (0-5, see table above) |
| `target` | int | Quality label (0=Clean, 1=Form Error 1, 2=Form Error 2) |
| `timestamp_ms` | float | Timestamp in milliseconds |
| `accelX`, `accelY`, `accelZ` | float | Raw accelerometer data |
| `gyroX`, `gyroY`, `gyroZ` | float | Raw gyroscope data |
| `filteredX`, `filteredY`, `filteredZ` | float | Filtered accelerometer data |
| `filteredMag` | float | Filtered magnitude (used for rep detection) |
| `source_file` | string | Original filename |

---

## 🔧 Technical Details

### Rep Segmentation Algorithm

1. **Signal Smoothing:** Savitzky-Golay filter
   - Window length: 11 (standard) or 21 (weight stack)
   - Polynomial order: 3

2. **Valley Detection:** Find local minima using inverted peak finding
   - Adaptive prominence based on signal characteristics
   - Minimum distance between valleys: 0.5s (standard) or 1.5s (weight stack)

3. **Rep Boundaries:** Valley-to-valley segmentation
   - Each rep starts at a valley (rest position)
   - Ends at the next valley (return to rest)

### Exercise-Specific Parameters

```python
# Dumbbell/Barbell (Concentration Curls, Overhead Extension, Bench Press, Back Squat)
prominence_factor = 0.1
min_prominence_floor = 0.1
min_rep_duration = 800-1200ms

# Weight Stack (Lateral Pulldown, Seated Leg Extension)
prominence_factor = 0.05
min_prominence_floor = 0.05
min_rep_duration = 1500ms
stronger_smoothing = True  # Window length: 21
```

### Classification Features

- Time-domain features (mean, std, min, max)
- Frequency-domain features (FFT)
- Statistical features (skewness, kurtosis)
- Sensor magnitude features
- Rep duration features

---

## 📈 Performance

### Model Metrics

- **Accuracy:** Typically 85-95% depending on exercise type
- **Cross-Validation:** 5-fold stratified K-fold
- **Class Imbalance Handling:** Automatic class weighting

### Common Issues & Solutions

**Issue:** Model confusing Clean and Uncontrolled classes
- **Solution:** `class_weight='balanced'` parameter added to Random Forest
- **Status:** ✅ Fixed in latest version

**Issue:** Only 2-4 reps detected instead of 20
- **Solution:** Per-source-file processing instead of merged dataset processing
- **Status:** ✅ Fixed in latest version

---

## 👥 Contributors

- **Marvin Cayosa** - Project Lead

---

## 📝 Notes

### Important Reminders

1. **Quality Labels are Exercise-Specific:**
   - Don't mix quality labels between exercise types
   - Each exercise has its own set of 3 quality classes

2. **Filename Convention:**
   - Merged datasets: `<EXERCISE_NAME>_<YYYYMMDD>_<HHMMSS>.csv`
   - Exercise name extracted from filename for parameter selection

3. **Resegmentation Behavior:**
   - Resegmentation **ONLY fixes rep boundaries**
   - Quality labels (`target` column) are **PRESERVED**, not changed
   - Exercise-specific parameters based on **exercise type**, not quality

4. **Data Processing Pipeline:**
   ```
   Raw CSV → Dataset Merger → Preprocessing → Resegmentation → Model Training → Predictions
   ```

---

## 🔮 Future Improvements

- [ ] Real-time exercise detection
- [ ] Mobile app integration
- [ ] Additional exercise types
- [ ] Deep learning models (LSTM, CNN)
- [ ] Form correction recommendations

---

## 📞 Support

For questions or issues, please contact the project team.

---

**Last Updated:** February 13, 2026
**Version:** 1.0.0
