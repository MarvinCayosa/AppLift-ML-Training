# Exercise Rep Waveform Visualizers

Two professional visualization tools for analyzing exercise repetition data with different quality classifications.

## Tools

### 1. `rep_visualizer.py` - Single File Analyzer
Visualize individual CSV files with detailed waveform analysis.

**Features:**
- Load any CSV file from your dataset
- Select specific reps to analyze
- View multiple signal types: Acceleration, Gyroscope, Orientation, Filtered
- Three-panel layout showing X, Y, Z axes simultaneously
- Displays classification, participant, and file information

**Usage:**
```bash
python rep_visualizer.py
```

### 2. `rep_comparison_visualizer.py` - Classification Comparison
Compare waveforms across different quality classifications side-by-side.

**Features:**
- Load three CSV files simultaneously (Clean, Uncontrolled Movement, Inclination Asymmetry)
- Side-by-side comparison in three panels
- Select signal type and specific axis
- Color-coded by classification (Green=Clean, Red=Uncontrolled, Orange=Asymmetry)
- Ideal for thesis presentations showing classification differences

**Usage:**
```bash
python rep_comparison_visualizer.py
```

## Signal Types Available

1. **Acceleration**: accelX, accelY, accelZ (m/s²)
2. **Gyroscope**: gyroX, gyroY, gyroZ (rad/s)
3. **Orientation**: roll, pitch, yaw (degrees)
4. **Filtered**: filteredX, filteredY, filteredZ (m/s²)

## Thesis Presentation Tips

### For Single Rep Analysis:
- Use `rep_visualizer.py` to show detailed waveform characteristics
- Compare different reps within the same classification
- Highlight specific features in the time series

### For Classification Comparison:
- Use `rep_comparison_visualizer.py` to show clear differences between classifications
- Select the most representative axis (usually Y for vertical movement in squats)
- Use "All" axis view to show multi-dimensional differences

## Example Workflow

1. **Load Clean Example:**
   - `Data/Barbell/Back_Squats/Clean/P001_Barbell_Back_Squats_Clean_0_01.csv`

2. **Load Uncontrolled Movement:**
   - `Data/Barbell/Back_Squats/Uncontrolled Movement/P001_Barbell_Back_Squats_Uncontrolled_Movement_1_01.csv`

3. **Load Inclination Asymmetry:**
   - `Data/Barbell/Back_Squats/Inclination Asymmetry/P001_Barbell_Back_Squats_Inclination_Asymmetry_2_01.csv`

4. **Select Signal Type:** Acceleration (most common for movement analysis)

5. **Select Axis:** Y (vertical movement) or All (comprehensive view)

## Requirements

```bash
pip install pandas matplotlib numpy
```

## Data Structure Expected

CSV files should contain:
- `participant`: Participant ID
- `rep`: Repetition number
- `equipment_code`: Equipment identifier
- `exercise_code`: Exercise type (3=Back Squats, 4=Front Squats, 5=Bench Press)
- `quality_code`: Quality classification (0=Clean, 1=Uncontrolled, 2=Asymmetry)
- `timestamp_ms`: Timestamp in milliseconds
- Signal columns: accelX/Y/Z, gyroX/Y/Z, roll/pitch/yaw, filteredX/Y/Z

## Styling

Both visualizers use thesis-appropriate styling:
- Clean, professional fonts
- Grid lines for readability
- Color-coded classifications
- Proper axis labels and units
- Comprehensive titles with metadata
