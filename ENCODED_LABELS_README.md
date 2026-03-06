# Encoded Labels Reference Guide

## Overview
This document provides a comprehensive reference for all encoded labels used in the AppLift ML Training system for exercise form classification. The system uses a hierarchical encoding structure with three main categories: Equipment, Exercise, and Quality.

## Encoding Structure

The labeling system follows a three-tier hierarchical structure:
```
Equipment Type → Exercise Type → Quality Assessment
     ↓               ↓              ↓
   Code 0-2        Code 0-5       Code 0-2
```

## Equipment Types Encoding

| Code | Equipment Type | Description |
|------|----------------|-------------|
| 0    | Dumbbell       | Free weight exercises using dumbbells |
| 1    | Barbell        | Free weight exercises using barbells |
| 2    | Weight Stack   | Machine-based exercises with weight stacks |

## Exercise Types Encoding

| Code | Exercise Name        | Equipment | Description |
|------|---------------------|-----------|-------------|
| 0    | Concentration Curls | Dumbbell  | Isolated bicep curl exercise |
| 1    | Overhead Extension  | Dumbbell  | Tricep extension exercise |
| 2    | Bench Press         | Barbell   | Chest pressing exercise |
| 3    | Back Squat          | Barbell   | Lower body squat exercise |
| 4    | Lateral Pulldown    | Weight Stack | Upper body pulling exercise |
| 5    | Seated Leg Extension| Weight Stack | Isolated quadricep exercise |

## Quality Assessment Encoding

The quality assessment varies by exercise type to capture exercise-specific form issues:

### Dumbbell Exercises (Concentration Curls & Overhead Extension)

| Code | Quality Label        | Description |
|------|---------------------|-------------|
| 0    | Clean               | Perfect form execution |
| 1    | Uncontrolled Movement | Loss of control during movement |
| 2    | Abrupt Initiation   | Sudden, jerky start of movement |

**Exercises:** Concentration Curls (0), Overhead Extension (1)

### Barbell Exercises (Bench Press & Back Squat)

| Code | Quality Label          | Description |
|------|------------------------|-------------|
| 0    | Clean                  | Perfect form execution |
| 1    | Uncontrolled Movement  | Loss of control during movement |
| 2    | Inclination Asymmetry  | Uneven bar path or body positioning |

**Exercises:** Bench Press (2), Back Squat (3)

### Weight Stack Exercises (Lateral Pulldown & Seated Leg Extension)

| Code | Quality Label      | Description |
|------|--------------------|-------------|
| 0    | Clean              | Perfect form execution |
| 1    | Pulling Too Fast   | Excessive speed during concentric phase |
| 2    | Releasing Too Fast | Excessive speed during eccentric phase |

**Exercises:** Lateral Pulldown (4), Seated Leg Extension (5)

## Complete Label Mapping

### Equipment-Exercise-Quality Hierarchy

```
DUMBBELL (0)
├── Concentration Curls (0)
│   ├── 0: Clean
│   ├── 1: Uncontrolled Movement
│   └── 2: Abrupt Initiation
└── Overhead Extension (1)
    ├── 0: Clean
    ├── 1: Uncontrolled Movement
    └── 2: Abrupt Initiation

BARBELL (1)
├── Bench Press (2)
│   ├── 0: Clean
│   ├── 1: Uncontrolled Movement
│   └── 2: Inclination Asymmetry
└── Back Squat (3)
    ├── 0: Clean
    ├── 1: Uncontrolled Movement
    └── 2: Inclination Asymmetry

WEIGHT STACK (2)
├── Lateral Pulldown (4)
│   ├── 0: Clean
│   ├── 1: Pulling Too Fast
│   └── 2: Releasing Too Fast
└── Seated Leg Extension (5)
    ├── 0: Clean
    ├── 1: Pulling Too Fast
    └── 2: Releasing Too Fast
```

## Default Quality Labels

For backward compatibility and unknown exercises, the system uses default quality labels:

| Code | Default Quality Label |
|------|--------------------|
| 0    | Clean              |
| 1    | Uncontrolled Movement |
| 2    | Abrupt Initiation  |

## Usage in Code

### Python Constants
These encodings are defined as constants in the main classifier files:

```python
EQUIPMENT_TYPES = {
    0: 'Dumbbell',
    1: 'Barbell', 
    2: 'Weight Stack'
}

EXERCISE_TYPES = {
    0: 'Concentration Curls',
    1: 'Overhead Extension',
    2: 'Bench Press',
    3: 'Back Squat',
    4: 'Lateral Pulldown',
    5: 'Seated Leg Extension'
}

QUALITY_NAMES_BY_EXERCISE = {
    0: {0: 'Clean', 1: 'Uncontrolled Movement', 2: 'Abrupt Initiation'},
    1: {0: 'Clean', 1: 'Uncontrolled Movement', 2: 'Abrupt Initiation'},
    2: {0: 'Clean', 1: 'Uncontrolled Movement', 2: 'Inclination Asymmetry'},
    3: {0: 'Clean', 1: 'Uncontrolled Movement', 2: 'Inclination Asymmetry'},
    4: {0: 'Clean', 1: 'Pulling Too Fast', 2: 'Releasing Too Fast'},
    5: {0: 'Clean', 1: 'Pulling Too Fast', 2: 'Releasing Too Fast'}
}
```

### Data File Naming Convention
The merged dataset files follow the naming pattern:
- `[EQUIPMENT]_[EXERCISE]_[QUALITY]_merged_cleaned_[TIMESTAMP].csv`
- Example: `Barbell_Back_Squat_Clean_merged_cleaned_20260218_231048.csv`

## Data Processing Pipeline

1. **Phase 1**: Raw data collection and initial labeling
2. **Phase 2**: Data segmentation and rep extraction
3. **Phase 3**: Data cleaning and quality assessment
4. **Final**: Model training with encoded labels

## Model Output Interpretation

When the trained model makes predictions, it outputs:
- **Exercise Type**: Integer 0-5 (maps to exercise name)
- **Quality Assessment**: Integer 0-2 (maps to quality description based on exercise type)

Use the `QUALITY_NAMES_BY_EXERCISE` mapping to convert predictions back to human-readable labels.

## Notes

- All encodings use zero-based indexing
- Quality labels are context-aware and exercise-specific
- The system maintains backward compatibility with default quality labels
- Label consistency is enforced across the entire pipeline

---
*Last updated: February 22, 2026*  
*AppLift ML Training Pipeline v2*