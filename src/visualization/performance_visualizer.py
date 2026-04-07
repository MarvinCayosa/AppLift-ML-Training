"""
AppLift ML Training - Performance Visualizer
=============================================
Comprehensive performance analysis including:
- Gyroscope-based fatigue analysis (F = 0.35·D_ω + 0.25·I_T + 0.20·I_J + 0.20·I_S)
- Consistency analysis (performance variability assessment)  
- Jerk-based smoothness tracking (movement quality)
- Range of Motion (ROM) per rep (degrees from accelerometer)
- Session-over-session comparisons
- Detailed performance reports with key findings

Features:
- Peak angular velocity |change| (D_ω) from gyroscope — detects both speed drop and surge
- Tempo / duration increase (I_T) for endurance tracking
- RMS jerk increase (I_J) for motor control degradation
- Shakiness increase (I_S) — within-rep angular acceleration RMS for tremor detection
- Coefficient of variation for consistency measurement
- Linear regression for trend analysis
- Comprehensive text-based performance reports
- Multi-panel visualizations with actionable insights

Technical Notes:
- Clean movements: Mean Jerk < 12, CV < 10%
- Uncontrolled movements: Mean Jerk > 18, CV > 20%
- Fatigue formula: 0.35·D_ω + 0.25·I_T + 0.20·I_J + 0.20·I_S (with boost for severe indicators)
- Fatigue levels: <10% minimal, 10-20% low, 20-35% moderate, 35-55% high, >55% severe

Author: AppLift ML Training Pipeline
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from scipy import stats
from scipy.signal import savgol_filter, find_peaks
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).parent.resolve()
OUTPUT_DIR = SCRIPT_DIR / 'output'
VISUALIZATIONS_DIR = OUTPUT_DIR / 'visualizations' / 'performance'
VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)

# Color schemes
COLORS = {
    'primary': '#2196F3',
    'success': '#4CAF50',
    'warning': '#FF9800',
    'danger': '#f44336',
    'purple': '#9C27B0',
    'teal': '#00BCD4',
    'gradient_good': '#4CAF50',
    'gradient_bad': '#f44336',
}

# Irregularity thresholds (higher = more irregular = less smooth)
# Combines: normalized jerk (jerk/ROM), direction changes, peak count
# Scale: 0-100 where 0=perfectly smooth, 100=extremely irregular
# Calibrated from real IMU sensor data:
# - Clean movements: low irregularity (few direction changes, single peak)
# - Uncontrolled movements: high irregularity (many direction changes, multi-peak)
JERK_THRESHOLDS = {
    'excellent': 20,   # < 20 = excellent smoothness (single smooth curve)
    'good': 35,        # 20-35 = good (minor irregularities)
    'fair': 55,        # 35-55 = fair (noticeable irregularities)
    'poor': 75         # > 55 = poor (very irregular, multi-peaked)
}

# Consistency thresholds based on Coefficient of Variation (CV)
# CV = std/mean — lower is more consistent
CONSISTENCY_CV_THRESHOLDS = {
    'excellent': 0.08,  # CV < 8% = excellent (92%+ consistency)
    'good': 0.15,       # CV 8-15% = good (70-92% consistency)
    'fair': 0.25,       # CV 15-25% = fair (50-70% consistency)
    'poor': 0.40        # CV > 25% = poor (< 50% consistency)
}


# =============================================================================
# METRIC COMPUTATION FUNCTIONS
# =============================================================================

def compute_angle_from_accelerometer(accel_x, accel_y, accel_z):
    """
    Compute tilt angle (in degrees) from accelerometer data.
    
    Uses the gravity component to estimate arm/limb orientation.
    For dumbbell exercises, this gives the angle of the arm relative to vertical.
    
    Parameters:
    - accel_x, accel_y, accel_z: Acceleration arrays for each axis (in m/s² or g)
    
    Returns:
    - angles: Array of angles in degrees (0° = vertical, 90° = horizontal)
    """
    try:
        accel_x = np.array(accel_x)
        accel_y = np.array(accel_y)
        accel_z = np.array(accel_z)
        
        # Calculate total magnitude
        magnitude = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
        
        # Avoid division by zero
        magnitude = np.where(magnitude == 0, 1e-10, magnitude)
        
        # Calculate pitch angle (rotation around X-axis)
        # This measures the angle of the sensor relative to horizontal
        # For a curl: 0° at bottom (arm down), increasing as arm curls up
        pitch = np.arctan2(accel_y, np.sqrt(accel_x**2 + accel_z**2))
        
        # Convert to degrees
        pitch_degrees = np.degrees(pitch)
        
        # Also calculate roll for more complete picture
        roll = np.arctan2(accel_x, accel_z)
        roll_degrees = np.degrees(roll)
        
        # For curl exercises, the primary motion is pitch
        # Normalize to 0-180 range (arm fully down to fully up)
        # Add 90 to convert from -90/+90 to 0/180 range
        angle = pitch_degrees + 90
        
        return angle
        
    except Exception as e:
        print(f"  ⚠️ Error computing angle: {e}")
        return np.zeros(len(accel_x))


def compute_rom_degrees(accel_x, accel_y, accel_z):
    """
    Compute Range of Motion in degrees from accelerometer data.
    
    Parameters:
    - accel_x, accel_y, accel_z: Acceleration arrays for each axis
    
    Returns:
    - rom_degrees: ROM in degrees
    - min_angle: Minimum angle (degrees)
    - max_angle: Maximum angle (degrees)
    - angle_array: Full angle array for visualization
    """
    try:
        angles = compute_angle_from_accelerometer(accel_x, accel_y, accel_z)
        
        min_angle = np.min(angles)
        max_angle = np.max(angles)
        rom_degrees = max_angle - min_angle
        
        return {
            'rom_degrees': rom_degrees,
            'min_angle': min_angle,
            'max_angle': max_angle,
            'mean_angle': np.mean(angles),
            'angle_array': angles
        }
        
    except Exception as e:
        print(f"  ⚠️ Error computing ROM degrees: {e}")
        return {
            'rom_degrees': 0,
            'min_angle': 0,
            'max_angle': 0,
            'mean_angle': 0,
            'angle_array': np.array([])
        }


def compute_ldlj(accel_x, accel_y, accel_z, timestamps_ms, filtered_mag=None):
    """
    Compute smoothness metrics combining NORMALIZED JERK and IRREGULARITY DETECTION.
    
    Key insight: Absolute jerk fails to detect jagged/irregular movements when amplitude
    is small. We use:
    1. Normalized jerk = jerk / ROM (relative jerkiness)
    2. Direction changes = zero crossings in velocity (irregularity)
    3. Peak count = number of peaks/valleys (multi-peaked = irregular)
    
    Parameters:
    - accel_x, accel_y, accel_z: Acceleration arrays for each axis
    - timestamps_ms: Timestamps in milliseconds
    - filtered_mag: Optional filtered magnitude signal (for better irregularity detection)
    
    Returns:
    - irregularity_score: Combined metric (higher = MORE irregular, LESS smooth)
    - smoothness_score: Inverted score (higher = smoother, 0-100 scale)
    """
    try:
        if len(accel_x) < 4:
            return 0, 50
        
        # Duration in seconds
        duration = (timestamps_ms[-1] - timestamps_ms[0]) / 1000.0
        
        if duration <= 0:
            return 0, 50
        
        # Compute time step (average sample interval in seconds)
        dt = duration / (len(timestamps_ms) - 1)
        
        # Compute acceleration magnitude (or use provided filtered_mag)
        if filtered_mag is not None and len(filtered_mag) == len(accel_x):
            signal = np.array(filtered_mag)
        else:
            signal = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
        
        # Range of motion (amplitude)
        rom = np.max(signal) - np.min(signal)
        if rom < 0.1:  # Avoid division by zero
            rom = 0.1
        
        # === METRIC 1: Normalized Jerk (jerk relative to ROM) ===
        # Velocity (first derivative)
        velocity = np.diff(signal) / dt
        # Jerk (second derivative) 
        jerk = np.abs(np.diff(velocity) / dt) if len(velocity) > 1 else np.array([0])
        mean_jerk = np.mean(jerk) if len(jerk) > 0 else 0
        normalized_jerk = mean_jerk / rom  # Typical range: clean=1.4-2.1, poor=4-6+
        
        # === METRIC 2: Direction Changes (zero crossings in velocity) ===
        # More direction changes = more irregular movement
        sign_changes = np.diff(np.sign(velocity))
        direction_changes = np.sum(sign_changes != 0)
        # Normalize by duration (changes per second)
        # Clean movement: 0.2-0.5/sec, Poor: 3+/sec
        direction_rate = direction_changes / duration if duration > 0 else 0
        
        # === METRIC 3: Peak Detection (multi-peaked = irregular) ===
        # A clean rep should have 1-2 peaks, irregular has many
        # Use adaptive prominence based on ROM
        prominence = rom * 0.05  # 5% of ROM is significant
        peaks, _ = find_peaks(signal, prominence=prominence)
        valleys, _ = find_peaks(-signal, prominence=prominence)
        total_peaks = len(peaks) + len(valleys)
        # Clean=1-2 peaks, Poor=7-13 peaks
        excess_peaks = max(0, total_peaks - 2)
        
        # === COMBINE INTO IRREGULARITY SCORE ===
        # Calibrated from MARVIN FATIGUE TEST data:
        # Clean reps: norm_jerk=1.4-2.1, dir_rate=0.2-0.5, peaks=1
        # Poor reps:  norm_jerk=4.1-6.1, dir_rate=3.1-3.3, peaks=7-13
        
        # Normalized jerk contribution (0-40 range)
        # Baseline ~1.5, then scale: 2.0→7, 3.0→20, 5.0→40
        jerk_contrib = min(40, max(0, (normalized_jerk - 1.5)) * 13.3)
        
        # Direction changes contribution (0-35 range)
        # Baseline ~0.5, then scale: 1.0→5, 2.0→15, 4.0→35
        dir_contrib = min(35, max(0, (direction_rate - 0.5)) * 10)
        
        # Excess peaks contribution (0-25 range)
        # 0 extra→0, 3 extra→10, 5 extra→17, 8+→25
        peaks_contrib = min(25, excess_peaks * 3.3)
        
        # Total irregularity score (0-100, higher = more irregular)
        irregularity_score = jerk_contrib + dir_contrib + peaks_contrib
        
        # === CONVERT TO SMOOTHNESS SCORE ===
        # Smoothness = 100 - irregularity (inverted)
        smoothness_score = max(0, min(100, 100 - irregularity_score))
        
        # Return irregularity as the "jerk-like" metric (higher = worse)
        # This maintains API compatibility (first return is "jerk", lower is better)
        return irregularity_score, smoothness_score
        
    except Exception as e:
        return 0, 50


def compute_range_of_motion(signal, signal_name='filteredMag'):
    """
    Compute Range of Motion (ROM) metrics for a rep.
    
    Parameters:
    - signal: Signal array (e.g., filteredMag, angle, position)
    - signal_name: Name of the signal for context
    
    Returns:
    - rom_metrics: Dictionary with ROM values
    """
    if len(signal) < 2:
        return {'rom': 0, 'rom_normalized': 0, 'peak': 0, 'trough': 0}
    
    # Basic ROM = max - min
    peak = np.max(signal)
    trough = np.min(signal)
    rom = peak - trough
    
    # Normalized ROM (0-100 scale based on typical ranges)
    rom_normalized = min(100, (rom / np.mean(np.abs(signal))) * 50) if np.mean(np.abs(signal)) > 0 else 0
    
    return {
        'rom': rom,
        'rom_normalized': rom_normalized,
        'peak': peak,
        'trough': trough,
        'mean': np.mean(signal),
        'std': np.std(signal)
    }


def compute_fatigue_indicators(rep_metrics_list):
    """
    Compute fatigue using gyroscope-based methodology.
    
    Scientifically-grounded fatigue score based on four indicators:
    
    1. D_ω (Peak Angular Velocity Change) - 35% weight
       Measures ABSOLUTE change in peak gyro magnitude from early to late reps.
       D_ω = |ω_peak_first - ω_peak_last| / ω_peak_first
       Both DROP (muscles slowing) and SURGE (compensatory swinging) = fatigue.
    
    2. I_T (Tempo / Duration Increase) - 25% weight
       Measures slowing of rep execution speed.
       I_T = (T_last - T_first) / T_first
    
    3. I_J (Jerk / Smoothness Increase) - 20% weight
       Measures increase in movement jerkiness.
       I_J = (J_last - J_first) / J_first
    
    4. I_S (Shakiness Increase) - 20% weight
       Measures increase in within-rep angular acceleration RMS.
       I_S = (S_last - S_first) / S_first
       Detects tremor/shaky movement quality degradation.
    
    Final: Fatigue = 0.35 * D_ω + 0.25 * I_T + 0.20 * I_J + 0.20 * I_S
    
    Parameters:
    - rep_metrics_list: List of dicts with metrics per rep
    
    Returns:
    - fatigue_analysis: Dictionary with comprehensive fatigue indicators
    """
    if len(rep_metrics_list) < 3:
        return {
            'fatigue_score': 0,
            'fatigue_level': 'insufficient_data',
            'D_omega': 0,
            'I_T': 0,
            'I_J': 0,
            'I_S': 0,
            'gyro_direction': 'stable',
            'consistency_score': 0,
            'rom_trend': 0,
            'smoothness_trend': 0,
            'duration_trend': 0,
            'peak_trend': 0,
            'rom_consistency': 0,
            'smoothness_consistency': 0,
            'duration_consistency': 0,
            'peak_consistency': 0,
            # Legacy fields for backward compatibility
            'rom_change_percent': 0,
            'smoothness_degradation_percent': 0,
            # New nested structure
            'early_vs_late': {
                'rom_change_percent': 0,
                'smoothness_degradation_percent': 0,
                'duration_increase_percent': 0,
                'peak_degradation_percent': 0,
                'gyro_peak_change_percent': 0,
                'jerk_increase_percent': 0,
                'shakiness_increase_percent': 0,
                'avg_gyro_peak_first': 0,
                'avg_gyro_peak_last': 0,
                'avg_shakiness_first': 0,
                'avg_shakiness_last': 0,
                'avg_rom_first': 0,
                'avg_rom_last': 0,
                'avg_smoothness_first': 0,
                'avg_smoothness_last': 0,
                'avg_duration_first_ms': 0,
                'avg_duration_last_ms': 0
            },
            'performance_report': {
                'session_quality': 'Insufficient Data',
                'consistency_rating': 'Unknown',
                'key_findings': ['⚠️ Need at least 3 reps for analysis']
            },
            'n_reps_analyzed': len(rep_metrics_list)
        }
        
    # Proceed with full analysis if enough reps available
    try:
        n_reps = len(rep_metrics_list)
        rep_indices = np.arange(n_reps)
        third = max(1, n_reps // 3)
        
        # =====================================================================
        # EXTRACT PER-REP METRICS
        # =====================================================================
        
        # Gyroscope peak angular velocity per rep
        gyro_peaks = np.array([m.get('gyro_peak', 0) for m in rep_metrics_list])
        has_gyro = any(g > 0 for g in gyro_peaks)
        
        # Shakiness per rep (angular acceleration RMS — within-rep tremor)
        shakiness_vals = np.array([m.get('shakiness', 0) for m in rep_metrics_list])
        has_shakiness = any(s > 0 for s in shakiness_vals)
        
        # Durations per rep (ms)
        durations = np.array([m.get('duration_ms', 0) for m in rep_metrics_list])
        
        # Jerk values per rep
        jerk_values = np.array([m.get('mean_jerk', 0) for m in rep_metrics_list])
        
        # ROM (prefer degrees)
        rom_degrees_vals = [m.get('rom_degrees', 0) for m in rep_metrics_list]
        has_rom_degrees = any(v > 0 for v in rom_degrees_vals)
        if has_rom_degrees:
            roms = np.array(rom_degrees_vals)
        else:
            roms = np.array([m.get('rom', 0) for m in rep_metrics_list])
        
        # Smoothness scores
        smoothness = np.array([m.get('smoothness_score', 50) for m in rep_metrics_list])
        peaks = np.array([m.get('peak', 0) for m in rep_metrics_list])
        
        # =====================================================================
        # FATIGUE FORMULA: F = 0.35*D_ω + 0.25*I_T + 0.20*I_J + 0.20*I_S
        # =====================================================================
        
        # --- D_ω: Peak Angular Velocity ABSOLUTE Change (35% weight) ---
        # Both drop (muscles slowing) AND surge (compensatory swinging) = fatigue
        # At rest accel ~9.8; concentric dips, eccentric returns.
        # Gyro peak increasing = person swinging harder = loss of control.
        if has_gyro:
            avg_gyro_first = np.mean(gyro_peaks[:third])
            avg_gyro_last = np.mean(gyro_peaks[-third:])
            D_omega = abs(avg_gyro_first - avg_gyro_last) / avg_gyro_first if avg_gyro_first > 0 else 0
            # Track direction for reporting
            gyro_direction = 'drop' if avg_gyro_last < avg_gyro_first else 'surge'
        else:
            # Fallback: use peak acceleration change if no gyro data
            avg_peak_first = np.mean(peaks[:third])
            avg_peak_last = np.mean(peaks[-third:])
            D_omega = abs(avg_peak_first - avg_peak_last) / avg_peak_first if avg_peak_first > 0 else 0
            gyro_direction = 'drop' if avg_peak_last < avg_peak_first else 'surge'
        
        # --- I_T: Tempo / Duration Increase (25% weight) ---
        avg_dur_first = np.mean(durations[:third])
        avg_dur_last = np.mean(durations[-third:])
        I_T = (avg_dur_last - avg_dur_first) / avg_dur_first if avg_dur_first > 0 else 0
        
        # --- I_J: Jerk Increase (20% weight) ---
        avg_jerk_first = np.mean(jerk_values[:third])
        avg_jerk_last = np.mean(jerk_values[-third:])
        I_J = (avg_jerk_last - avg_jerk_first) / avg_jerk_first if avg_jerk_first > 0 else 0
        
        # --- I_S: Shakiness Increase (20% weight) ---
        # RMS of angular acceleration within each rep — detects within-rep tremor
        if has_shakiness:
            avg_shaky_first = np.mean(shakiness_vals[:third])
            avg_shaky_last = np.mean(shakiness_vals[-third:])
            I_S = (avg_shaky_last - avg_shaky_first) / avg_shaky_first if avg_shaky_first > 0 else 0
        else:
            avg_shaky_first = 0
            avg_shaky_last = 0
            I_S = 0
        
        # --- Composite Fatigue Score ---
        # Clamp each component to [0, ∞) — only positive change = fatigue
        D_omega_clamped = max(0, D_omega)
        I_T_clamped = max(0, I_T)
        I_J_clamped = max(0, I_J)
        I_S_clamped = max(0, I_S)
        
        fatigue_raw = (0.35 * D_omega_clamped +
                       0.25 * I_T_clamped +
                       0.20 * I_J_clamped +
                       0.20 * I_S_clamped)
        
        # Boost factor: if any single indicator is severely elevated, boost score
        # This prevents "averaging down" when one component shows severe degradation
        worst_indicator = max(D_omega_clamped, I_T_clamped, I_J_clamped, I_S_clamped)
        if worst_indicator > 0.40:  # 40%+ in any single indicator
            boost = (worst_indicator - 0.40) * 0.5  # Add up to 30% boost for extreme cases
            fatigue_raw = min(1.0, fatigue_raw + boost)
        
        # Scale to 0-100 (raw=1.0 → 100% fatigue)
        fatigue_score = min(100, fatigue_raw * 100)
        
        # Determine fatigue level (tighter thresholds for realistic assessment)
        if fatigue_score < 10:
            fatigue_level = 'minimal'
        elif fatigue_score < 20:
            fatigue_level = 'low'
        elif fatigue_score < 35:
            fatigue_level = 'moderate'
        elif fatigue_score < 55:
            fatigue_level = 'high'
        else:
            fatigue_level = 'severe'
        
        # =====================================================================
        # TREND ANALYSIS (linear regression, kept for visualization)
        # =====================================================================
        
        def get_trend(values):
            if len(values) < 2 or np.std(values) == 0:
                return 0
            try:
                slope, _, r_value, _, _ = stats.linregress(rep_indices, values)
                if np.mean(values) != 0:
                    return (slope / np.mean(values)) * 100
                return 0
            except:
                return 0
        
        rom_trend = get_trend(roms)
        smoothness_trend = get_trend(smoothness)
        duration_trend = get_trend(durations)
        peak_trend = get_trend(peaks)
        gyro_trend = get_trend(gyro_peaks) if has_gyro else 0
        
        # =====================================================================
        # CONSISTENCY ANALYSIS (kept — separate from fatigue)
        # =====================================================================
        
        def get_consistency(values):
            """Convert CV to consistency score. More realistic mapping."""
            if len(values) < 2 or np.mean(values) == 0:
                return 0
            try:
                cv = np.std(values) / np.mean(values)
                # Realistic mapping: CV=0% → 100%, CV=30% → 0%
                # This means 10% CV → 67% consistency, 15% CV → 50%, 20% CV → 33%
                consistency = max(0, 100 - cv * 333)
                return min(100, consistency)
            except:
                return 0
        
        rom_consistency = get_consistency(roms)
        smoothness_consistency = get_consistency(smoothness)
        duration_consistency = get_consistency(durations)
        peak_consistency = get_consistency(peaks)
        gyro_consistency = get_consistency(gyro_peaks) if has_gyro else 0
        
        consistency_score = np.mean([rom_consistency, smoothness_consistency,
                                     duration_consistency, peak_consistency])
        
        # =====================================================================
        # EARLY vs LATE COMPARISON
        # =====================================================================
        
        first_third = rep_metrics_list[:third]
        last_third = rep_metrics_list[-third:]
        
        # ROM comparison
        if has_rom_degrees:
            avg_rom_first = np.mean([m.get('rom_degrees', 0) for m in first_third])
            avg_rom_last = np.mean([m.get('rom_degrees', 0) for m in last_third])
        else:
            avg_rom_first = np.mean([m.get('rom', 0) for m in first_third])
            avg_rom_last = np.mean([m.get('rom', 0) for m in last_third])
        # Positive = ROM increased (wider movement), Negative = ROM decreased (smaller movement)
        rom_change_pct = ((avg_rom_last - avg_rom_first) / avg_rom_first * 100) if avg_rom_first > 0 else 0
        
        # Smoothness comparison
        avg_smooth_first = np.mean([m.get('smoothness_score', 50) for m in first_third])
        avg_smooth_last = np.mean([m.get('smoothness_score', 50) for m in last_third])
        smoothness_degradation = ((avg_smooth_first - avg_smooth_last) / avg_smooth_first * 100) if avg_smooth_first > 0 else 0
        
        # Duration comparison
        duration_increase = ((avg_dur_last - avg_dur_first) / avg_dur_first * 100) if avg_dur_first > 0 else 0
        
        # Gyro peak comparison (absolute change, not just drop)
        if has_gyro:
            avg_gyro_first_val = np.mean(gyro_peaks[:third])
            avg_gyro_last_val = np.mean(gyro_peaks[-third:])
            gyro_change_pct = ((avg_gyro_last_val - avg_gyro_first_val) / avg_gyro_first_val * 100) if avg_gyro_first_val > 0 else 0
        else:
            avg_gyro_first_val = 0
            avg_gyro_last_val = 0
            gyro_change_pct = 0
        
        # Jerk comparison
        jerk_increase_pct = ((avg_jerk_last - avg_jerk_first) / avg_jerk_first * 100) if avg_jerk_first > 0 else 0
        
        # Shakiness comparison
        if has_shakiness:
            shakiness_increase_pct = ((avg_shaky_last - avg_shaky_first) / avg_shaky_first * 100) if avg_shaky_first > 0 else 0
        else:
            shakiness_increase_pct = 0
        
        # Peak comparison
        avg_peak_first = np.mean([m.get('peak', 0) for m in first_third])
        avg_peak_last = np.mean([m.get('peak', 0) for m in last_third])
        peak_degradation = ((avg_peak_first - avg_peak_last) / avg_peak_first * 100) if avg_peak_first > 0 else 0
        
        # =====================================================================
        # PERFORMANCE INTERPRETATION
        # =====================================================================
        
        def get_session_quality(fatigue_score):
            """Convert fatigue score to session quality label."""
            if fatigue_score < 10:
                return "Excellent"  # minimal fatigue
            elif fatigue_score < 20:
                return "Good"  # low fatigue
            elif fatigue_score < 35:
                return "Fair"  # moderate fatigue
            elif fatigue_score < 55:
                return "Poor"  # high fatigue
            else:
                return "Very Poor"  # severe fatigue
        
        def get_consistency_rating(consistency_score):
            """Convert CV-based consistency to rating label."""
            if consistency_score >= 70:
                return "Good"  # CV < 9%
            elif consistency_score >= 50:
                return "Fair"  # CV 9-15%
            else:
                return "Poor"  # CV > 15%
        
        performance_report = {
            'session_quality': get_session_quality(fatigue_score),
            'consistency_rating': get_consistency_rating(consistency_score),
            'fatigue_components': {
                'D_omega': round(D_omega * 100, 2),
                'I_T': round(I_T * 100, 2),
                'I_J': round(I_J * 100, 2),
                'I_S': round(I_S * 100, 2),
                'gyro_direction': gyro_direction,
                'formula': 'F = 0.35·D_ω + 0.25·I_T + 0.20·I_J + 0.20·I_S'
            },
            'trend_analysis': {
                'rom_trend_percent_per_rep': round(rom_trend, 2),
                'smoothness_trend_percent_per_rep': round(smoothness_trend, 2),
                'duration_trend_percent_per_rep': round(duration_trend, 2),
                'peak_trend_percent_per_rep': round(peak_trend, 2)
            },
            'consistency_breakdown': {
                'rom_consistency': round(rom_consistency, 1),
                'smoothness_consistency': round(smoothness_consistency, 1),
                'duration_consistency': round(duration_consistency, 1),
                'peak_consistency': round(peak_consistency, 1)
            },
            'key_findings': []
        }
        
        # Generate key findings
        findings = performance_report['key_findings']
        
        if fatigue_score < 10:
            findings.append("✅ Excellent fatigue resistance — stable speed and jerk throughout")
        elif fatigue_score > 45:
            findings.append("⚠️ Significant fatigue detected — movement quality degraded notably")
        
        if has_gyro and D_omega > 0.15:
            if gyro_direction == 'surge':
                findings.append(f"⚠️ Peak angular velocity SURGED {D_omega*100:.1f}% — compensatory swinging")
            else:
                findings.append(f"⚠️ Peak angular velocity dropped {D_omega*100:.1f}% — muscles slowing")
        elif has_gyro and D_omega < 0.05:
            findings.append(f"✅ Angular velocity stable (only {D_omega*100:.1f}% change)")
        
        if has_shakiness and I_S > 0.20:
            findings.append(f"⚠️ Within-rep shakiness increased {I_S*100:.1f}% — losing motor control")
        elif has_shakiness and I_S < 0.05:
            findings.append(f"✅ Movement shakiness stable throughout session")
        
        if I_T > 0.15:
            findings.append(f"⚠️ Rep duration increased {I_T*100:.1f}% — slowing down")
        
        if I_J > 0.20:
            findings.append(f"⚠️ Jerk increased {I_J*100:.1f}% — movement becoming less smooth")
        
        if consistency_score > 85:
            findings.append("✅ Highly consistent, controlled movements")
        elif consistency_score < 60:
            findings.append("⚠️ High variability — erratic rep execution")
        
        if rom_change_pct < -20:
            findings.append(f"⚠️ ROM decreased by {abs(rom_change_pct):.1f}% (early vs late reps) — losing range")
        elif rom_change_pct > 20:
            findings.append(f"⚠️ ROM increased by {rom_change_pct:.1f}% (early vs late reps) — possible compensatory swinging")
        
        if not findings:
            findings.append("📊 Moderate performance with mixed indicators")
        
        return {
            'fatigue_score': round(fatigue_score, 1),
            'fatigue_level': fatigue_level,
            'D_omega': round(D_omega, 4),
            'I_T': round(I_T, 4),
            'I_J': round(I_J, 4),
            'I_S': round(I_S, 4),
            'gyro_direction': gyro_direction,
            'has_gyro': has_gyro,
            'consistency_score': round(consistency_score, 1),
            'rom_trend': round(rom_trend, 2),
            'smoothness_trend': round(smoothness_trend, 2),
            'duration_trend': round(duration_trend, 2),
            'peak_trend': round(peak_trend, 2),
            'gyro_trend': round(gyro_trend, 2),
            'rom_consistency': round(rom_consistency, 1),
            'smoothness_consistency': round(smoothness_consistency, 1),
            'duration_consistency': round(duration_consistency, 1),
            'peak_consistency': round(peak_consistency, 1),
            'gyro_consistency': round(gyro_consistency, 1) if has_gyro else 0,
            # Legacy fields for backward compatibility with existing UI
            'rom_change_percent': round(rom_change_pct, 1),
            'smoothness_degradation_percent': round(smoothness_degradation, 1),
            # New nested structure with additional details
            'early_vs_late': {
                'rom_change_percent': round(rom_change_pct, 1),
                'smoothness_degradation_percent': round(smoothness_degradation, 1),
                'duration_increase_percent': round(duration_increase, 1),
                'peak_degradation_percent': round(peak_degradation, 1),
                'gyro_peak_change_percent': round(gyro_change_pct, 1),
                'jerk_increase_percent': round(jerk_increase_pct, 1),
                'shakiness_increase_percent': round(shakiness_increase_pct, 1),
                'avg_gyro_peak_first': round(avg_gyro_first_val, 4),
                'avg_gyro_peak_last': round(avg_gyro_last_val, 4),
                'avg_shakiness_first': round(avg_shaky_first if has_shakiness else 0, 4),
                'avg_shakiness_last': round(avg_shaky_last if has_shakiness else 0, 4),
                'avg_rom_first': round(avg_rom_first, 2),
                'avg_rom_last': round(avg_rom_last, 2),
                'avg_smoothness_first': round(avg_smooth_first, 1),
                'avg_smoothness_last': round(avg_smooth_last, 1),
                'avg_duration_first_ms': round(avg_dur_first, 0),
                'avg_duration_last_ms': round(avg_dur_last, 0)
            },
            'performance_report': performance_report,
            'n_reps_analyzed': n_reps
        }
        
    except Exception as e:
        # If any error occurs, return a safe default structure
        print(f"⚠️ Error in fatigue analysis: {e}")
        return {
            'fatigue_score': 0,
            'fatigue_level': 'error',
            'D_omega': 0,
            'I_T': 0,
            'I_J': 0,
            'I_S': 0,
            'gyro_direction': 'stable',
            'has_gyro': False,
            'consistency_score': 0,
            'rom_trend': 0,
            'smoothness_trend': 0,
            'duration_trend': 0,
            'peak_trend': 0,
            'gyro_trend': 0,
            'rom_consistency': 0,
            'smoothness_consistency': 0,
            'duration_consistency': 0,
            'peak_consistency': 0,
            'gyro_consistency': 0,
            # Legacy fields for backward compatibility
            'rom_change_percent': 0,
            'smoothness_degradation_percent': 0,
            # New nested structure
            'early_vs_late': {
                'rom_change_percent': 0,
                'smoothness_degradation_percent': 0,
                'duration_increase_percent': 0,
                'peak_degradation_percent': 0,
                'gyro_peak_change_percent': 0,
                'jerk_increase_percent': 0,
                'shakiness_increase_percent': 0,
                'avg_gyro_peak_first': 0,
                'avg_gyro_peak_last': 0,
                'avg_shakiness_first': 0,
                'avg_shakiness_last': 0,
                'avg_rom_first': 0,
                'avg_rom_last': 0,
                'avg_smoothness_first': 0,
                'avg_smoothness_last': 0,
                'avg_duration_first_ms': 0,
                'avg_duration_last_ms': 0
            },
            'performance_report': {
                'session_quality': 'Error',
                'consistency_rating': 'Unknown',
                'key_findings': [f'⚠️ Analysis error: {str(e)}']
            },
            'n_reps_analyzed': len(rep_metrics_list) if 'rep_metrics_list' in locals() else 0
        }


def analyze_session_data(df, signal_column='filteredMag'):
    """
    Analyze a full session of data, computing metrics per rep.
    
    Parameters:
    - df: DataFrame with sensor data (must have 'rep' column)
    - signal_column: Primary signal column to analyze
    
    Returns:
    - session_metrics: Dictionary with per-rep and overall metrics
    """
    if 'rep' not in df.columns:
        print("  ⚠️ No 'rep' column found in data")
        return None
    
    # Filter out rep 0
    df = df[df['rep'] > 0].copy()
    
    rep_metrics_list = []
    
    for rep_num in sorted(df['rep'].unique()):
        rep_data = df[df['rep'] == rep_num]
        
        if len(rep_data) < 3:
            continue
        
        metrics = {'rep': rep_num}
        
        # Duration
        if 'timestamp_ms' in rep_data.columns:
            timestamps = rep_data['timestamp_ms'].values
            metrics['duration_ms'] = timestamps[-1] - timestamps[0]
            metrics['sample_count'] = len(rep_data)
        
        # ROM from primary signal
        if signal_column in rep_data.columns:
            signal = rep_data[signal_column].values
            rom_metrics = compute_range_of_motion(signal, signal_column)
            metrics.update(rom_metrics)
        
        # Jerk-based smoothness metrics (using filtered magnitude for irregularity detection)
        accel_cols = ['accelX', 'accelY', 'accelZ']
        if all(col in rep_data.columns for col in accel_cols) and 'timestamp_ms' in rep_data.columns:
            # Pass filteredMag if available for better irregularity detection
            filtered_mag = rep_data['filteredMag'].values if 'filteredMag' in rep_data.columns else None
            irregularity, smoothness = compute_ldlj(
                rep_data['accelX'].values,
                rep_data['accelY'].values,
                rep_data['accelZ'].values,
                rep_data['timestamp_ms'].values,
                filtered_mag=filtered_mag
            )
            metrics['mean_jerk'] = irregularity  # Higher = more irregular (renamed semantically)
            metrics['smoothness_score'] = smoothness
            # Backward compatibility: add ldlj field for legacy UI code
            metrics['ldlj'] = -irregularity  # Convert to LDLJ-like scale for compatibility
            
            # Compute ROM in degrees from accelerometer
            rom_deg_metrics = compute_rom_degrees(
                rep_data['accelX'].values,
                rep_data['accelY'].values,
                rep_data['accelZ'].values
            )
            metrics['rom_degrees'] = rom_deg_metrics['rom_degrees']
            metrics['min_angle'] = rom_deg_metrics['min_angle']
            metrics['max_angle'] = rom_deg_metrics['max_angle']
            metrics['mean_angle'] = rom_deg_metrics['mean_angle']
        
        # Gyroscope-based features (angular velocity + shakiness)
        gyro_cols = ['gyroX', 'gyroY', 'gyroZ']
        if all(col in rep_data.columns for col in gyro_cols) and 'timestamp_ms' in rep_data.columns:
            gyro_x = rep_data['gyroX'].values
            gyro_y = rep_data['gyroY'].values
            gyro_z = rep_data['gyroZ'].values
            gyro_mag = np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)
            metrics['gyro_peak'] = float(np.max(gyro_mag))       # Peak angular velocity (rad/s)
            metrics['gyro_rms'] = float(np.sqrt(np.mean(gyro_mag**2)))  # RMS angular velocity
            metrics['gyro_std'] = float(np.std(gyro_mag))        # Within-rep gyro variability
            
            # Shakiness: RMS of angular acceleration (derivative of gyro magnitude)
            # High values = jerky/tremorous rotation within a single rep
            ts_rep = rep_data['timestamp_ms'].values
            dur_rep = ts_rep[-1] - ts_rep[0]
            if len(ts_rep) > 2 and dur_rep > 0:
                dt_rep = (dur_rep / 1000.0) / (len(ts_rep) - 1)
                angular_accel = np.diff(gyro_mag) / dt_rep
                metrics['shakiness'] = float(np.sqrt(np.mean(angular_accel**2)))
            else:
                metrics['shakiness'] = 0.0
        
        # Per-axis ROM (for asymmetry detection)
        for axis in ['X', 'Y', 'Z']:
            col = f'filtered{axis}'
            if col in rep_data.columns:
                axis_rom = compute_range_of_motion(rep_data[col].values, col)
                metrics[f'rom_{axis.lower()}'] = axis_rom['rom']
        
        rep_metrics_list.append(metrics)
    
    # Compute fatigue analysis
    fatigue_analysis = compute_fatigue_indicators(rep_metrics_list)
    
    # Compute average ROM in degrees if available
    rom_degrees_values = [m.get('rom_degrees', 0) for m in rep_metrics_list]
    has_rom_degrees = any(v > 0 for v in rom_degrees_values)
    
    return {
        'rep_metrics': rep_metrics_list,
        'fatigue_analysis': fatigue_analysis,
        'session_summary': {
            'total_reps': len(rep_metrics_list),
            'avg_rom': np.mean([m.get('rom', 0) for m in rep_metrics_list]),
            'avg_rom_degrees': np.mean(rom_degrees_values) if has_rom_degrees else 0,
            'has_rom_degrees': has_rom_degrees,
            'avg_smoothness': np.mean([m.get('smoothness_score', 50) for m in rep_metrics_list]),
            'avg_duration_ms': np.mean([m.get('duration_ms', 0) for m in rep_metrics_list]),
            'total_duration_ms': sum([m.get('duration_ms', 0) for m in rep_metrics_list])
        }
    }


def generate_comprehensive_performance_report(session_metrics, output_path=None):
    """
    Generate a detailed text-based performance report with all metrics and analysis.
    
    Parameters:
    - session_metrics: Dictionary from analyze_session_data()
    - output_path: Optional file path to save the report
    
    Returns:
    - report_text: Formatted string with comprehensive analysis
    """
    rep_metrics = session_metrics['rep_metrics']
    fatigue = session_metrics['fatigue_analysis']
    summary = session_metrics['session_summary']
    
    # Determine ROM display (degrees or raw units)
    if summary.get('has_rom_degrees', False) and summary.get('avg_rom_degrees', 0) > 0:
        rom_display = f"{summary['avg_rom_degrees']:.1f}°"
    else:
        rom_display = f"{summary['avg_rom']:.2f} units"
    
    report_text = f"""
AppLift ML Training - Comprehensive Performance Analysis Report
==============================================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

EXECUTIVE SUMMARY
─────────────────
Session Quality: {fatigue.get('performance_report', {}).get('session_quality', 'Unknown')}
Consistency Rating: {fatigue.get('performance_report', {}).get('consistency_rating', 'Unknown')}
Fatigue Level: {fatigue['fatigue_level'].title()} ({fatigue['fatigue_score']:.1f}/100)
Overall Consistency: {fatigue['consistency_score']:.1f}%

SESSION OVERVIEW
────────────────
Total Repetitions: {summary['total_reps']}
Total Session Time: {summary['total_duration_ms']/1000/60:.1f} minutes
Average Rep Duration: {summary['avg_duration_ms']/1000:.1f} seconds
Average Range of Motion: {rom_display}
Average Movement Smoothness: {summary['avg_smoothness']:.1f}%

FATIGUE ANALYSIS (Gyroscope-Based)
──────────────────────────────────
Formula: Fatigue = 0.35·D_ω + 0.25·I_T + 0.20·I_J + 0.20·I_S
Overall Fatigue Score: {fatigue['fatigue_score']:.1f}/100

Component Breakdown:
• D_ω (Peak Angular Velocity Change): {fatigue.get('D_omega', 0)*100:+.2f}%  [weight: 35%]  ({fatigue.get('gyro_direction', 'unknown')})
  — Measures absolute change in peak gyroscope magnitude from early to late reps
  — Detects both speed drop (muscles slowing) and surge (compensatory swinging)
• I_T (Tempo / Duration Increase): {fatigue.get('I_T', 0)*100:+.2f}%  [weight: 25%]
  — Measures slowing of rep execution speed
• I_J (Jerk Increase): {fatigue.get('I_J', 0)*100:+.2f}%  [weight: 20%]
  — Measures increase in movement jerkiness (loss of smoothness)
• I_S (Shakiness Increase): {fatigue.get('I_S', 0)*100:+.2f}%  [weight: 20%]
  — Measures increase in within-rep angular acceleration RMS (tremor/instability)

Early vs Late Performance Comparison:
• Gyro ω_peak Change: {fatigue.get('early_vs_late', {}).get('gyro_peak_change_percent', fatigue.get('early_vs_late', {}).get('gyro_peak_drop_percent', 0)):+.1f}% ({fatigue.get('gyro_direction', 'unknown')})
  — First-third avg: {fatigue.get('early_vs_late', {}).get('avg_gyro_peak_first', 0):.4f} rad/s
  — Last-third avg:  {fatigue.get('early_vs_late', {}).get('avg_gyro_peak_last', 0):.4f} rad/s
• Shakiness Change: {fatigue.get('early_vs_late', {}).get('shakiness_increase_percent', 0):+.1f}%
  — First-third avg: {fatigue.get('early_vs_late', {}).get('avg_shakiness_first', 0):.4f}
  — Last-third avg:  {fatigue.get('early_vs_late', {}).get('avg_shakiness_last', 0):.4f}
• ROM Change: {fatigue.get('early_vs_late', {}).get('rom_change_percent', 0):+.1f}%
• Smoothness Decline: {fatigue.get('early_vs_late', {}).get('smoothness_degradation_percent', 0):+.1f}%
• Duration Increase: {fatigue.get('early_vs_late', {}).get('duration_increase_percent', 0):+.1f}%
• Jerk Increase: {fatigue.get('early_vs_late', {}).get('jerk_increase_percent', 0):+.1f}%

Trend Analysis (% change per repetition):
• Range of Motion: {fatigue['rom_trend']:+.2f}% per rep
• Movement Smoothness: {fatigue['smoothness_trend']:+.2f}% per rep  
• Rep Duration: {fatigue['duration_trend']:+.2f}% per rep
• Peak Force/Acceleration: {fatigue['peak_trend']:+.2f}% per rep

CONSISTENCY ANALYSIS
────────────────────
Overall Consistency Score: {fatigue['consistency_score']:.1f}%

Individual Metric Consistency:
• Range of Motion: {fatigue['rom_consistency']:.1f}%
• Movement Smoothness: {fatigue['smoothness_consistency']:.1f}%
• Rep Duration: {fatigue['duration_consistency']:.1f}%
• Peak Force: {fatigue['peak_consistency']:.1f}%

REP-BY-REP BREAKDOWN
────────────────────
Rep | Duration(s) | ROM     | Smoothness | ω_peak   | Jerk     | Shakiness | Notes
────┼─────────────┼─────────┼────────────┼──────────┼──────────┼───────────┼──────────────────
"""
    
    for i, rep in enumerate(rep_metrics):
        rep_num = rep['rep']
        duration = rep.get('duration_ms', 0) / 1000
        rom = rep.get('rom', 0)
        smoothness = rep.get('smoothness_score', 0)
        gyro_peak = rep.get('gyro_peak', 0)
        mean_jerk = rep.get('mean_jerk', 0)
        shakiness = rep.get('shakiness', 0)
        
        # Add performance indicators
        notes = ""
        if smoothness >= 80:
            notes += "Excellent "
        elif smoothness < 50:
            notes += "Poor smoothness "
            
        if i > 0:  # Compare to previous rep
            prev_smoothness = rep_metrics[i-1].get('smoothness_score', 0)
            if smoothness < prev_smoothness - 10:
                notes += "Declining "
            elif smoothness > prev_smoothness + 10:
                notes += "Improving "
        
        report_text += f"{rep_num:3d} | {duration:7.1f}     | {rom:7.2f} | {smoothness:8.1f}%  | {gyro_peak:8.4f} | {mean_jerk:8.2f} | {shakiness:9.4f} | {notes}\n"
    
    report_text += f"""
KEY FINDINGS & RECOMMENDATIONS
──────────────────────────────
"""
    
    findings = fatigue.get('performance_report', {}).get('key_findings', [])
    for finding in findings:
        report_text += f"• {finding}\n"
    
    report_text += f"""

DETAILED INTERPRETATIONS
────────────────────────

Fatigue Score Methodology (Gyroscope-Based):
• Formula: Fatigue = 0.35·D_ω + 0.25·I_T + 0.20·I_J + 0.20·I_S
• D_ω = |ω_peak,first − ω_peak,last| / ω_peak,first
  → Measures absolute change in peak angular velocity from gyroscope
  → Speed drop = neuromuscular fatigue (muscles slowing)
  → Speed surge = compensatory swinging (using momentum instead of control)
• I_T = (T_last − T_first) / T_first
  → Measures increase in rep duration (slowing down)
  → Higher increase = more power/endurance fatigue
• I_J = (J_last − J_first) / J_first
  → Measures increase in movement jerkiness
  → Higher increase = loss of motor control coordination
• I_S = (shakiness_last − shakiness_first) / shakiness_first
  → Measures increase in within-rep angular acceleration RMS
  → Higher increase = more tremor/instability (fine motor control loss)

Fatigue Score Components:
• Weighted average: 0.35·D_ω + 0.25·I_T + 0.20·I_J + 0.20·I_S
• Boost applied when any single indicator exceeds 40% (prevents averaging down)

Fatigue Level Interpretation:
• Minimal (0-10): Excellent fatigue resistance, stable performance
• Low (10-20): Minor performance decline, within normal range
• Moderate (20-35): Noticeable fatigue effects on movement quality
• High (35-55): Significant fatigue impacting exercise performance
• Severe (55-100): Substantial neuromuscular fatigue, consider rest

Consistency Score Interpretation (based on CV):
• Excellent (>85): CV < 5%, highly repeatable movements
• Good (70-85): CV 5-10%, generally consistent
• Fair (50-70): CV 10-15%, moderate variability
• Poor (<50): CV > 15%, high inconsistency, indicates poor form or fatigue

TECHNICAL NOTES
───────────────
• Angular velocity (ω) computed from gyroX, gyroY, gyroZ: ω = √(gx² + gy² + gz²)
• Jerk calculated as derivative of 3-axis acceleration magnitude / dt
• Smoothness scored using mean jerk with thresholds: <12 excellent, 12-18 good, 18-25 fair, >25 poor
• Consistency measured using CV (std/mean): lower CV = more consistent
• Early/Late comparison uses first vs last third of repetitions
• ROM in degrees computed from accelerometer pitch: arctan2(ay, √(ax² + az²)) + 90°

Report Generated by AppLift ML Performance Analysis System
"""
    
    if output_path:
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"  ✓ Comprehensive report saved: {output_path}")
        except Exception as e:
            print(f"  ⚠️ Failed to save report: {e}")
    
    return report_text


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_fatigue_visualization(session_metrics, output_path=None, title="Fatigue Analysis"):
    """
    Create comprehensive fatigue visualization.
    """
    rep_metrics = session_metrics['rep_metrics']
    fatigue = session_metrics['fatigue_analysis']
    
    if len(rep_metrics) < 3:
        print("  ⚠️ Not enough reps for fatigue visualization (need at least 3)")
        return None
    
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    fig.suptitle(f'🏋️ {title}', fontsize=18, fontweight='bold', y=0.98)
    
    reps = [m['rep'] for m in rep_metrics]
    
    # 1. ROM Over Reps (Top Left - Large) - Use degrees if available
    ax1 = fig.add_subplot(gs[0, :2])
    
    # Check if ROM in degrees is available
    has_degrees = 'rom_degrees' in rep_metrics[0] and rep_metrics[0].get('rom_degrees', 0) > 0
    
    if has_degrees:
        roms = [m.get('rom_degrees', 0) for m in rep_metrics]
        rom_unit = '°'
        rom_label = 'Range of Motion (degrees)'
    else:
        roms = [m.get('rom', 0) for m in rep_metrics]
        rom_unit = ''
        rom_label = 'Range of Motion'
    
    # Color gradient based on fatigue
    colors = plt.cm.RdYlGn_r(np.linspace(0, 1, len(reps)))
    bars = ax1.bar(reps, roms, color=colors, edgecolor='black', linewidth=0.5)
    
    # Trend line
    if len(reps) > 2:
        z = np.polyfit(reps, roms, 1)
        p = np.poly1d(z)
        ax1.plot(reps, p(reps), 'r--', linewidth=2, label=f'Trend (slope: {z[0]:.2f})')
    
    ax1.set_xlabel('Rep Number', fontweight='bold')
    ax1.set_ylabel(rom_label, fontweight='bold')
    ax1.set_title(f'📏 Range of Motion per Rep{" (degrees)" if has_degrees else ""}', fontweight='bold', fontsize=12)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Add first/last comparison annotation
    rom_change = fatigue.get('rom_change_percent', 0)
    if rom_change != 0:
        rom_dir = '↑' if rom_change > 0 else '↓'
        # Red if ROM decreased (losing range) or increased too much (swinging)
        rom_color = 'red' if rom_change < -10 or rom_change > 30 else 'green' if abs(rom_change) < 10 else 'orange'
        ax1.annotate(f"ROM Change: {rom_change:+.1f}% ({rom_dir})",
                    xy=(0.02, 0.98), xycoords='axes fraction',
                    fontsize=10, fontweight='bold',
                    color=rom_color,
                    verticalalignment='top')
    
    # 2. Fatigue Score Gauge (Top Right)
    ax2 = fig.add_subplot(gs[0, 2])
    fatigue_score = fatigue['fatigue_score']
    
    # Create gauge
    theta = np.linspace(0, np.pi, 100)
    r = 1
    
    # Background arc (gray)
    ax2.plot(r * np.cos(theta), r * np.sin(theta), 'lightgray', linewidth=20)
    
    # Colored arc based on fatigue level
    n_colored = int(fatigue_score)
    theta_colored = np.linspace(0, np.pi * (fatigue_score / 100), n_colored)
    
    # Color gradient from green to red
    for i, t in enumerate(theta_colored[:-1]):
        color = plt.cm.RdYlGn_r(i / len(theta_colored))
        ax2.plot([r * np.cos(t), r * np.cos(theta_colored[i+1])],
                [r * np.sin(t), r * np.sin(theta_colored[i+1])],
                color=color, linewidth=20)
    
    # Needle
    needle_angle = np.pi * (1 - fatigue_score / 100)
    ax2.annotate('', xy=(0.8 * np.cos(needle_angle), 0.8 * np.sin(needle_angle)),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='black', lw=3))
    
    ax2.text(0, -0.3, f'{fatigue_score:.0f}%', ha='center', va='center',
            fontsize=24, fontweight='bold')
    ax2.text(0, -0.5, f'Fatigue Level: {fatigue["fatigue_level"].upper()}',
            ha='center', va='center', fontsize=12, fontweight='bold',
            color='red' if fatigue_score > 60 else 'orange' if fatigue_score > 30 else 'green')
    
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-0.7, 1.3)
    ax2.set_aspect('equal')
    ax2.axis('off')
    ax2.set_title('⚡ Fatigue Score', fontweight='bold', fontsize=12, y=1.05)
    
    # 3. Smoothness Over Reps (Middle Left)
    ax3 = fig.add_subplot(gs[1, :2])
    smoothness_scores = [m.get('smoothness_score', 50) for m in rep_metrics]
    
    # Area plot with gradient
    ax3.fill_between(reps, smoothness_scores, alpha=0.3, color=COLORS['primary'])
    ax3.plot(reps, smoothness_scores, 'o-', color=COLORS['primary'], linewidth=2, markersize=8)
    
    # Reference lines (smoothness = 100 - irregularity)
    # Excellent: irregularity < 20 → smoothness > 80
    # Good: irregularity 20-35 → smoothness 65-80
    # Fair: irregularity 35-55 → smoothness 45-65
    ax3.axhline(y=80, color='green', linestyle='--', alpha=0.5, label='Excellent (80)')
    ax3.axhline(y=65, color='#8BC34A', linestyle='--', alpha=0.5, label='Good (65)')
    ax3.axhline(y=45, color='orange', linestyle='--', alpha=0.5, label='Fair (45)')
    
    ax3.set_xlabel('Rep Number', fontweight='bold')
    ax3.set_ylabel('Smoothness Score (0-100)', fontweight='bold')
    ax3.set_title('🎯 Movement Smoothness (Jerk-based)', fontweight='bold', fontsize=12)
    ax3.set_ylim(0, 100)
    ax3.legend(loc='lower left')
    ax3.grid(alpha=0.3)
    
    # 4. Consistency Analysis (Middle Right)
    ax4 = fig.add_subplot(gs[1, 2])
    
    # Consistency metrics from fatigue analysis
    consistency_metrics = [
        fatigue.get('rom_consistency', 0),
        fatigue.get('smoothness_consistency', 0),
        fatigue.get('duration_consistency', 0),
        fatigue.get('peak_consistency', 0)
    ]
    
    labels = ['ROM', 'Smoothness', 'Duration', 'Peak']
    # Color thresholds aligned with CV-based consistency interpretation
    colors_cons = ['green' if c >= 70 else 'orange' if c >= 50 else 'red' for c in consistency_metrics]
    
    bars = ax4.barh(labels, consistency_metrics, color=colors_cons, edgecolor='black', linewidth=1)
    ax4.set_xlabel('Consistency Score (0-100)', fontweight='bold')
    ax4.set_title('📊 Performance Consistency (CV-based)', fontweight='bold', fontsize=12)
    ax4.set_xlim(0, 100)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, consistency_metrics)):
        ax4.text(val + 2, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', ha='left', va='center', fontweight='bold')
    
    # Overall consistency score
    overall_consistency = fatigue.get('consistency_score', 0)
    consistency_color = 'green' if overall_consistency >= 70 else 'orange' if overall_consistency >= 50 else 'red'
    ax4.text(0.98, 0.02, f'Overall: {overall_consistency:.1f}%', 
            transform=ax4.transAxes, ha='right', va='bottom',
            fontsize=12, fontweight='bold', color=consistency_color,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 5. Rep Duration Trend (Bottom Left)
    ax5 = fig.add_subplot(gs[2, 0])
    durations = [m.get('duration_ms', 0) / 1000 for m in rep_metrics]  # Convert to seconds
    
    ax5.plot(reps, durations, 's-', color=COLORS['purple'], linewidth=2, markersize=8)
    ax5.fill_between(reps, durations, alpha=0.2, color=COLORS['purple'])
    
    ax5.set_xlabel('Rep Number', fontweight='bold')
    ax5.set_ylabel('Duration (seconds)', fontweight='bold')
    ax5.set_title('⏱️ Rep Duration', fontweight='bold', fontsize=12)
    ax5.grid(alpha=0.3)
    
    # 6. Peak Angular Velocity Trend (Bottom Middle)
    ax6 = fig.add_subplot(gs[2, 1])
    
    # Use gyro peak if available, otherwise fall back to peak accel
    has_gyro = 'gyro_peak' in rep_metrics[0] and rep_metrics[0].get('gyro_peak', 0) > 0
    
    if has_gyro:
        gyro_peaks = [m.get('gyro_peak', 0) for m in rep_metrics]
        ax6.plot(reps, gyro_peaks, '^-', color=COLORS['teal'], linewidth=2, markersize=8)
        ax6.fill_between(reps, gyro_peaks, alpha=0.2, color=COLORS['teal'])
        
        # Trend line
        if len(reps) > 2:
            z = np.polyfit(reps, gyro_peaks, 1)
            p = np.poly1d(z)
            ax6.plot(reps, p(reps), 'r--', linewidth=2, alpha=0.7, label=f'Trend')
        
        ax6.set_ylabel('Peak ω (rad/s)', fontweight='bold')
        ax6.set_title('🔄 Peak Angular Velocity (ω_peak)', fontweight='bold', fontsize=12)
        
        # Annotate D_omega with direction
        gyro_change_pct = fatigue.get('early_vs_late', {}).get('gyro_peak_change_percent', 0)
        direction = fatigue.get('gyro_direction', 'stable')
        if direction == 'surge':
            label = f"D_ω: {abs(gyro_change_pct):.1f}% surge (↑ swinging)"
            color = 'red' if abs(gyro_change_pct) > 15 else 'orange'
        elif direction == 'drop':
            label = f"D_ω: {abs(gyro_change_pct):.1f}% drop (↓ slowing)"
            color = 'red' if abs(gyro_change_pct) > 15 else 'orange'
        else:
            label = f"D_ω: {abs(gyro_change_pct):.1f}% change"
            color = 'green'
        ax6.annotate(label,
                    xy=(0.02, 0.98), xycoords='axes fraction',
                    fontsize=10, fontweight='bold', color=color,
                    verticalalignment='top')
    else:
        peaks = [m.get('peak', 0) for m in rep_metrics]
        ax6.plot(reps, peaks, '^-', color=COLORS['teal'], linewidth=2, markersize=8)
        ax6.fill_between(reps, peaks, alpha=0.2, color=COLORS['teal'])
        ax6.set_ylabel('Peak Value', fontweight='bold')
        ax6.set_title('📈 Peak Force/Acceleration', fontweight='bold', fontsize=12)
    
    ax6.set_xlabel('Rep Number', fontweight='bold')
    ax6.legend()
    ax6.grid(alpha=0.3)
    
    # 7. Comprehensive Performance Report (Bottom Right)
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')
    
    summary = session_metrics['session_summary']
    report = fatigue.get('performance_report', {})
    early_late = fatigue.get('early_vs_late', {})
    
    summary_text = f"""📊 PERFORMANCE REPORT
─────────────────────────
Total Reps: {summary['total_reps']}
Session Quality: {report.get('session_quality', 'Unknown')}
Consistency Rating: {report.get('consistency_rating', 'Unknown')}

🔥 FATIGUE (Gyro-Based)
Score: {fatigue['fatigue_score']:.1f}/100
Level: {fatigue['fatigue_level'].title()}
Formula: F=.35·Dω+.25·IT+.20·IJ+.20·IS

📐 FATIGUE COMPONENTS
D_ω ({fatigue.get('gyro_direction','?')}): {fatigue.get('D_omega', 0)*100:+.1f}%
I_T (Tempo Rise): {fatigue.get('I_T', 0)*100:+.1f}%
I_J (Jerk Rise): {fatigue.get('I_J', 0)*100:+.1f}%
I_S (Shakiness): {fatigue.get('I_S', 0)*100:+.1f}%

📈 TREND ANALYSIS (% per rep)
ROM: {fatigue['rom_trend']:+.1f}%
Smoothness: {fatigue['smoothness_trend']:+.1f}%  
Duration: {fatigue['duration_trend']:+.1f}%

📊 CONSISTENCY SCORES
Overall: {fatigue['consistency_score']:.1f}%
ROM: {fatigue['rom_consistency']:.1f}%
Smoothness: {fatigue['smoothness_consistency']:.1f}%

🔍 KEY FINDINGS
"""
    
    # Add key findings from the report
    findings = report.get('key_findings', ['No specific findings'])
    for finding in findings[:3]:  # Limit to 3 findings for space
        summary_text += f"• {finding}\n"
    
    ax7.text(0.05, 0.98, summary_text, transform=ax7.transAxes,
            fontsize=8, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Saved: {output_path}")
    
    return fig


def create_ldlj_visualization(session_metrics, output_path=None, title="Movement Smoothness Analysis"):
    """
    Create detailed jerk/smoothness visualization.
    Uses RMS Jerk metric where LOWER jerk = smoother movement.
    """
    rep_metrics = session_metrics['rep_metrics']
    
    if len(rep_metrics) < 2:
        print("  ⚠️ Not enough reps for jerk visualization")
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'🎯 {title}', fontsize=16, fontweight='bold', y=1.02)
    
    reps = [m['rep'] for m in rep_metrics]
    jerk_values = [m.get('mean_jerk', 0) for m in rep_metrics]
    smoothness_scores = [m.get('smoothness_score', 50) for m in rep_metrics]
    
    # 1. Jerk Values (Lower = Better)
    ax1 = axes[0, 0]
    # Color based on jerk thresholds: lower jerk = green, higher jerk = red
    colors = ['green' if j < 12 else 'orange' if j < 18 else 'red' for j in jerk_values]
    bars = ax1.bar(reps, jerk_values, color=colors, edgecolor='black', linewidth=0.5)
    
    # Reference lines (lower = better) using JERK_THRESHOLDS
    ax1.axhline(y=JERK_THRESHOLDS['excellent'], color='green', linestyle='--', alpha=0.7, 
                label=f"Excellent (<{JERK_THRESHOLDS['excellent']})")
    ax1.axhline(y=JERK_THRESHOLDS['good'], color='orange', linestyle='--', alpha=0.7,
                label=f"Good (<{JERK_THRESHOLDS['good']})")
    ax1.axhline(y=JERK_THRESHOLDS['fair'], color='red', linestyle='--', alpha=0.7,
                label=f"Fair (<{JERK_THRESHOLDS['fair']})")
    
    ax1.set_xlabel('Rep Number', fontweight='bold')
    ax1.set_ylabel('Irregularity Score (0-100)', fontweight='bold')
    ax1.set_title('📊 Movement Irregularity (Lower = Smoother)', fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(axis='y', alpha=0.3)
    # No invert - lower is better, show naturally
    
    # 2. Smoothness Score Progression
    ax2 = axes[0, 1]
    
    # Create smoothed line
    if len(smoothness_scores) > 5:
        smoothed = savgol_filter(smoothness_scores, min(5, len(smoothness_scores)//2*2+1), 2)
    else:
        smoothed = smoothness_scores
    
    ax2.fill_between(reps, smoothness_scores, alpha=0.3, color=COLORS['primary'])
    ax2.plot(reps, smoothness_scores, 'o', color=COLORS['primary'], markersize=10, label='Actual')
    ax2.plot(reps, smoothed, '-', color='darkblue', linewidth=2, label='Trend')
    
    # Color zones (smoothness = 100 - irregularity)
    # Excellent: irregularity < 20 → smoothness > 80
    # Good: irregularity 20-35 → smoothness 65-80
    # Fair: irregularity 35-55 → smoothness 45-65
    # Poor: irregularity >= 55 → smoothness < 45
    ax2.axhspan(80, 100, alpha=0.1, color='green', label='Excellent')
    ax2.axhspan(65, 80, alpha=0.1, color='#8BC34A')
    ax2.axhspan(45, 65, alpha=0.1, color='yellow')
    ax2.axhspan(0, 45, alpha=0.1, color='red', label='Poor')
    
    ax2.set_xlabel('Rep Number', fontweight='bold')
    ax2.set_ylabel('Smoothness Score (0-100)', fontweight='bold')
    ax2.set_title('🎯 Smoothness Score Progression', fontweight='bold')
    ax2.set_ylim(0, 100)
    ax2.legend(loc='lower left')
    ax2.grid(alpha=0.3)
    
    # 3. Smoothness Distribution
    ax3 = axes[1, 0]
    
    ax3.hist(smoothness_scores, bins=10, color=COLORS['primary'], edgecolor='black', alpha=0.7)
    ax3.axvline(x=np.mean(smoothness_scores), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(smoothness_scores):.1f}')
    ax3.axvline(x=np.median(smoothness_scores), color='green', linestyle='--', linewidth=2,
               label=f'Median: {np.median(smoothness_scores):.1f}')
    
    ax3.set_xlabel('Smoothness Score', fontweight='bold')
    ax3.set_ylabel('Frequency', fontweight='bold')
    ax3.set_title('📈 Smoothness Distribution', fontweight='bold')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Movement Quality Pie Chart (based on mean jerk thresholds for consistency with bar chart)
    ax4 = axes[1, 1]
    
    # Use irregularity score with JERK_THRESHOLDS for consistency with the bar chart
    excellent = sum(1 for j in jerk_values if j < JERK_THRESHOLDS['excellent'])  # < 20
    good = sum(1 for j in jerk_values if JERK_THRESHOLDS['excellent'] <= j < JERK_THRESHOLDS['good'])  # 20-35
    fair = sum(1 for j in jerk_values if JERK_THRESHOLDS['good'] <= j < JERK_THRESHOLDS['fair'])  # 35-55
    poor = sum(1 for j in jerk_values if j >= JERK_THRESHOLDS['fair'])  # >= 55
    
    sizes = [excellent, good, fair, poor]
    labels = [f'Excellent\n({excellent})', f'Good\n({good})', f'Fair\n({fair})', f'Poor\n({poor})']
    colors_pie = ['#4CAF50', '#8BC34A', '#FF9800', '#f44336']
    explode = (0.05, 0, 0, 0.05)
    
    # Remove zero values
    non_zero = [(s, l, c, e) for s, l, c, e in zip(sizes, labels, colors_pie, explode) if s > 0]
    if non_zero:
        sizes, labels, colors_pie, explode = zip(*non_zero)
        ax4.pie(sizes, explode=explode, labels=labels, colors=colors_pie,
               autopct='%1.1f%%', shadow=True, startangle=90)
    
    ax4.set_title('🥧 Movement Quality (by Jerk)', fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Saved: {output_path}")
    
    return fig


def create_rom_visualization(session_metrics, output_path=None, title="Range of Motion Analysis"):
    """
    Create detailed Range of Motion visualization.
    """
    rep_metrics = session_metrics['rep_metrics']
    fatigue = session_metrics['fatigue_analysis']
    
    if len(rep_metrics) < 2:
        print("  ⚠️ Not enough reps for ROM visualization")
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'📏 {title}', fontsize=16, fontweight='bold', y=1.02)
    
    reps = [m['rep'] for m in rep_metrics]
    
    # Use degrees when available (consistent with fatigue chart)
    has_degrees = 'rom_degrees' in rep_metrics[0] and rep_metrics[0].get('rom_degrees', 0) > 0
    if has_degrees:
        roms = [m.get('rom_degrees', 0) for m in rep_metrics]
        rom_label = 'Range of Motion (degrees)'
    else:
        roms = [m.get('rom', 0) for m in rep_metrics]
        rom_label = 'Range of Motion'
    
    peaks = [m.get('peak', 0) for m in rep_metrics]
    troughs = [m.get('trough', 0) for m in rep_metrics]
    
    # 1. ROM per Rep with Error Bars (showing peak-trough)
    ax1 = axes[0, 0]
    
    # Color based on relative ROM (normalized to max)
    max_rom = max(roms) if max(roms) > 0 else 1
    colors = plt.cm.RdYlGn([r/max_rom for r in roms])
    
    bars = ax1.bar(reps, roms, color=colors, edgecolor='black', linewidth=0.5)
    
    # Add mean line
    mean_rom = np.mean(roms)
    ax1.axhline(y=mean_rom, color='blue', linestyle='--', linewidth=2, label=f'Mean: {mean_rom:.2f}')
    
    # Trend line
    if len(reps) > 2:
        z = np.polyfit(reps, roms, 1)
        p = np.poly1d(z)
        ax1.plot(reps, p(reps), 'r-', linewidth=2, alpha=0.7, label=f'Trend')
    
    ax1.set_xlabel('Rep Number', fontweight='bold')
    ax1.set_ylabel(rom_label, fontweight='bold')
    ax1.set_title(f'📊 ROM per Rep{" (degrees)" if has_degrees else ""}', fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Peak and Trough Values
    ax2 = axes[0, 1]
    
    ax2.fill_between(reps, peaks, troughs, alpha=0.3, color=COLORS['primary'], label='ROM Range')
    ax2.plot(reps, peaks, '^-', color='green', linewidth=2, markersize=8, label='Peak')
    ax2.plot(reps, troughs, 'v-', color='red', linewidth=2, markersize=8, label='Trough')
    
    ax2.set_xlabel('Rep Number', fontweight='bold')
    ax2.set_ylabel('Value', fontweight='bold')
    ax2.set_title('📈 Peak and Trough Values', fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # 3. ROM Consistency (Box plot simulation)
    ax3 = axes[1, 0]
    
    # Create a violin plot style visualization
    parts = ax3.violinplot([roms], positions=[1], showmeans=True, showmedians=True)
    parts['bodies'][0].set_facecolor(COLORS['primary'])
    parts['bodies'][0].set_alpha(0.7)
    
    # Add individual points
    ax3.scatter(np.ones(len(roms)) + np.random.uniform(-0.1, 0.1, len(roms)), 
               roms, alpha=0.6, color='darkblue', s=50)
    
    # Stats text
    stats_text = f"Mean: {np.mean(roms):.2f}\nStd: {np.std(roms):.2f}\nCV: {(np.std(roms)/np.mean(roms)*100):.1f}%"
    ax3.text(1.3, np.mean(roms), stats_text, fontsize=10, verticalalignment='center')
    
    ax3.set_ylabel(rom_label, fontweight='bold')
    ax3.set_title('📦 ROM Distribution & Consistency', fontweight='bold')
    ax3.set_xticks([1])
    ax3.set_xticklabels(['All Reps'])
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. ROM Change Analysis (Early vs Late)
    ax4 = axes[1, 1]
    
    # Split into thirds
    n = len(rep_metrics)
    third = max(1, n // 3)
    
    first_roms = roms[:third]
    mid_roms = roms[third:2*third]
    last_roms = roms[2*third:]
    
    positions = [1, 2, 3]
    bp = ax4.boxplot([first_roms, mid_roms, last_roms], positions=positions, patch_artist=True)
    
    colors_box = [COLORS['success'], COLORS['warning'], COLORS['danger']]
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax4.set_xticklabels(['First Third', 'Middle Third', 'Last Third'])
    ax4.set_ylabel(rom_label, fontweight='bold')
    ax4.set_title('📊 ROM Change Over Session', fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    # Use the same rom_change_percent from fatigue dict (consistent with fatigue chart)
    rom_chg = fatigue.get('rom_change_percent', 0)
    if rom_chg != 0:
        rom_dir = '↑' if rom_chg > 0 else '↓'
        color = 'red' if rom_chg < -10 or rom_chg > 30 else 'green' if abs(rom_chg) < 10 else 'orange'
        ax4.text(2, ax4.get_ylim()[1] * 0.95, f'ROM Change: {rom_chg:+.1f}% ({rom_dir})',
                ha='center', fontsize=12, fontweight='bold', color=color)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Saved: {output_path}")
    
    return fig


def create_session_comparison_visualization(sessions_data, output_path=None, title="Session Comparison"):
    """
    Compare performance across multiple sessions.
    
    Parameters:
    - sessions_data: List of session_metrics dicts with 'date' or 'session_id' keys
    """
    if len(sessions_data) < 2:
        print("  ⚠️ Need at least 2 sessions for comparison")
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'📊 {title}', fontsize=16, fontweight='bold', y=1.02)
    
    session_ids = [s.get('session_id', f'Session {i+1}') for i, s in enumerate(sessions_data)]
    
    # Extract summary metrics
    avg_roms = [s['session_summary']['avg_rom'] for s in sessions_data]
    avg_smoothness = [s['session_summary']['avg_smoothness'] for s in sessions_data]
    fatigue_scores = [s['fatigue_analysis']['fatigue_score'] for s in sessions_data]
    total_reps = [s['session_summary']['total_reps'] for s in sessions_data]
    
    # 1. ROM Progression
    ax1 = axes[0, 0]
    ax1.plot(session_ids, avg_roms, 'o-', color=COLORS['primary'], linewidth=2, markersize=10)
    ax1.fill_between(session_ids, avg_roms, alpha=0.3, color=COLORS['primary'])
    ax1.set_xlabel('Session', fontweight='bold')
    ax1.set_ylabel('Average ROM', fontweight='bold')
    ax1.set_title('📏 ROM Across Sessions', fontweight='bold')
    ax1.grid(alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 2. Smoothness Progression
    ax2 = axes[0, 1]
    ax2.plot(session_ids, avg_smoothness, 's-', color=COLORS['success'], linewidth=2, markersize=10)
    ax2.fill_between(session_ids, avg_smoothness, alpha=0.3, color=COLORS['success'])
    ax2.axhline(y=70, color='green', linestyle='--', alpha=0.5, label='Target')
    ax2.set_xlabel('Session', fontweight='bold')
    ax2.set_ylabel('Average Smoothness', fontweight='bold')
    ax2.set_title('🎯 Smoothness Across Sessions', fontweight='bold')
    ax2.set_ylim(0, 100)
    ax2.legend()
    ax2.grid(alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 3. Fatigue Score Trend
    ax3 = axes[1, 0]
    colors = [COLORS['success'] if f < 30 else COLORS['warning'] if f < 60 else COLORS['danger'] for f in fatigue_scores]
    bars = ax3.bar(session_ids, fatigue_scores, color=colors, edgecolor='black', linewidth=0.5)
    ax3.set_xlabel('Session', fontweight='bold')
    ax3.set_ylabel('Fatigue Score', fontweight='bold')
    ax3.set_title('⚡ Fatigue Levels Across Sessions', fontweight='bold')
    ax3.set_ylim(0, 100)
    ax3.grid(axis='y', alpha=0.3)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add value labels
    for bar, val in zip(bars, fatigue_scores):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.0f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Volume (Total Reps) Progression
    ax4 = axes[1, 1]
    ax4.bar(session_ids, total_reps, color=COLORS['purple'], edgecolor='black', linewidth=0.5, alpha=0.7)
    ax4.set_xlabel('Session', fontweight='bold')
    ax4.set_ylabel('Total Reps', fontweight='bold')
    ax4.set_title('🔢 Training Volume Across Sessions', fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Saved: {output_path}")
    
    return fig


# =============================================================================
# UI FUNCTIONS
# =============================================================================

def select_data_file():
    """Open file dialog to select a CSV file."""
    root = tk.Tk()
    root.withdraw()
    
    file_path = filedialog.askopenfilename(
        title="Select Exercise Data CSV",
        initialdir=str(OUTPUT_DIR),
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    
    root.destroy()
    return file_path


def performance_visualizer_ui():
    """
    Main UI for performance visualization.
    """
    root = tk.Tk()
    root.title("🏋️ AppLift Performance Visualizer")
    root.geometry("800x600")
    root.configure(bg='#f5f5f5')
    
    # State
    state = {
        'df': None,
        'session_metrics': None,
        'file_path': None
    }
    
    # Header
    header_frame = tk.Frame(root, bg='#2196F3', pady=15)
    header_frame.pack(fill=tk.X)
    
    tk.Label(header_frame, text="🏋️ AppLift Performance Visualizer",
            font=('Arial', 20, 'bold'), bg='#2196F3', fg='white').pack()
    tk.Label(header_frame, text="Fatigue Analysis • Movement Smoothness • Range of Motion",
            font=('Arial', 11), bg='#2196F3', fg='white').pack()
    
    # Main content
    content_frame = tk.Frame(root, bg='#f5f5f5')
    content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
    
    # File selection
    file_frame = tk.LabelFrame(content_frame, text="📁 Data Selection", 
                               font=('Arial', 11, 'bold'), bg='#f5f5f5')
    file_frame.pack(fill=tk.X, pady=10)
    
    file_label = tk.Label(file_frame, text="No file selected", 
                         font=('Arial', 10), bg='#f5f5f5', fg='#666')
    file_label.pack(side=tk.LEFT, padx=10, pady=10)
    
    def load_file():
        file_path = select_data_file()
        if file_path:
            try:
                state['df'] = pd.read_csv(file_path)
                state['file_path'] = file_path
                file_label.config(text=f"✓ {Path(file_path).name}", fg='green')
                
                # Analyze data
                state['session_metrics'] = analyze_session_data(state['df'])
                
                if state['session_metrics']:
                    summary = state['session_metrics']['session_summary']
                    fatigue = state['session_metrics']['fatigue_analysis']
                    
                    stats_text.config(state='normal')
                    stats_text.delete(1.0, tk.END)
                    stats_text.insert(tk.END, f"📊 SESSION ANALYSIS\n")
                    stats_text.insert(tk.END, "=" * 40 + "\n\n")
                    stats_text.insert(tk.END, f"Total Reps: {summary.get('total_reps', 0)}\n")
                    stats_text.insert(tk.END, f"Average ROM: {summary.get('avg_rom', 0):.2f}\n")
                    stats_text.insert(tk.END, f"Average Smoothness: {summary.get('avg_smoothness', 0):.1f}/100\n")
                    stats_text.insert(tk.END, f"Average Duration: {summary.get('avg_duration_ms', 0)/1000:.2f}s\n\n")
                    stats_text.insert(tk.END, f"⚡ FATIGUE ANALYSIS (Gyro-Based)\n")
                    stats_text.insert(tk.END, "-" * 40 + "\n")
                    stats_text.insert(tk.END, f"Fatigue Score: {fatigue.get('fatigue_score', 0):.0f}/100\n")
                    stats_text.insert(tk.END, f"Fatigue Level: {fatigue.get('fatigue_level', 'unknown').upper()}\n")
                    direction = fatigue.get('gyro_direction', 'drop')
                    dir_label = "Speed Surge" if direction == 'surge' else "Speed Drop"
                    stats_text.insert(tk.END, f"D_ω ({dir_label}): {fatigue.get('D_omega', 0)*100:+.1f}%\n")
                    stats_text.insert(tk.END, f"I_T (Tempo Rise): {fatigue.get('I_T', 0)*100:+.1f}%\n")
                    stats_text.insert(tk.END, f"I_J (Jerk Rise):  {fatigue.get('I_J', 0)*100:+.1f}%\n")
                    stats_text.insert(tk.END, f"I_S (Shakiness):  {fatigue.get('I_S', 0)*100:+.1f}%\n")
                    rom_chg = fatigue.get('rom_change_percent', 0)
                    rom_dir_lbl = '↑ increased' if rom_chg > 0 else '↓ decreased' if rom_chg < 0 else 'stable'
                    stats_text.insert(tk.END, f"ROM Change: {rom_chg:+.1f}% ({rom_dir_lbl})\n")
                    
                    # Add new metrics
                    if 'consistency_score' in fatigue:
                        stats_text.insert(tk.END, f"\n📊 CONSISTENCY ANALYSIS\n")
                        stats_text.insert(tk.END, "-" * 40 + "\n")
                        stats_text.insert(tk.END, f"Overall Consistency: {fatigue.get('consistency_score', 0):.1f}%\n")
                        stats_text.insert(tk.END, f"ROM Consistency: {fatigue.get('rom_consistency', 0):.1f}%\n")
                        stats_text.insert(tk.END, f"Smoothness Consistency: {fatigue.get('smoothness_consistency', 0):.1f}%\n")
                    
                    stats_text.config(state='disabled')
                    
                    # Enable visualization buttons
                    for btn in viz_buttons:
                        btn.config(state='normal')
                else:
                    messagebox.showerror("Error", "Could not analyze data. Make sure it has 'rep' column.")
                    
            except Exception as e:
                error_msg = str(e)
                print(f"Load file error: {error_msg}")  # Debug output
                messagebox.showerror("Error", f"Failed to load file: {error_msg}")
    
    tk.Button(file_frame, text="📂 Load Data File", command=load_file,
             font=('Arial', 10), bg='#4CAF50', fg='white',
             padx=15, pady=5).pack(side=tk.RIGHT, padx=10, pady=10)
    
    # Stats display
    stats_frame = tk.LabelFrame(content_frame, text="📊 Session Statistics",
                                font=('Arial', 11, 'bold'), bg='#f5f5f5')
    stats_frame.pack(fill=tk.BOTH, expand=True, pady=10)
    
    stats_text = tk.Text(stats_frame, height=12, font=('Courier', 10),
                        bg='#f0f0f0', state='disabled')
    stats_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Visualization buttons
    viz_frame = tk.LabelFrame(content_frame, text="📈 Visualizations",
                              font=('Arial', 11, 'bold'), bg='#f5f5f5')
    viz_frame.pack(fill=tk.X, pady=10)
    
    viz_buttons = []
    
    def create_viz(viz_type):
        if state['session_metrics'] is None:
            messagebox.showwarning("Warning", "Please load a data file first!")
            return
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if viz_type == 'fatigue':
            output_path = VISUALIZATIONS_DIR / f'fatigue_analysis_{timestamp}.png'
            fig = create_fatigue_visualization(state['session_metrics'], output_path,
                                               title=f"Fatigue Analysis - {Path(state['file_path']).stem}")
        elif viz_type == 'ldlj':
            output_path = VISUALIZATIONS_DIR / f'smoothness_analysis_{timestamp}.png'
            fig = create_ldlj_visualization(state['session_metrics'], output_path,
                                            title=f"Smoothness Analysis - {Path(state['file_path']).stem}")
        elif viz_type == 'rom':
            output_path = VISUALIZATIONS_DIR / f'rom_analysis_{timestamp}.png'
            fig = create_rom_visualization(state['session_metrics'], output_path,
                                           title=f"ROM Analysis - {Path(state['file_path']).stem}")
        elif viz_type == 'all':
            # Create all visualizations
            base_name = Path(state['file_path']).stem
            
            create_fatigue_visualization(state['session_metrics'],
                                        VISUALIZATIONS_DIR / f'fatigue_{base_name}_{timestamp}.png',
                                        title=f"Fatigue Analysis - {base_name}")
            create_ldlj_visualization(state['session_metrics'],
                                     VISUALIZATIONS_DIR / f'smoothness_{base_name}_{timestamp}.png',
                                     title=f"Smoothness Analysis - {base_name}")
            create_rom_visualization(state['session_metrics'],
                                    VISUALIZATIONS_DIR / f'rom_{base_name}_{timestamp}.png',
                                    title=f"ROM Analysis - {base_name}")
            
            messagebox.showinfo("Success", f"All visualizations saved to:\n{VISUALIZATIONS_DIR}")
            return
        
        if fig:
            plt.show()
            messagebox.showinfo("Saved", f"Visualization saved to:\n{output_path}")
    
    btn_configs = [
        ("⚡ Fatigue Analysis", 'fatigue', '#f44336'),
        ("🎯 Smoothness (LDLJ)", 'ldlj', '#2196F3'),
        ("📏 Range of Motion", 'rom', '#4CAF50'),
        ("📊 Generate All", 'all', '#9C27B0'),
    ]
    
    for text, viz_type, color in btn_configs:
        btn = tk.Button(viz_frame, text=text, command=lambda vt=viz_type: create_viz(vt),
                       font=('Arial', 10, 'bold'), bg=color, fg='white',
                       padx=15, pady=8, state='disabled')
        btn.pack(side=tk.LEFT, padx=5, pady=10)
        viz_buttons.append(btn)
    
    # Output folder button
    def open_output_folder():
        import os
        os.startfile(str(VISUALIZATIONS_DIR))
    
    tk.Button(viz_frame, text="📁 Open Output Folder", command=open_output_folder,
             font=('Arial', 10), bg='#757575', fg='white',
             padx=15, pady=8).pack(side=tk.RIGHT, padx=5, pady=10)
    
    # Footer
    footer_frame = tk.Frame(root, bg='#f5f5f5')
    footer_frame.pack(fill=tk.X, pady=10)
    
    tk.Label(footer_frame, text="💡 Tip: Load a resegmented CSV file for best results",
            font=('Arial', 9), bg='#f5f5f5', fg='#666').pack()
    
    # Center window
    root.update_idletasks()
    x = (root.winfo_screenwidth() - root.winfo_width()) // 2
    y = (root.winfo_screenheight() - root.winfo_height()) // 2
    root.geometry(f"+{x}+{y}")
    
    root.mainloop()


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def analyze_and_visualize(file_path, output_dir=None):
    """
    Analyze a data file and generate all visualizations.
    
    Parameters:
    - file_path: Path to CSV file
    - output_dir: Output directory for visualizations (optional)
    
    Returns:
    - session_metrics: Analysis results
    """
    print("\n" + "=" * 60)
    print("🏋️ APPLIFT PERFORMANCE ANALYZER")
    print("=" * 60)
    
    file_path = Path(file_path)
    
    if output_dir:
        viz_dir = Path(output_dir)
    else:
        viz_dir = VISUALIZATIONS_DIR
    
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📂 Loading: {file_path.name}")
    df = pd.read_csv(file_path)
    print(f"   Rows: {len(df):,} | Columns: {len(df.columns)}")
    
    print("\n🔍 Analyzing session data...")
    session_metrics = analyze_session_data(df)
    
    if session_metrics is None:
        print("❌ Analysis failed. Check that data has 'rep' column.")
        return None
    
    summary = session_metrics['session_summary']
    fatigue = session_metrics['fatigue_analysis']
    
    print(f"\n📊 SESSION SUMMARY")
    print("-" * 40)
    print(f"   Total Reps: {summary['total_reps']}")
    print(f"   Average ROM: {summary['avg_rom']:.2f}")
    print(f"   Average Smoothness: {summary['avg_smoothness']:.1f}/100")
    print(f"   Average Duration: {summary['avg_duration_ms']/1000:.2f}s")
    
    print(f"\n⚡ FATIGUE ANALYSIS")
    print("-" * 40)
    print(f"   Fatigue Score: {fatigue['fatigue_score']:.0f}/100")
    print(f"   Fatigue Level: {fatigue['fatigue_level'].upper()}")
    rom_chg = fatigue.get('rom_change_percent', 0)
    rom_dir_lbl = '↑ increased' if rom_chg > 0 else '↓ decreased' if rom_chg < 0 else 'stable'
    print(f"   ROM Change: {rom_chg:+.1f}% ({rom_dir_lbl})")
    print(f"   Smoothness Drop: {fatigue['smoothness_degradation_percent']:.1f}%")
    
    # Generate visualizations
    print("\n📈 Generating visualizations...")
    
    base_name = file_path.stem
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    create_fatigue_visualization(
        session_metrics,
        viz_dir / f'fatigue_{base_name}_{timestamp}.png',
        title=f"Fatigue Analysis - {base_name}"
    )
    
    create_ldlj_visualization(
        session_metrics,
        viz_dir / f'smoothness_{base_name}_{timestamp}.png',
        title=f"Smoothness Analysis - {base_name}"
    )
    
    create_rom_visualization(
        session_metrics,
        viz_dir / f'rom_{base_name}_{timestamp}.png',
        title=f"ROM Analysis - {base_name}"
    )
    
    print(f"\n✅ All visualizations saved to: {viz_dir}")
    print("=" * 60)
    
    return session_metrics


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Command line mode
        file_path = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else None
        analyze_and_visualize(file_path, output_dir)
    else:
        # UI mode
        performance_visualizer_ui()
