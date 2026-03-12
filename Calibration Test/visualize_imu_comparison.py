"""
IMU Comparison Visualizer
Visualizes the difference between Smartphone IMU and ESP32 IMU data from CSV
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

# Get the script's directory and project root
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent

# Paths for data and output
DATA_DIR = PROJECT_ROOT / 'data'
VIZ_DIR = PROJECT_ROOT / 'visualizations' / 'imu_comparison_plots'

# Ensure output directory exists
VIZ_DIR.mkdir(parents=True, exist_ok=True)

def load_comparison_data(csv_path: str) -> pd.DataFrame:
    """Load IMU comparison CSV data - use raw values as-is (phone in rad/s, ESP32 in deg/s)"""
    df = pd.read_csv(csv_path)
    
    # Convert timestamp to seconds from start
    if 'timestamp_ms' in df.columns:
        df['time_sec'] = (df['timestamp_ms'] - df['timestamp_ms'].iloc[0]) / 1000
    
    print("=== LOADING RAW DATA (NO CONVERSION) ===")
    print(f"Loaded {len(df)} samples")
    
    # Use raw gyro values as-is from CSV
    # CSV has: phone_gyroX_rad (rad/s), esp32_gyroX_deg (deg/s)
    for axis in ['X', 'Y', 'Z']:
        phone_rad_col = f'phone_gyro{axis}_rad'
        esp32_deg_col = f'esp32_gyro{axis}_deg'
        
        # Use phone rad/s as-is
        if phone_rad_col in df.columns:
            df[f'phone_gyro{axis}'] = df[phone_rad_col]
            print(f"Using {phone_rad_col} as phone_gyro{axis} (rad/s)")
        
        # Interpolate ESP32 deg/s to match phone sampling rate
        if esp32_deg_col in df.columns:
            # Forward fill then interpolate to smooth out the steps
            df[f'esp32_gyro{axis}_raw'] = df[esp32_deg_col]
            df[f'esp32_gyro{axis}'] = df[esp32_deg_col].interpolate(method='linear', limit_direction='both')
            print(f"Interpolated {esp32_deg_col} to match phone sampling rate")
        
        # Use pre-calculated difference from CSV (in rad/s)
        diff_rad_col = f'diff_gyro{axis}_rad'
        if diff_rad_col in df.columns:
            df[f'diff_gyro{axis}'] = df[diff_rad_col]
            print(f"Using {diff_rad_col} as diff_gyro{axis} (rad/s)")
    
    # Also interpolate ESP32 orientation data
    for axis in ['roll', 'pitch', 'yaw']:
        esp32_col = f'esp32_{axis}'
        if esp32_col in df.columns:
            df[f'{esp32_col}_raw'] = df[esp32_col]
            df[esp32_col] = df[esp32_col].interpolate(method='linear', limit_direction='both')
            print(f"Interpolated {esp32_col}")
    
    # Interpolate ESP32 accelerometer data
    for axis in ['X', 'Y', 'Z']:
        esp32_col = f'esp32_accel{axis}'
        if esp32_col in df.columns:
            df[f'{esp32_col}_raw'] = df[esp32_col]
            df[esp32_col] = df[esp32_col].interpolate(method='linear', limit_direction='both')
            print(f"Interpolated {esp32_col}")
    
    print("\nGyroscope data ranges (after interpolation):")
    for axis in ['X', 'Y', 'Z']:
        phone_col = f'phone_gyro{axis}'
        esp32_col = f'esp32_gyro{axis}'
        if phone_col in df.columns:
            print(f"  {phone_col}: min={df[phone_col].min():.4f} rad/s, max={df[phone_col].max():.4f} rad/s")
        if esp32_col in df.columns:
            print(f"  {esp32_col}: min={df[esp32_col].min():.4f} deg/s, max={df[esp32_col].max():.4f} deg/s")
    
    print("\nOrientation data ranges:")
    for axis in ['roll', 'pitch', 'yaw']:
        phone_col = f'phone_{axis}'
        esp32_col = f'esp32_{axis}'
        if phone_col in df.columns:
            print(f"  {phone_col}: min={df[phone_col].min():.2f}°, max={df[phone_col].max():.2f}°")
        if esp32_col in df.columns:
            print(f"  {esp32_col}: min={df[esp32_col].min():.2f}°, max={df[esp32_col].max():.2f}°")
    
    return df

def plot_yaw_correction_comparison(df: pd.DataFrame, save_path: str = None):
    """Plot yaw comparison - original values only (no correction applied)"""
    # This function is kept for compatibility but won't show correction since we use original values
    print("Skipping yaw correction comparison - using original values only")
    return

def plot_orientation_comparison(df: pd.DataFrame, save_path: str = None):
    """Plot orientation (Roll, Pitch, Yaw) comparison"""
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle('Orientation Comparison: Phone vs ESP32', fontsize=14, fontweight='bold')
    
    time = df['time_sec'] if 'time_sec' in df.columns else df.index
    
    # Roll
    axes[0, 0].plot(time, df['phone_roll'], label='Phone', color='#2196f3', alpha=0.8)
    axes[0, 0].plot(time, df['esp32_roll'], label='ESP32', color='#f44336', alpha=0.8)
    axes[0, 0].set_ylabel('Roll (°)')
    axes[0, 0].set_title('Roll Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(time, df['diff_roll'], color='#ff9800', linewidth=1)
    axes[0, 1].axhline(y=0, color='green', linestyle='--', alpha=0.5)
    axes[0, 1].axhline(y=5, color='orange', linestyle=':', alpha=0.5)
    axes[0, 1].axhline(y=-5, color='orange', linestyle=':', alpha=0.5)
    axes[0, 1].fill_between(time, -5, 5, alpha=0.1, color='green')
    axes[0, 1].set_ylabel('Δ Roll (°)')
    axes[0, 1].set_title(f'Roll Difference (Mean: {df["diff_roll"].mean():.2f}°, Std: {df["diff_roll"].std():.2f}°)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Pitch
    axes[1, 0].plot(time, df['phone_pitch'], label='Phone', color='#2196f3', alpha=0.8)
    axes[1, 0].plot(time, df['esp32_pitch'], label='ESP32', color='#f44336', alpha=0.8)
    axes[1, 0].set_ylabel('Pitch (°)')
    axes[1, 0].set_title('Pitch Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(time, df['diff_pitch'], color='#ff9800', linewidth=1)
    axes[1, 1].axhline(y=0, color='green', linestyle='--', alpha=0.5)
    axes[1, 1].axhline(y=5, color='orange', linestyle=':', alpha=0.5)
    axes[1, 1].axhline(y=-5, color='orange', linestyle=':', alpha=0.5)
    axes[1, 1].fill_between(time, -5, 5, alpha=0.1, color='green')
    axes[1, 1].set_ylabel('Δ Pitch (°)')
    axes[1, 1].set_title(f'Pitch Difference (Mean: {df["diff_pitch"].mean():.2f}°, Std: {df["diff_pitch"].std():.2f}°)')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Yaw
    axes[2, 0].plot(time, df['phone_yaw'], label='Phone', color='#2196f3', alpha=0.8)
    axes[2, 0].plot(time, df['esp32_yaw'], label='ESP32', color='#f44336', alpha=0.8)
    axes[2, 0].set_ylabel('Yaw (°)')
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 0].set_title('Yaw Comparison')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    axes[2, 1].plot(time, df['diff_yaw'], color='#ff9800', linewidth=1)
    axes[2, 1].axhline(y=0, color='green', linestyle='--', alpha=0.5)
    axes[2, 1].axhline(y=5, color='orange', linestyle=':', alpha=0.5)
    axes[2, 1].axhline(y=-5, color='orange', linestyle=':', alpha=0.5)
    axes[2, 1].fill_between(time, -5, 5, alpha=0.1, color='green')
    axes[2, 1].set_ylabel('Δ Yaw (°)')
    axes[2, 1].set_xlabel('Time (s)')
    axes[2, 1].set_title(f'Yaw Difference (Mean: {df["diff_yaw"].mean():.2f}°, Std: {df["diff_yaw"].std():.2f}°)')
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()

def plot_accelerometer_comparison(df: pd.DataFrame, save_path: str = None):
    """Plot accelerometer comparison"""
    # Check if accelerometer columns exist
    accel_cols = ['phone_accelX', 'phone_accelY', 'phone_accelZ', 'esp32_accelX', 'esp32_accelY', 'esp32_accelZ']
    if not all(col in df.columns for col in accel_cols):
        print("Accelerometer data not found in CSV")
        return
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle('Accelerometer Comparison: Phone vs ESP32', fontsize=14, fontweight='bold')
    
    time = df['time_sec'] if 'time_sec' in df.columns else df.index
    
    labels = ['X', 'Y', 'Z']
    colors_phone = '#2196f3'
    colors_esp32 = '#f44336'
    
    for i, axis in enumerate(['X', 'Y', 'Z']):
        phone_col = f'phone_accel{axis}'
        esp32_col = f'esp32_accel{axis}'
        diff_col = f'diff_accel{axis}'
        
        # Comparison plot
        axes[i, 0].plot(time, df[phone_col], label='Phone', color=colors_phone, alpha=0.8)
        axes[i, 0].plot(time, df[esp32_col], label='ESP32', color=colors_esp32, alpha=0.8)
        axes[i, 0].set_ylabel(f'Accel {axis} (m/s²)')
        axes[i, 0].set_title(f'Accelerometer {axis} Comparison')
        axes[i, 0].legend()
        axes[i, 0].grid(True, alpha=0.3)
        
        # Difference plot
        if diff_col in df.columns:
            axes[i, 1].plot(time, df[diff_col], color='#ff9800', linewidth=1)
            axes[i, 1].axhline(y=0, color='green', linestyle='--', alpha=0.5)
            axes[i, 1].axhline(y=1.0, color='orange', linestyle=':', alpha=0.5)
            axes[i, 1].axhline(y=-1.0, color='orange', linestyle=':', alpha=0.5)
            axes[i, 1].fill_between(time, -1.0, 1.0, alpha=0.1, color='green')
            axes[i, 1].set_ylabel(f'Δ Accel {axis} (m/s²)')
            axes[i, 1].set_title(f'Accel {axis} Difference (Mean: {df[diff_col].mean():.2f} m/s², Std: {df[diff_col].std():.2f} m/s²)')
            axes[i, 1].grid(True, alpha=0.3)
    
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 1].set_xlabel('Time (s)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()

def plot_gyroscope_comparison(df: pd.DataFrame, save_path: str = None):
    """Plot gyroscope comparison - phone in rad/s, ESP32 in deg/s (raw values)"""
    # Check if gyroscope columns exist
    gyro_cols = ['phone_gyroX', 'phone_gyroY', 'phone_gyroZ', 'esp32_gyroX', 'esp32_gyroY', 'esp32_gyroZ']
    if not all(col in df.columns for col in gyro_cols):
        print("Gyroscope data not found in CSV")
        return
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle('Gyroscope Comparison: Phone (rad/s) vs ESP32 (deg/s) - Raw Values', fontsize=14, fontweight='bold')
    
    time = df['time_sec'] if 'time_sec' in df.columns else df.index
    
    for i, axis in enumerate(['X', 'Y', 'Z']):
        phone_col = f'phone_gyro{axis}'
        esp32_col = f'esp32_gyro{axis}'
        diff_col = f'diff_gyro{axis}'
        
        # Comparison plot - dual y-axis since different units
        ax1 = axes[i, 0]
        ax2 = ax1.twinx()
        
        # Phone (rad/s) - left y-axis
        line1 = ax1.plot(time, df[phone_col], label='Phone (rad/s)', color='#2196f3', alpha=0.8, linewidth=1.5)
        ax1.set_ylabel(f'Phone Gyro {axis} (rad/s)', color='#2196f3')
        ax1.tick_params(axis='y', labelcolor='#2196f3')
        
        # ESP32 (deg/s) - right y-axis
        line2 = ax2.plot(time, df[esp32_col], label='ESP32 (deg/s)', color='#f44336', alpha=0.8, linewidth=1.5)
        ax2.set_ylabel(f'ESP32 Gyro {axis} (deg/s)', color='#f44336')
        ax2.tick_params(axis='y', labelcolor='#f44336')
        
        ax1.set_title(f'Gyroscope {axis} Comparison (Different Units)')
        ax1.grid(True, alpha=0.3)
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')
        
        # Difference plot (in rad/s as per CSV)
        if diff_col in df.columns:
            axes[i, 1].plot(time, df[diff_col], color='#ff9800', linewidth=1)
            axes[i, 1].axhline(y=0, color='green', linestyle='--', alpha=0.5)
            axes[i, 1].axhline(y=0.087, color='orange', linestyle=':', alpha=0.5, label='±5°/s (~0.087 rad/s)')
            axes[i, 1].axhline(y=-0.087, color='orange', linestyle=':', alpha=0.5)
            axes[i, 1].fill_between(time, -0.087, 0.087, alpha=0.1, color='green')
            axes[i, 1].set_ylabel(f'Δ Gyro {axis} (rad/s)')
            axes[i, 1].set_title(f'Gyro {axis} Difference (Mean: {df[diff_col].mean():.4f} rad/s, Std: {df[diff_col].std():.4f} rad/s)')
            axes[i, 1].legend(fontsize=8)
            axes[i, 1].grid(True, alpha=0.3)
    
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 1].set_xlabel('Time (s)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()

def plot_difference_histograms(df: pd.DataFrame, save_path: str = None):
    """Plot histograms of all differences"""
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    fig.suptitle('Distribution of Differences (ESP32 - Phone)', fontsize=14, fontweight='bold')
    
    # Orientation differences
    diff_cols = [
        ('diff_roll', 'Roll (°)', 5, axes[0, 0]),
        ('diff_pitch', 'Pitch (°)', 5, axes[0, 1]),
        ('diff_yaw', 'Yaw (°)', 5, axes[0, 2]),
    ]
    
    # Accelerometer differences
    if 'diff_accelX' in df.columns:
        diff_cols.extend([
            ('diff_accelX', 'Accel X (m/s²)', 1.0, axes[1, 0]),
            ('diff_accelY', 'Accel Y (m/s²)', 1.0, axes[1, 1]),
            ('diff_accelZ', 'Accel Z (m/s²)', 1.0, axes[1, 2]),
        ])
    
    # Gyroscope differences (in rad/s from CSV)
    if 'diff_gyroX' in df.columns:
        diff_cols.extend([
            ('diff_gyroX', 'Gyro X (rad/s)', 0.087, axes[2, 0]),
            ('diff_gyroY', 'Gyro Y (rad/s)', 0.087, axes[2, 1]),
            ('diff_gyroZ', 'Gyro Z (rad/s)', 0.087, axes[2, 2]),
        ])
    
    for col, label, threshold, ax in diff_cols:
        if col in df.columns:
            data = df[col].dropna()
            
            # Calculate statistics
            mean = data.mean()
            std = data.std()
            within_threshold = (data.abs() < threshold).sum() / len(data) * 100
            
            # Plot histogram
            ax.hist(data, bins=50, color='#ff9800', alpha=0.7, edgecolor='white')
            ax.axvline(x=0, color='green', linestyle='--', linewidth=2, label='Ideal (0)')
            ax.axvline(x=mean, color='red', linestyle='-', linewidth=2, label=f'Mean ({mean:.2f})')
            ax.axvline(x=threshold, color='orange', linestyle=':', alpha=0.7)
            ax.axvline(x=-threshold, color='orange', linestyle=':', alpha=0.7)
            
            ax.set_xlabel(f'Δ {label}')
            ax.set_ylabel('Count')
            ax.set_title(f'{label}\nMean: {mean:.2f}, Std: {std:.2f}\n{within_threshold:.1f}% within ±{threshold}')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()

def print_summary_statistics(df: pd.DataFrame, save_path: str = None):
    """Print summary statistics and optionally save to file"""
    summary_lines = []
    
    summary_lines.append("=" * 60)
    summary_lines.append("IMU COMPARISON SUMMARY (INTERPOLATED VALUES)")
    summary_lines.append("=" * 60)
    summary_lines.append("")
    summary_lines.append(f"Total Samples: {len(df)}")
    if 'time_sec' in df.columns:
        summary_lines.append(f"Duration: {df['time_sec'].max():.1f} seconds")
    summary_lines.append("")
    
    # Orientation
    summary_lines.append("📐 ORIENTATION DIFFERENCES:")
    summary_lines.append("-" * 40)
    for axis in ['roll', 'pitch', 'yaw']:
        col = f'diff_{axis}'
        if col in df.columns:
            data = df[col].dropna()
            mean = data.mean()
            std = data.std()
            abs_mean = data.abs().mean()
            max_diff = data.abs().max()
            within_5deg = (data.abs() < 5).sum() / len(data) * 100
            
            summary_lines.append(f"  {axis.capitalize():6} | Mean: {mean:7.2f}° | Std: {std:6.2f}° | Abs Mean: {abs_mean:6.2f}° | Max: {max_diff:6.2f}° | Within ±5°: {within_5deg:.1f}%")
    summary_lines.append("")
    
    # Accelerometer
    if 'diff_accelX' in df.columns:
        summary_lines.append("📊 ACCELEROMETER DIFFERENCES:")
        summary_lines.append("-" * 40)
        for axis in ['X', 'Y', 'Z']:
            col = f'diff_accel{axis}'
            if col in df.columns:
                data = df[col].dropna()
                mean = data.mean()
                std = data.std()
                abs_mean = data.abs().mean()
                max_diff = data.abs().max()
                within_1ms2 = (data.abs() < 1.0).sum() / len(data) * 100
                
                summary_lines.append(f"  Accel {axis} | Mean: {mean:7.2f} m/s² | Std: {std:6.2f} m/s² | Abs Mean: {abs_mean:6.2f} m/s² | Max: {max_diff:6.2f} m/s² | Within ±1 m/s²: {within_1ms2:.1f}%")
        summary_lines.append("")
    
    # Gyroscope (raw values: phone in rad/s, ESP32 in deg/s after interpolation)
    if 'diff_gyroX' in df.columns:
        summary_lines.append("🔄 GYROSCOPE (RAW VALUES - Phone: rad/s, ESP32: deg/s):")
        summary_lines.append("-" * 40)
        for axis in ['X', 'Y', 'Z']:
            phone_col = f'phone_gyro{axis}'
            esp32_col = f'esp32_gyro{axis}'
            diff_col = f'diff_gyro{axis}'
            
            if all(col in df.columns for col in [phone_col, esp32_col, diff_col]):
                phone_data = df[phone_col].dropna()
                esp32_data = df[esp32_col].dropna()
                diff_data = df[diff_col].dropna()
                
                phone_range = phone_data.max() - phone_data.min()
                esp32_range = esp32_data.max() - esp32_data.min()
                diff_mean = diff_data.mean()
                diff_std = diff_data.std()
                diff_abs_mean = diff_data.abs().mean()
                diff_max = diff_data.abs().max()
                within_0087rad = (diff_data.abs() < 0.087).sum() / len(diff_data) * 100  # 0.087 rad/s ≈ 5°/s
                
                summary_lines.append(f"  Gyro {axis}:")
                summary_lines.append(f"    Phone:  min={phone_data.min():7.4f} rad/s, max={phone_data.max():7.4f} rad/s, range={phone_range:.4f} rad/s")
                summary_lines.append(f"    ESP32:  min={esp32_data.min():7.4f} deg/s, max={esp32_data.max():7.4f} deg/s, range={esp32_range:.4f} deg/s")
                summary_lines.append(f"    Diff:   mean={diff_mean:7.4f} rad/s, std={diff_std:6.4f} rad/s, abs_mean={diff_abs_mean:.4f} rad/s, max={diff_max:.4f} rad/s")
                summary_lines.append(f"    Within ±0.087 rad/s (~5°/s): {within_0087rad:.1f}%")
                summary_lines.append("")
    
    # Overall assessment
    summary_lines.append("=" * 60)
    summary_lines.append("OVERALL ASSESSMENT:")
    summary_lines.append("-" * 40)
    
    # Calculate overall metrics
    if 'diff_roll' in df.columns:
        orientation_diffs = []
        for axis in ['roll', 'pitch', 'yaw']:
            col = f'diff_{axis}'
            if col in df.columns:
                orientation_diffs.extend(df[col].abs().dropna().tolist())
        if orientation_diffs:
            avg_orientation_diff = np.mean(orientation_diffs)
            summary_lines.append(f"Average Orientation Difference: {avg_orientation_diff:.2f}°")
    
    if 'diff_accelX' in df.columns:
        accel_diffs = []
        for axis in ['X', 'Y', 'Z']:
            col = f'diff_accel{axis}'
            if col in df.columns:
                accel_diffs.extend(df[col].abs().dropna().tolist())
        if accel_diffs:
            avg_accel_diff = np.mean(accel_diffs)
            summary_lines.append(f"Average Accelerometer Difference: {avg_accel_diff:.2f} m/s²")
    
    if 'diff_gyroX' in df.columns:
        gyro_diffs = []
        for axis in ['X', 'Y', 'Z']:
            col = f'diff_gyro{axis}'
            if col in df.columns:
                gyro_diffs.extend(df[col].abs().dropna().tolist())
        if gyro_diffs:
            avg_gyro_diff = np.mean(gyro_diffs)
            avg_gyro_diff_deg = avg_gyro_diff * (180 / np.pi)  # Convert to deg/s for context
            summary_lines.append(f"Average Gyroscope Difference: {avg_gyro_diff:.4f} rad/s (~{avg_gyro_diff_deg:.2f}°/s)")
    
    summary_lines.append("=" * 60)
    
    # Print to console
    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)
    
    # Save to file if path provided
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(summary_text)
        print(f"\nSummary saved to: {save_path}")

def main():
    # Find CSV files in the workspace
    workspace = PROJECT_ROOT
    
    # Look for IMU comparison CSV files in data directory first
    csv_files = list(DATA_DIR.glob("imu_comparison*.csv"))
    
    # Also check workspace root if not found
    if not csv_files:
        csv_files = list(workspace.glob("**/imu_comparison*.csv"))
    
    if not csv_files:
        print("No IMU comparison CSV files found.")
        print("\nUsage: python visualize_imu_comparison.py <path_to_csv>")
        print(f"\nOr place a CSV file named 'imu_comparison*.csv' in {DATA_DIR}")
        
        # Check if a file path was provided as argument
        if len(sys.argv) > 1:
            csv_path = sys.argv[1]
            if Path(csv_path).exists():
                csv_files = [Path(csv_path)]
            else:
                print(f"\nFile not found: {csv_path}")
                return
        else:
            return
    
    # Use the most recent file if multiple found
    csv_file = sorted(csv_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]
    print(f"Loading: {csv_file}")
    
    # Load data
    df = load_comparison_data(str(csv_file))
    print(f"Loaded {len(df)} samples")
    
    # Print available columns
    print(f"\nColumns: {', '.join(df.columns)}")
    
    # Use the organized output directory
    output_dir = VIZ_DIR
    output_dir.mkdir(exist_ok=True)
    
    # Print and save summary statistics
    summary_file = output_dir / "comparison_summary.txt"
    print_summary_statistics(df, str(summary_file))
    
    # Generate plots
    print("\nGenerating plots...")
    
    # Yaw correction comparison (if correction was applied)
    if 'esp32_yaw_original' in df.columns:
        plot_yaw_correction_comparison(df, str(output_dir / "yaw_correction_comparison.png"))
    
    # Orientation comparison
    plot_orientation_comparison(df, str(output_dir / "orientation_comparison.png"))
    
    # Accelerometer comparison (if data exists)
    if 'phone_accelX' in df.columns:
        plot_accelerometer_comparison(df, str(output_dir / "accelerometer_comparison.png"))
    
    # Gyroscope comparison (if data exists)
    if 'phone_gyroX' in df.columns:
        plot_gyroscope_comparison(df, str(output_dir / "gyroscope_comparison.png"))
    
    # Difference histograms
    plot_difference_histograms(df, str(output_dir / "difference_histograms.png"))
    
    print(f"\nPlots saved to: {output_dir}")

if __name__ == "__main__":
    main()
