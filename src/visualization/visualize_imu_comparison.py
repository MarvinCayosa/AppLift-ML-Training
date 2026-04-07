"""
Comprehensive IMU Sensor Comparison Report Generator
Analyzes and visualizes differences between Smartphone IMU and ESP32 IMU data
Includes orientation, accelerometer, gyroscope, and quaternion analysis
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
import os

plt.rcParams.update({
    'figure.titlesize': 11,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
})

# ====================================================================
# CONFIGURABLE AXIS MAPPING - EDIT HERE TO TEST DIFFERENT COMBINATIONS
# ====================================================================
AXIS_MAPPING_CONFIG = {
    'X': {'esp32_axis': 'Y', 'invert': True},   # Phone X ↔ ESP32 Y (inverted)
    'Y': {'esp32_axis': 'Z', 'invert': False},  # Phone Y ↔ ESP32 Z
    'Z': {'esp32_axis': 'X', 'invert': True}    # Phone Z ↔ ESP32 X (inverted)
}

QUATERNION_MAPPING_CONFIG = {
    'phone': {
        'qw': {'source': 'qw', 'invert': False},
        'qx': {'source': 'qx', 'invert': False},
        'qy': {'source': 'qy', 'invert': False},
        'qz': {'source': 'qz', 'invert': False},
    },
    'esp32': {
        'qw': {'source': 'qx', 'invert': False},
        'qx': {'source': 'qw', 'invert': False},
        'qy': {'source': 'qw', 'invert': False},
        'qz': {'source': 'qx', 'invert': True},
    },
}

# Alternative mappings to try (uncomment to test):
# Option 1: Original mapping
# AXIS_MAPPING_CONFIG = {
#     'X': {'esp32_axis': 'Y', 'invert': True},   # Phone X ↔ ESP32 Y (inverted)
#     'Y': {'esp32_axis': 'Z', 'invert': False},  # Phone Y ↔ ESP32 Z
#     'Z': {'esp32_axis': 'X', 'invert': False}   # Phone Z ↔ ESP32 X
# }

# Option 2: Direct mapping
# AXIS_MAPPING_CONFIG = {
#     'X': {'esp32_axis': 'X', 'invert': False},  # Phone X ↔ ESP32 X
#     'Y': {'esp32_axis': 'Y', 'invert': False},  # Phone Y ↔ ESP32 Y
#     'Z': {'esp32_axis': 'Z', 'invert': False}   # Phone Z ↔ ESP32 Z
# }

# Option 3: Custom test (modify as needed)
# AXIS_MAPPING_CONFIG = {
#     'X': {'esp32_axis': 'Z', 'invert': True},   # Phone X ↔ ESP32 Z (inverted)
#     'Y': {'esp32_axis': 'X', 'invert': False},  # Phone Y ↔ ESP32 X
#     'Z': {'esp32_axis': 'Y', 'invert': True}    # Phone Z ↔ ESP32 Y (inverted)
# }

def print_mapping_config():
    """Print current axis mapping configuration"""
    print("\n=== CURRENT AXIS MAPPING CONFIGURATION ===")
    for phone_axis, config in AXIS_MAPPING_CONFIG.items():
        esp32_axis = config['esp32_axis']
        invert_str = " (INVERTED)" if config['invert'] else ""
        print(f"  Phone {phone_axis} ↔ ESP32 {esp32_axis}{invert_str}")
    print("=" * 45)

def print_quaternion_mapping_config():
    """Print current quaternion mapping configuration"""
    print("\n=== CURRENT QUATERNION MAPPING ===")
    for device in ['phone', 'esp32']:
        print(f"  {device.upper()}:")
        for target_quat in ['qw', 'qx', 'qy', 'qz']:
            config = QUATERNION_MAPPING_CONFIG[device][target_quat]
            invert_str = " (INVERTED)" if config['invert'] else ""
            print(f"    {target_quat} ← {config['source']}{invert_str}")
    print("=" * 39)

def compute_quaternion_angle_difference(df: pd.DataFrame):
    """Recompute quaternion angle difference after any remapping."""
    quat_cols = [
        'phone_qw', 'phone_qx', 'phone_qy', 'phone_qz',
        'esp32_qw', 'esp32_qx', 'esp32_qy', 'esp32_qz',
    ]
    if not all(col in df.columns for col in quat_cols):
        return

    phone_quat = df[['phone_qw', 'phone_qx', 'phone_qy', 'phone_qz']].to_numpy(dtype=float)
    esp32_quat = df[['esp32_qw', 'esp32_qx', 'esp32_qy', 'esp32_qz']].to_numpy(dtype=float)

    phone_norm = np.linalg.norm(phone_quat, axis=1)
    esp32_norm = np.linalg.norm(esp32_quat, axis=1)
    valid_mask = (
        np.isfinite(phone_quat).all(axis=1)
        & np.isfinite(esp32_quat).all(axis=1)
        & (phone_norm > 0)
        & (esp32_norm > 0)
    )

    diff_angles = np.full(len(df), np.nan)
    if valid_mask.any():
        phone_unit = phone_quat[valid_mask] / phone_norm[valid_mask, np.newaxis]
        esp32_unit = esp32_quat[valid_mask] / esp32_norm[valid_mask, np.newaxis]
        dot_products = np.sum(phone_unit * esp32_unit, axis=1)
        dot_products = np.clip(np.abs(dot_products), -1.0, 1.0)
        diff_angles[valid_mask] = 2.0 * np.degrees(np.arccos(dot_products))

    df['diff_quat_angle_deg'] = diff_angles

def apply_quaternion_mapping(df: pd.DataFrame):
    """Apply quaternion remapping to phone and ESP32 components."""
    print("\n=== QUATERNION REMAPPING ===")
    print_quaternion_mapping_config()

    for device in ['phone', 'esp32']:
        source_values = {}
        for quat in ['qw', 'qx', 'qy', 'qz']:
            col = f'{device}_{quat}'
            if col not in df.columns:
                continue

            if f'{col}_raw' not in df.columns:
                df[f'{col}_raw'] = df[col]

            source_series = df[f'{col}_raw']
            if device == 'esp32':
                source_series = source_series.interpolate(method='linear', limit_direction='both')
            source_values[quat] = source_series.copy()

        for target_quat, config in QUATERNION_MAPPING_CONFIG[device].items():
            source_quat = config['source']
            if source_quat not in source_values:
                continue

            mapped_values = source_values[source_quat].copy()
            if config['invert']:
                mapped_values = -mapped_values

            df[f'{device}_{target_quat}'] = mapped_values
            invert_str = ' (INVERTED)' if config['invert'] else ''
            print(f"  {device}_{target_quat} ← {device}_{source_quat}{invert_str}")

    if all(col in df.columns for col in ['esp32_qw', 'esp32_qx', 'esp32_qy', 'esp32_qz']):
        df['esp32_quatNorm'] = np.sqrt(
            df['esp32_qw']**2 + df['esp32_qx']**2 + df['esp32_qy']**2 + df['esp32_qz']**2
        )

    compute_quaternion_angle_difference(df)
    df.attrs['quaternion_mapping'] = QUATERNION_MAPPING_CONFIG
    print("✅ Quaternion remapping applied successfully.")

def test_all_axis_combinations(df):
    """Test all possible axis combinations and show correlation results"""
    print("\n=== TESTING ALL AXIS COMBINATIONS ===")
    print("Finding best correlations between phone and ESP32 gyroscope axes...\n")
    
    results = []
    
    for phone_axis in ['X', 'Y', 'Z']:
        phone_col = f'phone_gyro{phone_axis}_rad'
        if phone_col not in df.columns:
            continue
            
        phone_data = df[phone_col].dropna()
        
        for esp32_axis in ['X', 'Y', 'Z']:
            esp32_col = f'esp32_gyro{esp32_axis}_deg'
            if esp32_col not in df.columns:
                continue
                
            esp32_data = df[esp32_col].interpolate(method='linear', limit_direction='both').dropna()
            esp32_rad = np.radians(esp32_data)
            
            # Test both normal and inverted
            if len(phone_data) == len(esp32_rad):
                corr_normal = np.corrcoef(phone_data, esp32_rad)[0, 1]
                corr_inverted = np.corrcoef(phone_data, -esp32_rad)[0, 1]
                
                results.append((phone_axis, esp32_axis, False, corr_normal))
                results.append((phone_axis, esp32_axis, True, corr_inverted))
    
    # Sort by absolute correlation (best first)
    results.sort(key=lambda x: abs(x[3]), reverse=True)
    
    print("📊 CORRELATION RESULTS (sorted by strength):")
    print("-" * 55)
    print("Phone → ESP32   │ Invert │ Correlation   │ Rating")
    print("-" * 55)
    
    for phone_axis, esp32_axis, invert, corr in results[:12]:  # Show top 12
        invert_str = "Yes" if invert else "No "
        abs_corr = abs(corr)
        if abs_corr > 0.9:
            rating = "Excellent"
        elif abs_corr > 0.7:
            rating = "Good     "
        elif abs_corr > 0.5:
            rating = "Fair     "
        else:
            rating = "Poor     "
        
        print(f"  {phone_axis}   →   {esp32_axis}    │   {invert_str}   │   {corr:7.3f}   │ {rating}")
    
    print("-" * 55)
    print("\n💡 TIP: Use the best correlations above to update AXIS_MAPPING_CONFIG\n")

# Get the script's directory and project root
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR  # Use current directory as project root

# Paths for data and output
DATA_DIR = PROJECT_ROOT / 'data'
VIZ_DIR = PROJECT_ROOT / 'visualizations' / 'imu_comparison_plots'

# Ensure output directory exists
VIZ_DIR.mkdir(parents=True, exist_ok=True)

def load_comparison_data(csv_path: str) -> pd.DataFrame:
    """Load IMU comparison CSV data including quaternion values"""
    df = pd.read_csv(csv_path)
    
    # Convert timestamp to seconds from start
    if 'timestamp_ms' in df.columns:
        df['time_sec'] = (df['timestamp_ms'] - df['timestamp_ms'].iloc[0]) / 1000
    
    print("=== LOADING IMU COMPARISON DATA ===")
    print(f"Loaded {len(df)} samples")
    
    # Custom axis remapping for ESP32 vs Phone gyroscopes
    print("\n=== GYROSCOPE AXIS REMAPPING ===")
    print_mapping_config()
    
    axis_mapping = AXIS_MAPPING_CONFIG
    
    for phone_axis in ['X', 'Y', 'Z']:
        phone_rad_col = f'phone_gyro{phone_axis}_rad'
        esp32_axis = axis_mapping[phone_axis]['esp32_axis']
        invert = axis_mapping[phone_axis]['invert']
        esp32_deg_col = f'esp32_gyro{esp32_axis}_deg'
        
        # Use phone rad/s as-is
        if phone_rad_col in df.columns:
            df[f'phone_gyro{phone_axis}'] = df[phone_rad_col]
            print(f"Using {phone_rad_col} as phone_gyro{phone_axis} (rad/s)")
        
        # Map ESP32 axis and interpolate
        if esp32_deg_col in df.columns:
            # Forward fill then interpolate to smooth out the steps
            df[f'esp32_gyro{phone_axis}_raw'] = df[esp32_deg_col]
            df[f'esp32_gyro{phone_axis}_temp'] = df[esp32_deg_col].interpolate(method='linear', limit_direction='both')
            
            # Apply inversion if needed
            if invert:
                df[f'esp32_gyro{phone_axis}'] = -df[f'esp32_gyro{phone_axis}_temp']
                print(f"📍 Mapped {esp32_deg_col} → esp32_gyro{phone_axis} (INVERTED)")
            else:
                df[f'esp32_gyro{phone_axis}'] = df[f'esp32_gyro{phone_axis}_temp']
                print(f"📍 Mapped {esp32_deg_col} → esp32_gyro{phone_axis}")
            
            # Calculate correlation with remapped data
            if phone_rad_col in df.columns:
                esp32_rad = np.radians(df[f'esp32_gyro{phone_axis}'])
                phone_rad = df[phone_rad_col]
                valid_mask = ~(phone_rad.isna() | esp32_rad.isna())
                if valid_mask.sum() > 1:
                    corr = np.corrcoef(phone_rad[valid_mask], esp32_rad[valid_mask])[0, 1]
                    print(f"   Correlation: {corr:.3f}")
        
        # Calculate differences with remapped data
        if f'phone_gyro{phone_axis}' in df.columns and f'esp32_gyro{phone_axis}' in df.columns:
            # Convert ESP32 deg/s to rad/s for difference calculation
            esp32_rad = np.radians(df[f'esp32_gyro{phone_axis}'])
            df[f'diff_gyro{phone_axis}'] = esp32_rad - df[f'phone_gyro{phone_axis}']
            print(f"Calculated difference for gyro{phone_axis}")
    
    # Store axis mapping info for reporting
    df.attrs['axis_mapping'] = axis_mapping
    print(f"\n✅ Custom axis remapping applied successfully.")
    
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
    
    apply_quaternion_mapping(df)
    
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
    
    print("\nQuaternion data ranges:")
    for quat in ['qw', 'qx', 'qy', 'qz']:
        phone_col = f'phone_{quat}'
        esp32_col = f'esp32_{quat}'
        if phone_col in df.columns:
            print(f"  {phone_col}: min={df[phone_col].min():.4f}, max={df[phone_col].max():.4f}")
        if esp32_col in df.columns:
            print(f"  {esp32_col}: min={df[esp32_col].min():.4f}, max={df[esp32_col].max():.4f}")
    
    if 'diff_quat_angle_deg' in df.columns:
        print(f"  diff_quat_angle_deg: min={df['diff_quat_angle_deg'].min():.2f}°, max={df['diff_quat_angle_deg'].max():.2f}°")
    
    return df

def plot_quaternion_comparison(df: pd.DataFrame, save_path: str = None):
    """Plot quaternion components comparison"""
    # Check if quaternion columns exist
    quat_cols = ['phone_qw', 'phone_qx', 'phone_qy', 'phone_qz', 'esp32_qw', 'esp32_qx', 'esp32_qy', 'esp32_qz']
    if not all(col in df.columns for col in quat_cols):
        print("Quaternion data not found in CSV")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5))
    fig.suptitle('Quaternion Components Comparison', fontsize=12, fontweight='semibold')
    
    time = df['time_sec'] if 'time_sec' in df.columns else df.index
    
    components = ['qw', 'qx', 'qy', 'qz']
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    
    for comp, (row, col) in zip(components, positions):
        phone_col = f'phone_{comp}'
        esp32_col = f'esp32_{comp}'
        
        axes[row, col].plot(time, df[phone_col], label='Phone', color='#2196f3', alpha=0.8, linewidth=1.4)
        axes[row, col].plot(time, df[esp32_col], label='ESP32', color='#f44336', alpha=0.8, linewidth=1.4)
        axes[row, col].set_ylabel(f'{comp.upper()}')
        axes[row, col].set_title(f'Quaternion {comp.upper()} Component', fontsize=10)
        axes[row, col].legend()
        axes[row, col].grid(True, alpha=0.3)
        
        # Add range information
        phone_range = df[phone_col].max() - df[phone_col].min()
        esp32_range = df[esp32_col].max() - df[esp32_col].min()
        axes[row, col].text(0.02, 0.98, f'Phone range: {phone_range:.4f}\nESP32 range: {esp32_range:.4f}', 
                           transform=axes[row, col].transAxes, fontsize=6, 
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 1].set_xlabel('Time (s)')
    
    fig.tight_layout(pad=2.0, w_pad=1.8, h_pad=1.8)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()

def plot_quaternion_analysis(df: pd.DataFrame, save_path: str = None):
    """Plot quaternion analysis including angle differences and norms"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5))
    fig.suptitle('Quaternion Analysis', fontsize=12, fontweight='semibold')
    
    time = df['time_sec'] if 'time_sec' in df.columns else df.index
    
    # Quaternion angle difference
    if 'diff_quat_angle_deg' in df.columns:
        axes[0, 0].plot(time, df['diff_quat_angle_deg'], color='#ff9800', linewidth=1.3)
        axes[0, 0].axhline(y=0, color='green', linestyle='--', alpha=0.5)
        axes[0, 0].axhline(y=5, color='orange', linestyle=':', alpha=0.5)
        axes[0, 0].axhline(y=10, color='red', linestyle=':', alpha=0.5)
        axes[0, 0].fill_between(time, 0, 5, alpha=0.1, color='green')
        axes[0, 0].fill_between(time, 5, 10, alpha=0.1, color='orange')
        axes[0, 0].set_ylabel('Angle Difference (°)')
        axes[0, 0].set_title(f'Quaternion Angle Difference\nMean: {df["diff_quat_angle_deg"].mean():.2f}°, Std: {df["diff_quat_angle_deg"].std():.2f}°', fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)
    
    # Calculate quaternion norms for both devices
    phone_norm = np.sqrt(df['phone_qw']**2 + df['phone_qx']**2 + df['phone_qy']**2 + df['phone_qz']**2)
    axes[0, 1].plot(time, phone_norm, label='Phone', color='#2196f3', alpha=0.8)
    if 'esp32_quatNorm' in df.columns:
        axes[0, 1].plot(time, df['esp32_quatNorm'], label='ESP32', color='#f44336', alpha=0.8)
    else:
        esp32_norm = np.sqrt(df['esp32_qw']**2 + df['esp32_qx']**2 + df['esp32_qy']**2 + df['esp32_qz']**2)
        axes[0, 1].plot(time, esp32_norm, label='ESP32', color='#f44336', alpha=0.8)
    axes[0, 1].axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Ideal (1.0)')
    axes[0, 1].set_ylabel('Quaternion Norm')
    axes[0, 1].set_title('Quaternion Normalization', fontsize=10)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Quaternion difference histogram
    if 'diff_quat_angle_deg' in df.columns:
        data = df['diff_quat_angle_deg'].dropna()
        axes[1, 0].hist(data, bins=50, color='#ff9800', alpha=0.7, edgecolor='white')
        axes[1, 0].axvline(x=data.mean(), color='red', linestyle='-', linewidth=2, label=f'Mean ({data.mean():.2f}°)')
        axes[1, 0].axvline(x=5, color='orange', linestyle=':', alpha=0.7, label='5° threshold')
        axes[1, 0].axvline(x=10, color='red', linestyle=':', alpha=0.7, label='10° threshold')
        axes[1, 0].set_xlabel('Angle Difference (°)')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Distribution of Quaternion Angle Differences', fontsize=10)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Quaternion component differences (if available or calculated)
    quat_diffs = []
    for comp in ['qw', 'qx', 'qy', 'qz']:
        phone_col = f'phone_{comp}'
        esp32_col = f'esp32_{comp}'
        if phone_col in df.columns and esp32_col in df.columns:
            diff = df[esp32_col] - df[phone_col]
            quat_diffs.append((comp, diff))
    
    if quat_diffs:
        for i, (comp, diff) in enumerate(quat_diffs):
            axes[1, 1].plot(time, diff, label=f'Δ{comp.upper()}', alpha=0.7)
        axes[1, 1].axhline(y=0, color='green', linestyle='--', alpha=0.5)
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Component Difference')
        axes[1, 1].set_title('Quaternion Component Differences', fontsize=10)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    fig.tight_layout(pad=2.0, w_pad=1.8, h_pad=1.8)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()

def plot_orientation_comparison(df: pd.DataFrame, save_path: str = None):
    """Plot orientation (Roll, Pitch, Yaw) comparison"""
    fig, axes = plt.subplots(3, 2, figsize=(15, 11))
    fig.suptitle('Orientation Comparison', fontsize=12, fontweight='semibold')
    
    time = df['time_sec'] if 'time_sec' in df.columns else df.index
    
    # Roll
    axes[0, 0].plot(time, df['phone_roll'], label='Phone', color='#2196f3', alpha=0.8, linewidth=1.3)
    axes[0, 0].plot(time, df['esp32_roll'], label='ESP32', color='#f44336', alpha=0.8, linewidth=1.3)
    axes[0, 0].set_ylabel('Roll (°)')
    axes[0, 0].set_title('Roll Comparison', fontsize=10)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(time, df['diff_roll'], color='#ff9800', linewidth=1)
    axes[0, 1].axhline(y=0, color='green', linestyle='--', alpha=0.5)
    axes[0, 1].axhline(y=5, color='orange', linestyle=':', alpha=0.5)
    axes[0, 1].axhline(y=-5, color='orange', linestyle=':', alpha=0.5)
    axes[0, 1].fill_between(time, -5, 5, alpha=0.1, color='green')
    axes[0, 1].set_ylabel('Δ Roll (°)')
    axes[0, 1].set_title(f'Roll Difference\nMean: {df["diff_roll"].mean():.2f}°, Std: {df["diff_roll"].std():.2f}°', fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Pitch
    axes[1, 0].plot(time, df['phone_pitch'], label='Phone', color='#2196f3', alpha=0.8, linewidth=1.3)
    axes[1, 0].plot(time, df['esp32_pitch'], label='ESP32', color='#f44336', alpha=0.8, linewidth=1.3)
    axes[1, 0].set_ylabel('Pitch (°)')
    axes[1, 0].set_title('Pitch Comparison', fontsize=10)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(time, df['diff_pitch'], color='#ff9800', linewidth=1)
    axes[1, 1].axhline(y=0, color='green', linestyle='--', alpha=0.5)
    axes[1, 1].axhline(y=5, color='orange', linestyle=':', alpha=0.5)
    axes[1, 1].axhline(y=-5, color='orange', linestyle=':', alpha=0.5)
    axes[1, 1].fill_between(time, -5, 5, alpha=0.1, color='green')
    axes[1, 1].set_ylabel('Δ Pitch (°)')
    axes[1, 1].set_title(f'Pitch Difference\nMean: {df["diff_pitch"].mean():.2f}°, Std: {df["diff_pitch"].std():.2f}°', fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Yaw
    axes[2, 0].plot(time, df['phone_yaw'], label='Phone', color='#2196f3', alpha=0.8, linewidth=1.3)
    axes[2, 0].plot(time, df['esp32_yaw'], label='ESP32', color='#f44336', alpha=0.8, linewidth=1.3)
    axes[2, 0].set_ylabel('Yaw (°)')
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 0].set_title('Yaw Comparison', fontsize=10)
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    axes[2, 1].plot(time, df['diff_yaw'], color='#ff9800', linewidth=1)
    axes[2, 1].axhline(y=0, color='green', linestyle='--', alpha=0.5)
    axes[2, 1].axhline(y=5, color='orange', linestyle=':', alpha=0.5)
    axes[2, 1].axhline(y=-5, color='orange', linestyle=':', alpha=0.5)
    axes[2, 1].fill_between(time, -5, 5, alpha=0.1, color='green')
    axes[2, 1].set_ylabel('Δ Yaw (°)')
    axes[2, 1].set_xlabel('Time (s)')
    axes[2, 1].set_title(f'Yaw Difference\nMean: {df["diff_yaw"].mean():.2f}°, Std: {df["diff_yaw"].std():.2f}°', fontsize=10)
    axes[2, 1].grid(True, alpha=0.3)
    
    fig.tight_layout(pad=2.0, w_pad=1.8, h_pad=1.8)
    
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
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 11))
    fig.suptitle('Accelerometer Comparison', fontsize=12, fontweight='semibold')
    
    time = df['time_sec'] if 'time_sec' in df.columns else df.index
    
    labels = ['X', 'Y', 'Z']
    colors_phone = '#2196f3'
    colors_esp32 = '#f44336'
    
    for i, axis in enumerate(['X', 'Y', 'Z']):
        phone_col = f'phone_accel{axis}'
        esp32_col = f'esp32_accel{axis}'
        diff_col = f'diff_accel{axis}'
        
        # Comparison plot
        axes[i, 0].plot(time, df[phone_col], label='Phone', color=colors_phone, alpha=0.8, linewidth=1.3)
        axes[i, 0].plot(time, df[esp32_col], label='ESP32', color=colors_esp32, alpha=0.8, linewidth=1.3)
        axes[i, 0].set_ylabel(f'Accel {axis} (m/s²)')
        axes[i, 0].set_title(f'Accelerometer {axis} Comparison', fontsize=10)
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
            axes[i, 1].set_title(f'Accel {axis} Difference\nMean: {df[diff_col].mean():.2f} m/s², Std: {df[diff_col].std():.2f} m/s²', fontsize=10)
            axes[i, 1].grid(True, alpha=0.3)
    
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 1].set_xlabel('Time (s)')
    
    fig.tight_layout(pad=2.0, w_pad=1.8, h_pad=1.8)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()

def plot_gyroscope_comparison(df: pd.DataFrame, save_path: str = None):
    """Plot gyroscope comparison with inversion correction applied"""
    # Check if gyroscope columns exist
    gyro_cols = ['phone_gyroX', 'phone_gyroY', 'phone_gyroZ', 'esp32_gyroX', 'esp32_gyroY', 'esp32_gyroZ']
    if not all(col in df.columns for col in gyro_cols):
        print("Gyroscope data not found in CSV")
        return
    
    # Store current backend
    current_backend = plt.get_backend()
    
    # Check if any axes were inverted
    title_suffix = ""
    if hasattr(df, 'attrs') and 'axis_mapping' in df.attrs:
        title_suffix = " (Custom Mapping)"
    elif hasattr(df, 'attrs') and 'axis_inversions' in df.attrs:
        title_suffix = " (Mapped Axes)"
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 11))
    fig.suptitle(f'Gyroscope Comparison{title_suffix}', fontsize=12, fontweight='semibold')
    
    time = df['time_sec'] if 'time_sec' in df.columns else df.index
    
    for i, axis in enumerate(['X', 'Y', 'Z']):
        phone_col = f'phone_gyro{axis}'
        esp32_col = f'esp32_gyro{axis}'
        diff_col = f'diff_gyro{axis}'
        
        # Comparison plot - dual y-axis since different units
        ax1 = axes[i, 0]
        ax2 = ax1.twinx()
        
        # Phone (rad/s) - left y-axis
        line1 = ax1.plot(time, df[phone_col], label='Phone', color='#2196f3', alpha=0.8, linewidth=1.3)
        ax1.set_ylabel(f'Phone Gyro {axis} (rad/s)', color='#2196f3')
        ax1.tick_params(axis='y', labelcolor='#2196f3')
        
        # ESP32 (deg/s) - right y-axis
        line2 = ax2.plot(time, df[esp32_col], label='ESP32', color='#f44336', alpha=0.8, linewidth=1.3)
        ax2.set_ylabel(f'ESP32 Gyro {axis} (deg/s)', color='#f44336')
        ax2.tick_params(axis='y', labelcolor='#f44336')
        
        # Add mapping indicator to title
        title_suffix = ""
        if hasattr(df, 'attrs') and 'axis_mapping' in df.attrs:
            mapping = df.attrs['axis_mapping'][axis]
            title_suffix = f" / ESP32 {mapping['esp32_axis']}"
        elif hasattr(df, 'attrs') and 'axis_inversions' in df.attrs:
            title_suffix = ""
        
        ax1.set_title(f'Gyroscope {axis} Comparison{title_suffix}', fontsize=10)
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
            axes[i, 1].set_title(f'Gyro {axis} Difference\nMean: {df[diff_col].mean():.4f} rad/s, Std: {df[diff_col].std():.4f} rad/s', fontsize=10)
            axes[i, 1].legend(fontsize=8)
            axes[i, 1].grid(True, alpha=0.3)
    
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 1].set_xlabel('Time (s)')
    
    fig.tight_layout(pad=2.0, w_pad=2.2, h_pad=1.8)
    
    # Save the plot FIRST with current backend
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    # Now switch to interactive backend for display
    plt.switch_backend('TkAgg')
    
    # Force show the plot - will display interactively
    plt.show(block=True)
    
    # Restore original backend
    plt.switch_backend(current_backend)

def plot_difference_histograms(df: pd.DataFrame, save_path: str = None):
    """Plot histograms of all differences including quaternion"""
    fig, axes = plt.subplots(4, 3, figsize=(15, 12.5))
    fig.suptitle('Distribution of Differences', fontsize=12, fontweight='semibold')
    
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
    
    # Quaternion differences
    if 'diff_quat_angle_deg' in df.columns:
        diff_cols.append(('diff_quat_angle_deg', 'Quat Angle (°)', 10.0, axes[3, 0]))
    
    # Calculate quaternion component differences if not in CSV
    quat_comp_diffs = []
    for i, comp in enumerate(['qw', 'qx', 'qy']):
        if i + 1 < axes.shape[1]:  # Ensure we don't go out of bounds
            phone_col = f'phone_{comp}'
            esp32_col = f'esp32_{comp}'
            if phone_col in df.columns and esp32_col in df.columns:
                diff_name = f'calc_diff_{comp}'
                df[diff_name] = df[esp32_col] - df[phone_col]
                threshold = 0.1 if comp == 'qw' else 0.05
                diff_cols.append((diff_name, f'Quat {comp.upper()}', threshold, axes[3, i+1]))
    
    # Hide unused subplot if qz is not included
    if len([col for col in diff_cols if 'quat' in col[1].lower()]) < 4:
        axes[3, 2].set_visible(False)
    
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
            ax.set_title(f'{label}\nMean: {mean:.2f}, Std: {std:.2f}\n{within_threshold:.1f}% within ±{threshold}', fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
    
    fig.tight_layout(pad=2.0, w_pad=1.8, h_pad=2.0)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()

def generate_comprehensive_report(df: pd.DataFrame, save_path: str = None):
    """Generate comprehensive sensor comparison report including quaternions"""
    summary_lines = []
    
    summary_lines.append("=" * 80)
    summary_lines.append("COMPREHENSIVE IMU SENSOR COMPARISON REPORT")
    summary_lines.append("ESP32 vs Smartphone Sensor Analysis with Quaternion Data")
    summary_lines.append("=" * 80)
    summary_lines.append("")
    summary_lines.append(f"📊 Dataset Information:")
    summary_lines.append(f"   Total Samples: {len(df):,}")
    if 'time_sec' in df.columns:
        duration = df['time_sec'].max()
        sampling_rate = len(df) / duration if duration > 0 else 0
        summary_lines.append(f"   Duration: {duration:.1f} seconds")
        summary_lines.append(f"   Average Sampling Rate: {sampling_rate:.1f} Hz")
    summary_lines.append(f"   Timestamp: {pd.to_datetime(df['timestamp_ms'].iloc[0], unit='ms').strftime('%Y-%m-%d %H:%M:%S')} to {pd.to_datetime(df['timestamp_ms'].iloc[-1], unit='ms').strftime('%Y-%m-%d %H:%M:%S')}")
    summary_lines.append("")

    # Orientation differences
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
    
    # Gyroscope analysis with axis remapping
    if 'diff_gyroX' in df.columns:
        summary_lines.append("🔄 GYROSCOPE ANALYSIS (with Custom Axis Mapping):")
        summary_lines.append("-" * 40)
        
        # Report axis mapping applied
        if hasattr(df, 'attrs') and 'axis_mapping' in df.attrs:
            summary_lines.append(f"  📍 CUSTOM AXIS REMAPPING APPLIED:")
            summary_lines.append(f"     Phone X ↔ ESP32 Y (inverted)")
            summary_lines.append(f"     Phone Y ↔ ESP32 Z")
            summary_lines.append(f"     Phone Z ↔ ESP32 X")
            summary_lines.append(f"     Analysis uses remapped axes for proper comparison.")
            summary_lines.append("")
        
        axis_mapping = AXIS_MAPPING_CONFIG
        
        for axis in ['X', 'Y', 'Z']:
            phone_col = f'phone_gyro{axis}'
            esp32_col = f'esp32_gyro{axis}'
            diff_col = f'diff_gyro{axis}'
            
            if all(col in df.columns for col in [phone_col, esp32_col, diff_col]):
                phone_data = df[phone_col].dropna()
                esp32_data = df[esp32_col].dropna()
                diff_data = df[diff_col].dropna()
                
                # Calculate correlation after any corrections
                if len(phone_data) > 0 and len(esp32_data) > 0:
                    esp32_rad = np.radians(esp32_data)
                    correlation = np.corrcoef(phone_data, esp32_rad)[0, 1] if len(phone_data) == len(esp32_rad) else np.nan
                else:
                    correlation = np.nan
                
                phone_range = phone_data.max() - phone_data.min()
                esp32_range = esp32_data.max() - esp32_data.min()
                diff_mean = diff_data.mean()
                diff_std = diff_data.std()
                diff_abs_mean = diff_data.abs().mean()
                diff_max = diff_data.abs().max()
                within_0087rad = (diff_data.abs() < 0.087).sum() / len(diff_data) * 100  # 0.087 rad/s ≈ 5°/s
                
                # Show axis mapping from configuration
                mapping_info = f" ↔ ESP32 {AXIS_MAPPING_CONFIG[axis]['esp32_axis']}"
                if AXIS_MAPPING_CONFIG[axis]['invert']:
                    mapping_info += " [INVERTED]"
                
                summary_lines.append(f"  Gyro {axis}{mapping_info}:")
                summary_lines.append(f"    Phone:  min={phone_data.min():7.4f} rad/s, max={phone_data.max():7.4f} rad/s, range={phone_range:.4f} rad/s")
                summary_lines.append(f"    ESP32:  min={esp32_data.min():7.4f} deg/s, max={esp32_data.max():7.4f} deg/s, range={esp32_range:.4f} deg/s")
                summary_lines.append(f"    Diff:   mean={diff_mean:7.4f} rad/s, std={diff_std:6.4f} rad/s, abs_mean={diff_abs_mean:.4f} rad/s, max={diff_max:.4f} rad/s")
                summary_lines.append(f"    Correlation: {correlation:.3f}")
                summary_lines.append(f"    Within ±0.087 rad/s (~5°/s): {within_0087rad:.1f}%")
                summary_lines.append("")
    
    # Quaternion Analysis
    if 'diff_quat_angle_deg' in df.columns:
        summary_lines.append("🔄 QUATERNION ANALYSIS:")
        summary_lines.append("-" * 40)
        if hasattr(df, 'attrs') and 'quaternion_mapping' in df.attrs:
            summary_lines.append("  Applied quaternion mapping:")
            summary_lines.append("    Phone: original component order")
            summary_lines.append("    ESP32: qw ← qy, qy ← qw, qx inverted")
            summary_lines.append("")
        quat_angle_data = df['diff_quat_angle_deg'].dropna()
        quat_mean = quat_angle_data.mean()
        quat_std = quat_angle_data.std()
        quat_abs_mean = quat_angle_data.abs().mean()
        quat_max = quat_angle_data.abs().max()
        within_5deg = (quat_angle_data.abs() < 5).sum() / len(quat_angle_data) * 100
        within_10deg = (quat_angle_data.abs() < 10).sum() / len(quat_angle_data) * 100
        
        summary_lines.append(f"  Quaternion Angle Difference:")
        summary_lines.append(f"    Mean: {quat_mean:7.2f}° | Std: {quat_std:6.2f}° | Abs Mean: {quat_abs_mean:6.2f}° | Max: {quat_max:6.2f}°")
        summary_lines.append(f"    Within ±5°:  {within_5deg:5.1f}%")
        summary_lines.append(f"    Within ±10°: {within_10deg:5.1f}%")
        
        # Quaternion component analysis
        summary_lines.append(f"\n  Quaternion Components:")
        for comp in ['qw', 'qx', 'qy', 'qz']:
            phone_col = f'phone_{comp}'
            esp32_col = f'esp32_{comp}'
            if phone_col in df.columns and esp32_col in df.columns:
                phone_data = df[phone_col].dropna()
                esp32_data = df[esp32_col].dropna()
                diff_data = esp32_data - phone_data
                
                phone_range = phone_data.max() - phone_data.min()
                esp32_range = esp32_data.max() - esp32_data.min()
                
                summary_lines.append(f"    {comp.upper()}:")
                summary_lines.append(f"      Phone:  min={phone_data.min():7.4f}, max={phone_data.max():7.4f}, range={phone_range:.4f}")
                summary_lines.append(f"      ESP32:  min={esp32_data.min():7.4f}, max={esp32_data.max():7.4f}, range={esp32_range:.4f}")
                summary_lines.append(f"      Diff:   mean={diff_data.mean():7.4f}, std={diff_data.std():6.4f}, abs_mean={diff_data.abs().mean():.4f}")
        
        # Quaternion normalization analysis
        phone_norm = np.sqrt(df['phone_qw']**2 + df['phone_qx']**2 + df['phone_qy']**2 + df['phone_qz']**2)
        phone_norm_error = (phone_norm - 1.0).abs().mean()
        
        summary_lines.append(f"\n  Quaternion Normalization:")
        summary_lines.append(f"    Phone norm error (from 1.0):  {phone_norm_error:.6f}")
        
        if 'esp32_quatNorm' in df.columns:
            esp32_norm_data = df['esp32_quatNorm'].dropna()
            esp32_norm_error = (esp32_norm_data - 1.0).abs().mean()
            summary_lines.append(f"    ESP32 norm error (from 1.0):  {esp32_norm_error:.6f}")
        
        summary_lines.append("")
    
    # Data Quality Assessment
    summary_lines.append("📈 DATA QUALITY ASSESSMENT:")
    summary_lines.append("-" * 40)
    
    # Missing data analysis
    total_samples = len(df)
    missing_data = {}
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            missing_data[col] = (missing_count, missing_count/total_samples*100)
    
    if missing_data:
        summary_lines.append("  Missing Data:")
        for col, (count, pct) in missing_data.items():
            summary_lines.append(f"    {col}: {count} samples ({pct:.1f}%)")
    else:
        summary_lines.append("  ✓ No missing data detected")
    
    # Outlier detection for key metrics
    summary_lines.append(f"\n  Outlier Analysis (>3 std dev):")
    outlier_cols = ['diff_roll', 'diff_pitch', 'diff_yaw']
    if 'diff_quat_angle_deg' in df.columns:
        outlier_cols.append('diff_quat_angle_deg')
    
    for col in outlier_cols:
        if col in df.columns:
            data = df[col].dropna()
            mean = data.mean()
            std = data.std()
            outliers = data[data.abs() > (mean + 3*std)]
            outlier_pct = len(outliers) / len(data) * 100 if len(data) > 0 else 0
            summary_lines.append(f"    {col}: {len(outliers)} outliers ({outlier_pct:.1f}%)")
    
    summary_lines.append("")
    
    # Overall assessment
    summary_lines.append("=" * 80)
    summary_lines.append("OVERALL SENSOR PERFORMANCE ASSESSMENT:")
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
    
    if 'diff_quat_angle_deg' in df.columns:
        avg_quat_diff = df['diff_quat_angle_deg'].abs().mean()
        summary_lines.append(f"Average Quaternion Angle Difference: {avg_quat_diff:.2f}°")
    
    # Performance rating
    summary_lines.append("\n📊 PERFORMANCE RATING:")
    
    # Calculate overall performance score
    score_components = []
    
    if 'diff_roll' in df.columns and 'diff_pitch' in df.columns and 'diff_yaw' in df.columns:
        orientation_score = 0
        for axis in ['roll', 'pitch', 'yaw']:
            col = f'diff_{axis}'
            within_5deg = (df[col].abs() < 5).sum() / len(df[col].dropna()) * 100
            orientation_score += within_5deg
        orientation_score /= 3
        score_components.append(('Orientation Accuracy', orientation_score, 'Excellent' if orientation_score > 90 else 'Good' if orientation_score > 80 else 'Fair' if orientation_score > 70 else 'Poor'))
    
    if 'diff_quat_angle_deg' in df.columns:
        quat_within_10deg = (df['diff_quat_angle_deg'].abs() < 10).sum() / len(df['diff_quat_angle_deg'].dropna()) * 100
        quat_rating = 'Excellent' if quat_within_10deg > 95 else 'Good' if quat_within_10deg > 85 else 'Fair' if quat_within_10deg > 75 else 'Poor'
        score_components.append(('Quaternion Accuracy', quat_within_10deg, quat_rating))
    
    for component, score, rating in score_components:
        summary_lines.append(f"  {component}: {score:.1f}% ({rating})")
    
    if score_components:
        overall_score = np.mean([score for _, score, _ in score_components])
        overall_rating = 'Excellent' if overall_score > 90 else 'Good' if overall_score > 80 else 'Fair' if overall_score > 70 else 'Poor'
        summary_lines.append(f"\n  🎯 OVERALL PERFORMANCE: {overall_score:.1f}% ({overall_rating})")
    
    summary_lines.append("\n" + "=" * 80)
    
    # Print to console
    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)
    
    # Save to file if path provided
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(summary_text)
        print(f"\nComprehensive report saved to: {save_path}")
    
    return summary_text

def print_summary_statistics(df: pd.DataFrame, save_path: str = None):
    """Legacy function - redirects to comprehensive report"""
    return generate_comprehensive_report(df, save_path)

def main():
    # Check for test mode
    test_mode = False
    csv_path = None
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test" or sys.argv[1] == "-t":
            test_mode = True
            if len(sys.argv) > 2:
                csv_path = sys.argv[2]
            else:
                print("Usage: python visualize_imu_comparison.py --test <path_to_csv>")
                print("   or: python visualize_imu_comparison.py <path_to_csv>")
                return
        else:
            csv_path = sys.argv[1]
    
    # Find CSV files in the workspace
    workspace = SCRIPT_DIR
    
    # Look for IMU comparison CSV files in current directory
    csv_files = list(workspace.glob("imu_comparison*.csv"))
    
    if csv_path:
        if Path(csv_path).exists():
            csv_files = [Path(csv_path)]
        else:
            print(f"File not found: {csv_path}")
            return
    elif not csv_files:
        print("No IMU comparison CSV files found in current directory.")
        print("\nUsage: python visualize_imu_comparison.py <path_to_csv>")
        print("   or: python visualize_imu_comparison.py --test <path_to_csv>")
        return
    
    # Use the most recent file if multiple found
    csv_file = sorted(csv_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]
    print(f"Loading: {csv_file}")
    
    # Load data
    df = load_comparison_data(str(csv_file))
    print(f"Loaded {len(df)} samples")
    
    # Print available columns
    print(f"\nColumns: {', '.join(df.columns)}")
    
    # If test mode, run axis combination testing and exit
    if test_mode:
        test_all_axis_combinations(df)
        return
    
    # Use the organized output directory
    output_dir = VIZ_DIR
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Generate comprehensive report
    summary_file = output_dir / "comprehensive_sensor_report.txt"
    report_text = generate_comprehensive_report(df, str(summary_file))
    
    # Generate plots
    print("\nGenerating comprehensive visualizations...")
    
    # Quaternion analysis plots
    if 'phone_qw' in df.columns and 'esp32_qw' in df.columns:
        plot_quaternion_comparison(df, str(output_dir / "quaternion_comparison.png"))
        plot_quaternion_analysis(df, str(output_dir / "quaternion_analysis.png"))
    
    # Orientation comparison
    plot_orientation_comparison(df, str(output_dir / "orientation_comparison.png"))
    
    # Accelerometer comparison (if data exists)
    if 'phone_accelX' in df.columns:
        plot_accelerometer_comparison(df, str(output_dir / "accelerometer_comparison.png"))
    
    # Gyroscope comparison (if data exists)
    if 'phone_gyroX' in df.columns:
        plot_gyroscope_comparison(df, str(output_dir / "gyroscope_comparison.png"))
    
    # Difference histograms with quaternion data
    plot_difference_histograms(df, str(output_dir / "difference_histograms.png"))
    
    print(f"\nComprehensive analysis complete!")
    print(f"Report and visualizations saved to: {output_dir}")
    print(f"\nGenerated files:")
    print(f"  📄 Report: {summary_file}")
    print(f"  📊 Plots: {len(list(output_dir.glob('*.png')))} visualization files")

if __name__ == "__main__":
    main()