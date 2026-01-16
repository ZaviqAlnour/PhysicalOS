"""
Motion Interpolation Script - "Ghost Predictor"
Cubic Spline Interpolation for Hardware-Ready Motion Control

Purpose: Upsample motion data from 30 FPS to 400 FPS using smooth interpolation.
         Fills intermediate frames for precise robot motion control.

Usage:
    python interpolate_motion.py input.csv output.csv
    python interpolate_motion.py input.csv output.csv --target-fps 400
"""

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from pathlib import Path
import argparse
import sys


def interpolate_motion_data(input_csv, output_csv, source_fps=30.0, target_fps=400.0):
    """
    Interpolate motion data using cubic splines.
    
    Args:
        input_csv: Path to input CSV file
        output_csv: Path to output CSV file
        source_fps: Original frame rate (default: 30 FPS)
        target_fps: Target frame rate (default: 400 FPS)
        
    Process:
        1. Read original timestamps and all numeric columns
        2. For each column, fit a cubic spline
        3. Evaluate spline at new (higher frequency) timestamps
        4. Write interpolated data to new CSV
    """
    print("=" * 70)
    print("Motion Interpolation - Cubic Spline Upsampling")
    print("=" * 70)
    print(f"[INPUT]  {input_csv}")
    print(f"[OUTPUT] {output_csv}")
    print(f"[FPS]    {source_fps} → {target_fps}")
    print("=" * 70)
    
    # Read input CSV
    df = pd.read_csv(input_csv)
    
    if 'timestamp' not in df.columns:
        raise ValueError("CSV must contain 'timestamp' column")
    
    # Original data
    original_timestamps = df['timestamp'].values
    n_original = len(original_timestamps)
    duration = original_timestamps[-1] - original_timestamps[0]
    
    # Create new timestamps at target FPS
    dt_new = 1.0 / target_fps
    new_timestamps = np.arange(
        original_timestamps[0],
        original_timestamps[-1] + dt_new,
        dt_new
    )
    n_new = len(new_timestamps)
    
    print(f"\n[INTERPOLATION]")
    print(f"  Original frames: {n_original}")
    print(f"  Target frames:   {n_new}")
    print(f"  Upsampling ratio: {n_new / n_original:.1f}x")
    print(f"  Duration: {duration:.2f} seconds")
    
    # Create new dataframe
    new_df = pd.DataFrame()
    new_df['timestamp'] = new_timestamps
    
    # Identify numeric columns to interpolate
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'timestamp' in numeric_cols:
        numeric_cols.remove('timestamp')
    
    # Interpolate each column using cubic splines
    print(f"\n[PROCESSING] Interpolating {len(numeric_cols)} columns...")
    
    for i, col in enumerate(numeric_cols):
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(numeric_cols)} columns")
        
        # Get original values
        original_values = df[col].values
        
        # Handle NaN values (interpolate over them or use linear fill)
        if np.any(np.isnan(original_values)):
            # Fill NaN with linear interpolation first
            mask = np.isnan(original_values)
            original_values[mask] = np.interp(
                np.flatnonzero(mask),
                np.flatnonzero(~mask),
                original_values[~mask]
            )
        
        # Create cubic spline
        # bc_type='natural' ensures smooth second derivatives at boundaries
        cs = CubicSpline(original_timestamps, original_values, bc_type='natural')
        
        # Evaluate at new timestamps
        interpolated_values = cs(new_timestamps)
        
        # Add to new dataframe
        new_df[col] = interpolated_values
    
    # Handle non-numeric columns (copy from nearest frame)
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    if non_numeric_cols:
        print(f"\n[CATEGORICAL] Copying {len(non_numeric_cols)} non-numeric columns...")
        for col in non_numeric_cols:
            # For each new timestamp, find nearest original timestamp
            indices = np.searchsorted(original_timestamps, new_timestamps)
            indices = np.clip(indices, 0, len(original_timestamps) - 1)
            new_df[col] = df[col].iloc[indices].values
    
    # Save output
    new_df.to_csv(output_csv, index=False)
    
    print(f"\n[COMPLETE]")
    print(f"  Saved {len(new_df)} frames to: {output_csv}")
    print(f"  File size: {Path(output_csv).stat().st_size / 1024:.1f} KB")
    print("=" * 70)
    
    return new_df


def validate_interpolation_smoothness(df, column_name='RIGHT_WRIST_Y'):
    """
    Validate that interpolation is smooth (no discontinuities).
    
    Args:
        df: Interpolated dataframe
        column_name: Column to check for smoothness
        
    Returns:
        bool: True if smooth, False if discontinuities detected
    """
    if column_name not in df.columns:
        print(f"[WARNING] Column {column_name} not found for validation")
        return True
    
    values = df[column_name].values
    
    # Calculate first derivative (velocity)
    velocity = np.diff(values)
    
    # Calculate second derivative (acceleration)
    acceleration = np.diff(velocity)
    
    # Check for large jumps (potential discontinuities)
    # A smooth function should have bounded derivatives
    vel_max = np.max(np.abs(velocity))
    acc_max = np.max(np.abs(acceleration))
    
    print(f"\n[VALIDATION] Smoothness check for {column_name}:")
    print(f"  Max velocity:     {vel_max:.6f}")
    print(f"  Max acceleration: {acc_max:.6f}")
    
    # Heuristic: if acceleration is extremely high, there may be discontinuities
    if acc_max > 100 * np.std(acceleration):
        print(f"  [WARNING] Large acceleration spike detected!")
        return False
    else:
        print(f"  [PASS] Interpolation appears smooth")
        return True


def main():
    """Command-line interface for motion interpolation."""
    parser = argparse.ArgumentParser(
        description="Interpolate motion data using cubic splines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: 30 FPS → 400 FPS
  python interpolate_motion.py input.csv output.csv
  
  # Custom target FPS
  python interpolate_motion.py input.csv output.csv --target-fps 240
  
  # Specify source FPS
  python interpolate_motion.py input.csv output.csv --source-fps 60 --target-fps 400
        """
    )
    
    parser.add_argument('input_csv', type=str,
                       help='Path to input CSV file')
    parser.add_argument('output_csv', type=str,
                       help='Path to output CSV file')
    parser.add_argument('--source-fps', type=float, default=30.0,
                       help='Source frame rate (default: 30 FPS)')
    parser.add_argument('--target-fps', type=float, default=400.0,
                       help='Target frame rate (default: 400 FPS)')
    parser.add_argument('--validate', action='store_true',
                       help='Run smoothness validation after interpolation')
    
    args = parser.parse_args()
    
    # Check input file exists
    if not Path(args.input_csv).exists():
        print(f"[ERROR] Input file not found: {args.input_csv}")
        sys.exit(1)
    
    # Run interpolation
    try:
        df = interpolate_motion_data(
            args.input_csv,
            args.output_csv,
            source_fps=args.source_fps,
            target_fps=args.target_fps
        )
        
        # Validation
        if args.validate:
            validate_interpolation_smoothness(df)
        
    except Exception as e:
        print(f"\n[ERROR] Interpolation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
