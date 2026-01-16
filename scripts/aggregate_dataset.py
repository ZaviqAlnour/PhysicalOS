"""
Dataset Aggregation Script
Combines multiple CSV files into a single training dataset.

Purpose: Build "Library of Human Intent" for machine learning applications.
         Aggregates all processed motion data with metadata.

Usage:
    python aggregate_dataset.py
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import sys


def aggregate_training_data(training_data_dir="Training_Data", output_name="aggregated_data.csv"):
    """
    Combine all CSV files in Training_Data/processed_csv/ into single dataset.
    
    Args:
        training_data_dir: Path to Training_Data directory
        output_name: Name of output aggregated CSV
        
    Returns:
        pandas.DataFrame: Combined dataset
    """
    training_path = Path(training_data_dir)
    processed_dir = training_path / "processed_csv"
    
    print("=" * 70)
    print("Dataset Aggregation - Library of Human Intent")
    print("=" * 70)
    
    if not processed_dir.exists():
        print(f"[ERROR] Directory not found: {processed_dir}")
        print(f"[TIP] Process videos first using Option 4 in main menu")
        sys.exit(1)
    
    # Find all CSV files
    csv_files = list(processed_dir.glob("*.csv"))
    
    if not csv_files:
        print(f"[WARNING] No CSV files found in {processed_dir}")
        sys.exit(0)
    
    print(f"[FOUND] {len(csv_files)} CSV files")
    
    # Load manifest if exists
    manifest_path = training_path / "manifest.json"
    manifest = {}
    
    if manifest_path.exists():
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        print(f"[MANIFEST] Loaded metadata for {len(manifest)} videos")
    
    # Aggregate data
    all_data = []
    
    for i, csv_file in enumerate(csv_files):
        print(f"\n[{i+1}/{len(csv_files)}] Processing: {csv_file.name}")
        
        # Read CSV
        df = pd.read_csv(csv_file)
        
        # Add video_id column
        video_id = csv_file.stem
        df['video_id'] = video_id
        
        # Add task_label from manifest if available
        if video_id in manifest:
            df['task_label'] = manifest[video_id].get('task_label', 'unknown')
            df['subject_id'] = manifest[video_id].get('subject_id', 'unknown')
        else:
            df['task_label'] = 'unknown'
            df['subject_id'] = 'unknown'
        
        # Add frame_id (local frame number within video)
        df['frame_id'] = range(len(df))
        
        all_data.append(df)
        
        print(f"  Frames: {len(df)}")
        print(f"  Task: {df['task_label'].iloc[0]}")
    
    # Combine all dataframes
    print(f"\n[AGGREGATING] Combining {len(all_data)} datasets...")
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Save aggregated dataset
    output_path = training_path / output_name
    combined_df.to_csv(output_path, index=False)
    
    # Statistics
    print(f"\n[COMPLETE]")
    print(f"  Total frames: {len(combined_df):,}")
    print(f"  Total videos: {combined_df['video_id'].nunique()}")
    print(f"  Unique tasks: {combined_df['task_label'].nunique()}")
    print(f"  Columns: {len(combined_df.columns)}")
    print(f"  Output: {output_path}")
    print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Task distribution
    print(f"\n[TASK DISTRIBUTION]")
    task_counts = combined_df.groupby('task_label').size()
    for task, count in task_counts.items():
        print(f"  {task}: {count} frames ({count/len(combined_df)*100:.1f}%)")
    
    print("=" * 70)
    
    return combined_df


def create_manifest_template(training_data_dir="Training_Data"):
    """
    Create a template manifest.json file for user to fill in.
    
    Manifest format:
    {
        "video_name": {
            "task_label": "walking" | "sitting" | "reaching" | etc.,
            "subject_id": "person_01",
            "notes": "Optional description"
        }
    }
    """
    training_path = Path(training_data_dir)
    manifest_path = training_path / "manifest.json"
    
    if manifest_path.exists():
        print(f"[INFO] Manifest already exists: {manifest_path}")
        return
    
    # Create template
    template = {
        "_README": "Fill in task_label and subject_id for each video",
        "_EXAMPLE_video_name": {
            "task_label": "walking",
            "subject_id": "person_01",
            "notes": "Optional description or metadata"
        }
    }
    
    with open(manifest_path, 'w') as f:
        json.dump(template, f, indent=2)
    
    print(f"[CREATED] Manifest template: {manifest_path}")
    print(f"[TODO] Edit manifest.json to add task labels for your videos")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Aggregate training dataset from processed CSVs"
    )
    parser.add_argument('--training-dir', type=str, default="Training_Data",
                       help='Path to Training_Data directory')
    parser.add_argument('--output', type=str, default="aggregated_data.csv",
                       help='Name of output aggregated CSV')
    parser.add_argument('--create-manifest', action='store_true',
                       help='Create manifest template without aggregating')
    
    args = parser.parse_args()
    
    if args.create_manifest:
        create_manifest_template(args.training_dir)
    else:
        aggregate_training_data(args.training_dir, args.output)
