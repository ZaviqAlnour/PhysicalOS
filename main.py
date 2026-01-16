"""
PhysicalOS_V1 - Main Orchestrator (ENHANCED)
Converts human motion videos to robot-readable joint angle data.

NEW FEATURES:
- Smoothed joint angles (removes jitter)
- 3D relative coordinates (camera-independent)
- Intent-based motion data (inverse kinematics ready)
- Dual CSV export (original + enhanced formats)

Fully offline motion capture pipeline for Windows 11 / Python 3.11.9
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

from vision_engine import PoseDetector
from kinematics import calculate_elbow_angle, calculate_shoulder_flexion
from data_logger import DataLogger
from utils import normalize_to_torso

# NEW MODULES
from smoothing import OneEuroFilter, MovingAverageFilter
from relative_coordinates import (
    get_end_effector_positions, 
    get_key_joint_positions,
    format_positions_for_csv
)
from enhanced_data_logger import DualLogger


def process_video(video_path, visual_mode=False, smoothing_mode='one_euro', 
                  export_3d_coords=True):
    """
    Process a single video file through the ENHANCED pipeline.
    
    Args:
        video_path: Path to input MP4 file
        visual_mode: If True, display skeleton overlay during processing
        smoothing_mode: 'one_euro', 'moving_average', or 'none'
        export_3d_coords: If True, export 3D relative coordinates
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Validate input
    if not os.path.exists(video_path):
        print(f"[ERROR] Video not found: {video_path}")
        return False
    
    video_path = str(Path(video_path).resolve())
    video_name = Path(video_path).stem
    
    print(f"\n{'='*60}")
    print(f"[PROCESSING] {video_name}")
    print(f"[MODE] {'Visual Display' if visual_mode else 'Headless Batch'}")
    print(f"[SMOOTHING] {smoothing_mode}")
    print(f"[3D COORDS] {'Enabled' if export_3d_coords else 'Disabled'}")
    print(f"{'='*60}")
    
    # Initialize vision engine
    detector = PoseDetector()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return False
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[VIDEO INFO] FPS: {fps:.2f}, Total Frames: {total_frames}")
    
    # Setup output
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("output_data")
    output_dir.mkdir(exist_ok=True)
    
    output_csv = output_dir / f"{video_name}_{timestamp_str}_motion.csv"
    
    # Initialize smoothing filter
    if smoothing_mode == 'one_euro':
        smoother = OneEuroFilter(min_cutoff=1.0, beta=0.007, d_cutoff=1.0)
        print("[SMOOTHING] One Euro Filter initialized (adaptive)")
    elif smoothing_mode == 'moving_average':
        smoother = MovingAverageFilter(window_size=5)
        print("[SMOOTHING] Moving Average Filter initialized (window=5)")
    else:
        smoother = None
        print("[SMOOTHING] Disabled (raw angles)")
    
    # Define joint angles to export
    joint_names = ['R_Shoulder_Flex', 'R_Elbow', 'L_Shoulder_Flex', 'L_Elbow']
    
    # Define 3D positions to export (end effectors + key joints)
    position_names = []
    if export_3d_coords:
        # End effectors (hands and feet)
        end_effector_joints = ['LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_ANKLE', 'RIGHT_ANKLE']
        
        # Key body joints for full motion
        key_joints = [
            'LEFT_SHOULDER', 'RIGHT_SHOULDER',
            'LEFT_ELBOW', 'RIGHT_ELBOW',
            'LEFT_HIP', 'RIGHT_HIP',
            'LEFT_KNEE', 'RIGHT_KNEE',
            'MID_HIP'
        ]
        
        # Combine unique joints
        all_export_joints = list(set(end_effector_joints + key_joints))
        
        # Create column names (X, Y, Z for each joint)
        for joint in all_export_joints:
            position_names.extend([f"{joint}_X", f"{joint}_Y", f"{joint}_Z"])
    
    # Initialize data logger (dual mode if 3D coords enabled)
    if export_3d_coords and position_names:
        logger = DualLogger(str(output_csv), joint_names, position_names)
    else:
        logger = DataLogger(str(output_csv), joint_names)
    
    frame_count = 0
    processed_count = 0
    
    print("[PIPELINE] Starting frame-by-frame processing...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        timestamp = frame_count / fps
        
        # Progress indicator (every 30 frames)
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"[PROGRESS] {frame_count}/{total_frames} frames ({progress:.1f}%)")
        
        # PHASE 1: Vision - Extract 33 landmarks
        landmarks = detector.detect_pose(frame)
        
        if landmarks is None:
            # No pose detected in this frame
            if visual_mode:
                cv2.putText(frame, "NO POSE DETECTED", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow('PhysicalOS_V1 - Enhanced Motion Capture', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            continue
        
        # PHASE 2: Normalize coordinates (torso-centered)
        normalized = normalize_to_torso(landmarks, left_hip_idx=23, right_hip_idx=24)
        
        # PHASE 3: Kinematics - Calculate joint angles (RAW)
        raw_angles = {}
        
        # Right arm angles
        r_shoulder = normalized[12]  # RIGHT_SHOULDER
        r_elbow = normalized[14]     # RIGHT_ELBOW
        r_wrist = normalized[16]     # RIGHT_WRIST
        r_hip = normalized[24]       # RIGHT_HIP
        
        raw_angles['R_Elbow'] = calculate_elbow_angle(r_shoulder, r_elbow, r_wrist)
        raw_angles['R_Shoulder_Flex'] = calculate_shoulder_flexion(r_hip, r_shoulder, r_elbow)
        
        # Left arm angles
        l_shoulder = normalized[11]  # LEFT_SHOULDER
        l_elbow = normalized[13]     # LEFT_ELBOW
        l_wrist = normalized[15]     # LEFT_WRIST
        l_hip = normalized[23]       # LEFT_HIP
        
        raw_angles['L_Elbow'] = calculate_elbow_angle(l_shoulder, l_elbow, l_wrist)
        raw_angles['L_Shoulder_Flex'] = calculate_shoulder_flexion(l_hip, l_shoulder, l_elbow)
        
        # PHASE 4: Smoothing - Apply filter to reduce jitter
        if smoother is not None:
            if smoothing_mode == 'one_euro':
                smoothed_angles = smoother.filter_dict(raw_angles, timestamp)
            else:  # moving_average
                smoothed_angles = smoother.filter_dict(raw_angles)
        else:
            smoothed_angles = raw_angles
        
        # PHASE 5: 3D Relative Coordinates - Intent-based motion data
        positions_dict = {}
        if export_3d_coords:
            # Get all key joint positions relative to Mid-Hip
            key_positions = get_key_joint_positions(landmarks)
            
            # Format for CSV export (flatten to X, Y, Z columns)
            positions_dict = format_positions_for_csv(key_positions)
        
        # PHASE 6: Data Logging
        if export_3d_coords and positions_dict:
            logger.log_frame(timestamp, smoothed_angles, positions_dict)
        else:
            logger.log_frame(timestamp, smoothed_angles)
        
        processed_count += 1
        
        # PHASE 7: Visual feedback (if enabled)
        if visual_mode:
            # Draw skeleton
            frame_with_skeleton = detector.draw_skeleton(frame, landmarks)
            
            # Overlay smoothed joint angles
            y_offset = 30
            for joint_name in joint_names:
                raw_angle = raw_angles.get(joint_name, 0)
                smooth_angle = smoothed_angles.get(joint_name, 0)
                
                # Show both raw and smoothed for comparison
                text = f"{joint_name}: {smooth_angle:.1f}° (raw: {raw_angle:.1f}°)"
                color = (0, 255, 0) if 0 <= smooth_angle <= 180 else (0, 0, 255)
                cv2.putText(frame_with_skeleton, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y_offset += 20
            
            # Show 3D coordinate info
            if export_3d_coords:
                cv2.putText(frame_with_skeleton, 
                           f"3D Coords: {len(positions_dict)//3} joints tracked", 
                           (10, y_offset + 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Frame info
            cv2.putText(frame_with_skeleton, 
                       f"Frame: {frame_count}/{total_frames} | Smoothing: {smoothing_mode}", 
                       (10, frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('PhysicalOS_V1 - Enhanced Motion Capture', frame_with_skeleton)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[USER] Stopped by user (pressed 'q')")
                break
    
    # Cleanup
    cap.release()
    logger.close()
    if visual_mode:
        cv2.destroyAllWindows()
    
    print(f"\n{'='*60}")
    print(f"[COMPLETE] Processed {processed_count}/{frame_count} frames")
    if export_3d_coords:
        print(f"[OUTPUT] Original CSV: {output_csv}")
        enhanced_csv = output_csv.parent / (output_csv.stem + "_enhanced" + output_csv.suffix)
        print(f"[OUTPUT] Enhanced CSV: {enhanced_csv}")
        print(f"         (includes 3D coordinates for {len(positions_dict)//3} joints)")
    else:
        print(f"[OUTPUT] {output_csv}")
    print(f"{'='*60}\n")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="PhysicalOS_V1 - Enhanced Offline Motion Capture Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visual mode with One Euro smoothing and 3D coords
  python main.py --video input_videos/motion.mp4 --visual
  
  # Headless with moving average smoothing
  python main.py --video input_videos/motion.mp4 --smoothing moving_average
  
  # Batch process all videos (headless, default settings)
  python main.py
  
  # Disable smoothing (raw angles only)
  python main.py --video input_videos/motion.mp4 --smoothing none
  
  # Disable 3D coordinates export
  python main.py --video input_videos/motion.mp4 --no-3d-coords

Smoothing Options:
  - one_euro (default): Adaptive filter, best for reducing jitter while maintaining responsiveness
  - moving_average: Simple averaging, good for general smoothing
  - none: No smoothing, export raw angles
        """
    )
    
    parser.add_argument('--video', type=str, 
                       help='Path to specific video file to process')
    parser.add_argument('--visual', action='store_true',
                       help='Enable visual mode (display skeleton overlay)')
    parser.add_argument('--smoothing', type=str, default='one_euro',
                       choices=['one_euro', 'moving_average', 'none'],
                       help='Smoothing filter to apply (default: one_euro)')
    parser.add_argument('--no-3d-coords', action='store_true',
                       help='Disable 3D coordinate export (angles only)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("PhysicalOS_V1 - ENHANCED Motion Capture Pipeline")
    print("Photons → Geometry → Physics → Memory → Intent")
    print("=" * 70)
    print("\nNEW FEATURES:")
    print("  ✓ Smoothed joint angles (reduces jitter)")
    print("  ✓ 3D relative coordinates (camera-independent)")
    print("  ✓ Intent-based motion data (inverse kinematics ready)")
    print("  ✓ Dual CSV export (original + enhanced)")
    print("=" * 70)
    
    # Create output directories
    Path("output_data").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    
    export_3d = not args.no_3d_coords
    
    if args.video:
        # Single video mode
        success = process_video(
            args.video, 
            visual_mode=args.visual,
            smoothing_mode=args.smoothing,
            export_3d_coords=export_3d
        )
        sys.exit(0 if success else 1)
    else:
        # Batch processing mode (all videos in input_videos/)
        input_dir = Path("input_videos")
        
        if not input_dir.exists():
            print(f"[ERROR] Input directory not found: {input_dir}")
            print("Create it with: mkdir input_videos")
            sys.exit(1)
        
        video_files = list(input_dir.glob("*.mp4")) + list(input_dir.glob("*.MP4"))
        
        if not video_files:
            print(f"[WARNING] No MP4 files found in {input_dir}")
            print("Add videos and run again.")
            sys.exit(0)
        
        print(f"\n[BATCH MODE] Found {len(video_files)} video(s)")
        print(f"[SMOOTHING] {args.smoothing}")
        print(f"[3D COORDS] {'Enabled' if export_3d else 'Disabled'}")
        print("[NOTE] Headless mode (no display for performance)")
        
        success_count = 0
        for video_path in video_files:
            if process_video(
                str(video_path), 
                visual_mode=False,
                smoothing_mode=args.smoothing,
                export_3d_coords=export_3d
            ):
                success_count += 1
        
        print("\n" + "=" * 70)
        print(f"[BATCH COMPLETE] {success_count}/{len(video_files)} videos processed")
        print("=" * 70)


if __name__ == "__main__":
    main()