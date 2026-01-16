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
from physics_intelligence import PhysicalIntelligence

# ADVANCED ANALYSIS MODULES
from visualizer_live import AdvancedVisualizer
from dynamics_engine import DynamicsEngine
from grf_estimator import GRFEstimator


def process_video(video_path, visual_mode=False, smoothing_mode='one_euro', 
                  export_3d_coords=True, export_com=True, body_mass=70.0):
    """
    Process a single video file through the ENHANCED pipeline.
    
    Args:
        video_path: Path to input MP4 file
        visual_mode: If True, display skeleton overlay during processing
        smoothing_mode: 'one_euro', 'moving_average', or 'none'
        export_3d_coords: If True, export 3D relative coordinates
        export_com: If True, calculate and export Center of Mass
        body_mass: Total body mass in kg for CoM calculation
    
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
    print(f"[CoM CALC] {'Enabled' if export_com else 'Disabled'}")
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
    
    # Initialize physical intelligence (CoM calculator)
    physics = None
    if export_com:
        physics = PhysicalIntelligence(body_mass=body_mass)
        print(f"[PHYSICS] Center of Mass calculator initialized (mass={body_mass}kg)")
    
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
        
        # Add CoM columns if enabled
        if export_com:
            position_names.extend(['CoM_X', 'CoM_Y', 'CoM_Z'])
    
    # Initialize data logger (dual mode if 3D coords enabled)
    if export_3d_coords and position_names:
        logger = DualLogger(str(output_csv), joint_names, position_names)
    else:
        logger = DataLogger(str(output_csv), joint_names)
    
    frame_count = 0
    processed_count = 0
    
    print("[PIPELINE] Starting frame-by-frame processing...")
    
    # Initialize Advanced Visualizer
    viz = None
    if visual_mode:
        viz = AdvancedVisualizer(history_size=100)
    
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
            
            # Calculate Center of Mass if enabled
            if export_com and physics is not None:
                com = physics.com_calculator.calculate_com_relative_to_hip(key_positions)
                if com is not None:
                    key_positions['CoM'] = com
            
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
                
                # Show CoM if enabled
                if export_com and 'CoM_Y' in positions_dict:
                    cv2.putText(frame_with_skeleton,
                               f"CoM Height: {positions_dict['CoM_Y']:.3f}",
                               (10, y_offset + 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Frame info
            cv2.putText(frame_with_skeleton, 
                       f"Frame: {frame_count}/{total_frames} | Smoothing: {smoothing_mode}", 
                       (10, frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Update Advanced Visualizer (Graphs)
            if viz:
                # Extract CoM_Y and Right_Elbow angle for graphs
                com_y = 0
                if export_com and 'CoM' in key_positions:
                    com_y = key_positions['CoM'][1] # Y is index 1
                
                elbow_angle = smoothed_angles.get('R_Elbow', 0)
                
                # Update visualizer with 3D landmarks and graph data
                viz.update(landmarks, com_y, elbow_angle, timestamp)
            
            cv2.imshow('PhysicalOS_V1 - Enhanced Motion Capture', frame_with_skeleton)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[USER] Stopped by user (pressed 'q')")
                break
    
    # Cleanup
    cap.release()
    logger.close()
    if viz:
        viz.close()
    cv2.destroyAllWindows()
    
    # PHASE 8: Advanced Post-Processing
    print("\n[ANALYSIS] Running Advanced Dynamics & GRF Engine...")
    try:
        # Enriched data generation
        enhanced_path = logger.enhanced_path if hasattr(logger, 'enhanced_path') else output_csv
        
        # 1. Dynamics (Velocity/Acceleration)
        dyn_engine = DynamicsEngine(fps=fps)
        dyn_engine.process_csv_data(str(enhanced_path))
        
        # 2. GRF (Support Leg Detection)
        grf_engine = GRFEstimator()
        grf_engine.process_csv_data(str(enhanced_path))
        
        print("[ANALYSIS] Dynamics and GRF data successfully merged.")
    except Exception as e:
        print(f"[WARNING] Advanced analysis failed: {e}")
    
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


def show_menu():
    """
    Display interactive menu and get user choice for video source.
    
    Returns:
        str: User choice ('1', '2', or '3')
    """
    print("\n" + "=" * 60)
    print("SELECT VIDEO SOURCE")
    print("=" * 60)
    print("1. Live Video (Webcam)")
    print("2. Recorded Video (Single File)")
    print("3. All Input Files (Batch Processing)")
    print("4. Training Data Library (Batch Aggregate)")
    print("=" * 60)
    
    while True:
        choice = input("\nEnter your choice (1/2/3/4): ").strip()
        if choice in ['1', '2', '3', '4']:
            return choice
        print("[ERROR] Invalid choice. Please enter 1, 2, 3, or 4.")


def process_live_camera(smoothing_mode='one_euro', export_3d_coords=True, 
                        export_com=True, body_mass=70.0):
    """
    Process live webcam feed with real-time data saving.
    Uses same pipeline logic as process_video() but with camera input.
    
    Args:
        smoothing_mode: 'one_euro', 'moving_average', or 'none'
        export_3d_coords: If True, export 3D relative coordinates
        export_com: If True, calculate and export Center of Mass
        body_mass: Total body mass in kg for CoM calculation
    
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"[PROCESSING] LIVE CAMERA")
    print(f"[MODE] Visual Display with Real-Time Data Saving")
    print(f"[SMOOTHING] {smoothing_mode}")
    print(f"[3D COORDS] {'Enabled' if export_3d_coords else 'Disabled'}")
    print(f"[CoM CALC] {'Enabled' if export_com else 'Disabled'}")
    print(f"[INFO] Press 'q' to stop recording")
    print(f"{'='*60}")
    
    # Initialize vision engine
    detector = PoseDetector()
    
    # Open webcam (device index 0)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open webcam")
        print(f"[TIP] Make sure your camera is connected and not in use by another app")
        return False
    
    # Get camera properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:  # Some cameras return 0
        fps = 30.0  # Default assumption
    print(f"[CAMERA INFO] FPS: {fps:.2f}")
    
    # Setup output with timestamp
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("output_data")
    output_dir.mkdir(exist_ok=True)
    
    output_csv = output_dir / f"live_camera_{timestamp_str}_motion.csv"
    
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
    
    # Initialize physical intelligence (CoM calculator)
    physics = None
    if export_com:
        physics = PhysicalIntelligence(body_mass=body_mass)
        print(f"[PHYSICS] Center of Mass calculator initialized (mass={body_mass}kg)")
    
    # Define joint angles to export
    joint_names = ['R_Shoulder_Flex', 'R_Elbow', 'L_Shoulder_Flex', 'L_Elbow']
    
    # Define 3D positions to export
    position_names = []
    if export_3d_coords:
        end_effector_joints = ['LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_ANKLE', 'RIGHT_ANKLE']
        key_joints = [
            'LEFT_SHOULDER', 'RIGHT_SHOULDER',
            'LEFT_ELBOW', 'RIGHT_ELBOW',
            'LEFT_HIP', 'RIGHT_HIP',
            'LEFT_KNEE', 'RIGHT_KNEE',
            'MID_HIP'
        ]
        all_export_joints = list(set(end_effector_joints + key_joints))
        for joint in all_export_joints:
            position_names.extend([f"{joint}_X", f"{joint}_Y", f"{joint}_Z"])
        if export_com:
            position_names.extend(['CoM_X', 'CoM_Y', 'CoM_Z'])
    
    # Initialize data logger
    if export_3d_coords and position_names:
        logger = DualLogger(str(output_csv), joint_names, position_names)
    else:
        logger = DataLogger(str(output_csv), joint_names)
    
    frame_count = 0
    processed_count = 0
    start_time = datetime.now()
    
    print("[PIPELINE] Camera active - Recording started...")
    
    # Initialize Advanced Visualizer
    viz = AdvancedVisualizer(history_size=100)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Failed to read frame from camera")
            break
        
        frame_count += 1
        # Calculate elapsed time for timestamp
        elapsed = (datetime.now() - start_time).total_seconds()
        timestamp = elapsed
        
        # PHASE 1: Vision - Extract 33 landmarks
        landmarks = detector.detect_pose(frame)
        
        if landmarks is None:
            # No pose detected
            cv2.putText(frame, "NO POSE DETECTED", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow('PhysicalOS_V1 - Live Camera', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        
        # PHASE 2: Normalize coordinates
        normalized = normalize_to_torso(landmarks, left_hip_idx=23, right_hip_idx=24)
        
        # PHASE 3: Kinematics - Calculate joint angles (RAW)
        raw_angles = {}
        
        # Right arm angles
        r_shoulder = normalized[12]
        r_elbow = normalized[14]
        r_wrist = normalized[16]
        r_hip = normalized[24]
        
        raw_angles['R_Elbow'] = calculate_elbow_angle(r_shoulder, r_elbow, r_wrist)
        raw_angles['R_Shoulder_Flex'] = calculate_shoulder_flexion(r_hip, r_shoulder, r_elbow)
        
        # Left arm angles
        l_shoulder = normalized[11]
        l_elbow = normalized[13]
        l_wrist = normalized[15]
        l_hip = normalized[23]
        
        raw_angles['L_Elbow'] = calculate_elbow_angle(l_shoulder, l_elbow, l_wrist)
        raw_angles['L_Shoulder_Flex'] = calculate_shoulder_flexion(l_hip, l_shoulder, l_elbow)
        
        # PHASE 4: Smoothing
        if smoother is not None:
            if smoothing_mode == 'one_euro':
                smoothed_angles = smoother.filter_dict(raw_angles, timestamp)
            else:
                smoothed_angles = smoother.filter_dict(raw_angles)
        else:
            smoothed_angles = raw_angles
        
        # PHASE 5: 3D Relative Coordinates
        positions_dict = {}
        if export_3d_coords:
            key_positions = get_key_joint_positions(landmarks)
            if export_com and physics is not None:
                com = physics.com_calculator.calculate_com_relative_to_hip(key_positions)
                if com is not None:
                    key_positions['CoM'] = com
            positions_dict = format_positions_for_csv(key_positions)
        
        # PHASE 6: Data Logging
        if export_3d_coords and positions_dict:
            logger.log_frame(timestamp, smoothed_angles, positions_dict)
        else:
            logger.log_frame(timestamp, smoothed_angles)
        
        processed_count += 1
        
        # PHASE 7: Visual feedback
        frame_with_skeleton = detector.draw_skeleton(frame, landmarks)
        
        # Display angles and info
        y_offset = 30
        for joint_name in joint_names:
            smooth_angle = smoothed_angles.get(joint_name, 0)
            text = f"{joint_name}: {smooth_angle:.1f} deg"
            color = (0, 255, 0) if 0 <= smooth_angle <= 180 else (0, 0, 255)
            cv2.putText(frame_with_skeleton, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += 20
        
        # Recording indicator
        cv2.circle(frame_with_skeleton, (frame.shape[1] - 30, 30), 10, (0, 0, 255), -1)
        cv2.putText(frame_with_skeleton, "REC", (frame.shape[1] - 70, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        cv2.putText(frame_with_skeleton, 
                   f"Frames: {processed_count} | Time: {elapsed:.1f}s | Press 'q' to stop", 
                   (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Update Advanced Visualizer
        if viz:
            com_y = 0
            if export_com and 'CoM' in key_positions:
                com_y = key_positions['CoM'][1]
            elbow_angle = smoothed_angles.get('R_Elbow', 0)
            viz.update(landmarks, com_y, elbow_angle, timestamp)
            
        cv2.imshow('PhysicalOS_V1 - Live Camera', frame_with_skeleton)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n[USER] Recording stopped by user (pressed 'q')")
            break
    
    # Cleanup
    cap.release()
    logger.close()
    if viz:
        viz.close()
    cv2.destroyAllWindows()
    
    # PHASE 8: Advanced Post-Processing
    print("\n[ANALYSIS] Running Advanced Dynamics & GRF Engine...")
    try:
        # Enriched data generation
        enhanced_path = logger.enhanced_path if hasattr(logger, 'enhanced_path') else output_csv
        
        # 1. Dynamics (Velocity/Acceleration)
        dyn_engine = DynamicsEngine(fps=fps)
        dyn_df = dyn_engine.process_csv_data(str(enhanced_path))
        
        # 2. GRF (Support Leg Detection)
        grf_engine = GRFEstimator()
        grf_engine.process_csv_data(str(enhanced_path))
        
        print("[ANALYSIS] Dynamics and GRF data successfully merged.")
    except Exception as e:
        print(f"[WARNING] Advanced analysis failed: {e}")
    
    print(f"\n{'='*60}")
    print(f"[COMPLETE] Processed {processed_count} frames in {elapsed:.1f}s")
    if export_3d_coords:
        print(f"[OUTPUT] Original CSV: {output_csv}")
        enhanced_csv = output_csv.parent / (output_csv.stem + "_enhanced" + output_csv.suffix)
        print(f"[OUTPUT] Enhanced CSV: {enhanced_csv}")
    else:
        print(f"[OUTPUT] {output_csv}")
    print(f"{'='*60}\n")
    
    return True


def main():
    """Interactive main function with menu-driven video source selection."""
    print("=" * 70)
    print("PhysicalOS_V1 - ENHANCED Motion Capture Pipeline")
    print("Photons -> Geometry -> Physics -> Memory -> Intent")
    print("=" * 70)
    print("\nFEATURES:")
    print("  + Smoothed joint angles (reduces jitter)")
    print("  + 3D relative coordinates (camera-independent)")
    print("  + Intent-based motion data (inverse kinematics ready)")
    print("  + Center of Mass calculation (biomechanics)")
    print("  + Dual CSV export (original + enhanced)")
    print("=" * 70)
    
    # Create output directories
    Path("output_data").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    
    # Show interactive menu
    choice = show_menu()
    
    # Default settings (can be modified if needed)
    smoothing_mode = 'one_euro'
    export_3d_coords = True
    export_com = True
    body_mass = 70.0
    
    if choice == '1':
        # Live Video (Webcam)
        success = process_live_camera(
            smoothing_mode=smoothing_mode,
            export_3d_coords=export_3d_coords,
            export_com=export_com,
            body_mass=body_mass
        )
        sys.exit(0 if success else 1)
    
    elif choice == '2':
        # Recorded Video (Single File)
        print("\n" + "=" * 60)
        video_name = input("Enter video filename (in input_videos/): ").strip()
        print("=" * 60)
        
        # Construct full path
        video_path = Path("input_videos") / video_name
        
        # Check if file exists
        if not video_path.exists():
            # Try adding .mp4 extension if not present
            if not video_name.endswith('.mp4'):
                video_path = Path("input_videos") / f"{video_name}.mp4"
        
        success = process_video(
            str(video_path),
            visual_mode=True,  # Enable visual mode for single video
            smoothing_mode=smoothing_mode,
            export_3d_coords=export_3d_coords,
            export_com=export_com,
            body_mass=body_mass
        )
        sys.exit(0 if success else 1)
    
    elif choice == '3':
        # All Input Files (Batch Processing)
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
        print(f"[SMOOTHING] {smoothing_mode}")
        print(f"[3D COORDS] {'Enabled' if export_3d_coords else 'Disabled'}")
        print(f"[CoM CALC] {'Enabled' if export_com else 'Disabled'}")
        print("[NOTE] Headless mode (no display for performance)")
        
        success_count = 0
        for video_path in video_files:
            if process_video(
                str(video_path),
                visual_mode=False,  # Headless for batch
                smoothing_mode=smoothing_mode,
                export_3d_coords=export_3d_coords,
                export_com=export_com,
                body_mass=body_mass
            ):
                success_count += 1
        
        print("\n" + "=" * 70)
        print(f"[BATCH COMPLETE] {success_count}/{len(video_files)} videos processed")
        print("=" * 70)

    elif choice == '4':
        # Training Data Library (Batch Aggregate)
        print("\n" + "=" * 60)
        print("TRAINING DATA BATCH PROCESSING")
        print("=" * 60)
        
        training_dir = Path("Training_Data")
        raw_videos_dir = training_dir / "raw_videos"
        processed_csv_dir = training_dir / "processed_csv"
        
        if not raw_videos_dir.exists():
            print(f"[ERROR] Raw videos directory not found: {raw_videos_dir}")
            sys.exit(1)
            
        video_files = list(raw_videos_dir.glob("*.mp4")) + list(raw_videos_dir.glob("*.MP4"))
        
        if not video_files:
            print(f"[WARNING] No videos found in {raw_videos_dir}")
            sys.exit(0)
            
        print(f"[INFO] Processing {len(video_files)} videos from training library...")
        
        success_count = 0
        for video_path in video_files:
            # Process video and save data to processed_csv_dir
            # We override output directory for this mode
            success = process_video(
                str(video_path),
                visual_mode=False,
                smoothing_mode=smoothing_mode,
                export_3d_coords=export_3d_coords,
                export_com=export_com,
                body_mass=body_mass
            )
            if success:
                success_count += 1
                
        print(f"\n[INFO] {success_count} videos processed. Aggregating dataset...")
        
        # Run aggregation script
        try:
            from scripts.aggregate_dataset import aggregate_training_data
            aggregate_training_data()
        except ImportError:
            print("[ERROR] Could not run aggregation script. Run 'python scripts/aggregate_dataset.py' manually.")
            
        print("\n" + "=" * 70)
        print(f"[TRAINING DATA COMPLETE] Dataset ready in {training_dir}")
        print("=" * 70)


if __name__ == "__main__":
    main()