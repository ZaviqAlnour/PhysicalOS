"""
Stress Testing Module - Pipeline Robustness Verification
Tests the motion capture pipeline under adverse conditions:
- Low-light videos
- Loose clothing / occlusions
- Fast movements
- Large batch processing (100+ videos)
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from datetime import datetime


class VideoStressTester:
    """
    Test video processing under various challenging conditions.
    """
    
    def __init__(self):
        """Initialize stress tester."""
        self.test_results = []
    
    def analyze_video_quality(self, video_path):
        """
        Analyze video characteristics that might affect processing.
        
        Args:
            video_path: Path to video file
            
        Returns:
            dict: Quality metrics
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return None
        
        metrics = {
            'brightness': [],
            'contrast': [],
            'motion': [],
            'blur': []
        }
        
        prev_frame = None
        frame_count = 0
        
        # Sample every 10th frame for efficiency
        while frame_count < 100:  # Limit to first 100 frames
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames
            if frame_count % 10 != 0:
                frame_count += 1
                continue
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Brightness (mean intensity)
            brightness = np.mean(gray)
            metrics['brightness'].append(brightness)
            
            # Contrast (standard deviation)
            contrast = np.std(gray)
            metrics['contrast'].append(contrast)
            
            # Blur (Laplacian variance)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            blur_score = laplacian.var()
            metrics['blur'].append(blur_score)
            
            # Motion (frame difference)
            if prev_frame is not None:
                diff = cv2.absdiff(gray, prev_frame)
                motion = np.mean(diff)
                metrics['motion'].append(motion)
            
            prev_frame = gray
            frame_count += 1
        
        cap.release()
        
        # Aggregate metrics
        summary = {
            'avg_brightness': np.mean(metrics['brightness']),
            'avg_contrast': np.mean(metrics['contrast']),
            'avg_blur': np.mean(metrics['blur']),
            'avg_motion': np.mean(metrics['motion']) if metrics['motion'] else 0,
            'frames_analyzed': frame_count
        }
        
        # Quality assessment
        summary['is_low_light'] = summary['avg_brightness'] < 80
        summary['is_low_contrast'] = summary['avg_contrast'] < 30
        summary['is_blurry'] = summary['avg_blur'] < 100
        summary['is_high_motion'] = summary['avg_motion'] > 20
        
        return summary
    
    def test_single_video(self, video_path, detector):
        """
        Test pose detection success rate on a single video.
        
        Args:
            video_path: Path to video
            detector: PoseDetector instance
            
        Returns:
            dict: Test results
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return None
        
        total_frames = 0
        detected_frames = 0
        detection_confidences = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            total_frames += 1
            
            # Detect pose
            landmarks = detector.detect_pose(frame)
            
            if landmarks is not None:
                detected_frames += 1
                # Average visibility as confidence
                avg_confidence = np.mean(landmarks[:, 3])
                detection_confidences.append(avg_confidence)
        
        cap.release()
        
        success_rate = (detected_frames / total_frames * 100) if total_frames > 0 else 0
        avg_confidence = np.mean(detection_confidences) if detection_confidences else 0
        
        return {
            'total_frames': total_frames,
            'detected_frames': detected_frames,
            'success_rate': success_rate,
            'avg_confidence': avg_confidence
        }


class BatchProcessingTester:
    """
    Test pipeline stability with large batches of videos.
    """
    
    def __init__(self):
        """Initialize batch tester."""
        self.results = []
    
    def test_batch(self, video_paths, process_func, max_videos=None):
        """
        Test processing a batch of videos.
        
        Args:
            video_paths: List of video file paths
            process_func: Function to process each video
            max_videos: Maximum number to test (None = all)
            
        Returns:
            dict: Batch test results
        """
        if max_videos:
            video_paths = video_paths[:max_videos]
        
        print(f"\n[BATCH TEST] Testing {len(video_paths)} videos...")
        
        results = {
            'total_videos': len(video_paths),
            'successful': 0,
            'failed': 0,
            'processing_times': [],
            'errors': []
        }
        
        start_time = time.time()
        
        for idx, video_path in enumerate(video_paths):
            print(f"[{idx+1}/{len(video_paths)}] Processing: {Path(video_path).name}")
            
            video_start = time.time()
            
            try:
                success = process_func(video_path)
                
                if success:
                    results['successful'] += 1
                else:
                    results['failed'] += 1
                    results['errors'].append({
                        'video': str(video_path),
                        'error': 'Processing returned False'
                    })
                
                processing_time = time.time() - video_start
                results['processing_times'].append(processing_time)
                
            except Exception as e:
                results['failed'] += 1
                results['errors'].append({
                    'video': str(video_path),
                    'error': str(e)
                })
                print(f"[ERROR] {e}")
        
        total_time = time.time() - start_time
        
        results['total_time'] = total_time
        results['avg_time_per_video'] = np.mean(results['processing_times']) if results['processing_times'] else 0
        results['success_rate'] = (results['successful'] / results['total_videos'] * 100) if results['total_videos'] > 0 else 0
        
        return results
    
    def print_summary(self, results):
        """Print batch test summary."""
        print("\n" + "="*60)
        print("BATCH PROCESSING TEST SUMMARY")
        print("="*60)
        print(f"Total Videos:      {results['total_videos']}")
        print(f"Successful:        {results['successful']}")
        print(f"Failed:            {results['failed']}")
        print(f"Success Rate:      {results['success_rate']:.1f}%")
        print(f"Total Time:        {results['total_time']:.2f}s")
        print(f"Avg Time/Video:    {results['avg_time_per_video']:.2f}s")
        
        if results['errors']:
            print(f"\nErrors ({len(results['errors'])}):")
            for err in results['errors'][:5]:  # Show first 5
                print(f"  - {Path(err['video']).name}: {err['error']}")
            if len(results['errors']) > 5:
                print(f"  ... and {len(results['errors']) - 5} more")
        
        print("="*60)


class FilterRobustnessTester:
    """
    Test smoothing filter performance under noisy conditions.
    """
    
    @staticmethod
    def add_noise(signal, noise_level=5.0):
        """
        Add Gaussian noise to signal.
        
        Args:
            signal: Clean signal array
            noise_level: Standard deviation of noise
            
        Returns:
            numpy array: Noisy signal
        """
        noise = np.random.normal(0, noise_level, len(signal))
        return signal + noise
    
    @staticmethod
    def add_outliers(signal, outlier_ratio=0.05, outlier_magnitude=50.0):
        """
        Add random outliers to signal.
        
        Args:
            signal: Clean signal array
            outlier_ratio: Fraction of points that are outliers
            outlier_magnitude: Size of outliers
            
        Returns:
            numpy array: Signal with outliers
        """
        noisy_signal = signal.copy()
        n_outliers = int(len(signal) * outlier_ratio)
        outlier_indices = np.random.choice(len(signal), n_outliers, replace=False)
        
        for idx in outlier_indices:
            noisy_signal[idx] += np.random.choice([-1, 1]) * outlier_magnitude
        
        return noisy_signal
    
    @staticmethod
    def test_filter_performance(smoother, clean_signal, noisy_signal, timestamps):
        """
        Test filter performance on noisy data.
        
        Args:
            smoother: Smoothing filter instance
            clean_signal: Ground truth signal
            noisy_signal: Signal with noise
            timestamps: Time values
            
        Returns:
            dict: Performance metrics
        """
        from smoothing import OneEuroFilter
        
        # Reset filter state
        if hasattr(smoother, 'reset'):
            smoother.reset()
        
        # Apply filter
        smoothed_signal = []
        for i, (t, value) in enumerate(zip(timestamps, noisy_signal)):
            if isinstance(smoother, OneEuroFilter):
                smoothed = smoother.filter('test_joint', value, t)
            else:
                smoothed = smoother.filter('test_joint', value)
            smoothed_signal.append(smoothed)
        
        smoothed_signal = np.array(smoothed_signal)
        
        # Calculate metrics
        # RMSE compared to clean signal
        rmse_noisy = np.sqrt(np.mean((noisy_signal - clean_signal) ** 2))
        rmse_smoothed = np.sqrt(np.mean((smoothed_signal - clean_signal) ** 2))
        
        # Improvement
        improvement = ((rmse_noisy - rmse_smoothed) / rmse_noisy * 100)
        
        return {
            'rmse_noisy': rmse_noisy,
            'rmse_smoothed': rmse_smoothed,
            'improvement_percent': improvement,
            'smoothed_signal': smoothed_signal
        }


def run_comprehensive_stress_test(input_dir='input_videos'):
    """
    Run all stress tests on available videos.
    
    Args:
        input_dir: Directory containing test videos
    """
    print("="*70)
    print("COMPREHENSIVE STRESS TEST - PhysicalOS_V1")
    print("="*70)
    
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"[ERROR] Input directory not found: {input_dir}")
        return
    
    video_files = list(input_path.glob("*.mp4")) + list(input_path.glob("*.MP4"))
    
    if not video_files:
        print(f"[WARNING] No videos found in {input_dir}")
        return
    
    print(f"\n[INFO] Found {len(video_files)} video(s)")
    
    # Test 1: Video Quality Analysis
    print("\n" + "="*70)
    print("TEST 1: Video Quality Analysis")
    print("="*70)
    
    quality_tester = VideoStressTester()
    
    for video_path in video_files[:5]:  # Test first 5 videos
        print(f"\nAnalyzing: {video_path.name}")
        quality = quality_tester.analyze_video_quality(str(video_path))
        
        if quality:
            print(f"  Brightness:  {quality['avg_brightness']:.1f} {'[LOW LIGHT]' if quality['is_low_light'] else ''}")
            print(f"  Contrast:    {quality['avg_contrast']:.1f} {'[LOW CONTRAST]' if quality['is_low_contrast'] else ''}")
            print(f"  Blur Score:  {quality['avg_blur']:.1f} {'[BLURRY]' if quality['is_blurry'] else ''}")
            print(f"  Motion:      {quality['avg_motion']:.1f} {'[HIGH MOTION]' if quality['is_high_motion'] else ''}")
    
    # Test 2: Pose Detection Robustness
    print("\n" + "="*70)
    print("TEST 2: Pose Detection Robustness")
    print("="*70)
    
    from vision_engine import PoseDetector
    detector = PoseDetector(model_complexity=1)
    
    for video_path in video_files[:3]:  # Test first 3 videos
        print(f"\nTesting: {video_path.name}")
        results = quality_tester.test_single_video(str(video_path), detector)
        
        if results:
            print(f"  Total Frames:    {results['total_frames']}")
            print(f"  Detected Frames: {results['detected_frames']}")
            print(f"  Success Rate:    {results['success_rate']:.1f}%")
            print(f"  Avg Confidence:  {results['avg_confidence']:.3f}")
    
    detector.close()
    
    # Test 3: Filter Robustness
    print("\n" + "="*70)
    print("TEST 3: Filter Robustness Under Noise")
    print("="*70)
    
    from smoothing import OneEuroFilter, MovingAverageFilter
    
    # Create synthetic clean signal
    t = np.linspace(0, 3, 100)
    clean = 90 + 30 * np.sin(2 * np.pi * t)
    
    # Add severe noise
    noisy = FilterRobustnessTester.add_noise(clean, noise_level=10.0)
    noisy_with_outliers = FilterRobustnessTester.add_outliers(noisy, outlier_ratio=0.1, outlier_magnitude=50)
    
    # Test One Euro Filter
    euro_filter = OneEuroFilter(min_cutoff=1.0, beta=0.007)
    euro_results = FilterRobustnessTester.test_filter_performance(
        euro_filter, clean, noisy_with_outliers, t
    )
    
    print("\nOne Euro Filter:")
    print(f"  RMSE (noisy):    {euro_results['rmse_noisy']:.2f}")
    print(f"  RMSE (filtered): {euro_results['rmse_smoothed']:.2f}")
    print(f"  Improvement:     {euro_results['improvement_percent']:.1f}%")
    
    # Test Moving Average
    ma_filter = MovingAverageFilter(window_size=5)
    ma_results = FilterRobustnessTester.test_filter_performance(
        ma_filter, clean, noisy_with_outliers, t
    )
    
    print("\nMoving Average Filter:")
    print(f"  RMSE (noisy):    {ma_results['rmse_noisy']:.2f}")
    print(f"  RMSE (filtered): {ma_results['rmse_smoothed']:.2f}")
    print(f"  Improvement:     {ma_results['improvement_percent']:.1f}%")
    
    print("\n" + "="*70)
    print("STRESS TEST COMPLETE")
    print("="*70)


if __name__ == "__main__":
    run_comprehensive_stress_test()