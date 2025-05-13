#!/usr/bin/env python3
import cv2
import os
import argparse
import numpy as np
from glob import glob
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_first_last_frames(video_path):
    """Extract the first and last frames of a video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Unable to open video: {video_path}")
        return None, None
    
    # Get total frame count
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 1:
        logger.error(f"Insufficient frames in video: {video_path}")
        return None, None
    
    # Read the first frame
    ret, first_frame = cap.read()
    if not ret:
        logger.error(f"Unable to read first frame: {video_path}")
        return None, None
    
    # Seek to the last frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
    ret, last_frame = cap.read()
    if not ret:
        logger.error(f"Unable to read last frame: {video_path}")
        return None, None
    
    cap.release()
    return first_frame, last_frame

def calculate_frame_similarity(frame1, frame2):
    """Calculate the similarity between two frames."""
    if frame1 is None or frame2 is None:
        return 0
    
    # Resize frames to ensure consistent dimensions
    frame1_resized = cv2.resize(frame1, (256, 256))
    frame2_resized = cv2.resize(frame2, (256, 256))
    
    # Convert to grayscale to reduce computation
    gray1 = cv2.cvtColor(frame1_resized, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2_resized, cv2.COLOR_BGR2GRAY)
    
    # Compute histograms
    hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
    
    # Normalize histograms
    cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
    
    # Calculate similarity (correlation coefficient)
    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    
    # Also calculate structural similarity (SSIM)
    ssim = calculate_ssim(gray1, gray2)
    
    # Combine scores
    return (similarity + ssim) / 2

def calculate_ssim(img1, img2):
    """Calculate Structural Similarity Index (SSIM)."""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return np.mean(ssim_map)

def find_optimal_video_sequence(video_files, first_video_path=None):
    """Find the optimal sequence for stitching videos, with an optional starting video."""
    if not video_files:
        return []
    
    logger.info("Analyzing video frames to find the best stitching sequence...")
    
    # Ensure first_video_path is in video_files
    if first_video_path is not None:
        if first_video_path not in video_files:
            if os.path.exists(first_video_path):
                video_files.append(first_video_path)
                logger.info(f"Added specified starting video: {first_video_path}")
            else:
                logger.warning(f"Specified starting video does not exist: {first_video_path}")
                first_video_path = None
    
    # Extract first and last frames for each video
    video_frames = {}
    for video in video_files:
        first_frame, last_frame = extract_first_last_frames(video)
        if first_frame is not None and last_frame is not None:
            video_frames[video] = (first_frame, last_frame)
        else:
            logger.warning(f"Skipping invalid video: {video}")
    
    if not video_frames:
        logger.error("No valid videos available for stitching")
        return []
    
    # Create similarity matrix
    videos = list(video_frames.keys())
    n = len(videos)
    similarity_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                # Calculate similarity between last frame of video i and first frame of video j
                similarity = calculate_frame_similarity(
                    video_frames[videos[i]][1],  # Last frame of video i
                    video_frames[videos[j]][0]   # First frame of video j
                )
                similarity_matrix[i, j] = similarity
    
    # Find optimal path (greedy algorithm)
    used = [False] * n
    sequence = []
    
    # Start with specified video if provided
    if first_video_path is not None and first_video_path in videos:
        current = videos.index(first_video_path)
    else:
        # Otherwise start from any video
        current = 0
        
    used[current] = True
    sequence.append(videos[current])
    
    # Greedily select the next most similar video
    while len(sequence) < n:
        next_video = -1
        max_similarity = -1
        
        for i in range(n):
            if not used[i] and similarity_matrix[current, i] > max_similarity:
                max_similarity = similarity_matrix[current, i]
                next_video = i
        
        if next_video == -1:  # If no next video found, pick any remaining one
            for i in range(n):
                if not used[i]:
                    next_video = i
                    break
        
        used[next_video] = True
        sequence.append(videos[next_video])
        current = next_video
    
    # Log the stitching sequence
    logger.info("Optimal stitching sequence:")
    for i, video in enumerate(sequence):
        logger.info(f"{i+1}. {os.path.basename(video)}")
    
    return sequence

def merge_videos_smart(video_dir, output_path, pattern="*.mp4", fps=30, crossfade_frames=15, first_video_path=None):
    """
    Smart video merging: Determine optimal stitching order based on frame similarity, with optional starting video.
    """
    logger.info(f"Searching for video files: {video_dir}/{pattern}")
    
    # Get all video files
    video_files = glob(os.path.join(video_dir, pattern))
    if not video_files:
        raise Exception(f"No video files matching {pattern} found in {video_dir}")
    
    logger.info(f"Found {len(video_files)} video files")
    
    # Determine optimal stitching sequence
    optimal_sequence = find_optimal_video_sequence(video_files, first_video_path)
    if not optimal_sequence:
        raise Exception("Unable to determine stitching sequence, please check video files")
    
    # Read the first video to get dimensions
    cap = cv2.VideoCapture(optimal_sequence[0])
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    # Create video writer object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process each video file
    frames_by_video = []
    for i, video_file in enumerate(optimal_sequence):
        frames = []
        cap = cv2.VideoCapture(video_file)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        frames_by_video.append(frames)
        logger.info(f"Read video {i+1}/{len(optimal_sequence)}: {len(frames)} frames - {os.path.basename(video_file)}")
    
    # Merge videos with transition effects
    for i, frames in enumerate(frames_by_video):
        if i == 0:
            # Write all frames of the first video
            for frame in frames:
                out.write(frame)
        else:
            # Add transition effect for subsequent videos
            prev_frames = frames_by_video[i-1]
            
            # Ensure crossfade frames do not exceed video length
            actual_crossfade = min(crossfade_frames, len(prev_frames), len(frames))
            
            # Crossfade between last frames of previous video and first frames of current video
            for j in range(actual_crossfade):
                # Calculate blending weight
                alpha = j / actual_crossfade
                
                # Get frames to blend
                prev_frame = prev_frames[-actual_crossfade+j]
                curr_frame = frames[j]
                
                # Blend frames
                blended_frame = cv2.addWeighted(prev_frame, 1 - alpha, curr_frame, alpha, 0)
                out.write(blended_frame)
            
            # Write remaining frames
            for j in range(actual_crossfade, len(frames)):
                out.write(frames[j])
    
    # Release resources
    out.release()
    logger.info(f"Video merging completed, saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Smart merging of multiple videos into one')
    parser.add_argument('--video_dir', type=str, default="/h3cstore_ns/ydchen/code/wan_2_1/Wan2.1/video_result_chair", help='Directory containing video files')
    parser.add_argument('--output', type=str, default="/h3cstore_ns/ydchen/code/wan_2_1/Wan2.1/video_result_chair/merge", help='Output video file path')
    parser.add_argument('--pattern', type=str, default='*.mp4', help='File matching pattern')
    parser.add_argument('--fps', type=int, default=24, help='Output video frame rate')
    parser.add_argument('--crossfade', type=int, default=5, help='Number of frames for transition effect')
    parser.add_argument('--first_video', type=str, default="/h3cstore_ns/ydchen/code/wan_2_1/Wan2.1/video_result_chair/flf2v-14B_1280*720_1_1_Animate_the_transition:_smoothly_zoom_into_the_woo_20250505_175510.mp4", help='Full path to the starting video for merging')
    args = parser.parse_args()
    args.output = os.path.join(args.video_dir, 'merge')
    os.makedirs(args.output, exist_ok=True)
    args.output = os.path.join(args.output, f"merged_{args.fps}fps.mp4")
    merge_videos_smart(args.video_dir, args.output, args.pattern, args.fps, args.crossfade, args.first_video)

if __name__ == "__main__":
    main()