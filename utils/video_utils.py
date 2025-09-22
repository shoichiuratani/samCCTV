"""
Video processing utilities for Grounded-SAM-2
"""

import cv2
import os
import glob
from typing import List, Tuple, Optional

def create_video_from_images(image_dir: str, output_path: str, fps: int = 30, 
                           image_pattern: str = "*.jpg") -> bool:
    """
    Create video from sequence of images
    
    Args:
        image_dir: Directory containing images
        output_path: Output video path
        fps: Frames per second
        image_pattern: Pattern to match image files
        
    Returns:
        Success status
    """
    try:
        # Get all image files
        image_files = sorted(glob.glob(os.path.join(image_dir, image_pattern)))
        
        if not image_files:
            print(f"No images found in {image_dir} with pattern {image_pattern}")
            return False
        
        # Read first image to get dimensions
        first_image = cv2.imread(image_files[0])
        if first_image is None:
            print(f"Could not read first image: {image_files[0]}")
            return False
        
        height, width, _ = first_image.shape
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            print(f"Could not open video writer for {output_path}")
            return False
        
        # Write all images to video
        for image_file in image_files:
            img = cv2.imread(image_file)
            if img is not None:
                # Resize if needed
                if img.shape[:2] != (height, width):
                    img = cv2.resize(img, (width, height))
                out.write(img)
            else:
                print(f"Warning: Could not read image {image_file}")
        
        out.release()
        print(f"Video created successfully: {output_path}")
        print(f"Total frames: {len(image_files)}")
        
        return True
        
    except Exception as e:
        print(f"Error creating video: {e}")
        return False

def extract_frames_from_video(video_path: str, output_dir: str, 
                            max_frames: Optional[int] = None,
                            frame_skip: int = 1) -> List[str]:
    """
    Extract frames from video
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save frames
        max_frames: Maximum number of frames to extract
        frame_skip: Skip every N frames
        
    Returns:
        List of frame paths
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        frame_paths = []
        frame_count = 0
        saved_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames if needed
            if frame_count % frame_skip == 0:
                if max_frames is None or saved_count < max_frames:
                    frame_filename = f"{saved_count:05d}.jpg"
                    frame_path = os.path.join(output_dir, frame_filename)
                    cv2.imwrite(frame_path, frame)
                    frame_paths.append(frame_path)
                    saved_count += 1
                else:
                    break
            
            frame_count += 1
        
        cap.release()
        print(f"Extracted {len(frame_paths)} frames from video")
        
        return frame_paths
        
    except Exception as e:
        print(f"Error extracting frames: {e}")
        return []

def get_video_info(video_path: str) -> dict:
    """
    Get video information
    
    Args:
        video_path: Path to video file
        
    Returns:
        Video information dictionary
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        info = {
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
        }
        
        cap.release()
        return info
        
    except Exception as e:
        print(f"Error getting video info: {e}")
        return {}

def resize_video_frames(input_dir: str, output_dir: str, target_size: Tuple[int, int],
                       image_pattern: str = "*.jpg") -> bool:
    """
    Resize all frames in a directory
    
    Args:
        input_dir: Input directory with frames
        output_dir: Output directory for resized frames
        target_size: Target size (width, height)
        image_pattern: Pattern to match image files
        
    Returns:
        Success status
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        image_files = sorted(glob.glob(os.path.join(input_dir, image_pattern)))
        
        if not image_files:
            print(f"No images found in {input_dir}")
            return False
        
        for image_file in image_files:
            img = cv2.imread(image_file)
            if img is not None:
                resized_img = cv2.resize(img, target_size)
                output_path = os.path.join(output_dir, os.path.basename(image_file))
                cv2.imwrite(output_path, resized_img)
        
        print(f"Resized {len(image_files)} frames to {target_size}")
        return True
        
    except Exception as e:
        print(f"Error resizing frames: {e}")
        return False

def create_video_preview(video_path: str, output_path: str, 
                        duration: int = 10, fps: int = 2) -> bool:
    """
    Create a preview video with selected frames
    
    Args:
        video_path: Input video path
        output_path: Output preview path
        duration: Preview duration in seconds
        fps: Frames per second for preview
        
    Returns:
        Success status
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate frame interval for preview
        frames_needed = duration * fps
        frame_interval = max(1, total_frames // frames_needed)
        
        # Get video dimensions
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                out.write(frame)
            
            frame_count += 1
            
            if frame_count >= total_frames:
                break
        
        cap.release()
        out.release()
        
        print(f"Created preview video: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error creating preview: {e}")
        return False