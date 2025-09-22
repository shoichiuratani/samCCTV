"""
Utility functions for Grounded-SAM-2 integration
"""

import os
import json
import logging
import subprocess
from typing import Dict, Any, Optional
import torch

logger = logging.getLogger(__name__)


def setup_models(checkpoints_dir: str = "checkpoints") -> Dict[str, str]:
    """
    Setup and verify model checkpoints
    
    Args:
        checkpoints_dir: Directory containing model checkpoints
        
    Returns:
        Dictionary with paths to model checkpoints and configs
    """
    model_paths = {
        'sam2_checkpoint': None,
        'sam2_config': None,
        'gdino_checkpoint': None,
        'gdino_config': None
    }
    
    # Create checkpoints directory if it doesn't exist
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    # Check for SAM 2 checkpoints
    sam2_checkpoints = [
        'sam2_hiera_large.pt',
        'sam2_hiera_base_plus.pt',
        'sam2_hiera_small.pt'
    ]
    
    for checkpoint in sam2_checkpoints:
        checkpoint_path = os.path.join(checkpoints_dir, checkpoint)
        if os.path.exists(checkpoint_path):
            model_paths['sam2_checkpoint'] = checkpoint_path
            logger.info(f"Found SAM 2 checkpoint: {checkpoint_path}")
            break
    
    # Check for Grounding DINO checkpoints
    gdino_checkpoints = [
        'groundingdino_swint_ogc.pth',
        'groundingdino_swinb_cogcoor.pth'
    ]
    
    for checkpoint in gdino_checkpoints:
        checkpoint_path = os.path.join(checkpoints_dir, checkpoint)
        if os.path.exists(checkpoint_path):
            model_paths['gdino_checkpoint'] = checkpoint_path
            logger.info(f"Found Grounding DINO checkpoint: {checkpoint_path}")
            break
    
    return model_paths


def download_checkpoints(checkpoints_dir: str = "checkpoints") -> bool:
    """
    Download required model checkpoints
    
    Args:
        checkpoints_dir: Directory to save checkpoints
        
    Returns:
        True if successful, False otherwise
    """
    try:
        os.makedirs(checkpoints_dir, exist_ok=True)
        
        # This is a placeholder for checkpoint download
        # In real implementation, this would download from official sources
        
        logger.info("Model checkpoints download is handled separately.")
        logger.info("Please run the official download scripts:")
        logger.info("1. For SAM 2: cd checkpoints && bash download_ckpts.sh")
        logger.info("2. For Grounding DINO: cd gdino_checkpoints && bash download_ckpts.sh")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to download checkpoints: {e}")
        return False


def get_device() -> str:
    """
    Get the best available device for model inference
    
    Returns:
        Device string ('cuda' or 'cpu')
    """
    if torch.cuda.is_available():
        device = 'cuda'
        logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        logger.info("Using CPU for inference")
    
    return device


def validate_video_file(video_path: str) -> Dict[str, Any]:
    """
    Validate video file and get basic information
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with video information
    """
    import cv2
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Open video to get properties
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        video_info = {
            'path': video_path,
            'width': width,
            'height': height,
            'fps': fps,
            'frame_count': frame_count,
            'duration_seconds': duration,
            'file_size_bytes': os.path.getsize(video_path)
        }
        
        logger.info(f"Video validated: {width}x{height} @ {fps:.2f} FPS, {duration:.2f}s")
        
        return video_info
        
    finally:
        cap.release()


def create_analysis_config(
    text_prompt: str,
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
    frame_step: int = 1,
    prompt_type: str = "mask"
) -> Dict[str, Any]:
    """
    Create analysis configuration
    
    Args:
        text_prompt: Text prompt for object detection
        box_threshold: Box confidence threshold
        text_threshold: Text matching threshold
        frame_step: Process every nth frame
        prompt_type: Type of prompt for SAM 2 ('mask', 'box', 'point')
        
    Returns:
        Analysis configuration dictionary
    """
    config = {
        'text_prompt': text_prompt,
        'box_threshold': box_threshold,
        'text_threshold': text_threshold,
        'frame_step': frame_step,
        'prompt_type': prompt_type,
        'timestamp': None
    }
    
    # Validate prompt type
    valid_prompt_types = ['mask', 'box', 'point']
    if prompt_type not in valid_prompt_types:
        raise ValueError(f"Invalid prompt_type: {prompt_type}. Must be one of {valid_prompt_types}")
    
    # Validate thresholds
    if not 0.0 <= box_threshold <= 1.0:
        raise ValueError("box_threshold must be between 0.0 and 1.0")
    
    if not 0.0 <= text_threshold <= 1.0:
        raise ValueError("text_threshold must be between 0.0 and 1.0")
    
    # Validate frame step
    if frame_step < 1:
        raise ValueError("frame_step must be >= 1")
    
    logger.info(f"Analysis config created: {config}")
    
    return config


def save_results_json(results: Dict[str, Any], output_path: str) -> None:
    """
    Save analysis results to JSON file
    
    Args:
        results: Analysis results dictionary
        output_path: Path to save JSON file
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Results saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        raise


def load_results_json(json_path: str) -> Dict[str, Any]:
    """
    Load analysis results from JSON file
    
    Args:
        json_path: Path to JSON file
        
    Returns:
        Analysis results dictionary
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        logger.info(f"Results loaded from: {json_path}")
        return results
        
    except Exception as e:
        logger.error(f"Failed to load results: {e}")
        raise


def get_system_info() -> Dict[str, Any]:
    """
    Get system information for debugging
    
    Returns:
        System information dictionary
    """
    import platform
    import psutil
    
    try:
        system_info = {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2),
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
        }
        
        if torch.cuda.is_available():
            system_info['cuda_version'] = torch.version.cuda
            system_info['gpu_name'] = torch.cuda.get_device_name(0)
            system_info['gpu_memory_gb'] = round(
                torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
            )
        
        return system_info
        
    except Exception as e:
        logger.error(f"Failed to get system info: {e}")
        return {'error': str(e)}


def check_dependencies() -> Dict[str, bool]:
    """
    Check if required dependencies are available
    
    Returns:
        Dictionary showing availability of dependencies
    """
    dependencies = {}
    
    # Check Python packages
    packages = [
        'torch', 'torchvision', 'transformers', 'opencv-python',
        'Pillow', 'numpy', 'scipy', 'matplotlib'
    ]
    
    for package in packages:
        try:
            __import__(package.replace('-', '_'))
            dependencies[package] = True
        except ImportError:
            dependencies[package] = False
    
    # Check CUDA
    dependencies['cuda'] = torch.cuda.is_available()
    
    # Check FFmpeg (for video processing)
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                               capture_output=True, text=True, timeout=5)
        dependencies['ffmpeg'] = result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        dependencies['ffmpeg'] = False
    
    logger.info(f"Dependencies check: {dependencies}")
    
    return dependencies