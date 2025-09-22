"""
Video Processing Module
Integrates with Grounded-SAM-2 for CCTV video analysis
"""

import os
import sys
import logging
import traceback
from typing import Dict, Any, Optional

# Add parent directory to path
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from grounded_sam2 import VideoAnalyzer
from grounded_sam2.utils import (
    get_device, 
    validate_video_file, 
    create_analysis_config,
    setup_models,
    check_dependencies
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CCTVVideoProcessor:
    """Main class for processing CCTV videos with Grounded-SAM-2"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize video processor
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Get device
        self.device = self.config.get('device') or get_device()
        
        # Model paths
        self.model_paths = setup_models(
            self.config.get('checkpoints_dir', 'checkpoints')
        )
        
        # Create analyzer
        self.analyzer = VideoAnalyzer(
            device=self.device,
            sam2_checkpoint=self.model_paths.get('sam2_checkpoint'),
            sam2_config=self.model_paths.get('sam2_config'),
            gdino_checkpoint=self.model_paths.get('gdino_checkpoint'),
            gdino_config=self.model_paths.get('gdino_config')
        )
        
        # Analysis parameters
        self.box_threshold = self.config.get('box_threshold', 0.35)
        self.text_threshold = self.config.get('text_threshold', 0.25)
        self.frame_step = self.config.get('frame_step', 1)
        self.prompt_type = self.config.get('prompt_type', 'mask')
        
        logger.info(f"CCTVVideoProcessor initialized on {self.device}")

    def process_video(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process video file with Grounded-SAM-2
        
        Args:
            task: Task dictionary containing file path and text prompt
            
        Returns:
            Analysis results
        """
        try:
            video_path = task['file_path']
            text_prompt = task['text_prompt']
            task_id = task['id']
            
            logger.info(f"Starting video analysis for task {task_id}")
            logger.info(f"Video: {video_path}")
            logger.info(f"Prompt: {text_prompt}")
            
            # Validate video file
            video_info = validate_video_file(video_path)
            logger.info(f"Video validated: {video_info['width']}x{video_info['height']}")
            
            # Create output directory
            output_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), 
                'outputs', 
                task_id
            )
            os.makedirs(output_dir, exist_ok=True)
            
            # Create analysis configuration
            analysis_config = create_analysis_config(
                text_prompt=text_prompt,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                frame_step=self.frame_step,
                prompt_type=self.prompt_type
            )
            
            # Update analyzer parameters
            self.analyzer.box_threshold = analysis_config['box_threshold']
            self.analyzer.text_threshold = analysis_config['text_threshold']
            self.analyzer.frame_step = analysis_config['frame_step']
            
            # Run analysis
            logger.info("Running Grounded-SAM-2 analysis...")
            results = self.analyzer.analyze_video(
                video_path=video_path,
                text_prompt=text_prompt,
                output_dir=output_dir
            )
            
            # Add video info and config to results
            results['video_info'] = video_info
            results['analysis_config'] = analysis_config
            results['task_id'] = task_id
            
            logger.info(f"Analysis completed for task {task_id}")
            logger.info(f"Found {len(results['detected_objects'])} detections")
            
            return results
            
        except Exception as e:
            logger.error(f"Video processing failed: {e}")
            logger.error(traceback.format_exc())
            raise

    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and capabilities"""
        try:
            dependencies = check_dependencies()
            
            status = {
                'device': self.device,
                'model_paths': self.model_paths,
                'dependencies': dependencies,
                'config': {
                    'box_threshold': self.box_threshold,
                    'text_threshold': self.text_threshold,
                    'frame_step': self.frame_step,
                    'prompt_type': self.prompt_type
                }
            }
            
            # Check if models are ready
            models_ready = all([
                path is not None 
                for path in self.model_paths.values() 
                if path is not None
            ])
            
            status['models_ready'] = models_ready
            status['ready'] = models_ready and dependencies.get('torch', False)
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {
                'error': str(e),
                'ready': False
            }

    def validate_input(self, video_path: str, text_prompt: str) -> Dict[str, Any]:
        """
        Validate input parameters
        
        Args:
            video_path: Path to video file
            text_prompt: Text prompt for detection
            
        Returns:
            Validation results
        """
        validation = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        try:
            # Check video file
            if not os.path.exists(video_path):
                validation['errors'].append(f"Video file not found: {video_path}")
                validation['valid'] = False
            else:
                try:
                    video_info = validate_video_file(video_path)
                    validation['video_info'] = video_info
                    
                    # Check video duration
                    max_duration = self.config.get('max_video_duration', 300)  # 5 minutes
                    if video_info['duration_seconds'] > max_duration:
                        validation['warnings'].append(
                            f"Video duration ({video_info['duration_seconds']:.1f}s) "
                            f"exceeds recommended maximum ({max_duration}s)"
                        )
                    
                    # Check video resolution
                    if video_info['width'] > 1920 or video_info['height'] > 1080:
                        validation['warnings'].append(
                            f"High resolution video ({video_info['width']}x{video_info['height']}) "
                            "may require longer processing time"
                        )
                        
                except Exception as e:
                    validation['errors'].append(f"Invalid video file: {e}")
                    validation['valid'] = False
            
            # Check text prompt
            if not text_prompt or not text_prompt.strip():
                validation['errors'].append("Text prompt is required")
                validation['valid'] = False
            else:
                # Parse and validate prompt
                classes = [cls.strip() for cls in text_prompt.split(',')]
                if len(classes) > 10:
                    validation['warnings'].append(
                        f"Many object classes ({len(classes)}) may slow down processing"
                    )
                
                validation['parsed_classes'] = classes
            
            return validation
            
        except Exception as e:
            logger.error(f"Input validation failed: {e}")
            validation['valid'] = False
            validation['errors'].append(f"Validation error: {e}")
            return validation


# Global processor instance
processor = None


def get_processor(config: Optional[Dict[str, Any]] = None) -> CCTVVideoProcessor:
    """Get or create global processor instance"""
    global processor
    
    if processor is None:
        processor = CCTVVideoProcessor(config)
    
    return processor


def process_video_task(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process video task (function interface for backend)
    
    Args:
        task: Task dictionary
        
    Returns:
        Analysis results
    """
    processor = get_processor()
    return processor.process_video(task)


def validate_video_input(video_path: str, text_prompt: str) -> Dict[str, Any]:
    """
    Validate video input (function interface for backend)
    
    Args:
        video_path: Path to video file
        text_prompt: Text prompt for detection
        
    Returns:
        Validation results
    """
    processor = get_processor()
    return processor.validate_input(video_path, text_prompt)


def get_system_status() -> Dict[str, Any]:
    """Get system status (function interface for backend)"""
    processor = get_processor()
    return processor.get_system_status()


# Initialize processor on module load
try:
    get_processor()
    logger.info("Video processor initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize video processor: {e}")
    logger.error(traceback.format_exc())