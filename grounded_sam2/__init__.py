"""
Grounded-SAM-2 Integration Module
Provides interface for CCTV video analysis using Grounded-SAM-2
"""

from .analyzer import VideoAnalyzer
from .utils import setup_models, download_checkpoints

__version__ = "1.0.0"
__all__ = ["VideoAnalyzer", "setup_models", "download_checkpoints"]