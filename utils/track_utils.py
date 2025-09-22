"""
Tracking utilities for Grounded-SAM-2
Based on official implementation
"""

import numpy as np
from typing import List, Tuple

def sample_points_from_masks(masks: np.ndarray, num_points: int = 10) -> List[np.ndarray]:
    """
    Sample positive points from masks for video tracking
    
    Args:
        masks: Array of masks with shape (n, H, W)
        num_points: Number of points to sample per mask
        
    Returns:
        List of point arrays for each mask
    """
    all_sample_points = []
    
    for mask in masks:
        # Get coordinates of positive pixels
        positive_coords = np.where(mask > 0)
        
        if len(positive_coords[0]) == 0:
            # No positive pixels, create a single point at center
            h, w = mask.shape
            points = np.array([[w//2, h//2]], dtype=np.float32)
            all_sample_points.append(points)
            continue
        
        # Convert to (y, x) format and then to (x, y) for consistency
        positive_pixels = np.column_stack((positive_coords[1], positive_coords[0]))  # (x, y)
        
        # Sample points uniformly
        if len(positive_pixels) <= num_points:
            # Use all available points
            sample_points = positive_pixels.astype(np.float32)
        else:
            # Uniform sampling
            indices = np.linspace(0, len(positive_pixels) - 1, num_points, dtype=int)
            sample_points = positive_pixels[indices].astype(np.float32)
        
        all_sample_points.append(sample_points)
    
    return all_sample_points

def filter_masks_by_area(masks: np.ndarray, scores: np.ndarray, 
                        min_area: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter masks by minimum area
    
    Args:
        masks: Array of masks
        scores: Corresponding scores
        min_area: Minimum area threshold
        
    Returns:
        Filtered masks and scores
    """
    valid_indices = []
    
    for i, mask in enumerate(masks):
        area = np.sum(mask > 0)
        if area >= min_area:
            valid_indices.append(i)
    
    if len(valid_indices) == 0:
        return np.array([]), np.array([])
    
    return masks[valid_indices], scores[valid_indices]

def compute_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Compute IoU between two masks
    
    Args:
        mask1: First mask
        mask2: Second mask
        
    Returns:
        IoU value
    """
    intersection = np.sum((mask1 > 0) & (mask2 > 0))
    union = np.sum((mask1 > 0) | (mask2 > 0))
    
    if union == 0:
        return 0.0
    
    return intersection / union

def merge_overlapping_masks(masks: np.ndarray, scores: np.ndarray, 
                           iou_threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Merge overlapping masks based on IoU threshold
    
    Args:
        masks: Array of masks
        scores: Corresponding scores
        iou_threshold: IoU threshold for merging
        
    Returns:
        Merged masks and scores
    """
    if len(masks) <= 1:
        return masks, scores
    
    keep_indices = []
    merged_masks = []
    merged_scores = []
    
    # Sort by score (descending)
    sorted_indices = np.argsort(scores)[::-1]
    
    for i in sorted_indices:
        mask_i = masks[i]
        score_i = scores[i]
        
        # Check if this mask overlaps significantly with any kept mask
        should_keep = True
        
        for j, kept_mask in enumerate(merged_masks):
            iou = compute_mask_iou(mask_i, kept_mask)
            if iou > iou_threshold:
                # Merge with higher score mask
                if score_i > merged_scores[j]:
                    merged_masks[j] = mask_i
                    merged_scores[j] = score_i
                should_keep = False
                break
        
        if should_keep:
            merged_masks.append(mask_i)
            merged_scores.append(score_i)
    
    return np.array(merged_masks), np.array(merged_scores)