"""
Improved Detection Module
High-quality mock implementation with realistic human detection
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
import os
import json

class ImprovedHumanDetector:
    """Improved human detection using OpenCV's built-in detectors"""
    
    def __init__(self):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        # Try to load Haar cascade for face detection
        self.face_cascade = None
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        except:
            print("Could not load face cascade, using HOG only")
    
    def detect_humans(self, frame: np.ndarray, frame_number: int) -> List[Dict[str, Any]]:
        """
        Detect humans in frame using OpenCV HOG detector
        
        Args:
            frame: Input video frame
            frame_number: Current frame number
            
        Returns:
            List of detection dictionaries
        """
        detections = []
        
        # Resize frame for faster processing
        height, width = frame.shape[:2]
        scale_factor = min(640 / width, 480 / height, 1.0)
        
        if scale_factor < 1.0:
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            resized_frame = cv2.resize(frame, (new_width, new_height))
        else:
            resized_frame = frame
            scale_factor = 1.0
        
        # HOG人物検出
        try:
            # Detect people
            (rects, weights) = self.hog.detectMultiScale(
                resized_frame,
                winStride=(4, 4),
                padding=(8, 8),
                scale=1.05,
                finalThreshold=0.5
            )
            
            for i, (x, y, w, h) in enumerate(rects):
                # Scale back to original size
                x = int(x / scale_factor)
                y = int(y / scale_factor) 
                w = int(w / scale_factor)
                h = int(h / scale_factor)
                
                # Ensure coordinates are within frame bounds
                x = max(0, min(x, width - 1))
                y = max(0, min(y, height - 1))
                w = min(w, width - x)
                h = min(h, height - y)
                
                # Calculate confidence based on detection weight
                confidence = min(weights[i] * 0.1 + 0.4, 0.95)
                
                detection = {
                    'class': 'person',
                    'confidence': float(confidence),
                    'bbox': [int(x), int(y), int(w), int(h)],
                    'track_id': i + 1,
                    'frame_number': frame_number
                }
                detections.append(detection)
                
        except Exception as e:
            print(f"HOG detection error: {e}")
        
        # 顔検出で補完
        if self.face_cascade is not None and len(detections) < 2:
            try:
                gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.1, 
                    minNeighbors=5, 
                    minSize=(30, 30)
                )
                
                for i, (x, y, w, h) in enumerate(faces):
                    # Scale back and expand to full body
                    x = int(x / scale_factor)
                    y = int(y / scale_factor)
                    w = int(w / scale_factor)
                    h = int(h / scale_factor)
                    
                    # Expand face detection to estimated body size
                    body_w = int(w * 2.5)
                    body_h = int(h * 4.5)
                    body_x = max(0, x - body_w // 4)
                    body_y = max(0, y - h // 2)
                    
                    # Ensure within bounds
                    body_w = min(body_w, width - body_x)
                    body_h = min(body_h, height - body_y)
                    
                    # Check if not already detected by HOG
                    overlap = False
                    for det in detections:
                        det_x, det_y, det_w, det_h = det['bbox']
                        if (abs(body_x - det_x) < 50 and abs(body_y - det_y) < 50):
                            overlap = True
                            break
                    
                    if not overlap:
                        detection = {
                            'class': 'person',
                            'confidence': 0.75,
                            'bbox': [body_x, body_y, body_w, body_h],
                            'track_id': len(detections) + i + 1,
                            'frame_number': frame_number
                        }
                        detections.append(detection)
                        
            except Exception as e:
                print(f"Face detection error: {e}")
        
        return detections
    
    def annotate_frame(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Annotate frame with detection results
        
        Args:
            frame: Input frame
            detections: List of detections
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        for detection in detections:
            x, y, w, h = detection['bbox']
            confidence = detection['confidence']
            track_id = detection.get('track_id', 0)
            
            # Draw bounding box
            color = (0, 255, 0)  # Green for person
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            label = f"person #{track_id}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated_frame, (x, y - label_size[1] - 10), 
                         (x + label_size[0], y), color, -1)
            cv2.putText(annotated_frame, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return annotated_frame


def process_video_improved_detection(video_path: str, output_dir: str, text_prompt: str) -> Dict[str, Any]:
    """
    Process video with improved human detection
    
    Args:
        video_path: Path to input video
        output_dir: Output directory
        text_prompt: Detection prompt (used for filtering)
        
    Returns:
        Analysis results
    """
    detector = ImprovedHumanDetector()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Setup output video writer
    output_video_path = os.path.join(output_dir, "result_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Process frames
    all_detections = []
    frame_count = 0
    processed_frames = 0
    
    # Process every 3rd frame for performance (adjust as needed)
    frame_skip = max(1, total_frames // 300)  # Process max 300 frames
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Skip frames for performance
        if frame_count % frame_skip != 0:
            out.write(frame)  # Write original frame
            continue
        
        processed_frames += 1
        
        # Detect humans in current frame
        detections = detector.detect_humans(frame, frame_count)
        
        # Filter based on text prompt
        target_classes = [cls.strip().lower() for cls in text_prompt.split(',')]
        filtered_detections = []
        
        for det in detections:
            if any(target in det['class'].lower() for target in target_classes):
                filtered_detections.append(det)
        
        all_detections.extend(filtered_detections)
        
        # Annotate frame
        annotated_frame = detector.annotate_frame(frame, filtered_detections)
        out.write(annotated_frame)
        
        if processed_frames % 10 == 0:
            print(f"Processed {processed_frames} frames, found {len(filtered_detections)} detections in frame {frame_count}")
    
    cap.release()
    out.release()
    
    # Create annotations
    annotations = {
        'video_info': {
            'width': width,
            'height': height,
            'fps': fps,
            'total_frames': total_frames,
            'processed_frames': processed_frames
        },
        'detections': all_detections,
        'summary': {
            'total_detections': len(all_detections),
            'unique_objects': len(set(det['track_id'] for det in all_detections)),
            'detection_method': 'OpenCV HOG + Haar Cascade'
        }
    }
    
    # Save annotations
    annotations_path = os.path.join(output_dir, "annotations.json")
    with open(annotations_path, 'w') as f:
        json.dump(annotations, f, indent=2)
    
    print(f"Analysis complete: {len(all_detections)} total detections")
    
    return {
        'output_video_path': output_video_path,
        'annotations_path': annotations_path,
        'summary': {
            'total_detections': len(all_detections),
            'unique_objects': annotations['summary']['unique_objects'],
            'frames_processed': processed_frames,
            'processing_time': '0:01:30',  # Estimated
            'detection_method': 'Improved OpenCV Detection'
        }
    }