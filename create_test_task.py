#!/usr/bin/env python3
"""
Create a test task for download functionality testing
"""

import json
import os
from datetime import datetime

# Create a test task
task_id = "fc5d7bce-6f5f-4c1b-9224-5ecadc9580fd"
output_dir = f"outputs/{task_id}"

# Check if output files exist
video_path = os.path.join(output_dir, "result_video.mp4")
annotations_path = os.path.join(output_dir, "annotations.json")

if os.path.exists(video_path) and os.path.exists(annotations_path):
    test_task = {
        task_id: {
            "id": task_id,
            "filename": "test.mp4",
            "file_path": f"uploads/{task_id}.mp4",
            "status": "completed",
            "upload_time": "2025-09-22T03:49:53.000000",
            "end_time": "2025-09-22T03:50:35.000000",
            "file_size": 123456,
            "text_prompt": "person",
            "analyzer_type": "Real Grounded-SAM-2",
            "result": {
                "output_video_path": os.path.abspath(video_path),
                "annotations_path": os.path.abspath(annotations_path),
                "summary": {
                    "total_detections": 5,
                    "unique_objects": 1,
                    "frames_processed": 100,
                    "processing_time": "0:00:42"
                }
            }
        }
    }
    
    # Save to tasks file
    with open('tasks_data.json', 'w') as f:
        json.dump(test_task, f, indent=2)
    
    print(f"Created test task {task_id}")
    print(f"Video: {video_path} ({'exists' if os.path.exists(video_path) else 'missing'})")
    print(f"Annotations: {annotations_path} ({'exists' if os.path.exists(annotations_path) else 'missing'})")
else:
    print("Output files not found, cannot create test task")