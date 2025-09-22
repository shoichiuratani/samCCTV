"""
Simplified CCTV Video Analysis Application
Single file version for easy deployment
"""

import os
import sys
import json
import uuid
import cv2
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
import traceback
from PIL import Image

app = Flask(__name__, 
           template_folder='templates',
           static_folder='static')
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 512 * 1024 * 1024  # 512MB max file size
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['OUTPUT_FOLDER'] = os.path.join(os.path.dirname(__file__), 'outputs')
app.config['SECRET_KEY'] = 'cctv-analysis-secret-key'

# Allowed file extensions
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv'}

# Ensure upload and output directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Store for analysis tasks (in production, use a database)
analysis_tasks = {}

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_video_mock(task):
    """Mock video processing function"""
    import time
    time.sleep(3)  # Simulate processing time
    
    # Create output directory for this task
    output_dir = os.path.join(app.config['OUTPUT_FOLDER'], task['id'])
    os.makedirs(output_dir, exist_ok=True)
    
    # Parse text prompt
    text_prompt = task.get('text_prompt', '')
    classes = [cls.strip() for cls in text_prompt.split(',')]
    
    # Generate mock results based on the prompt
    detected_objects = []
    for i, class_name in enumerate(classes[:3]):  # Limit to 3 objects
        for frame in range(10, 101, 10):  # Every 10 frames
            if np.random.random() > 0.6:  # 40% chance of detection per frame
                detected_objects.append({
                    'frame': frame,
                    'class_name': class_name.strip(),
                    'bbox': [
                        np.random.randint(50, 300),
                        np.random.randint(50, 200),
                        np.random.randint(350, 500),
                        np.random.randint(250, 400)
                    ],
                    'confidence': np.random.uniform(0.6, 0.95),
                    'track_id': i + 1
                })
    
    # Mock result structure
    result = {
        'detected_objects': detected_objects,
        'output_video_path': os.path.join(output_dir, 'result_video.mp4'),
        'annotations_path': os.path.join(output_dir, 'annotations.json'),
        'summary': {
            'total_frames': 100,
            'processed_frames': 100,
            'unique_objects': len(classes),
            'tracking_duration': '3.0s'
        }
    }
    
    # Save annotations to file
    with open(result['annotations_path'], 'w', encoding='utf-8') as f:
        json.dump(result['detected_objects'], f, indent=2, ensure_ascii=False)
    
    # Create a simple mock output video (copy of input for demo)
    input_path = task['file_path']
    if os.path.exists(input_path):
        try:
            import shutil
            shutil.copy2(input_path, result['output_video_path'])
        except:
            # If copy fails, create a placeholder
            with open(result['output_video_path'], 'w') as f:
                f.write("Mock video file")
    
    return result

@app.route('/')
def index():
    """Main page with upload interface"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle video file upload"""
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': f'File type not allowed. Supported: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
        
        # Generate unique filename
        file_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        file_extension = filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{file_id}.{file_extension}"
        
        # Save file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        
        # Create task record
        task_id = file_id
        analysis_tasks[task_id] = {
            'id': task_id,
            'filename': filename,
            'file_path': file_path,
            'status': 'uploaded',
            'upload_time': datetime.now().isoformat(),
            'file_size': os.path.getsize(file_path)
        }
        
        return jsonify({
            'task_id': task_id,
            'filename': filename,
            'file_size': analysis_tasks[task_id]['file_size'],
            'status': 'uploaded',
            'message': 'File uploaded successfully'
        })
        
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/analyze', methods=['POST'])
def analyze_video():
    """Start video analysis with text prompt"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        task_id = data.get('task_id')
        text_prompt = data.get('text_prompt', '').strip()
        
        if not task_id or task_id not in analysis_tasks:
            return jsonify({'error': 'Invalid task ID'}), 400
        
        if not text_prompt:
            return jsonify({'error': 'Text prompt is required'}), 400
        
        # Update task status
        task = analysis_tasks[task_id]
        task['text_prompt'] = text_prompt
        task['status'] = 'processing'
        task['start_time'] = datetime.now().isoformat()
        
        # Process video
        try:
            result = process_video_mock(task)
            task['status'] = 'completed'
            task['end_time'] = datetime.now().isoformat()
            task['result'] = result
            
            return jsonify({
                'task_id': task_id,
                'status': 'completed',
                'message': 'Analysis completed successfully',
                'result': result
            })
            
        except Exception as e:
            task['status'] = 'error'
            task['error'] = str(e)
            task['end_time'] = datetime.now().isoformat()
            raise e
            
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/status/<task_id>')
def get_task_status(task_id):
    """Get analysis task status"""
    if task_id not in analysis_tasks:
        return jsonify({'error': 'Task not found'}), 404
    
    task = analysis_tasks[task_id]
    return jsonify({
        'task_id': task_id,
        'status': task['status'],
        'filename': task.get('filename'),
        'text_prompt': task.get('text_prompt'),
        'upload_time': task.get('upload_time'),
        'start_time': task.get('start_time'),
        'end_time': task.get('end_time'),
        'error': task.get('error'),
        'result': task.get('result')
    })

@app.route('/results/<task_id>')
def get_results(task_id):
    """Get analysis results"""
    if task_id not in analysis_tasks:
        return jsonify({'error': 'Task not found'}), 404
    
    task = analysis_tasks[task_id]
    if task['status'] != 'completed':
        return jsonify({'error': 'Analysis not completed'}), 400
    
    return jsonify(task['result'])

@app.route('/download/<task_id>/<file_type>')
def download_result(task_id, file_type):
    """Download result files (video or annotations)"""
    try:
        if task_id not in analysis_tasks:
            return jsonify({'error': 'Task not found'}), 404
        
        task = analysis_tasks[task_id]
        if task['status'] != 'completed':
            return jsonify({'error': 'Analysis not completed'}), 400
        
        result = task.get('result', {})
        
        if file_type == 'video':
            file_path = result.get('output_video_path')
            if file_path and os.path.exists(file_path):
                filename = f"result_video_{task_id}.mp4"
                return send_file(file_path, 
                               as_attachment=True, 
                               download_name=filename,
                               mimetype='video/mp4')
        elif file_type == 'annotations':
            file_path = result.get('annotations_path')
            if file_path and os.path.exists(file_path):
                filename = f"annotations_{task_id}.json"
                return send_file(file_path, 
                               as_attachment=True, 
                               download_name=filename,
                               mimetype='application/json')
        
        return jsonify({'error': 'File not found'}), 404
        
    except Exception as e:
        print(f"Download error: {e}")
        return jsonify({'error': f'Download failed: {str(e)}'}), 500

@app.route('/tasks')
def list_tasks():
    """List all analysis tasks"""
    tasks = []
    for task_id, task in analysis_tasks.items():
        task_info = {
            'task_id': task_id,
            'filename': task.get('filename'),
            'status': task['status'],
            'upload_time': task.get('upload_time'),
            'text_prompt': task.get('text_prompt')
        }
        
        # Add file availability info for completed tasks
        if task['status'] == 'completed' and 'result' in task:
            result = task['result']
            task_info['files_available'] = {
                'video': os.path.exists(result.get('output_video_path', '')),
                'annotations': os.path.exists(result.get('annotations_path', ''))
            }
        
        tasks.append(task_info)
    
    # Sort by upload time (newest first)
    tasks.sort(key=lambda x: x.get('upload_time', ''), reverse=True)
    return jsonify(tasks)

@app.route('/debug/files/<task_id>')
def debug_files(task_id):
    """Debug endpoint to check file availability"""
    if task_id not in analysis_tasks:
        return jsonify({'error': 'Task not found'}), 404
    
    task = analysis_tasks[task_id]
    result = task.get('result', {})
    
    debug_info = {
        'task_id': task_id,
        'status': task['status'],
        'result_exists': 'result' in task,
        'output_dir': os.path.join(app.config['OUTPUT_FOLDER'], task_id),
        'files': {}
    }
    
    if 'result' in task:
        video_path = result.get('output_video_path')
        annotations_path = result.get('annotations_path')
        
        debug_info['files'] = {
            'video_path': video_path,
            'video_exists': os.path.exists(video_path) if video_path else False,
            'annotations_path': annotations_path,
            'annotations_exists': os.path.exists(annotations_path) if annotations_path else False
        }
        
        # List actual files in output directory
        output_dir = debug_info['output_dir']
        if os.path.exists(output_dir):
            debug_info['actual_files'] = os.listdir(output_dir)
        else:
            debug_info['actual_files'] = []
    
    return jsonify(debug_info)

@app.route('/system/status')
def system_status():
    """Get system status and capabilities"""
    return jsonify({
        'device': 'cpu',
        'models_ready': True,
        'ready': True,
        'message': 'Simple mock implementation running'
    })

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    return jsonify({'error': 'File too large. Maximum size is 512MB.'}), 413

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle internal server errors"""
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("Starting CCTV Video Analysis Application (Simple Version)...")
    print(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"Output folder: {app.config['OUTPUT_FOLDER']}")
    print("Server will run on http://0.0.0.0:5000")
    
    port = int(os.environ.get('PORT', 5001))  # Use port 5001 as fallback
    app.run(host='0.0.0.0', port=port, debug=False)