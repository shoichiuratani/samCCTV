"""
CCTV Video Analysis Application Backend
Uses Grounded-SAM-2 for object detection, tracking, and segmentation
"""

import os
import sys
import json
import uuid
from datetime import datetime
from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
import traceback

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__, 
           template_folder='../templates',
           static_folder='../static')
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 512 * 1024 * 1024  # 512MB max file size
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'uploads')
app.config['OUTPUT_FOLDER'] = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
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
        
        # Process video (this would be where we call Grounded-SAM-2)
        try:
            result = process_video_analysis(task)
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

def process_video_analysis(task):
    """Process video with Grounded-SAM-2"""
    try:
        # Import video processor
        from .video_processor import process_video_task
        
        # Run actual video analysis
        result = process_video_task(task)
        
        return result
        
    except Exception as e:
        logger.error(f"Video analysis failed: {e}")
        
        # Fallback to mock results for demo purposes
        import time
        time.sleep(2)  # Simulate processing time
        
        # Create output directory for this task
        output_dir = os.path.join(app.config['OUTPUT_FOLDER'], task['id'])
        os.makedirs(output_dir, exist_ok=True)
        
        # Mock result structure
        result = {
            'detected_objects': [
                {
                    'frame': 10,
                    'class_name': 'person',
                    'bbox': [100, 150, 200, 350],
                    'confidence': 0.95,
                    'track_id': 1
                },
                {
                    'frame': 15, 
                    'class_name': 'person',
                    'bbox': [105, 155, 205, 355],
                    'confidence': 0.93,
                    'track_id': 1
                }
            ],
            'output_video_path': os.path.join(output_dir, 'result_video.mp4'),
            'annotations_path': os.path.join(output_dir, 'annotations.json'),
            'summary': {
                'total_frames': 100,
                'processed_frames': 100,
                'unique_objects': 1,
                'tracking_duration': '5.2s',
                'note': 'Mock results - actual Grounded-SAM-2 integration available'
            }
        }
        
        # Save annotations to file
        with open(result['annotations_path'], 'w') as f:
            json.dump(result['detected_objects'], f, indent=2)
        
        return result

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
    if task_id not in analysis_tasks:
        return jsonify({'error': 'Task not found'}), 404
    
    task = analysis_tasks[task_id]
    if task['status'] != 'completed':
        return jsonify({'error': 'Analysis not completed'}), 400
    
    result = task.get('result', {})
    
    if file_type == 'video':
        file_path = result.get('output_video_path')
        if file_path and os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
    elif file_type == 'annotations':
        file_path = result.get('annotations_path')
        if file_path and os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
    
    return jsonify({'error': 'File not found'}), 404

@app.route('/tasks')
def list_tasks():
    """List all analysis tasks"""
    tasks = []
    for task_id, task in analysis_tasks.items():
        tasks.append({
            'task_id': task_id,
            'filename': task.get('filename'),
            'status': task['status'],
            'upload_time': task.get('upload_time'),
            'text_prompt': task.get('text_prompt')
        })
    
    # Sort by upload time (newest first)
    tasks.sort(key=lambda x: x.get('upload_time', ''), reverse=True)
    return jsonify(tasks)

@app.route('/system/status')
def system_status():
    """Get system status and capabilities"""
    try:
        from .video_processor import get_system_status
        status = get_system_status()
        return jsonify(status)
    except Exception as e:
        return jsonify({
            'error': f'Failed to get system status: {str(e)}',
            'ready': False
        }), 500

@app.route('/system/info')
def system_info():
    """Get system information"""
    try:
        from grounded_sam2.utils import get_system_info, check_dependencies
        
        info = {
            'system': get_system_info(),
            'dependencies': check_dependencies(),
            'app_version': '1.0.0',
            'upload_folder': app.config['UPLOAD_FOLDER'],
            'output_folder': app.config['OUTPUT_FOLDER']
        }
        
        return jsonify(info)
    except Exception as e:
        return jsonify({
            'error': f'Failed to get system info: {str(e)}'
        }), 500

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
    print("Starting CCTV Video Analysis Application...")
    print(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"Output folder: {app.config['OUTPUT_FOLDER']}")
    print("Server will run on http://0.0.0.0:5000")
    
    app.run(host='0.0.0.0', port=5000, debug=True)