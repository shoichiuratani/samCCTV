"""
CCTV Video Analysis Application with Official Grounded-SAM-2
Premium version with official implementation and streaming downloads
"""

import os
import sys
import json
import uuid
import traceback
from datetime import datetime
from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS
from werkzeug.utils import secure_filename
from streaming_download import streaming_handler, ChunkedUploadHandler

# Import analyzers
try:
    from official_grounded_sam2 import create_official_analyzer
    OFFICIAL_SAM2_AVAILABLE = True
except ImportError as e:
    print(f"Official SAM2 not available: {e}")
    OFFICIAL_SAM2_AVAILABLE = False

from improved_detection import process_video_improved_detection

app = Flask(__name__, 
           template_folder='templates',
           static_folder='static')
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1GB max file size
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['OUTPUT_FOLDER'] = os.path.join(os.path.dirname(__file__), 'outputs')

# Allowed file extensions
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm'}

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Initialize chunked upload handler
chunked_upload = ChunkedUploadHandler(app.config['UPLOAD_FOLDER'])

# Store for analysis tasks with persistence
analysis_tasks = {}
TASKS_FILE = os.path.join(os.path.dirname(__file__), 'tasks_data_official.json')

def load_tasks():
    """Load tasks from file on startup"""
    global analysis_tasks
    try:
        if os.path.exists(TASKS_FILE):
            with open(TASKS_FILE, 'r') as f:
                analysis_tasks = json.load(f)
                print(f"Loaded {len(analysis_tasks)} existing tasks")
                # Verify files still exist
                valid_tasks = {}
                for task_id, task in analysis_tasks.items():
                    if task.get('status') == 'completed':
                        result = task.get('result', {})
                        video_path = result.get('output_video_path')
                        if video_path and os.path.exists(video_path):
                            valid_tasks[task_id] = task
                        else:
                            print(f"Task {task_id} output file missing, removing")
                    else:
                        valid_tasks[task_id] = task
                analysis_tasks = valid_tasks
    except Exception as e:
        print(f"Error loading tasks: {e}")
        analysis_tasks = {}

def save_tasks():
    """Save tasks to file"""
    try:
        with open(TASKS_FILE, 'w') as f:
            json.dump(analysis_tasks, f, indent=2)
    except Exception as e:
        print(f"Error saving tasks: {e}")

# Initialize analyzers
device = "cpu"
try:
    import torch
    if torch.cuda.is_available():
        device = "cuda"
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
except ImportError:
    pass

print(f"Detected device: {device}")

# Initialize Official Grounded-SAM-2 analyzer
official_analyzer = None
official_ready = False

if OFFICIAL_SAM2_AVAILABLE:
    try:
        official_analyzer = create_official_analyzer(device=device)
        if official_analyzer and official_analyzer.get_model_info().get('model_ready', False):
            official_ready = True
            print("Official Grounded-SAM-2 analyzer initialized successfully!")
        else:
            print("Official Grounded-SAM-2 models not ready")
    except Exception as e:
        print(f"Failed to initialize Official Grounded-SAM-2: {e}")
        print(f"Error details: {traceback.format_exc()}")

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_video_official_sam2(task):
    """Process video with Official Grounded-SAM-2 or fallback"""
    video_path = task['file_path']
    text_prompt = task['text_prompt']
    task_id = task['id']
    
    # Create output directory for this task
    output_dir = os.path.join(app.config['OUTPUT_FOLDER'], task_id)
    os.makedirs(output_dir, exist_ok=True)
    
    # Priority order: Official SAM2 -> Improved Detection -> Mock
    if official_ready and official_analyzer is not None:
        try:
            print(f"🚀 Starting Official Grounded-SAM-2 analysis for task {task_id}")
            result = official_analyzer.analyze_video(
                video_path=video_path,
                text_prompt=text_prompt,
                output_dir=output_dir
            )
            print(f"✅ Official SAM2 analysis completed for task {task_id}")
            return result
        except Exception as e:
            print(f"❌ Official SAM2 failed: {e}, trying improved detection")
            print(f"Error traceback: {traceback.format_exc()}")
    
    # Fallback to improved OpenCV detection
    try:
        print(f"🔍 Starting improved OpenCV detection for task {task_id}")
        result = process_video_improved_detection(video_path, output_dir, text_prompt)
        print(f"✅ Improved detection completed for task {task_id}")
        return result
    except Exception as e:
        print(f"❌ All analysis methods failed: {e}")
        return {
            'error': f'Analysis failed: {str(e)}',
            'summary': {
                'total_detections': 0,
                'processing_time': '0:00:10',
                'detection_method': 'Failed'
            }
        }

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload with chunking support"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        # Get chunking parameters
        chunk_number = request.form.get('chunk_number', 0, type=int)
        total_chunks = request.form.get('total_chunks', 1, type=int)
        unique_id = request.form.get('unique_id')
        
        filename = secure_filename(file.filename)
        
        # Handle chunked upload
        if total_chunks > 1:
            result = chunked_upload.handle_chunked_upload(
                file=file,
                filename=filename,
                chunk_number=chunk_number,
                total_chunks=total_chunks,
                unique_id=unique_id
            )
            
            if result['status'] == 'completed':
                task_id = result['unique_id']
                file_path = result['file_path']
            else:
                return jsonify(result)
        else:
            # Single file upload
            task_id = str(uuid.uuid4())
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{task_id}_{filename}")
            file.save(file_path)
        
        # Create task record
        analysis_tasks[task_id] = {
            'id': task_id,
            'filename': filename,
            'file_path': file_path,
            'status': 'uploaded',
            'upload_time': datetime.now().isoformat(),
            'file_size': os.path.getsize(file_path)
        }
        save_tasks()  # Persist task data
        
        return jsonify({
            'success': True,
            'message': 'ファイルのアップロードが完了しました！',
            'task_id': task_id,
            'filename': filename,
            'file_size': analysis_tasks[task_id]['file_size']
        })
        
    except Exception as e:
        print(f"Upload error: {e}")
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/analyze', methods=['POST'])
def analyze_video():
    """Start video analysis"""
    try:
        data = request.get_json()
        task_id = data.get('task_id')
        text_prompt = data.get('text_prompt', 'person')
        
        if not task_id or task_id not in analysis_tasks:
            return jsonify({'error': 'Invalid task ID'}), 400
        
        task = analysis_tasks[task_id]
        
        if task['status'] != 'uploaded':
            return jsonify({'error': 'Task already processed or in progress'}), 400
        
        # Update task status
        task['status'] = 'processing'
        task['start_time'] = datetime.now().isoformat()
        task['text_prompt'] = text_prompt
        task['analyzer_type'] = 'Official Grounded-SAM-2' if official_ready else 'Enhanced Detection'
        
        # Process video with Official Grounded-SAM-2
        try:
            result = process_video_official_sam2(task)
            task['status'] = 'completed'
            task['end_time'] = datetime.now().isoformat()
            task['result'] = result
            save_tasks()  # Persist completed task
            
            return jsonify({
                'success': True,
                'message': '解析が完了しました！',
                'task_id': task_id,
                'status': 'completed',
                'analyzer_type': task['analyzer_type']
            })
            
        except Exception as e:
            task['status'] = 'failed'
            task['error'] = str(e)
            task['end_time'] = datetime.now().isoformat()
            save_tasks()
            
            print(f"Analysis error for task {task_id}: {e}")
            print(f"Error traceback: {traceback.format_exc()}")
            
            return jsonify({
                'error': f'Analysis failed: {str(e)}',
                'task_id': task_id,
                'status': 'failed'
            }), 500
        
    except Exception as e:
        print(f"Analyze request error: {e}")
        return jsonify({'error': f'Request processing failed: {str(e)}'}), 500

@app.route('/status/<task_id>')
def get_status(task_id):
    """Get task status"""
    try:
        if task_id not in analysis_tasks:
            return jsonify({'error': 'Task not found'}), 404
        
        task = analysis_tasks[task_id]
        return jsonify({
            'task_id': task_id,
            'status': task['status'],
            'filename': task.get('filename'),
            'analyzer_type': task.get('analyzer_type'),
            'progress': 100 if task['status'] == 'completed' else (50 if task['status'] == 'processing' else 0)
        })
        
    except Exception as e:
        print(f"Status error: {e}")
        return jsonify({'error': f'Status check failed: {str(e)}'}), 500

@app.route('/result/<task_id>')
def get_result(task_id):
    """Get analysis results"""
    try:
        if task_id not in analysis_tasks:
            return jsonify({'error': 'Task not found'}), 404
        
        task = analysis_tasks[task_id]
        if task['status'] != 'completed':
            return jsonify({'error': 'Analysis not completed'}), 400
        
        return jsonify(task['result'])
        
    except Exception as e:
        print(f"Result error: {e}")
        return jsonify({'error': f'Result retrieval failed: {str(e)}'}), 500

@app.route('/download/<task_id>/<file_type>')
def download_result(task_id, file_type):
    """Download result files with streaming support"""
    try:
        print(f"🔽 Download request: task_id={task_id}, file_type={file_type}")
        
        if task_id not in analysis_tasks:
            print(f"Task {task_id} not found in analysis_tasks")
            return jsonify({'error': 'Task not found'}), 404
        
        task = analysis_tasks[task_id]
        print(f"Task status: {task['status']}")
        
        if task['status'] != 'completed':
            return jsonify({'error': 'Analysis not completed'}), 400
        
        result = task.get('result', {})
        print(f"Task result keys: {list(result.keys())}")
        
        if file_type == 'video':
            file_path = result.get('output_video_path')
            print(f"Video file path: {file_path}")
            
            if file_path and os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                print(f"📹 Streaming video file: {file_size} bytes")
                
                filename = f"official_sam2_video_{task_id}.mp4"
                
                # Use streaming handler for large files
                return streaming_handler.stream_file(
                    file_path=file_path,
                    as_attachment=True,
                    download_name=filename
                )
            else:
                print(f"Video file not found or does not exist: {file_path}")
                return jsonify({'error': 'Video file not found'}), 404
                
        elif file_type == 'annotations':
            file_path = result.get('annotations_path')
            print(f"Annotations file path: {file_path}")
            
            if file_path and os.path.exists(file_path):
                filename = f"official_sam2_annotations_{task_id}.json"
                
                # Use streaming for consistency
                return streaming_handler.stream_file(
                    file_path=file_path,
                    as_attachment=True,
                    download_name=filename
                )
            else:
                print(f"Annotations file not found: {file_path}")
                return jsonify({'error': 'Annotations file not found'}), 404
        
        return jsonify({'error': f'Invalid file type: {file_type}'}), 400
        
    except Exception as e:
        print(f"Download error: {e}")
        print(f"Error traceback: {traceback.format_exc()}")
        return jsonify({'error': f'Download failed: {str(e)}'}), 500

@app.route('/tasks')
def list_tasks():
    """List all analysis tasks"""
    try:
        tasks = []
        for task_id, task in analysis_tasks.items():
            task_info = {
                'task_id': task_id,
                'filename': task.get('filename'),
                'status': task['status'],
                'upload_time': task.get('upload_time'),
                'text_prompt': task.get('text_prompt'),
                'analyzer_type': task.get('analyzer_type', 'Unknown')
            }
            
            # Add file availability info for completed tasks
            if task['status'] == 'completed' and 'result' in task:
                result = task['result']
                task_info['files_available'] = {
                    'video': os.path.exists(result.get('output_video_path', '')),
                    'annotations': os.path.exists(result.get('annotations_path', ''))
                }
                
                # Add summary info
                summary = result.get('summary', {})
                task_info['summary'] = {
                    'total_detections': summary.get('total_detections', 0),
                    'processing_time': summary.get('processing_time', 'Unknown'),
                    'detection_method': summary.get('detection_method', 'Unknown')
                }
            
            tasks.append(task_info)
        
        # Sort by upload time (newest first)
        tasks.sort(key=lambda x: x.get('upload_time', ''), reverse=True)
        
        return jsonify(tasks)
        
    except Exception as e:
        print(f"Tasks list error: {e}")
        return jsonify([])

@app.route('/system/status')
def system_status():
    """Get system status and capabilities"""
    status = {
        'device': device,
        'official_sam2_ready': official_ready,
        'models_ready': official_ready,
        'ready': True,
        'message': 'Official Grounded-SAM-2 Ready' if official_ready else 'Enhanced Detection Ready',
        'streaming_enabled': True,
        'chunked_upload_enabled': True
    }
    
    if official_ready and official_analyzer:
        try:
            model_info = official_analyzer.get_model_info()
            status['model_info'] = model_info
        except:
            pass
    
    return jsonify(status)

@app.route('/system/info')
def system_info():
    """Get detailed system information"""
    info = {
        'app_version': '3.0.0-official-sam2',
        'analyzer_type': 'Official Grounded-SAM-2' if official_ready else 'Enhanced Detection',
        'device': device,
        'upload_folder': app.config['UPLOAD_FOLDER'],
        'output_folder': app.config['OUTPUT_FOLDER'],
        'supported_formats': list(ALLOWED_EXTENSIONS),
        'max_file_size_gb': 1,
        'features': {
            'official_sam2': official_ready,
            'streaming_download': True,
            'chunked_upload': True,
            'memory_efficient': True
        }
    }
    
    # Add model paths if available
    if official_ready and official_analyzer:
        try:
            model_info = official_analyzer.get_model_info()
            info['model_paths'] = {
                'sam2_checkpoint': model_info.get('sam2_checkpoint'),
                'grounding_checkpoint': model_info.get('grounding_checkpoint')
            }
        except:
            pass
    
    return jsonify(info)

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    return jsonify({'error': 'File too large. Maximum size is 1GB.'}), 413

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle internal server errors"""
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Load existing tasks on startup
    load_tasks()
    
    print("=" * 60)
    print("🚀 Starting CCTV Video Analysis with Official Grounded-SAM-2")
    print("=" * 60)
    
    print(f"📁 Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"📁 Output folder: {app.config['OUTPUT_FOLDER']}")
    print(f"🤖 AI Analyzer: {'Official Grounded-SAM-2' if official_ready else 'Enhanced Detection'}")
    print(f"💻 Device: {device}")
    
    if official_ready:
        print("✅ Official Grounded-SAM-2 models loaded successfully!")
        print("🎯 Ready for authentic AI inference")
    else:
        print("⚠️  Official Grounded-SAM-2 not available, using enhanced detection")
    
    print("🔄 Streaming downloads enabled")
    print("📤 Chunked uploads enabled")
    print("🌐 Server will run on http://0.0.0.0:5003")
    print("=" * 60)
    
    port = int(os.environ.get('PORT', 5003))  # Use port 5003 for official version
    app.run(host='0.0.0.0', port=port, debug=False)