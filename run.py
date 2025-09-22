#!/usr/bin/env python3
"""
CCTV Video Analysis Application Entry Point
"""

import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'backend'))

# Import and run the Flask app
from backend.app import app

if __name__ == '__main__':
    print("Starting CCTV Video Analysis Application...")
    print(f"Project root: {project_root}")
    print(f"Python path: {sys.path[:3]}")
    
    app.run(host='0.0.0.0', port=5000, debug=False)