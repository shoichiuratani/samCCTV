"""
Streaming Download Handler
Solves client-side memory issues with server-side streaming and chunked downloads
"""

import os
import mimetypes
from flask import Response, request, abort
from typing import Generator, Optional
import logging

logger = logging.getLogger(__name__)

class StreamingDownloadHandler:
    """
    Handles large file downloads with streaming to prevent client-side memory issues
    """
    
    def __init__(self, chunk_size: int = 1024 * 1024):  # 1MB chunks
        """
        Initialize streaming handler
        
        Args:
            chunk_size: Size of each chunk in bytes (default: 1MB)
        """
        self.chunk_size = chunk_size
    
    def stream_file(self, file_path: str, as_attachment: bool = True, 
                   download_name: Optional[str] = None) -> Response:
        """
        Stream a file with support for range requests (partial content)
        
        Args:
            file_path: Path to the file to stream
            as_attachment: Whether to download as attachment
            download_name: Custom filename for download
            
        Returns:
            Flask Response with streaming content
        """
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            abort(404)
        
        file_size = os.path.getsize(file_path)
        
        # Determine MIME type
        mimetype = mimetypes.guess_type(file_path)[0] or 'application/octet-stream'
        
        # Handle range requests for partial content (important for large files)
        range_header = request.headers.get('Range')
        
        if range_header:
            return self._handle_range_request(file_path, file_size, mimetype, 
                                            range_header, as_attachment, download_name)
        else:
            return self._handle_full_request(file_path, file_size, mimetype, 
                                           as_attachment, download_name)
    
    def _handle_range_request(self, file_path: str, file_size: int, mimetype: str,
                             range_header: str, as_attachment: bool, 
                             download_name: Optional[str]) -> Response:
        """Handle HTTP Range request for partial content"""
        try:
            # Parse range header: bytes=start-end
            byte_start = 0
            byte_end = file_size - 1
            
            if range_header:
                match = range_header.replace('bytes=', '').split('-')
                if len(match) == 2:
                    if match[0]:
                        byte_start = int(match[0])
                    if match[1]:
                        byte_end = int(match[1])
            
            # Validate range
            if byte_start >= file_size or byte_end >= file_size or byte_start > byte_end:
                abort(416)  # Range Not Satisfiable
            
            content_length = byte_end - byte_start + 1
            
            def generate_partial():
                with open(file_path, 'rb') as f:
                    f.seek(byte_start)
                    remaining = content_length
                    
                    while remaining > 0:
                        chunk_size = min(self.chunk_size, remaining)
                        chunk = f.read(chunk_size)
                        if not chunk:
                            break
                        remaining -= len(chunk)
                        yield chunk
            
            response = Response(
                generate_partial(),
                206,  # Partial Content
                mimetype=mimetype,
                direct_passthrough=True
            )
            
            # Set headers for partial content
            response.headers['Content-Range'] = f'bytes {byte_start}-{byte_end}/{file_size}'
            response.headers['Content-Length'] = str(content_length)
            response.headers['Accept-Ranges'] = 'bytes'
            
            if as_attachment:
                filename = download_name or os.path.basename(file_path)
                response.headers['Content-Disposition'] = f'attachment; filename="{filename}"'
            
            # Add cache control
            response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
            response.headers['Pragma'] = 'no-cache'
            response.headers['Expires'] = '0'
            
            logger.info(f"Streaming partial content: {byte_start}-{byte_end}/{file_size}")
            
            return response
            
        except Exception as e:
            logger.error(f"Error handling range request: {e}")
            abort(500)
    
    def _handle_full_request(self, file_path: str, file_size: int, mimetype: str,
                           as_attachment: bool, download_name: Optional[str]) -> Response:
        """Handle full file request with streaming"""
        
        def generate_full():
            with open(file_path, 'rb') as f:
                while True:
                    chunk = f.read(self.chunk_size)
                    if not chunk:
                        break
                    yield chunk
        
        response = Response(
            generate_full(),
            200,
            mimetype=mimetype,
            direct_passthrough=True
        )
        
        # Set headers
        response.headers['Content-Length'] = str(file_size)
        response.headers['Accept-Ranges'] = 'bytes'
        
        if as_attachment:
            filename = download_name or os.path.basename(file_path)
            response.headers['Content-Disposition'] = f'attachment; filename="{filename}"'
        
        # Add cache control for better handling
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        
        # Add ETag for caching validation
        response.headers['ETag'] = f'"{hash(file_path)}-{file_size}"'
        
        logger.info(f"Streaming full file: {file_size} bytes")
        
        return response
    
    def create_progress_stream(self, file_path: str, progress_callback=None) -> Generator[bytes, None, None]:
        """
        Create a streaming generator with progress tracking
        
        Args:
            file_path: Path to file
            progress_callback: Callback function for progress updates
            
        Yields:
            File chunks
        """
        file_size = os.path.getsize(file_path)
        bytes_sent = 0
        
        with open(file_path, 'rb') as f:
            while True:
                chunk = f.read(self.chunk_size)
                if not chunk:
                    break
                
                bytes_sent += len(chunk)
                
                if progress_callback:
                    progress = (bytes_sent / file_size) * 100
                    progress_callback(progress, bytes_sent, file_size)
                
                yield chunk


class ChunkedUploadHandler:
    """
    Handles large file uploads in chunks to prevent server-side memory issues
    """
    
    def __init__(self, upload_dir: str, max_chunk_size: int = 10 * 1024 * 1024):  # 10MB
        """
        Initialize chunked upload handler
        
        Args:
            upload_dir: Directory to store uploaded files
            max_chunk_size: Maximum chunk size in bytes
        """
        self.upload_dir = upload_dir
        self.max_chunk_size = max_chunk_size
        os.makedirs(upload_dir, exist_ok=True)
    
    def handle_chunked_upload(self, file, filename: str, chunk_number: int = 0, 
                            total_chunks: int = 1, unique_id: str = None) -> dict:
        """
        Handle a single chunk of a file upload
        
        Args:
            file: File chunk from request
            filename: Original filename
            chunk_number: Current chunk number (0-based)
            total_chunks: Total number of chunks
            unique_id: Unique identifier for this upload session
            
        Returns:
            Upload status dictionary
        """
        try:
            if not unique_id:
                import uuid
                unique_id = str(uuid.uuid4())
            
            # Create temporary directory for this upload
            temp_dir = os.path.join(self.upload_dir, f"chunks_{unique_id}")
            os.makedirs(temp_dir, exist_ok=True)
            
            # Save chunk
            chunk_path = os.path.join(temp_dir, f"chunk_{chunk_number}")
            file.save(chunk_path)
            
            logger.info(f"Saved chunk {chunk_number}/{total_chunks} for {filename}")
            
            # Check if all chunks are uploaded
            uploaded_chunks = len([f for f in os.listdir(temp_dir) if f.startswith("chunk_")])
            
            if uploaded_chunks == total_chunks:
                # Reassemble file
                final_path = self._reassemble_file(temp_dir, filename, total_chunks, unique_id)
                return {
                    'status': 'completed',
                    'file_path': final_path,
                    'unique_id': unique_id,
                    'message': 'File upload completed successfully'
                }
            else:
                return {
                    'status': 'partial',
                    'uploaded_chunks': uploaded_chunks,
                    'total_chunks': total_chunks,
                    'unique_id': unique_id,
                    'message': f'Chunk {chunk_number} uploaded successfully'
                }
                
        except Exception as e:
            logger.error(f"Error handling chunk upload: {e}")
            return {
                'status': 'error',
                'message': f'Upload failed: {str(e)}'
            }
    
    def _reassemble_file(self, temp_dir: str, filename: str, total_chunks: int, unique_id: str) -> str:
        """Reassemble chunks into final file"""
        final_path = os.path.join(self.upload_dir, f"{unique_id}_{filename}")
        
        with open(final_path, 'wb') as output_file:
            for i in range(total_chunks):
                chunk_path = os.path.join(temp_dir, f"chunk_{i}")
                if os.path.exists(chunk_path):
                    with open(chunk_path, 'rb') as chunk_file:
                        output_file.write(chunk_file.read())
                else:
                    raise ValueError(f"Missing chunk {i}")
        
        # Clean up chunks
        import shutil
        shutil.rmtree(temp_dir)
        
        logger.info(f"Reassembled file: {final_path}")
        return final_path


def create_memory_efficient_response(data: bytes, mimetype: str = 'application/octet-stream',
                                   filename: str = None, chunk_size: int = 1024 * 1024) -> Response:
    """
    Create a memory-efficient response for large data
    
    Args:
        data: Data to send
        mimetype: MIME type
        filename: Filename for download
        chunk_size: Chunk size for streaming
        
    Returns:
        Streaming response
    """
    def generate_chunks():
        for i in range(0, len(data), chunk_size):
            yield data[i:i + chunk_size]
    
    response = Response(generate_chunks(), mimetype=mimetype)
    
    if filename:
        response.headers['Content-Disposition'] = f'attachment; filename="{filename}"'
    
    response.headers['Content-Length'] = str(len(data))
    response.headers['Cache-Control'] = 'no-cache'
    
    return response


# Global streaming handler instance
streaming_handler = StreamingDownloadHandler()
chunked_upload_handler = None  # Will be initialized with upload directory