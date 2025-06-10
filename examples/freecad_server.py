import socket
import sys
import io
import traceback
import threading
import queue
from PySide2 import QtCore

HOST = "localhost"
PORT = 12345
BUFFER_SIZE = 4096

# Queue for GUI tasks
gui_task_queue = queue.Queue()
result_queues = {}  # Store result queues by task ID

def process_gui_queue():
    """Process tasks in the GUI thread."""
    try:
        while not gui_task_queue.empty():
            task_id, fn = gui_task_queue.get_nowait()
            try:
                fn()
                # Force GUI update
                QtCore.QCoreApplication.processEvents()
                # Signal completion
                if task_id in result_queues:
                    result_queues[task_id].put("SUCCESS")
            except Exception as e:
                error_msg = f"[FreeCAD GUI Queue] Error: {str(e)}\n{traceback.format_exc()}"
                print(error_msg)
                if task_id in result_queues:
                    result_queues[task_id].put(error_msg)
            finally:
                if task_id in result_queues:
                    del result_queues[task_id]
    except Exception as e:
        print(f"[FreeCAD GUI Queue] Critical error: {e}")

# Set up a timer to run in the GUI thread
timer = QtCore.QTimer()
timer.timeout.connect(process_gui_queue)
timer.start(50)  # Check every 50ms for better responsiveness
print("[FreeCAD] GUI queue processor started.")

def execute_in_gui_thread(code):
    """Execute code in the GUI thread and wait for completion."""
    task_id = id(code)  # Use code object id as task identifier
    result_queue = queue.Queue()
    result_queues[task_id] = result_queue
    
    def execute():
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        try:
            sys_stdout_orig = sys.stdout
            sys_stderr_orig = sys.stderr
            sys.stdout = stdout_buf
            sys.stderr = stderr_buf
            
            # Execute the code
            exec(code, globals(), locals())
            
            # Force document recompute
            if hasattr(FreeCAD, 'ActiveDocument') and FreeCAD.ActiveDocument:
                FreeCAD.ActiveDocument.recompute()
                # Force GUI update
                QtCore.QCoreApplication.processEvents()
                
        except Exception as e:
            traceback.print_exc(file=stderr_buf)
        finally:
            sys.stdout = sys_stdout_orig
            sys.stderr = sys_stderr_orig
            
        output = stdout_buf.getvalue()
        errors = stderr_buf.getvalue()
        return output + errors
    
    # Add task to GUI queue
    gui_task_queue.put((task_id, execute))
    
    # Wait for result with timeout
    try:
        result = result_queue.get(timeout=500.0)  # Increased timeout
        if result == "SUCCESS":
            return "Command executed successfully"
        return result
    except queue.Empty:
        return "[ERROR] Execution timed out after 30 seconds"
    finally:
        if task_id in result_queues:
            del result_queues[task_id]

def handle_client(conn):
    """Handle client connection and execute received code."""
    try:
        conn.settimeout(35.0)  # Increased timeout to match execution timeout
        code = conn.recv(BUFFER_SIZE).decode()
        print("[FreeCAD] Received code block...")
        
        # Execute code in GUI thread
        result = execute_in_gui_thread(code)
        
        # Send result back to client
        try:
            conn.sendall(result.encode())
        except Exception as e:
            print(f"[FreeCAD] Failed to send response: {e}")
            
    except Exception as e:
        print(f"[FreeCAD] Error handling client: {e}")
    finally:
        conn.close()

def server_loop():
    """Main server loop."""
    print(f"[FreeCAD] Starting socket server on {HOST}:{PORT}")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_sock:
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind((HOST, PORT))
        server_sock.listen(1)
        print("[FreeCAD] Listening for connections...")

        while True:
            try:
                conn, addr = server_sock.accept()
                print(f"[FreeCAD] Connection from {addr}")
                handle_client(conn)
            except Exception as e:
                print(f"[FreeCAD] Error accepting connection: {e}")

def start_custom_server():
    """Start the server in a background thread."""
    print("[FreeCAD] Launching server in background thread")
    thread = threading.Thread(target=server_loop, daemon=True)
    thread.start()

if __name__ == "__main__":
    start_custom_server()