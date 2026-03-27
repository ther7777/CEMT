"""Isolated reward scoring server.

This FastAPI app runs a "clean-before-run" workflow: before handling each request,
it aggressively terminates known GPU-occupying worker processes to ensure exclusive
resource usage during scoring.
"""
import uvicorn
import logging
import argparse
import json
import subprocess
import tempfile
import os
import time
import uuid
import socket
import shutil
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from pathlib import Path
import psutil 

# Logging / app setup.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SafeRewardServer")
app = FastAPI(title="Safe Decoupled Reward Server")

# Request schema.
class RewardRequest(BaseModel):
    sources: List[str]
    mts: List[str]
    references: List[str] = None

# Globals.
MODEL_NAME = None
LOCK_FILE = Path("/tmp/comet_is_busy.lock")
OCCUPANCY_SCRIPT_NAME = "base.py"

def find_and_terminate_process(name: str):
    """
    Find Python processes whose cmdline contains the given name and kill them.

    This is intentionally aggressive to prevent GPU/worker contention.
    """
    procs_to_terminate = []
    # 1) Find target processes
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'python' in proc.info['name'].lower() and name in " ".join(proc.info['cmdline']):
                procs_to_terminate.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    
    if not procs_to_terminate:
        logger.info("No occupancy processes found; nothing to clean.")
        return

    logger.warning(f"Detected {len(procs_to_terminate)} occupancy processes; sending KILL...")
    
    # 2) Kill
    for proc in procs_to_terminate:
        try:
            proc.kill()
        except psutil.NoSuchProcess:
            continue
    
    # 3) Wait for exit (ensures resources are actually released)
    gone, alive = psutil.wait_procs(procs_to_terminate, timeout=10)
    
    if alive:
        logger.error(f"Critical: {len(alive)} processes are still alive after KILL. Check permissions.")

    logger.info("Occupancy processes cleaned up.")


@app.post("/predict", response_model=Dict[str, List[float]])
async def predict(request: RewardRequest):
    # Create a globally unique request id for traceable logs.
    hostname = socket.gethostname()
    timestamp = int(time.time() * 1000000)  # microsecond timestamp
    unique_id = str(uuid.uuid4())[:8]
    request_id = f"{hostname}_{timestamp}_{unique_id}"
    
    logger.info(f"[{request_id}] Start request. Batch size: {len(request.sources)}")
    
    # Kept for cleanup in finally.
    temp_dir = None
    
    try:
        # Lock + pre-clean.
        find_and_terminate_process(OCCUPANCY_SCRIPT_NAME)
        LOCK_FILE.touch()
        logger.info("Lock file created; server is busy.")

        # Use a unique temp directory per request.
        temp_base_dir = "/tmp"
        temp_dir_name = f"comet_{MODEL_NAME}_{request_id}"
        temp_dir = os.path.join(temp_base_dir, temp_dir_name)
        
        # Extra safety: ensure unique name even if a stale dir exists.
        if os.path.exists(temp_dir):
            logger.warning(f"[{request_id}] Temp dir already exists; adding PID suffix")
            temp_dir = f"{temp_dir}_{os.getpid()}"
        
        os.makedirs(temp_dir, exist_ok=False)
        logger.info(f"[{request_id}] Created unique temp dir: {temp_dir}")

        # Unique file names.
        input_path = os.path.join(temp_dir, f"input_{request_id}.jsonl")
        output_path = os.path.join(temp_dir, f"output_{request_id}.json")
        
        logger.info(f"[{request_id}] Input file: {input_path}")
        logger.info(f"[{request_id}] Output file: {output_path}")

        # Write input JSONL.
        try:
            with open(input_path, 'w', encoding='utf-8') as f:
                for src, mt, ref in zip(request.sources, request.mts, request.references or [None]*len(request.sources)):
                    line_data = {"src": src, "mt": mt, "ref": ref}
                    f.write(json.dumps(line_data) + '\n')
            
            # Sanity-check.
            if not os.path.exists(input_path):
                raise Exception(f"Failed to create input file: {input_path}")
            
            file_size = os.path.getsize(input_path)
            logger.info(f"[{request_id}] Input file created. Size: {file_size} bytes")
            
        except Exception as e:
            logger.error(f"[{request_id}] Failed to create input file: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to create input file: {e}")
        
        script_path = os.path.join(os.path.dirname(__file__), "score_worker.py")
        
        # Pass request-id to the worker for better traceability.
        cmd = [
            "python3", script_path, 
            "--model-name", MODEL_NAME, 
            "--input-file", input_path, 
            "--output-file", output_path,
            "--request-id", request_id  # 新增请求ID参数
        ]
        
        # Ensure the subprocess uses this temp directory.
        env = os.environ.copy()
        env['COMET_REQUEST_ID'] = request_id
        env['COMET_TEMP_DIR'] = temp_dir
        env['TMPDIR'] = temp_dir
        env['TEMP'] = temp_dir
        env['TMP'] = temp_dir
        
        logger.info(f"[{request_id}] Command: {' '.join(cmd)}")
        
        # Run the scoring worker.
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=1800, env=env)
            logger.info(f"[{request_id}] Worker process finished successfully.")
            if result.stdout:
                logger.info(f"[{request_id}] Worker stdout: {result.stdout}")
        except subprocess.CalledProcessError as e:
            logger.error(f"[{request_id}] Worker process failed with return code {e.returncode}.")
            logger.error(f"[{request_id}] --- Worker Stderr Log ---\n{e.stderr}\n--- End of Worker Stderr Log ---")
            raise HTTPException(status_code=500, detail=f"Worker process failed. Check server logs for details.")
        except subprocess.TimeoutExpired as e:
            logger.error(f"[{request_id}] Worker process timed out.")
            logger.error(f"[{request_id}] --- Worker Stderr on Timeout ---\n{e.stderr}\n--- End of Stderr ---")
            raise HTTPException(status_code=500, detail="Worker process timed out after 30 minutes.")

        # Validate output file.
        if not os.path.exists(output_path):
            raise Exception(f"Output file not created: {output_path}")
        
        output_size = os.path.getsize(output_path)
        if output_size == 0:
            raise Exception(f"Output file is empty: {output_path}")
        
        logger.info(f"[{request_id}] Output validated. Size: {output_size} bytes")

        with open(output_path, 'r', encoding='utf-8') as f:
            scores_data = json.load(f)
        
        logger.info(f"[{request_id}] Loaded scores")
        return scores_data
            
    except Exception as e:
        if not isinstance(e, HTTPException):
            logger.error(f"[{request_id}] An unexpected error occurred in the server: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"An unexpected server error occurred: {str(e)}")
        else:
            raise e

    finally:
        # Cleanup.
        if temp_dir and os.path.exists(temp_dir):
            try:
                files_to_delete = []
                for root, dirs, files in os.walk(temp_dir):
                    files_to_delete.extend([os.path.join(root, f) for f in files])
                
                logger.info(f"[{request_id}] Cleaning temp dir with {len(files_to_delete)} files")
                shutil.rmtree(temp_dir)
                logger.info(f"[{request_id}] Temp dir removed: {temp_dir}")
                
            except Exception as cleanup_error:
                logger.error(f"[{request_id}] Failed to clean temp dir: {cleanup_error}")
                logger.error(f"[{request_id}] Unremoved temp dir: {temp_dir}")
        
        # Remove lock.
        if LOCK_FILE.exists():
            LOCK_FILE.unlink()
            logger.info("Lock file removed; server is idle.")

def main():
    global MODEL_NAME
    parser = argparse.ArgumentParser(description="Run the Safe Decoupled Reward Server.")
    parser.add_argument("--model-name", type=str, required=True, choices=["XCOMET", "CometKiwi"])
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    
    MODEL_NAME = args.model_name
    
    logger.info(f"Starting SAFE server for model '{MODEL_NAME}' on port {args.port}...")
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()