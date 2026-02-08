# app/api/endpoints/evolver.py
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Depends
from fastapi.responses import FileResponse # For downloads
# NEW: Import BaseModel for response model
from pydantic import BaseModel
from typing import Annotated, Optional
import json
from app.models.common import TaskResponse, TaskStatus
from app.models.evolver import EvolverConfig
from tasks.evolution_tasks import run_evolution_task
from celery.result import AsyncResult
from app.core.celery_app import celery_app
from app.core.config import settings
import os
import uuid
import logging
from werkzeug.utils import secure_filename
import aioredis

logger = logging.getLogger(__name__)

os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.RESULT_DIR, exist_ok=True)

router = APIRouter()

# --- Helper for Secure Filename (Unchanged) ---
def generate_secure_path(original_filename: str, sub_dir: str = "") -> tuple[str, str]:
    if not original_filename: raise ValueError("Original filename cannot be empty.")
    base_name = secure_filename(original_filename)
    if not base_name: base_name = "unnamed_file"
    unique_prefix = str(uuid.uuid4())[:8]
    secure_name = f"{unique_prefix}_{base_name}"
    target_dir = os.path.join(settings.UPLOAD_DIR, sub_dir)
    os.makedirs(target_dir, exist_ok=True)
    full_path = os.path.join(target_dir, secure_name)
    # Check for collisions - less likely but good practice
    attempts = 0
    while os.path.exists(full_path) and attempts < 5:
        unique_prefix = str(uuid.uuid4())[:8]
        secure_name = f"{unique_prefix}_{base_name}"
        full_path = os.path.join(target_dir, secure_name)
        attempts += 1
    if os.path.exists(full_path): # Extremely unlikely after 5 attempts
        raise IOError(f"Could not generate unique filename for {original_filename} after multiple attempts.")
    logger.debug(f"Generated secure path: {full_path} for original: {original_filename}")
    return full_path, secure_name

# --- NEW: Response Model for Termination ---
class TerminateResponse(BaseModel):
    message: str
    task_id: str
# --- End New Model ---

@router.post("/start", response_model=TaskResponse)
async def start_evolution(
    model_definition: Annotated[UploadFile, File(...)],
    config_json: Annotated[str, Form(...)],
    task_evaluation: Annotated[Optional[UploadFile], File()] = None,
    initial_weights: Annotated[Optional[UploadFile], File()] = None,
    use_standard_eval: Annotated[bool, Form(...)] = False,
):
    logger.info("Received request to start evolution task.")
    logger.debug(f"Backend Received Raw config_json snippet: {config_json[:200]}...") # Avoid logging huge configs

    # --- Config Validation (Unchanged) ---
    try:
        config_data = json.loads(config_json)
        # Validate against the Pydantic model (this includes nsga2_enabled, use_fuzzy, etc.)
        config = EvolverConfig.model_validate(config_data)
        logger.info("Configuration JSON validated successfully.")
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON config received.", exc_info=True)
        raise HTTPException(status_code=400, detail="Invalid JSON configuration string.")
    except Exception as e:
        logger.error(f"Invalid config data structure: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Invalid configuration: {e}")

    # --- Evaluation Script Validation (Unchanged) ---
    if not use_standard_eval and task_evaluation is None:
        logger.error("Custom eval requested but no file provided.")
        raise HTTPException(status_code=400, detail="Custom evaluation script required if not using standard.")
    
    if use_standard_eval and task_evaluation is not None:
        logger.warning("Standard eval requested but custom file also provided. Custom file will be ignored.")
        task_evaluation = None

    # --- Securely save uploaded files (Unchanged logic) ---
    saved_files = {}
    model_def_path: str | None = None
    eval_path: str | None = None
    weights_path: str | None = None

    try:
        # 1. Save Model Definition
        if model_definition and model_definition.filename:
            model_def_path, _ = generate_secure_path(model_definition.filename, "model_defs")
            logger.info(f"Saving model definition to: {model_def_path}")
            try:
                await model_definition.seek(0)
                contents = await model_definition.read()
                if not contents: 
                    raise ValueError("Model definition file is empty.")
                with open(model_def_path, "wb") as f: 
                    f.write(contents)
                saved_files['model_def'] = model_def_path
            finally: 
                await model_definition.close()
        else: 
            raise HTTPException(status_code=400, detail="Model definition file or filename missing.")

        # 2. Save Custom Evaluation Script
        if task_evaluation and not use_standard_eval and task_evaluation.filename:
            eval_path, _ = generate_secure_path(task_evaluation.filename, "eval_scripts")
            logger.info(f"Saving custom eval script to: {eval_path}")
            try:
                await task_evaluation.seek(0)
                contents = await task_evaluation.read()
                if not contents: 
                    raise ValueError("Custom evaluation file is empty.")
                with open(eval_path, "wb") as f: 
                    f.write(contents)
                saved_files['eval'] = eval_path
            finally: 
                await task_evaluation.close()
        elif task_evaluation and not use_standard_eval and not task_evaluation.filename:
            logger.warning("Custom eval file object provided but filename missing.")
            raise HTTPException(status_code=400, detail="Custom evaluation filename missing.")

        # 3. Save Initial Weights
        if initial_weights and initial_weights.filename:
            weights_path, _ = generate_secure_path(initial_weights.filename, "weights")
            logger.info(f"Saving initial weights to: {weights_path}")
            try:
                await initial_weights.seek(0)
                contents = await initial_weights.read()
                if not contents: 
                    raise ValueError("Initial weights file is empty.")
                with open(weights_path, "wb") as f: 
                    f.write(contents)
                saved_files['weights'] = weights_path
            finally: 
                await initial_weights.close()
        elif initial_weights and not initial_weights.filename:
             logger.warning("Initial weights file provided but filename missing. Skipping.")

    except HTTPException as http_exc:
        # Cleanup on known errors
        for file_path in saved_files.values():
            if file_path and os.path.exists(file_path): 
                try: os.remove(file_path) 
                except OSError: pass
        raise http_exc
    except Exception as e:
        logger.error(f"Unexpected error during file processing: {e}", exc_info=True)
        # Cleanup on unexpected errors
        for file_path in saved_files.values():
             if file_path and os.path.exists(file_path): 
                try: os.remove(file_path) 
                except OSError: pass
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded files.")

    # --- Launch Celery task ---
    logger.info("Dispatching evolution task to Celery...")
    try:
        if not model_def_path: # Final safety check
            raise RuntimeError("Model definition path missing before dispatch.")

        # Convert the Pydantic model back to a dictionary to send to Celery
        # This includes any new fields like nsga2_enabled or num_fuzzy_params
        config_to_send = config.model_dump() 

        task = run_evolution_task.delay(
            model_definition_path=model_def_path,
            task_evaluation_path=eval_path,
            use_standard_eval=use_standard_eval,
            initial_weights_path=weights_path,
            config=config_to_send 
        )
        
        logger.info(f"Task dispatched with ID: {task.id}")
        return TaskResponse(task_id=task.id, status="PENDING")

    except Exception as dispatch_err:
         logger.error(f"Celery task dispatch failed: {dispatch_err}", exc_info=True)
         # Final cleanup if dispatch fails
         for file_path in saved_files.values():
             if file_path and os.path.exists(file_path): 
                try: os.remove(file_path) 
                except OSError: pass
         raise HTTPException(status_code=500, detail=f"Failed to start evolution task: {dispatch_err}")

# --- get_evolution_status (Unchanged) ---
@router.get("/status/{task_id}", response_model=TaskStatus)
async def get_evolution_status(task_id: str):
    """
    Fetches the current status of the evolution task.
    Updated to handle Multi-Objective (NSGA-II) progress data.
    """
    task_result = AsyncResult(task_id, app=celery_app)
    
    # Optional: Log the status check
    # logger.debug(f"Checking status for Task ID: {task_id}. Current state: {task_result.state}")

    if task_result.state == 'PENDING':
        return TaskStatus(task_id=task_id, status='PENDING', message="Task is in queue...")

    elif task_result.state == 'PROGRESS':
        info = task_result.info or {}
        
        # --- NSGA-II Multi-Objective Mapping ---
        # The worker now sends 'avg_obj_scores' (e.g., [0.95, 0.88, 15.2])
        # and 'generation' / 'total_generations'
        current_gen = info.get('generation', 0)
        total_gens = info.get('total_generations', 1)
        
        # Calculate a safe progress float for the frontend progress bar
        progress_percentage = (current_gen / total_gens) if total_gens > 0 else 0.0

        return TaskStatus(
            task_id=task_id,
            status='PROGRESS',
            progress=progress_percentage,
            message=info.get('message', f"Evolving Generation {current_gen}..."),
            # 'info' contains the 'avg_obj_scores' and 'diversity' for the frontend charts
            info=info 
        )

    elif task_result.state == 'SUCCESS':
        logger.info(f"Task {task_id} completed successfully.")
        return TaskStatus(
            task_id=task_id, 
            status='SUCCESS', 
            result=task_result.result,
            progress=1.0
        )

    elif task_result.state == 'FAILURE':
        error_msg = str(task_result.info)
        logger.error(f"Task {task_id} failed: {error_msg}")
        return TaskStatus(task_id=task_id, status='FAILURE', message="Evolution failed.", result={"error": error_msg})

    # For REVOKED or other states
    return TaskStatus(task_id=task_id, status=task_result.state, message=f"Task state: {task_result.state}")


# --- NEW: Redis-based Cooperative Halt Endpoint ---
@router.post("/tasks/{task_id}/terminate", response_model=TerminateResponse)
async def terminate_evolution_task_endpoint(task_id: str):
    """
    Requests graceful termination of a running evolution Celery task by setting a Redis halt flag.
    """
    logger.info(f"Received termination request for task ID: {task_id}")

    # Get the RedisDsn object from settings
    redis_url_setting = settings.REDIS_URL
    # --- FIX: Convert RedisDsn object to string ---
    redis_url_str = str(redis_url_setting)
    # --- End FIX ---

    redis = None # Initialize redis connection variable
    try:
        # Use the string URL to connect
        logger.debug(f"Connecting to Redis at: {redis_url_str}")
        redis = await aioredis.from_url(redis_url_str)
    except Exception as e:
        logger.error(f"Failed to connect to Redis at {redis_url_str}: {e}", exc_info=True)
        # Raise HTTPException if Redis connection fails, as halt cannot be set
        raise HTTPException(status_code=500, detail="Internal server error: Redis connection failed.")

    halt_key = f"task:halt:{task_id}"
    try:
        # Set the halt flag with an expiration (e.g., 1 hour)
        # Use await for aioredis v2+
        await redis.set(halt_key, "1", ex=3600)
        logger.info(f"Set halt flag in Redis for task {task_id} (key: {halt_key})")
        # Close the Redis connection pool when done
        await redis.close()
    except Exception as e:
        logger.error(f"Failed to set halt flag in Redis for task {task_id}: {e}", exc_info=True)
        if redis: await redis.close() # Ensure connection is closed on error too
        raise HTTPException(status_code=500, detail="Internal server error: Failed to set halt flag.")

    return TerminateResponse(
        message=f"Termination requested for task {task_id}. Task will halt shortly if it checks the flag.",
        task_id=task_id
    )



# --- download_evolution_result (Unchanged) ---
@router.get("/results/{task_id}/download")
async def download_evolution_result(task_id: str):
    """ Downloads the final evolved model (.pth) for a completed task. """
    logger.info(f"Received download request for task: {task_id}")
    try:
        task_result = AsyncResult(task_id, app=celery_app)
        if task_result is None: raise HTTPException(status_code=404, detail=f"Task ID {task_id} not found.")
        if task_result.status != 'SUCCESS': raise HTTPException(status_code=400, detail=f"Task {task_id} not completed successfully (Status: {task_result.status}).")
        result_data = task_result.result
        if not isinstance(result_data, dict) or 'final_model_path' not in result_data or not result_data['final_model_path']:
            logger.error(f"Download failed: Task {task_id} succeeded but missing 'final_model_path'. Result: {result_data}")
            raise HTTPException(status_code=500, detail="Result file path missing.")
        relative_file_path = result_data['final_model_path']
        # Use os.path.basename to prevent directory traversal via relative_file_path
        secure_basename = os.path.basename(relative_file_path)
        abs_file_path = os.path.abspath(os.path.join(settings.RESULT_DIR, secure_basename))
        abs_result_dir = os.path.abspath(settings.RESULT_DIR)
        # Ensure the final path is still within the RESULT_DIR
        if not abs_file_path.startswith(abs_result_dir + os.sep):
             logger.error(f"Security Violation: Invalid path construction. Path: {abs_file_path}")
             raise HTTPException(status_code=403, detail="Access denied.")
        if not os.path.exists(abs_file_path) or not os.path.isfile(abs_file_path):
            logger.error(f"Download failed: File not found at: {abs_file_path}")
            raise HTTPException(status_code=404, detail="Result file not found.")
        filename = os.path.basename(abs_file_path) # Use final safe basename
        logger.info(f"Sending file: {abs_file_path} as download: {filename}")
        return FileResponse(path=abs_file_path, filename=filename, media_type='application/octet-stream')
    except HTTPException as http_exc: raise http_exc
    except Exception as e:
        logger.error(f"Error processing download for task {task_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Could not process download: {e}")
