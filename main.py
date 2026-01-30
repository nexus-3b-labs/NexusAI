"""
Backend for 3B Model Training and Inference.
This implementation provides the logic for local model management,
real-time training metrics, and human-in-the-loop scoring.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import time
import random
import uuid
import threading
import os
import gc
import json

app = FastAPI(title="Nexus 3B Backend")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Internal State ---

class EngineState:
    def __init__(self):
        self.is_training = False
        self.metrics = {"loss": 0.0, "step": 0, "perplexity": 0.0}
        self.knowledge_base = []
        self.feedback_log = []
        # Model state
        self.model = None
        self.tokenizer = None
        self.model_name = None
        self.adapter_loaded = False
        self.active_adapter = None
        self.adapter_supports_thinking = False  # True if adapter was trained with <think> tags
        # Default system prompt
        self.system_prompt = "You are a helpful AI assistant."

        
        # Loading state
        self.loading_status = "idle" # idle, downloading, loading, ready, error
        self.loading_progress = 0.0 # 0-100
        self.loading_error = ""
        
        # Settings
        # Default to 'models' directory in the current working directory
        self.cache_dir = os.path.join(os.getcwd(), "models")
        
        # Inference Parameters
        self.params = {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_new_tokens": 1024,
            "top_k": 50,
            "repetition_penalty": 1.1,
            "min_new_tokens": 0,
        }

state = EngineState()

# --- Schemas ---

class ChatRequest(BaseModel):
    message: str
    enable_thinking: bool = True

class ScoreRequest(BaseModel):
    message_id: str
    score: int
    prompt: str = ""
    response: str = ""

@app.post("/v1/score")
async def submit_score(request: ScoreRequest):
    """
    Records human feedback. 
    If score >= 7 and prompt/response are provided, appends to training_data.jsonl.
    """
    print(f"DEBUG SCORE REQUEST: {request}")
    # Log the raw feedback
    state.feedback_log.append({
        "id": request.message_id,
        "score": request.score,
        "timestamp": time.time()
    })
    
    saved = False
    if request.score >= 7 and request.prompt and request.response:
        entry = {
            "prompt": request.prompt,
            "response": request.response,
            "score": request.score,
            "source": "human_feedback"
        }
        try:
            data_file = os.path.join(_project_root(), "training_data.jsonl")
            with open(data_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
            saved = True
        except Exception as e:
            print(f"Failed to save training data: {e}")
            
    return {"status": "Score accepted", "saved_for_training": saved}

class TrainingDataRequest(BaseModel):
    data: str

@app.post("/v1/training/data")
async def upload_training_data(request: TrainingDataRequest):
    """Appends valid JSONL data to the training file."""
    lines = request.data.strip().split("\n")
    added_count = 0
    data_file = os.path.join(_project_root(), "training_data.jsonl")
    with open(data_file, "a") as f:
        for line in lines:
            line = line.strip()
            if not line: continue
            try:
                # Validate JSON structure
                entry = json.loads(line)
                if "prompt" in entry and "response" in entry:
                    # Ensure score/source exist or add defaults
                    if "score" not in entry: entry["score"] = 10
                    if "source" not in entry: entry["source"] = "manual_upload"
                    
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    added_count += 1
            except json.JSONDecodeError:
                pass # Skip invalid lines
                
    return {"status": "Data uploaded", "added": added_count}

class LoadModelRequest(BaseModel):
    model_id: str

class SettingsRequest(BaseModel):
    cache_dir: str

class ModelParamsRequest(BaseModel):
    temperature: float
    top_p: float
    max_new_tokens: int
    top_k: int = 50
    repetition_penalty: float = 1.1
    min_new_tokens: int = 0

class LoadAdapterRequest(BaseModel):
    system_prompt: str = ""
    adapter_name: str = ""
    supports_thinking: bool = False  # True if adapter was trained with <think> format

@app.post("/v1/adapter/load")
async def load_adapter_handler(request: LoadAdapterRequest):
    """Loads a fine-tuned adapter. Supports nested structure."""
    if not state.model:
        raise HTTPException(status_code=400, detail="Base model not loaded.")
    
    # 1. Determine Model Safe Name
    model_safe = state.model_name.replace("/", "--")
    
    # 2. Determine Adapter Path (under project root for consistency with training output)
    adapters_root = os.path.join(_project_root(), "nexus_adapters")
    if request.adapter_name:
        adapter_path = os.path.join(adapters_root, model_safe, request.adapter_name)
    else:
        adapter_path = adapters_root

    if not os.path.exists(adapter_path):
        raise HTTPException(status_code=404, detail=f"Adapter '{request.adapter_name}' not found at {adapter_path}.")

    try:
        print(f"Loading Adapter from {adapter_path}...")
        # is_trainable=False for inference-only; keeps adapter frozen and avoids training state
        state.model = PeftModel.from_pretrained(
            state.model,
            adapter_path,
            is_trainable=False,
        )
        state.model.eval()
        state.adapter_loaded = True
        state.active_adapter = request.adapter_name or "Legacy Adapter"
        state.adapter_supports_thinking = request.supports_thinking
        state.system_prompt = request.system_prompt if request.system_prompt else "You are a helpful AI assistant."
        print(f"Adapter loaded. System Prompt: {state.system_prompt}, Supports Thinking: {state.adapter_supports_thinking}")
        return {"status": "Adapter loaded", "adapter": state.active_adapter, "supports_thinking": state.adapter_supports_thinking}
    except Exception as e:
        print(f"Error loading adapter: {e}")
        state.adapter_loaded = False
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/adapter/list")
async def list_adapters():
    """Lists available adapters for the CURRENTLY LOADED model."""
    if not state.model_name:
        return {"adapters": []}
    
    print(f"model_name: {state.model_name}")
    model_safe = state.model_name.replace("/", "--")
    base_path = os.path.join(_project_root(), "nexus_adapters", model_safe)
    print(f"base_path: {base_path}")
    adapters = []
    if os.path.exists(base_path):
        try:
             adapters = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
        except Exception as e:
             print(f"Error listing adapters: {e}")
        
    return {"adapters": adapters}



# --- Logic: Training Loop ---

# Import the training module (Make sure dependencies are installed)
try:
    import train as train_module
except ImportError:
    train_module = None

import subprocess
import sys

def _project_root():
    """Project root: directory containing main.py. Used for absolute data/cwd paths."""
    return os.path.dirname(os.path.abspath(__file__))


def real_training_loop(model_id, adapter_name="nexus_adapter", data_path=None, cache_dir=None):
    """
    Executes the real training process in a separate SUBPROCESS.
    This ensures proper memory reclamation (OS kills the process).
    """
    state.is_training = True
    state.metrics = {"loss": 0.0, "step": 0, "perplexity": 0.0, "status": "Starting..."}
    
    print(f"Starting Real Training (Subprocess) for {model_id} -> {adapter_name}...")
    
    project_root = _project_root()
    data_path = data_path or os.path.join(project_root, "training_data.jsonl")
    data_path = os.path.abspath(data_path)
    cache_dir = cache_dir or state.cache_dir
    cache_dir = os.path.abspath(cache_dir) if cache_dir else None

    # OUTPUT STRUCTURE: nexus_adapters/<model_name_safe>/<adapter_name>
    # Sanitize model_id (e.g. "stabilityai/stablelm-zephyr-3b" -> "stabilityai--stablelm-zephyr-3b")
    model_name_safe = model_id.replace("/", "--")
    output_dir = os.path.join(project_root, "nexus_adapters", model_name_safe, adapter_name)
    
    # Run train.py as a separate process with explicit --data and --cache_dir
    cmd = [sys.executable, "train.py", "--model_id", model_id, "--output", output_dir, "--data", data_path]
    if cache_dir:
        cmd.extend(["--cache_dir", cache_dir])
    
    try:
        # Use Popen to stream stdout; cwd=project_root so paths are predictable
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, # Merge stderr to stdout for capturing errors too
            text=True,
            bufsize=1,
            cwd=project_root
        )
        
        # Stream output
        for line in process.stdout:
            line = line.strip()
            if not line: continue
            
            # Check for our special JSON marker
            if line.startswith("JSON_LOG:"):
                try:
                    json_str = line[len("JSON_LOG:"):]
                    logs = json.loads(json_str)
                    
                    if "loss" in logs:
                        state.metrics["loss"] = logs["loss"]
                        state.metrics["perplexity"] = 2.718 ** logs["loss"]
                    if "epoch" in logs:
                        state.metrics["step"] = logs["epoch"]
                        
                except:
                    pass
            # Also capture status from normal logs if useful, or just print them
            print(f"[TRAIN]: {line}")
            
            # Update status for user feedback based on simple keywords
            if "Loading Model" in line:
                 state.metrics["status"] = "Loading Model..."
            elif "Tokenizing" in line:
                 state.metrics["status"] = "Tokenizing..."
            elif "Saving adapters" in line:
                 state.metrics["status"] = "Saving Weights..."

        # Wait for finish
        ret_code = process.wait()
        
        if ret_code == 0:
            state.metrics["status"] = "Complete"
            print("Training subprocess finished successfully.")
        else:
            state.metrics["status"] = "Failed"
            print(f"Training subprocess failed with code {ret_code}")
            
    except Exception as e:
        print(f"Training launch failed: {e}")
        state.metrics["status"] = f"Error: {str(e)}"
    
    state.is_training = False

# --- Endpoints ---

def _background_model_load(model_id: str):
    """Background task to load model."""
    global state
    try:
        print(f"Starting background load for {model_id}")
        state.loading_status = "downloading"
        state.loading_progress = 10.0
        
        # Determine device first
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
            
        state.loading_progress = 20.0
        
        # Load Tokenizer
        print(f"Loading tokenizer (cache: {state.cache_dir})...")
        state.tokenizer = AutoTokenizer.from_pretrained(
            model_id, 
            cache_dir=state.cache_dir,
            trust_remote_code=True
        )
        state.loading_progress = 40.0
        
        # Load Model
        print(f"Loading model on {device} (cache: {state.cache_dir})...")
        state.loading_status = "loading"
        state.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            device_map=device,
            cache_dir=state.cache_dir,
            trust_remote_code=True
        )
        state.model_name = model_id
        
        state.loading_progress = 100.0
        state.loading_status = "ready"
        print(f"Model {model_id} ready.")
        
    except Exception as e:
        print(f"Error in background load: {e}")
        state.loading_status = "error"
        state.loading_error = str(e)

@app.post("/v1/model/load")
async def load_model_handler(request: LoadModelRequest, background_tasks: BackgroundTasks):
    """Triggers background model loading."""
    if state.loading_status in ["downloading", "loading"]:
        raise HTTPException(status_code=400, detail="Model already loading.")
    
    # Reset state
    state.loading_status = "starting"
    state.loading_progress = 0.0
    state.loading_error = ""
    
    background_tasks.add_task(_background_model_load, request.model_id)
    return {"status": "Loading started", "model": request.model_id}

@app.get("/v1/model/status")
async def get_model_status():
    return {
        "status": state.loading_status,
        "progress": state.loading_progress,
        "error": state.loading_error,
        "current_model": state.model_name,
        "active_adapter": state.active_adapter,
        "adapter_loaded": state.adapter_loaded,
        "adapter_supports_thinking": state.adapter_supports_thinking,
        # Thinking is supported if: no adapter loaded, OR adapter was trained with thinking
        "thinking_supported": not state.adapter_loaded or state.adapter_supports_thinking,
    }

@app.post("/v1/model/unload")
async def unload_model():
    """Unloads the model and frees memory."""
    if state.model is None:
        return {"status": "No model to unload"}
        
    print("Unloading model...")
    state.model = None
    state.tokenizer = None
    state.model_name = None
    state.loading_status = "idle"
    state.loading_progress = 0.0
    
    # Force Garbage Collection
    gc.collect()
    
    # Clear CUDA Cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("CUDA cache cleared.")
    
    # Clear MPS Cache (if applicable/available in future PyTorch versions)
    if torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
            print("MPS cache cleared.")
        except:
             pass 

    print("Model unloaded successfully.")
    print("Model unloaded successfully.")
    return {"status": "Model unloaded"}



@app.post("/v1/adapter/unload")
async def unload_adapter_handler():
    """Unloads the adapter and reverts to the base model for inference."""
    if not state.adapter_loaded or not state.model:
        return {"status": "No adapter active"}

    try:
        print("Unloading adapter...")
        # PEFT: unload() returns the base model with adapter removed. Use it so inference uses pure base weights.
        if hasattr(state.model, "unload"):
            base = state.model.unload()
            if base is not None:
                state.model = base
            else:
                # In-place unload: get the inner base transformer (e.g. base_model.model)
                state.model = getattr(
                    getattr(state.model, "base_model", state.model),
                    "model",
                    state.model,
                )
        state.model.eval()
        state.adapter_loaded = False
        state.active_adapter = None
        state.adapter_supports_thinking = False
        state.system_prompt = "You are a helpful AI assistant."
        print("Adapter unloaded. Reverted to base model.")
        return {"status": "Adapter unloaded"}
    except Exception as e:
        print(f"Error unloading adapter: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/settings/update")
async def update_settings(request: SettingsRequest):
    # If empty, revert to default 'models' directory
    default_path = os.path.join(os.getcwd(), "models")
    state.cache_dir = request.cache_dir if request.cache_dir.strip() else default_path
    print(f"Cache dir updated to: {state.cache_dir}")
    return {"status": "Settings updated", "cache_dir": state.cache_dir}

@app.get("/v1/settings/get")
async def get_settings():
    return {"cache_dir": state.cache_dir or ""}

@app.get("/v1/model/list")
async def list_models():
    """Lists downloaded models in the cache directory."""
    if not state.cache_dir or not os.path.exists(state.cache_dir):
        return {"models": []}
    
    models = []
    try:
        for folder in os.listdir(state.cache_dir):
            if folder.startswith("models--"):
                # Parse "models--org--repo" -> "org/repo"
                parts = folder.split("--")
                if len(parts) >= 3:
                    org = parts[1]
                    repo = parts[2]
                    models.append(f"{org}/{repo}")
    except Exception as e:
        print(f"Error listing models: {e}")
        
    return {"models": models}

@app.post("/v1/parameters/update")
async def update_parameters(request: ModelParamsRequest):
    state.params["temperature"] = request.temperature
    state.params["top_p"] = request.top_p
    state.params["max_new_tokens"] = request.max_new_tokens
    state.params["top_k"] = request.top_k
    state.params["repetition_penalty"] = request.repetition_penalty
    state.params["min_new_tokens"] = request.min_new_tokens
    print(f"Parameters updated: {state.params}")
    return {"status": "Parameters updated", "params": state.params}

@app.get("/v1/parameters/get")
async def get_parameters():
    return {"params": state.params}

@app.post("/v1/chat")
async def chat_handler(request: ChatRequest):
    """
    Handles inference. Uses local model if loaded, otherwise falls back to mock.
    """
    msg_id = str(uuid.uuid4())
    
    # 1. Fallback if no model loaded
    if not state.model or not state.tokenizer:
        return {
            "id": msg_id,
            "role": "assistant",
            "content": "System: No model loaded. Please load a model via /v1/model/load first. (Mock Mode Active)"
        }

    # 2. Real Inference
    try:
        print(f"Generating response for: {request.message}")
        
        # Use Chat Template if available
        if state.tokenizer.chat_template:
            messages = [{"role": "system", "content": state.system_prompt}]
            
            # Determine thinking mode based on adapter state:
            # 1. No adapter + thinking enabled → use Qwen native thinking
            # 2. Adapter with thinking support → let adapter handle it (no native thinking, no prompt modification)
            # 3. Adapter without thinking support → direct response only
            
            adapter_handles_thinking = state.adapter_loaded and state.adapter_supports_thinking
            use_native_thinking = request.enable_thinking and not state.adapter_loaded
            use_direct_response = state.adapter_loaded and not state.adapter_supports_thinking
            
            print(f"[DEBUG] adapter_loaded={state.adapter_loaded}, adapter_supports_thinking={state.adapter_supports_thinking}, "
                  f"request.enable_thinking={request.enable_thinking}, adapter_handles_thinking={adapter_handles_thinking}, "
                  f"use_native_thinking={use_native_thinking}, use_direct_response={use_direct_response}")
            
            if use_native_thinking:
                # No adapter: use Qwen's native thinking with prompt instructions
                messages[0]["content"] += "\n\nYou MUST begin by reasoning step-by-step inside <think>...</think> tags. Do NOT speak to the user inside the tags. usage: <think>internal thought</think> final response"
                messages.append({"role": "user", "content": "Hello"})
                messages.append({"role": "assistant", "content": "<think>The user is greeting me. I should respond in character.</think>Greetings. I am ready to assist."})
            elif adapter_handles_thinking:
                # Adapter trained with thinking: let it handle naturally, no modifications needed
                # The adapter learned <think>...</think>response format from training data
                pass
            elif use_direct_response:
                # Adapter without thinking support: force direct response
                messages[0]["content"] += "\n\nAnswer directly without showing your thinking process."
            else:
                # User disabled thinking, no adapter
                messages[0]["content"] += "\n\nAnswer directly without showing your thinking process."

            messages.append({"role": "user", "content": request.message})
            
            # Build kwargs for apply_chat_template
            chat_template_kwargs = {
                "return_tensors": "pt",
                "add_generation_prompt": True,
            }
            # Qwen3 native enable_thinking: only use when no adapter and user wants thinking
            # For adapters with thinking support, set False so adapter's trained format is used
            chat_template_kwargs["enable_thinking"] = use_native_thinking
            
            try:
                input_ids = state.tokenizer.apply_chat_template(messages, **chat_template_kwargs).to(state.model.device)
            except TypeError:
                # Tokenizer doesn't support enable_thinking param — use standard call
                input_ids = state.tokenizer.apply_chat_template(
                    messages, return_tensors="pt", add_generation_prompt=True
                ).to(state.model.device)
            # Explicit attention_mask (all 1s for single sequence) so the model doesn't warn when pad_token_id == eos_token_id
            attention_mask = input_ids.new_ones(input_ids.shape, dtype=torch.long)
        else:
            # Fallback for base models
            encoded = state.tokenizer(request.message, return_tensors="pt")
            input_ids = encoded.input_ids.to(state.model.device)
            attention_mask = encoded.attention_mask.to(state.model.device) if encoded.get("attention_mask") is not None else input_ids.new_ones(input_ids.shape, dtype=torch.long)
        
        gen_kwargs = {
            "max_new_tokens": state.params["max_new_tokens"],
            "do_sample": True,
            "temperature": state.params["temperature"],
            "top_p": state.params["top_p"],
            "repetition_penalty": state.params["repetition_penalty"],
            "min_new_tokens": state.params["min_new_tokens"],
            "pad_token_id": state.tokenizer.eos_token_id,
        }
        if state.params.get("top_k", 0) > 0:
            gen_kwargs["top_k"] = state.params["top_k"]
        outputs = state.model.generate(input_ids, attention_mask=attention_mask, **gen_kwargs)
        
        # Decode only the new tokens? 
        # Easier to decode all and strip the prompt if we know the length, 
        # or just decode the slice of new tokens:
        new_tokens = outputs[0][input_ids.shape[1]:]
        response_text = state.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        print(f"Response generated: {response_text}")

        return {
            "id": msg_id,
            "role": "assistant",
            "content": response_text
        }
    except Exception as e:
        print(f"Inference error: {e}")
        return {
            "id": msg_id,
            "role": "assistant",
            "content": f"Error during inference: {str(e)}"
        }

class StartTrainingRequest(BaseModel):
    model_id: str = ""
    adapter_name: str = "nexus_adapter"

@app.post("/v1/train/start")
async def start_training(request: StartTrainingRequest, background_tasks: BackgroundTasks):
    """Triggers the training lab process in the background."""
    if state.is_training:
        return {"status": "Already training"}
    
    # Use current loaded model if not specified, or default
    target_model = request.model_id or state.model_name or "stabilityai/stablelm-zephyr-3b"
    adapter_name = request.adapter_name or "nexus_adapter"
    
    data_path = os.path.join(_project_root(), "training_data.jsonl")
    background_tasks.add_task(real_training_loop, target_model, adapter_name, data_path, state.cache_dir)
    return {"status": "Training started", "model": target_model, "adapter": adapter_name}

@app.get("/v1/metrics")
async def get_metrics():
    """Returns the current state of the model for frontend visualization."""
    return {
        "is_training": state.is_training,
        "metrics": state.metrics
    }



if __name__ == "__main__":
    import uvicorn
    print("Nexus 3B Backend starting at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)