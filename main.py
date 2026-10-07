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
import asyncio
import shutil
import tempfile
import fnmatch
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import HfApi, get_token, login, logout, snapshot_download, try_to_load_from_cache
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

# Import the training module (Make sure dependencies are installed)
try:
    import train as train_module
except ImportError:
    train_module = None

TOTAL_EPOCHS = getattr(train_module, "NUM_EPOCHS", 3)
DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant."

app = FastAPI(title="Nexus 3B Backend")

# All model work (loading, generation, adapters) runs on this single thread. It keeps the
# event loop free for status/metrics polling and serialises access to the model, which is
# not thread-safe (PyTorch on MPS crashes when driven from several threads).
model_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="model")


def _on_model_thread(fn, *args, **kwargs):
    return asyncio.get_running_loop().run_in_executor(model_executor, lambda: fn(*args, **kwargs))

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
        self.metrics = _fresh_metrics("Idle")
        self.knowledge_base = []
        self.feedback_log = []
        # Model state
        self.model = None
        self.tokenizer = None
        self.model_name = None
        self.adapter_loaded = False
        self.active_adapter = None
        # True when the active adapter was trained on responses with their own <think> block
        self.adapter_supports_thinking = False
        # Default system prompt
        self.system_prompt = DEFAULT_SYSTEM_PROMPT

        
        # Loading state
        self.loading_status = "idle" # idle, starting, downloading, loading, ready, error
        self.loading_progress = 0.0 # 0-100
        self.loading_error = ""
        # Detail behind loading_status, so the UI can show what is actually happening
        self.load = _fresh_load()
        # Recent backend activity (model loading, training), newest last
        self.log = deque(maxlen=80)
        # Set by the cancel endpoint; the download loop checks it
        self.cancel_download = threading.Event()
        # Hugging Face token status, refreshed by _check_hf_token (the token is never stored here)
        self.hf_token_valid = False
        self.hf_username = None
        
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

def _detect_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# Steps of a model load, in order. loading_status stays coarse; the stage is precise.
LOAD_STAGES = ["checking", "downloading", "tokenizer", "weights", "device"]


def _fresh_load(model_id=None):
    return {
        "model": model_id,
        "stage": None,            # one of LOAD_STAGES while loading
        "stage_started": 0.0,
        "started": time.time(),
        "failed_stage": None,
        "download": {
            "needed": False, "offline": False, "downloaded": 0, "total": 0, "speed": 0.0, "eta": None,
            # True when the download is slow and no Hugging Face token is set
            "slow_without_token": False,
        },
    }


def _log(message):
    """Records a line in the activity log shown in the UI, and prints it."""
    print(message)
    state.log.append({"time": time.time(), "message": message})


def _set_stage(stage, progress, message):
    state.load["stage"] = stage
    state.load["stage_started"] = time.time()
    state.loading_progress = progress
    _log(message)


def _format_bytes(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit in ("B", "KB") else f"{n:.1f} {unit}"
        n /= 1024


def _dir_size(path, follow_links=True):
    """Bytes of the files under path, counting each underlying file once.

    The Hugging Face cache links files in two ways: snapshots/ links into the model's
    blobs/, and newer versions link those blobs into a store shared by the whole cache.
    follow_links=True resolves both, which gives a model's real size. follow_links=False
    counts only real files, which is what a whole-cache total needs.
    """
    total = 0
    seen = set()
    for root, _dirs, files in os.walk(path):
        for name in files:
            full = os.path.join(root, name)
            try:
                if not follow_links and os.path.islink(full):
                    continue
                st = os.stat(full)
            except OSError:
                continue  # broken link or file removed mid-walk
            key = (st.st_dev, st.st_ino)
            if key not in seen:
                seen.add(key)
                total += st.st_size
    return total


def _repo_cache_path(model_id):
    return os.path.join(state.cache_dir, "models--" + model_id.replace("/", "--"))


def _plan_download(model_id):
    """Works out which files a load needs and how many bytes are not in the cache yet.

    Returns (patterns, total_bytes, missing_bytes). Needs the network; callers fall
    back to the cache when it fails.
    """
    info = HfApi().model_info(model_id, files_metadata=True)
    files = [(f.rfilename, f.size or 0) for f in info.siblings]
    # Prefer safetensors; skip other weight formats (GGUF, ONNX, ...) we never load
    weights = "*.safetensors" if any(n.endswith(".safetensors") for n, _ in files) else "*.bin"
    patterns = ["*.json", "*.txt", "*.model", "*.jinja", "*.py", weights]
    wanted = [(n, size) for n, size in files if any(fnmatch.fnmatch(n, p) for p in patterns)]
    missing = [
        (n, size) for n, size in wanted
        if not isinstance(try_to_load_from_cache(model_id, n, cache_dir=state.cache_dir), str)
    ]
    return patterns, sum(size for _, size in wanted), sum(size for _, size in missing)


def _blob_links(repo_path):
    """Files outside a model's folder that its blobs link to (the cache's shared blob store)."""
    targets = set()
    blobs = os.path.join(repo_path, "blobs")
    if os.path.isdir(blobs):
        for name in os.listdir(blobs):
            full = os.path.join(blobs, name)
            if os.path.islink(full):
                targets.add(os.path.realpath(full))
    return targets


def _delete_model_files(model_id):
    """Removes a model's folder from the cache and returns the bytes freed.

    Newer huggingface_hub versions keep large files in a store shared by every model in
    the cache, so those are removed too, unless another model still links to them.
    """
    cache = os.path.realpath(state.cache_dir)
    repo = os.path.realpath(_repo_cache_path(model_id))
    # Only ever delete a model folder that sits directly inside the cache folder
    if os.path.dirname(repo) != cache or not os.path.basename(repo).startswith("models--") or not os.path.isdir(repo):
        raise FileNotFoundError(model_id)

    before = _dir_size(cache, follow_links=False)
    shared = {t for t in _blob_links(repo) if t.startswith(cache + os.sep)}
    shutil.rmtree(repo)

    still_used = set()
    for folder in os.listdir(cache):
        if folder.startswith("models--"):
            still_used |= _blob_links(os.path.join(cache, folder))
    for target in shared - still_used:
        try:
            os.remove(target)
        except OSError:
            pass
    return before - _dir_size(cache, follow_links=False)


# A download counts as slow below this average speed, once it has run long enough to judge
SLOW_DOWNLOAD_BYTES_PER_SEC = 1024 * 1024
SLOW_DOWNLOAD_AFTER_SEC = 15


def _slow_without_token(elapsed, speed, remaining):
    """Hugging Face throttles anonymous downloads harder, so a token is worth suggesting."""
    return (
        elapsed >= SLOW_DOWNLOAD_AFTER_SEC
        and speed < SLOW_DOWNLOAD_BYTES_PER_SEC
        and remaining > 0
        and not state.hf_token_valid
    )


def _check_hf_token():
    """Asks Hugging Face who the saved token belongs to. Updates and returns the token status.

    The token itself never leaves the backend: the UI only learns whether one is set,
    whether it works, and the account name.
    """
    token = get_token()
    username, valid = None, False
    if token:
        try:
            username = HfApi().whoami(token=token).get("name")
            valid = True
        except Exception:
            valid = False
    state.hf_token_valid = valid
    state.hf_username = username
    return _hf_token_status()


def _hf_token_status():
    return {
        "configured": get_token() is not None,
        "valid": state.hf_token_valid,
        "username": state.hf_username,
        # A token from the environment cannot be changed from the app
        "from_environment": bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")),
    }


class DownloadCancelled(Exception):
    """The user cancelled a model download."""


# Runs in its own process so a download can be stopped at any moment by ending the process.
_DOWNLOAD_SCRIPT = """
import json, sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], cache_dir=sys.argv[2], allow_patterns=json.loads(sys.argv[3]))
"""


def _download_with_progress(model_id, patterns, missing_bytes):
    """Downloads the model files in a child process, reporting bytes written to the cache.

    Raises DownloadCancelled if the user cancels.
    """
    # Measure the whole cache folder: partial files may land in the model's own folder
    # or in the cache's shared blob store, depending on the huggingface_hub version.
    def on_disk():
        return _dir_size(state.cache_dir, follow_links=False)

    baseline = on_disk()
    download = state.load["download"]
    download.update({"needed": True, "downloaded": 0, "total": missing_bytes, "speed": 0.0, "eta": None})
    started = time.time()

    with tempfile.TemporaryFile(mode="w+") as errors:
        process = subprocess.Popen(
            [sys.executable, "-c", _DOWNLOAD_SCRIPT, model_id, state.cache_dir, json.dumps(patterns)],
            stdout=subprocess.DEVNULL,
            stderr=errors,
        )
        try:
            while process.poll() is None:
                if state.cancel_download.wait(1.0):
                    raise DownloadCancelled()
                got = max(0, on_disk() - baseline)
                elapsed = time.time() - started
                # Files reach the disk in bursts, so the average since the start is the
                # steadier figure; an instantaneous rate swings between zero and huge.
                speed = got / elapsed if elapsed > 0 else 0.0
                download["downloaded"] = min(got, missing_bytes) if missing_bytes else got
                download["speed"] = speed
                remaining = missing_bytes - got
                # No estimate until there is enough data for it to mean something
                settled = elapsed >= 3 and got >= 0.01 * missing_bytes
                download["eta"] = remaining / speed if settled and speed > 0 and remaining > 0 else None
                if not download["slow_without_token"] and _slow_without_token(elapsed, speed, remaining):
                    download["slow_without_token"] = True
                    _log("This download is slow and no Hugging Face token is set. Adding one in Settings raises the rate limits.")
                if missing_bytes:
                    # Downloading spans 5% to 70% of the overall bar
                    state.loading_progress = 5.0 + 65.0 * min(1.0, got / missing_bytes)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()

        if process.returncode != 0:
            errors.seek(0)
            lines = [line.strip() for line in errors.read().splitlines() if line.strip()]
            raise RuntimeError(lines[-1] if lines else f"Download failed (exit code {process.returncode}).")

    download["downloaded"] = download["total"]
    download["speed"], download["eta"] = 0.0, None


def _fresh_metrics(status):
    return {
        "loss": 0.0, "step": 0, "perplexity": 0.0, "status": status,
        "progress": 0.0, "total_epochs": TOTAL_EPOCHS, "loss_history": [],
        "error": "", "model": None, "adapter": None,
    }


state = EngineState()

# --- Schemas ---

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    enable_thinking: bool = True
    # Earlier turns of the conversation (oldest first), so the model has context
    history: list[ChatMessage] = []

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

def _data_file():
    return os.path.join(_project_root(), "training_data.jsonl")


def _read_data_lines():
    path = _data_file()
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return f.read().splitlines()


@app.get("/v1/training/data")
async def get_training_data():
    """Lists every example in the dataset, newest first. `line` identifies an entry for deletion."""
    entries = []
    for line_no, line in enumerate(_read_data_lines()):
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(entry, dict) and entry.get("prompt") and entry.get("response"):
            score = entry.get("score", 0)
            entries.append({
                "line": line_no,
                "prompt": entry["prompt"],
                "response": entry["response"],
                "score": score,
                "source": entry.get("source", ""),
                # Same filter train.py applies
                "usable": score >= 7,
                "thinking": _has_thinking(entry["response"]),
            })
    eligible = sum(1 for e in entries if e["usable"])
    thinking = sum(1 for e in entries if e["usable"] and e["thinking"])
    return {"total": len(entries), "eligible": eligible, "thinking": thinking, "entries": entries[::-1]}


class DeleteExampleRequest(BaseModel):
    # The prompt the caller saw on that line; guards against deleting the wrong
    # entry if the file changed since it was listed.
    prompt: str


@app.post("/v1/training/data/{line_no}/delete")
async def delete_training_example(line_no: int, request: DeleteExampleRequest):
    """Removes one example from the dataset."""
    if state.is_training:
        raise HTTPException(status_code=409, detail="Training is running. Wait for it to finish before editing the dataset.")
    lines = _read_data_lines()
    try:
        entry = json.loads(lines[line_no]) if 0 <= line_no < len(lines) else None
    except json.JSONDecodeError:
        entry = None
    if not isinstance(entry, dict) or entry.get("prompt") != request.prompt:
        raise HTTPException(status_code=409, detail="The dataset changed. Refresh and try again.")
    del lines[line_no]
    with open(_data_file(), "w") as f:
        f.write("".join(line + "\n" for line in lines))
    return {"status": "Example deleted"}


@app.post("/v1/training/data/clear")
async def clear_training_data():
    """Removes every example from the dataset."""
    if state.is_training:
        raise HTTPException(status_code=409, detail="Training is running. Wait for it to finish before editing the dataset.")
    removed = len(_read_data_lines())
    if os.path.exists(_data_file()):
        os.remove(_data_file())
    return {"status": "Dataset cleared", "removed": removed}

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
    # None = use what training recorded for this adapter; set it to override
    # (e.g. for adapters trained outside the app).
    supports_thinking: bool | None = None


ADAPTER_META_FILE = getattr(train_module, "ADAPTER_META_FILE", "nexus_adapter.json")


def _adapter_meta(adapter_path):
    """Reads what training recorded about an adapter; empty for adapters without metadata."""
    try:
        with open(os.path.join(adapter_path, ADAPTER_META_FILE), "r") as f:
            meta = json.load(f)
        return meta if isinstance(meta, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _has_thinking(response):
    return "<think>" in response and "</think>" in response

class SystemPromptRequest(BaseModel):
    system_prompt: str = ""

@app.post("/v1/system_prompt")
async def set_system_prompt(request: SystemPromptRequest):
    """Sets the system prompt for the base model or the active adapter."""
    state.system_prompt = request.system_prompt.strip() or DEFAULT_SYSTEM_PROMPT
    return {"status": "System prompt updated", "system_prompt": state.system_prompt}

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
        state.model = await _on_model_thread(
            PeftModel.from_pretrained,
            state.model,
            adapter_path,
            is_trainable=False,
        )
        state.model.eval()
        state.adapter_loaded = True
        state.active_adapter = request.adapter_name or "Legacy Adapter"
        if request.supports_thinking is None:
            state.adapter_supports_thinking = bool(_adapter_meta(adapter_path).get("supports_thinking", False))
        else:
            state.adapter_supports_thinking = request.supports_thinking
        # Keep the current system prompt unless the caller sends a new one
        if request.system_prompt:
            state.system_prompt = request.system_prompt
        print(f"Adapter loaded. System Prompt: {state.system_prompt}")
        return {
            "status": "Adapter loaded",
            "adapter": state.active_adapter,
            "supports_thinking": state.adapter_supports_thinking,
        }
    except Exception as e:
        print(f"Error loading adapter: {e}")
        state.adapter_loaded = False
        state.adapter_supports_thinking = False
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/adapter/list")
async def list_adapters():
    """Lists available adapters for the CURRENTLY LOADED model."""
    if not state.model_name:
        return {"adapters": [], "details": []}
    
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
        
    details = []
    for name in sorted(adapters):
        meta = _adapter_meta(os.path.join(base_path, name))
        details.append({
            "name": name,
            "supports_thinking": bool(meta.get("supports_thinking", False)),
            "examples": meta.get("examples"),
        })
    return {"adapters": sorted(adapters), "details": details}



# --- Logic: Training Loop ---

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
    state.metrics = _fresh_metrics("Starting...")
    state.metrics["model"] = model_id
    state.metrics["adapter"] = adapter_name
    
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
                        state.metrics["loss_history"].append(logs["loss"])
                        state.metrics["status"] = "Training..."
                    if "epoch" in logs:
                        state.metrics["step"] = logs["epoch"]
                        state.metrics["progress"] = min(100.0, 100.0 * logs["epoch"] / TOTAL_EPOCHS)
                        
                except:
                    pass
            # Also capture status from normal logs if useful, or just print them
            print(f"[TRAIN]: {line}")
            
            # Update status for user feedback based on simple keywords
            if line.startswith(("Loaded ", "Training Error:", "Error:", "No training data", "Training Complete", "Warning:")) or "examples include <think>" in line:
                 _log(f"Training: {line}")
            if "Loading Model" in line:
                 state.metrics["status"] = "Loading Model..."
            elif "Tokenizing" in line:
                 state.metrics["status"] = "Tokenizing..."
            elif "Saving adapters" in line:
                 state.metrics["status"] = "Saving Weights..."
            elif line.startswith(("Training Error:", "Error:", "No training data")):
                 state.metrics["error"] = line

        # Wait for finish
        ret_code = process.wait()
        
        if ret_code == 0:
            state.metrics["status"] = "Complete"
            state.metrics["progress"] = 100.0
            print("Training subprocess finished successfully.")
        else:
            state.metrics["status"] = "Failed"
            state.metrics["error"] = state.metrics["error"] or f"Training process exited with code {ret_code}"
            print(f"Training subprocess failed with code {ret_code}")
            
    except Exception as e:
        print(f"Training launch failed: {e}")
        state.metrics["status"] = "Failed"
        state.metrics["error"] = str(e)
    
    state.is_training = False

# --- Endpoints ---

def _background_model_load(model_id: str):
    """Loads a model on the model thread, reporting each stage as it goes."""
    global state
    started = time.time()
    try:
        device = _detect_device()

        # A new base model replaces whatever was loaded, including its adapter
        state.model = None
        state.tokenizer = None
        state.model_name = None
        state.adapter_loaded = False
        state.active_adapter = None
        state.adapter_supports_thinking = False
        gc.collect()

        _check_hf_token()

        # 1. What do we need, and how much of it is already on disk?
        _set_stage("checking", 2.0, f"Checking which files {model_id} needs...")
        state.loading_status = "downloading"
        try:
            patterns, total_bytes, missing_bytes = _plan_download(model_id)
        except GatedRepoError:
            raise RuntimeError(
f"{model_id} is a gated model. Accept its terms on its huggingface.co page, "
                "then add your Hugging Face access token in Settings."
            )
        except RepositoryNotFoundError:
            # Local folders and already-cached private models can still load; anything else is a bad ID
            if not os.path.isdir(model_id) and not os.path.isdir(_repo_cache_path(model_id)):
                raise RuntimeError(
                    f"{model_id} was not found on Hugging Face. Check the ID (org/model). "
                    "If it is a private model, add your Hugging Face access token in Settings."
                )
            patterns, total_bytes, missing_bytes = None, 0, 0
        except Exception as e:
            # Offline or the Hub is unreachable: carry on if the cache can serve the model
            patterns, total_bytes, missing_bytes = None, 0, 0
            state.load["download"]["offline"] = True
            _log(f"Could not reach Hugging Face ({type(e).__name__}); trying the local cache.")

        # 2. Download whatever is missing
        if missing_bytes > 0:
            _set_stage(
                "downloading", 5.0,
                f"Downloading {_format_bytes(missing_bytes)} of {_format_bytes(total_bytes)} to {state.cache_dir}",
            )
            _download_with_progress(model_id, patterns, missing_bytes)
            _log("Download complete.")
        elif patterns is not None:
            _log(f"All files already downloaded ({_format_bytes(total_bytes)}).")

        # 3. Tokenizer
        state.loading_status = "loading"
        _set_stage("tokenizer", 72.0, "Loading tokenizer...")
        state.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=state.cache_dir,
            trust_remote_code=True
        )

        # 4. Weights into memory
        _set_stage("weights", 78.0, "Loading weights into memory...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            cache_dir=state.cache_dir,
            trust_remote_code=True
        )

        # 5. Onto the device. Load on CPU, then move in one step: newer transformers copies
        # weights to the device from several threads at once, which hangs on MPS.
        _set_stage("device", 92.0, f"Moving model to {device}...")
        model.to(device)
        state.model = model
        state.model_name = model_id

        state.load["stage"] = None
        state.loading_progress = 100.0
        state.loading_status = "ready"
        _log(f"{model_id} is ready ({time.time() - started:.0f}s).")

    except DownloadCancelled:
        got = state.load["download"]["downloaded"]
        state.load = _fresh_load()
        state.loading_progress = 0.0
        state.loading_status = "idle"
        _log(f"Download of {model_id} cancelled after {_format_bytes(got)}.")

    except Exception as e:
        state.load["failed_stage"] = state.load["stage"]
        state.load["stage"] = None
        state.loading_status = "error"
        state.loading_error = str(e)
        _log(f"Model load failed: {e}")

@app.post("/v1/model/load")
async def load_model_handler(request: LoadModelRequest):
    """Triggers background model loading."""
    if state.loading_status in ["starting", "downloading", "loading"]:
        raise HTTPException(status_code=400, detail="Model already loading.")
    
    # Reset state
    state.loading_status = "starting"
    state.loading_progress = 0.0
    state.loading_error = ""
    state.load = _fresh_load(request.model_id)
    state.cancel_download.clear()
    
    model_executor.submit(_background_model_load, request.model_id)
    return {"status": "Loading started", "model": request.model_id}

@app.post("/v1/model/load/cancel")
async def cancel_model_download():
    """Stops a model download that is in progress. Only the download step can be cancelled."""
    busy = state.loading_status in ("starting", "downloading", "loading")
    if not busy or state.load["stage"] != "downloading":
        raise HTTPException(status_code=409, detail="No download is in progress.")
    state.cancel_download.set()
    return {"status": "Cancelling download", "model": state.load["model"]}


class HfTokenRequest(BaseModel):
    token: str


@app.get("/v1/settings/hf_token")
async def get_hf_token_status():
    """Whether a Hugging Face token is set and working. Never returns the token."""
    return await asyncio.to_thread(_check_hf_token)


@app.post("/v1/settings/hf_token")
async def set_hf_token(request: HfTokenRequest):
    """Validates a Hugging Face access token and saves it where huggingface_hub keeps it."""
    token = request.token.strip()
    if not token:
        raise HTTPException(status_code=400, detail="Paste a token first.")
    try:
        await asyncio.to_thread(lambda: HfApi().whoami(token=token))
    except Exception:
        raise HTTPException(status_code=400, detail="Hugging Face rejected this token. Check that it was copied in full and has not been revoked.")
    # Same place `hf auth login` writes to, so downloads (and the hf CLI) pick it up
    await asyncio.to_thread(login, token=token, add_to_git_credential=False)
    status = await asyncio.to_thread(_check_hf_token)
    _log(f"Hugging Face token saved for {status['username']}.")
    return status


@app.post("/v1/settings/hf_token/remove")
async def remove_hf_token():
    """Removes the saved Hugging Face token from this machine."""
    await asyncio.to_thread(logout)
    status = await asyncio.to_thread(_check_hf_token)
    _log("Hugging Face token removed.")
    return status


class DeleteModelRequest(BaseModel):
    model_id: str


@app.post("/v1/model/delete")
async def delete_model(request: DeleteModelRequest):
    """Deletes a downloaded model from the cache folder. This cannot be undone."""
    model_id = request.model_id
    if model_id == state.model_name:
        raise HTTPException(status_code=409, detail="This model is loaded. Eject it first.")
    if state.loading_status in ("starting", "downloading", "loading") and state.load["model"] == model_id:
        raise HTTPException(status_code=409, detail="This model is being loaded. Wait for it to finish or cancel the download.")
    if state.is_training and state.metrics.get("model") == model_id:
        raise HTTPException(status_code=409, detail="This model is being used for training.")
    try:
        freed = await asyncio.to_thread(_delete_model_files, model_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"{model_id} is not in the model folder.")
    _log(f"Deleted {model_id} from disk ({_format_bytes(freed)} freed).")
    return {"status": "Model deleted", "model": model_id, "freed_bytes": freed}


def _load_report():
    """The load detail for the UI, with elapsed times computed server-side."""
    load = state.load
    now = time.time()
    busy = state.loading_status in ("starting", "downloading", "loading")
    return {
        "model": load["model"],
        "stage": load["stage"],
        "stages": LOAD_STAGES,
        "failed_stage": load["failed_stage"],
        "stage_elapsed": now - load["stage_started"] if busy and load["stage"] else 0.0,
        "elapsed": now - load["started"] if busy else 0.0,
        "download": load["download"],
    }


@app.get("/v1/model/info")
async def model_info(model_id: str):
    """Size of a model and how much of it would have to be downloaded."""
    try:
        _patterns, total_bytes, missing_bytes = await asyncio.to_thread(_plan_download, model_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Could not look up {model_id}: {type(e).__name__}")
    return {"model": model_id, "total_bytes": total_bytes, "missing_bytes": missing_bytes}


@app.get("/v1/model/status")
async def get_model_status():
    return {
        "status": state.loading_status,
        "progress": state.loading_progress,
        "error": state.loading_error,
        "load": _load_report(),
        "log": list(state.log)[-50:],
        "current_model": state.model_name,
        "device": _detect_device(),
        "system_prompt": state.system_prompt,
        "is_training": state.is_training,
        "active_adapter": state.active_adapter,
        "adapter_loaded": state.adapter_loaded,
        # Thinking mode is disabled when an adapter is loaded (adapters aren't trained on <think> format)
        "adapter_supports_thinking": state.adapter_supports_thinking,
        # Thinking is available on the base model and on adapters trained with <think> examples
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
    state.adapter_loaded = False
    state.active_adapter = None
    state.adapter_supports_thinking = False
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
            base = await _on_model_thread(state.model.unload)
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
    """Lists downloaded models in the cache directory, with their size on disk."""
    if not state.cache_dir or not os.path.exists(state.cache_dir):
        return {"models": [], "details": []}
    
    details = []
    try:
        for folder in sorted(os.listdir(state.cache_dir)):
            if folder.startswith("models--"):
                # Parse "models--org--repo" -> "org/repo"
                parts = folder.split("--")
                if len(parts) >= 3:
                    size = await asyncio.to_thread(_dir_size, os.path.join(state.cache_dir, folder))
                    details.append({"id": f"{parts[1]}/{parts[2]}", "size_bytes": size})
    except Exception as e:
        print(f"Error listing models: {e}")
        
    return {"models": [d["id"] for d in details], "details": details}

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
    return await _on_model_thread(_generate_reply, request)


def _generate_reply(request: ChatRequest):
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
            
            # Three cases:
            #  - base model: we ask for <think> reasoning with instructions and a one-shot example
            #  - adapter trained with <think> examples: it writes its own reasoning, so the
            #    prompt is left untouched (injected instructions would override its voice)
            #  - any other adapter: direct replies only; it was never shown <think> and
            #    would stop after </think>
            adapter_thinks = state.adapter_loaded and state.adapter_supports_thinking
            use_thinking = request.enable_thinking and (not state.adapter_loaded or adapter_thinks)

            if use_thinking and adapter_thinks:
                pass
            elif use_thinking:
                messages[0]["content"] += "\n\nYou MUST begin by reasoning step-by-step inside <think>...</think> tags. Do NOT speak to the user inside the tags. usage: <think>internal thought</think> final response"
                # One-shot example to guide the model
                messages.append({"role": "user", "content": "Hello"})
                messages.append({"role": "assistant", "content": "<think>The user is greeting me. I should respond in character.</think>Greetings. I am ready to assist."})
            else:
                messages[0]["content"] += "\n\nAnswer directly without showing your thinking process."

            for turn in request.history:
                if turn.role in ("user", "assistant") and turn.content.strip():
                    messages.append({"role": turn.role, "content": turn.content})
            messages.append({"role": "user", "content": request.message})
            # enable_thinking is read by templates with native reasoning (e.g. Qwen3) and ignored by the rest
            input_ids = state.tokenizer.apply_chat_template(
                messages, return_tensors="pt", add_generation_prompt=True, return_dict=False,
                enable_thinking=use_thinking,
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
    if not os.path.exists(data_path) or os.path.getsize(data_path) == 0:
        raise HTTPException(status_code=400, detail="No training data yet. Add examples to the dataset first.")
    state.is_training = True  # set before returning so the UI's first poll sees it
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