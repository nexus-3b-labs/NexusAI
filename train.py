import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

import argparse
import gc

import argparse
from transformers import TrainerCallback

# Configuration (Defaults)
DEFAULT_MODEL_ID = "stabilityai/stablelm-zephyr-3b"
DATA_FILE = "training_data.jsonl"
OUTPUT_DIR = "nexus_adapters"
MAX_LENGTH = 512

class ProgressCallback(TrainerCallback):
    """Custom callback to report metrics to the backend state."""
    def __init__(self, callback_fn):
        self.callback_fn = callback_fn

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and self.callback_fn:
            # logs typically contains 'loss', 'learning_rate', 'epoch'
            self.callback_fn(logs)

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a model using LoRA and collected data.")
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID, help="HuggingFace Model ID to fine-tune")
    parser.add_argument("--data", type=str, default=DATA_FILE, help="Path to training data (JSONL)")
    parser.add_argument("--output", type=str, default=OUTPUT_DIR, help="Directory to save adapter weights")
    return parser.parse_args()

def load_data(data_path):
    """Loads and filters training data from JSONL."""
    data = []
    try:
        with open(data_path, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    # Data Filtering Logic (Score > 7)
                    if entry.get("score", 0) >= 7 and entry.get("prompt") and entry.get("response"):
                        text = f"<|user|>\n{entry['prompt']}<|endoftext|>\n<|assistant|>\n{entry['response']}<|endoftext|>"
                        data.append({"text": text})
                except:
                    continue
    except FileNotFoundError:
        print(f"Error: {data_path} not found.")
        return []
    
    print(f"Loaded {len(data)} high-quality training examples.")
    return data

def run_training(model_id=DEFAULT_MODEL_ID, data_path=DATA_FILE, output_dir=OUTPUT_DIR, progress_callback=None):
    """
    Main entry point for training from external scripts.
    """
    data = load_data(data_path)
    if not data:
        print("No training data found or loaded.")
        return False

    # Convert to HuggingFace Dataset
    dataset = Dataset.from_list(data)

    # Detect Device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"

    # Initialize variables for cleanup safety
    model = None
    tokenizer = None
    trainer = None

    try:
        print(f"Loading Tokenizer for {model_id}...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token

        def tokenize_function(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH)

        print("Tokenizing dataset...")
        tokenized_datasets = dataset.map(tokenize_function, batched=True)

        print(f"Loading Model on {device}...")
        
        # MPS/CPU usually doesn't support 4-bit via bitsandbytes unless specifically built
        use_4bit = (device == "cuda")

        if use_4bit:
            print("Using 4-bit quantization (CUDA detected).")
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map="auto",
                    load_in_4bit=True,
                    torch_dtype=torch.float16
                )
            except ImportError:
                print("Warning: bitsandbytes not found/installed. Falling back to fp16.")
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map=device,
                    torch_dtype=torch.float16
                )
        else:
            print(f"Using fp16 for {device} (Quantization disabled).")
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map=device,
                torch_dtype=torch.float16
            )

        # Apply LoRA
        print("Applying LoRA Config...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=8, 
            lora_alpha=32, 
            lora_dropout=0.1
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            num_train_epochs=3,
            learning_rate=2e-4,
            logging_steps=1,
            save_strategy="epoch",
            # Stability Fixes for Mac (MPS)
            fp16=(device == "cuda"),
            bf16=False, # MPS support for bf16 is experimental/limited on some devices
            use_cpu=(device == "cpu"),
            dataloader_num_workers=0, # Prevent multiprocessing crashes on Mac
            # use_mps_device is deprecated and causes issues if explicitly set to False/True incorrectly in some versions
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
            callbacks=[ProgressCallback(progress_callback)] if progress_callback else []
        )

        print("Starting Training...")
        trainer.train()

        print(f"Saving adapters to {output_dir}...")
        model.save_pretrained(output_dir)
        print("Training Complete!")
        return True

    except Exception as e:
        print(f"Training Error: {e}")
        return False
        
    finally:
        print("Cleaning up training memory...")
        # Explicit deletion
        if trainer: del trainer
        if model: del model
        if tokenizer: del tokenizer
        
        # Garbage Collection
        gc.collect()
        
        # Cache Clearing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except:
                pass
        print("Memory released.")

if __name__ == "__main__":
    args = parse_args()
    
    def json_logger(logs):
        # Prefix with specific marker to avoid parsing garbage logs
        print(f"JSON_LOG:{json.dumps(logs)}", flush=True)

    success = run_training(args.model_id, args.data, args.output, progress_callback=json_logger)
    if not success:
        exit(1)
