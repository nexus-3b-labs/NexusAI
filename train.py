import json
import os
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

import argparse
import gc
from transformers import TrainerCallback

# Configuration (Defaults)
DEFAULT_MODEL_ID = "stabilityai/stablelm-zephyr-3b"
DATA_FILE = "training_data.jsonl"
OUTPUT_DIR = "nexus_adapters"
MAX_LENGTH = 512
NUM_EPOCHS = 3
# Written next to the adapter weights so the app knows how the adapter was trained
ADAPTER_META_FILE = "nexus_adapter.json"

class ProgressCallback(TrainerCallback):
    """Custom callback to report metrics to the backend state."""
    def __init__(self, callback_fn):
        self.callback_fn = callback_fn

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and self.callback_fn:
            # logs typically contains 'loss', 'learning_rate', 'epoch'
            self.callback_fn(logs)


class DataCollatorForCausalLMWithLabels:
    """Pads the batch and preserves precomputed labels (e.g. prompt-masked). Masks padding in labels with -100."""
    def __init__(self, tokenizer, pad_to_multiple_of=None):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        # We already padded to max_length in tokenize_function, so all same length; just stack.
        batch = {}
        for key in ("input_ids", "attention_mask", "labels"):
            if key not in features[0]:
                continue
            tensors = [torch.tensor(f[key], dtype=torch.long if key != "attention_mask" else torch.long) for f in features]
            batch[key] = torch.stack(tensors)
        return batch

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a model using LoRA and collected data.")
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID, help="HuggingFace Model ID to fine-tune")
    parser.add_argument("--data", type=str, default=DATA_FILE, help="Path to training data (JSONL)")
    parser.add_argument("--output", type=str, default=OUTPUT_DIR, help="Directory to save adapter weights")
    parser.add_argument("--cache_dir", type=str, default=None, help="HuggingFace cache directory (default: ~/.cache/huggingface)")
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
                        data.append({"prompt": entry["prompt"], "response": entry["response"]})
                except:
                    continue
    except FileNotFoundError:
        print(f"Error: {data_path} not found.")
        return []
    
    print(f"Loaded {len(data)} high-quality training examples.")
    return data

def has_thinking(response):
    """True if a training response carries its own <think>...</think> reasoning."""
    return "<think>" in response and "</think>" in response

def format_example(tokenizer, prompt, response, thinking=False):
    """Returns (prefix, full_text) in the same chat format the model sees at inference time.

    thinking=False: the adapter answers directly, so the prompt ends the way a
    "thinking off" prompt does (Qwen3 pre-fills an empty think block).
    thinking=True: the responses contain their own <think> block, so the prompt is
    left open for the adapter to write it.
    """
    if tokenizer.chat_template:
        prefix = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
            enable_thinking=thinking,
        )
    else:
        prefix = f"<|user|>\n{prompt}<|endoftext|>\n<|assistant|>\n"
    return prefix, prefix + response + tokenizer.eos_token

def run_training(model_id=DEFAULT_MODEL_ID, data_path=DATA_FILE, output_dir=OUTPUT_DIR, progress_callback=None, cache_dir=None):
    """
    Main entry point for training from external scripts.
    """
    data = load_data(data_path)
    if not data:
        print("No training data found or loaded.")
        return False

    # An adapter learns to think only if its examples show it how
    thinking_examples = sum(1 for d in data if has_thinking(d["response"]))
    supports_thinking = thinking_examples > 0
    if supports_thinking:
        print(f"{thinking_examples} of {len(data)} examples include <think> reasoning: training a thinking adapter.")
        if thinking_examples < len(data):
            print("Warning: the dataset mixes thinking and direct examples; the adapter may be inconsistent.")

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
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token

        # Prompt masking below assumes the prompt sits at the start of the sequence
        tokenizer.padding_side = "right"

        def tokenize_function(examples):
            prefixes, texts = [], []
            for prompt, response in zip(examples["prompt"], examples["response"]):
                prefix, text = format_example(tokenizer, prompt, response, thinking=supports_thinking)
                prefixes.append(prefix)
                texts.append(text)
            full_enc = tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=MAX_LENGTH
            )
            # Loss only on the assistant response: mask prompt tokens and padding
            labels = []
            for i, prefix in enumerate(prefixes):
                plen = len(tokenizer(prefix, truncation=True, max_length=MAX_LENGTH)["input_ids"])
                lab = [
                    tok if mask else -100
                    for tok, mask in zip(full_enc["input_ids"][i], full_enc["attention_mask"][i])
                ]
                for j in range(min(plen, len(lab))):
                    lab[j] = -100
                labels.append(lab)
            full_enc["labels"] = labels
            return full_enc

        print("Tokenizing dataset...")
        tokenized_datasets = dataset.map(tokenize_function, batched=True)

        print(f"Loading Model on {device}...")
        
        # MPS/CPU usually doesn't support 4-bit via bitsandbytes unless specifically built
        use_4bit = (device == "cuda")

        kwargs_common = {"cache_dir": cache_dir, "trust_remote_code": True}
        if use_4bit:
            print("Using 4-bit quantization (CUDA detected).")
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map="auto",
                    load_in_4bit=True,
                    torch_dtype=torch.float16,
                    **kwargs_common
                )
            except ImportError:
                print("Warning: bitsandbytes not found/installed. Falling back to fp16.")
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map=device,
                    torch_dtype=torch.float16,
                    **kwargs_common
                )
        else:
            print(f"Using fp16 for {device} (Quantization disabled).")
            # Load on CPU, then move in one step (threaded device placement hangs on MPS)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                **kwargs_common
            )
            model.to(device)

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
            num_train_epochs=NUM_EPOCHS,
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
            data_collator=DataCollatorForCausalLMWithLabels(tokenizer),
            callbacks=[ProgressCallback(progress_callback)] if progress_callback else []
        )

        print("Starting Training...")
        trainer.train()

        print(f"Saving adapters to {output_dir}...")
        model.save_pretrained(output_dir)
        with open(os.path.join(output_dir, ADAPTER_META_FILE), "w") as f:
            json.dump({
                "base_model": model_id,
                "examples": len(data),
                "thinking_examples": thinking_examples,
                "supports_thinking": supports_thinking,
            }, f, indent=2)
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

    success = run_training(
        args.model_id,
        args.data,
        args.output,
        progress_callback=json_logger,
        cache_dir=args.cache_dir
    )
    if not success:
        exit(1)
