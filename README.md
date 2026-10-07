# NexusAI 🧠

**NexusAI** is a powerful, local laboratory for experimenting with Large Language Models (LLMs). It bridges the gap between complex Python scripts and a user-friendly interface, allowing you to chat with, fine-tune, and customize AI models directly from your desktop.

Designed for developers and hobbyists who want to explore **LoRA (Low-Rank Adaptation)** fine-tuning, persona engineering, and local inference without relying on cloud providers.

---

## 🌟 Key Features

### 💬 **Interactive Chat Interface**
- **Local Inference**: Run models like `SystemZephyr-3B` or `Qwen` entirely offline.
- **Chain-of-Thought Visualization**: See the "hidden" reasoning steps of the model with a collapsible **Thinking Process** UI.
- **Conversation Memory**: Earlier turns are sent with each message, so the model keeps context.
- **Customizable UI**: Light and dark themes, a resizable configuration panel, and adjustable text size.

### 🛠️ **No-Code Fine-Tuning**
- **Train Custom Adapters**: Upload a `training_data.jsonl` file and train the model on your own data.
- **LoRA Support**: Uses Parameter-Efficient Fine-Tuning to create lightweight adapters (~10MB) instead of retraining the whole model.
- **Real-time Metrics**: Watch the training loss and progress bar update live in the UI.

### 🎭 **Persona Engineering**
- **System Prompts**: Define the "soul" of your AI (e.g., "You are a pirate", "You are a coding assistant").
- **Adapter Management**: Load and unload different personality adapters on the fly without restarting the app.

---

## 🏗️ Tech Stack

- **Frontend**: [React](https://react.dev/), [Vite](https://vitejs.dev/), [TailwindCSS](https://tailwindcss.com/), [Lucide React](https://lucide.dev/) (Icons).
- **Backend**: [Python](https://www.python.org/), [FastAPI](https://fastapi.tiangolo.com/), [HuggingFace Transformers](https://huggingface.co/docs/transformers/index), [PEFT](https://github.com/huggingface/peft).
- **Training**: PyTorch, BitsAndBytes (Quantization).

---

## 🚀 Getting Started

### Prerequisites
- **Python 3.10+** installed.
- **Node.js 16+** installed.
- (Optional) NVIDIA GPU (CUDA) or Mac M1/M2/M3 (MPS) for faster training/inference.

### 1. Backend Setup (API & Model Engine)

Navigate to the root directory and install Python dependencies.

```bash
# It is recommended to create a virtual environment first
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install fastapi uvicorn torch transformers peft datasets scipy accelerate bitsandbytes
```

Start the backend server:

```bash
python main.py
```
*The server will start at `http://localhost:8000`.*

### 2. Frontend Setup (UI)

Open a new terminal, navigate to the UI folder, and start the development server.

```bash
cd nexus-lab-ui

# Install dependencies
npm install

# Start the UI
npm run dev
```
*Access the application at `http://localhost:5173`.*

---

## 📖 Usage Guide

### 1. Chatting with the Model
- Pick a model in the **selector in the top bar**. Downloaded models are listed; paste a Hugging Face ID (`org/model`) to fetch a new one.
- Type your message and hit Enter (Shift+Enter adds a new line).
- Rate replies with the thumbs: a thumbs-up saves the prompt and reply to your training dataset.
- Text size and theme live in **Settings**.

### 2. Fine-Tuning a New Persona
1.  **Prepare Data**: Create a file named `training_data.jsonl`.
    ```json
    {"prompt": "hi", "response": "Hey you 😊 I was hoping you'd show up. How’s your day going so far?", "score": 10, "source": "human_feedback"}
    {"prompt": "hello", "response": "Hi! It’s nice to see you here. What are we talking about today?", "score": 9, "source": "human_feedback"}
    ```
2.  **Add it**: Go to **Train**, paste or import the file under **Dataset**, and click **Add to dataset**.
3.  **Train**: Choose the base model and an adapter name, then click **Start training**.
4.  **Wait**: Monitor the progress bar and loss curve.
5.  **Load**: Once finished, click **Try in Chat**, or pick the adapter in the Chat **Configuration** panel.

### 3. Using System Prompts
- In the Chat **Configuration** panel (the sliders icon), you can enter a custom System Prompt (e.g., *"You are a helpful coding assistant"*).
- This works in tandem with your loaded adapter to steer the model's behavior.

### 4. Thinking with Adapters
- An adapter can only think if its training replies show it how. Put the reasoning inside the reply:
    ```json
    {"prompt": "Hello", "response": "<think>A visitor greets me. I shall welcome them.</think>Greetings, seeker of truth."}
    ```
- Train on replies like these (see `datasets/tesla_thinking.jsonl`) and the adapter is recorded as a **thinking adapter**. With thinking on, it reasons in its own trained voice and no instructions are injected.
- Adapters trained on plain replies keep thinking locked off, because they would stop after the reasoning.
- Use thinking in all examples or none. The Train page warns when a dataset mixes both.
- For adapters trained outside the app, tick **Trained with `<think>` examples** when applying them.

### 5. Watching a Model Load
- While a model loads, a panel under the model selector shows each step: checking files, downloading, loading the tokenizer, loading weights, and moving to the device.
- Downloads show size, speed and time remaining. The model list shows each model's size on disk, and a pasted Hugging Face ID shows how much will be downloaded before you start.
- **Activity log** in the same panel lists what the backend did, including errors.

- **Cancel download** stops a download in progress, and the trash icon in the model list deletes a downloaded model from disk.

### 6. Hugging Face Access Token
- Gated and private models need a token. Add one under **Settings → Hugging Face access**; create it with read access at `huggingface.co/settings/tokens`.
- The token is checked with Hugging Face, saved where the `hf` command-line tool keeps it, and never shown in the app again.
- If a download is slow and no token is set, the progress panel suggests adding one, because Hugging Face gives anonymous downloads lower rate limits.

### 7. Inference Parameters
In the Chat **Configuration** panel you can tune how the model generates text; changes apply automatically. Use the **Guide** section for full descriptions.

| Parameter | What it does |
|-----------|--------------|
| **Temperature** | Higher = more creative/random; lower = more deterministic (e.g. 0.1 for code, 0.7 for chat). |
| **Top P** | Nucleus sampling: only sample from the top fraction of likely tokens (e.g. 0.9). |
| **Max New Tokens** | Maximum length of each reply in tokens. |
| **Top K** | Only sample from the top K tokens (0 = no limit). |
| **Repetition Penalty** | Discourages repeating the same tokens (e.g. 1.1). |
| **Min New Tokens** | Don’t stop before generating at least this many tokens. |

---

## 📂 Project Structure

```
NexusAI/
├── main.py                 # Backend API Entry Point
├── train.py                # Standalone Training Script
├── requirements-train.txt  # Python Dependencies
├── nexus_adapters/         # Storage for trained LoRA adapters
├── nexus-lab-ui/           # Frontend React Application
│   ├── src/
│   │   ├── App.jsx         # App shell and state
│   │   ├── api.js          # Backend API client
│   │   ├── components/     # Chat, Train, model picker, config panel
│   │   └── ...
│   └── ...
└── README.md               # You are here
```

---

## ⚠️ Troubleshooting

**"Transformers architecture not recognized"**
- If you see errors about `qwen3` or other new architectures, your `transformers` library might be outdated.
- Run: `pip install --upgrade transformers`

**"CUDA out of memory"**
- Try reducing the `batch_size` in `train.py` or training on a smaller model.

---

*Built with ❤️ by the NexusAI Team.*
