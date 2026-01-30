# Thinking Mode with Adapters

This document explains how the thinking mode (`<think>` tags) works in NexusAI, especially when using fine-tuned adapters.

---

## Overview

NexusAI supports a "thinking mode" where the model shows its reasoning process before responding:

```
<think>User is asking about electricity. I should explain in Tesla's voice...</think>
Alternating current flows in harmony with nature's rhythms...
```

This feature works differently depending on whether you're using the **base model** or a **fine-tuned adapter**.

---

## How It Works

### 1. Base Model (No Adapter)

When no adapter is loaded:
- Uses **Qwen's native thinking** via `enable_thinking=True` in the chat template
- Adds thinking instructions to the system prompt
- Includes a one-shot example to guide the format

The model generates its own reasoning style.

### 2. Adapter WITHOUT Thinking Support

When an adapter is loaded but wasn't trained with `<think>` tags:
- Thinking mode is **automatically disabled**
- The toggle button turns amber and is locked
- Model uses direct response format

This prevents the model from generating incomplete responses (stopping after `</think>`).

### 3. Adapter WITH Thinking Support

When an adapter is trained with `<think>` tags in the training data:
- Check "Adapter trained with `<think>` format" when loading
- Qwen's native thinking is **disabled** (`enable_thinking=False`)
- No thinking instructions added to prompt
- The adapter generates thinking **naturally from its training**

This ensures the adapter uses its own trained thinking style (e.g., Tesla's voice) rather than Qwen's generic reasoning.

---

## Training Data Format

### Standard Format (No Thinking)

```json
{"prompt": "Hello", "response": "Hi there! How can I help?", "score": 10}
```

### Thinking Format

```json
{"prompt": "Hello", "response": "<think>User greeted me warmly.</think>Hi there! How can I help?", "score": 10}
```

### Example: Tesla Persona with Thinking

```json
{"prompt": "Hello Tesla", "response": "<think>A visitor greets me. I shall welcome them in my characteristic manner.</think>Greetings, seeker of truth. What stirs your mind today?", "score": 10}
{"prompt": "Tell me about AC", "response": "<think>They wish to learn of alternating current. I shall explain with passion.</think>Alternating current flows in harmony with nature's rhythms—efficient and transformable.", "score": 10}
```

---

## Technical Implementation

### Backend Logic (`main.py`)

The chat handler determines thinking mode based on three scenarios:

```python
# 1. No adapter + thinking enabled → use Qwen native thinking
use_native_thinking = request.enable_thinking and not state.adapter_loaded

# 2. Adapter with thinking support → let adapter handle it
adapter_handles_thinking = state.adapter_loaded and state.adapter_supports_thinking

# 3. Adapter without thinking support → direct response only
use_direct_response = state.adapter_loaded and not state.adapter_supports_thinking
```

#### Chat Template Parameters

| Scenario | `enable_thinking` | Prompt Modification |
|----------|-------------------|---------------------|
| Base model + thinking | `True` | Add instructions + one-shot |
| Adapter with thinking | `False` | None (adapter trained) |
| Adapter without thinking | `False` | "Answer directly..." |
| User disabled thinking | `False` | "Answer directly..." |

### API Changes

#### Load Adapter Request

```json
POST /v1/adapter/load
{
  "adapter_name": "tesla_adapter",
  "system_prompt": "You are Nikola Tesla...",
  "supports_thinking": true
}
```

#### Model Status Response

```json
GET /v1/model/status
{
  "adapter_loaded": true,
  "adapter_supports_thinking": true,
  "thinking_supported": true,
  ...
}
```

### Frontend Changes

- Added checkbox: "Adapter trained with `<think>` format"
- Thinking toggle enabled when:
  - No adapter loaded, OR
  - Adapter loaded with `supports_thinking=true`
- Thinking toggle disabled (amber) when:
  - Adapter loaded without thinking support

---

## Why This Design?

### Problem: Qwen's Native Thinking Conflicts with Trained Adapters

Qwen3 models have built-in thinking support. When `enable_thinking=True`:
- Qwen generates its **own** reasoning style
- This overrides whatever the adapter was trained on
- Result: Generic reasoning instead of persona-specific thinking

### Solution: Let Adapters Control Their Own Thinking

When an adapter is trained with `<think>` tags:
1. Disable Qwen's native thinking (`enable_thinking=False`)
2. Don't add any thinking instructions to the prompt
3. The adapter naturally generates `<think>...</think>response` from training

This preserves the adapter's unique voice and reasoning style.

---

## Quick Reference

| State | Thinking Toggle | Behavior |
|-------|-----------------|----------|
| Base model | Enabled | Qwen native thinking |
| Base model | Disabled | Direct response |
| Adapter (no thinking) | Locked/Disabled | Direct response |
| Adapter (with thinking) | Enabled | Adapter's trained thinking |
| Adapter (with thinking) | Disabled | Direct response |

---

## Files Modified

- `main.py` — Backend logic for thinking mode
- `nexus-lab-ui/src/App.jsx` — Frontend checkbox and toggle logic
- `training_data.jsonl` — Example data with `<think>` tags

---

## Troubleshooting

### Adapter generates Qwen-style thinking instead of trained style

**Cause:** `enable_thinking=True` is being passed to Qwen's chat template.

**Fix:** Ensure "Adapter trained with `<think>` format" is checked when loading.

### Model stops after `</think>` with no response

**Cause:** Adapter wasn't trained with `<think>` tags but thinking mode is enabled.

**Fix:** Either:
1. Uncheck "Adapter trained with `<think>` format", or
2. Retrain the adapter with `<think>` tags in responses

### Thinking toggle is locked/amber

**Expected:** This happens when an adapter without thinking support is loaded.

**To enable:** Load an adapter trained with thinking, or unload the adapter.

---

*Last updated: January 2026*
