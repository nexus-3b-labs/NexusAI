export const API_BASE = "http://localhost:8000";

export const DEFAULT_PARAMS = {
  temperature: 0.7,
  top_p: 0.9,
  max_new_tokens: 1024,
  top_k: 50,
  repetition_penalty: 1.1,
  min_new_tokens: 0,
};

async function request(path, body) {
  const res = await fetch(`${API_BASE}${path}`, body === undefined ? undefined : {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `Request failed (${res.status})`);
  return data;
}

export const api = {
  status: () => request('/v1/model/status'),
  models: () => request('/v1/model/list'),
  loadModel: (model_id) => request('/v1/model/load', { model_id }),
  unloadModel: () => request('/v1/model/unload', {}),
  adapters: () => request('/v1/adapter/list'),
  loadAdapter: (adapter_name) => request('/v1/adapter/load', { adapter_name }),
  unloadAdapter: () => request('/v1/adapter/unload', {}),
  setSystemPrompt: (system_prompt) => request('/v1/system_prompt', { system_prompt }),
  params: () => request('/v1/parameters/get'),
  setParams: (params) => request('/v1/parameters/update', params),
  settings: () => request('/v1/settings/get'),
  setSettings: (cache_dir) => request('/v1/settings/update', { cache_dir }),
  chat: (message, history, enable_thinking) => request('/v1/chat', { message, history, enable_thinking }),
  score: (payload) => request('/v1/score', payload),
  dataset: () => request('/v1/training/data'),
  addData: (data) => request('/v1/training/data', { data }),
  deleteExample: (line, prompt) => request(`/v1/training/data/${line}/delete`, { prompt }),
  clearDataset: () => request('/v1/training/data/clear', {}),
  startTraining: (model_id, adapter_name) => request('/v1/train/start', { model_id, adapter_name }),
  metrics: () => request('/v1/metrics'),
};

/** Splits a raw model reply into its <think> block and the visible answer. */
export function splitThinking(content = "") {
  const match = content.match(/<think>([\s\S]*?)(<\/think>|$)/);
  const thought = match ? match[1].trim() : "";
  const answer = content.replace(/<think>[\s\S]*?(<\/think>|$)/, '').trim();
  return { thought, answer };
}
