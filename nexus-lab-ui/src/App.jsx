import React, { useCallback, useEffect, useRef, useState } from 'react';
import { BrainCircuit, MessageSquare, FlaskConical, Settings, BookOpen, Sun, Moon, Cpu } from 'lucide-react';
import { api, splitThinking, formatBytes, DEFAULT_PARAMS } from './api';
import { Toasts } from './components/ui';
import ModelPicker from './components/ModelPicker';
import ChatView from './components/ChatView';
import ConfigPanel from './components/ConfigPanel';
import TrainView from './components/TrainView';
import { SettingsView, HelpView } from './components/Pages';

const NAV = [
  { id: 'chat', label: 'Chat', icon: MessageSquare },
  { id: 'train', label: 'Train', icon: FlaskConical },
  { id: 'settings', label: 'Settings', icon: Settings },
  { id: 'help', label: 'Guide', icon: BookOpen },
];

const DEVICE_LABELS = { cuda: 'NVIDIA GPU', mps: 'Apple GPU', cpu: 'CPU' };
const LOADING = ['starting', 'downloading', 'loading'];
const IDLE_STATUS = { status: 'idle', progress: 0, error: '', current_model: null, active_adapter: null, adapter_supports_thinking: false, thinking_supported: true, device: null, system_prompt: '', is_training: false };
const IDLE_METRICS = { loss: 0, step: 0, perplexity: 0, status: 'Idle', progress: 0, loss_history: [], error: '' };

function stored(key, fallback) {
  try {
    const raw = localStorage.getItem(key);
    return raw === null ? fallback : JSON.parse(raw);
  } catch {
    return fallback;
  }
}

function usePersisted(key, fallback) {
  const [value, setValue] = useState(() => stored(key, fallback));
  useEffect(() => {
    try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* storage unavailable */ }
  }, [key, value]);
  return [value, setValue];
}

export default function App() {
  const [view, setView] = useState('chat');
  const [theme, setTheme] = usePersisted('nexus.theme', 'dark');
  const [zoom, setZoom] = usePersisted('nexus.zoom', 1);
  const [thinking, setThinking] = usePersisted('nexus.thinking', true);
  const [configOpen, setConfigOpen] = usePersisted('nexus.configOpen', window.innerWidth >= 1100);
  const [panelWidth, setPanelWidth] = usePersisted('nexus.panelWidth', 340);

  const [online, setOnline] = useState(true);
  const [status, setStatus] = useState(IDLE_STATUS);
  const [models, setModels] = useState([]);
  const [modelSizes, setModelSizes] = useState({});
  const [adapters, setAdapters] = useState([]);
  const [params, setParams] = useState(DEFAULT_PARAMS);
  const [cacheDir, setCacheDir] = useState('');
  const [hfToken, setHfToken] = useState({ configured: false, valid: false, username: null, from_environment: false });
  const [dataset, setDataset] = useState({ total: 0, eligible: 0, thinking: 0, entries: [] });
  const [metrics, setMetrics] = useState(IDLE_METRICS);

  const [messages, setMessages] = useState([]);
  const [busy, setBusy] = useState(false);
  const [toasts, setToasts] = useState([]);
  const [pickerOpen, setPickerOpen] = useState(false);

  const prevStatus = useRef(IDLE_STATUS);
  const pendingAdapter = useRef(null); // { model, adapter } to apply once that model is ready
  const paramsTimer = useRef(null);
  const failedPolls = useRef(0);

  const toast = useCallback((kind, text) => {
    const id = `${Date.now()}-${Math.random()}`;
    setToasts((t) => [...t.slice(-3), { id, kind, text }]);
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), kind === 'error' ? 8000 : 4000);
  }, []);

  // --- Appearance ---
  useEffect(() => { document.documentElement.classList.toggle('dark', theme === 'dark'); }, [theme]);
  useEffect(() => { document.documentElement.style.zoom = zoom; }, [zoom]);

  // --- Backend sync ---
  const refreshStatus = useCallback(async () => {
    try {
      const data = await api.status();
      setStatus(data);
      failedPolls.current = 0;
      setOnline(true);
    } catch {
      // One slow poll (the backend is busy loading or training) is not an outage
      failedPolls.current += 1;
      if (failedPolls.current >= 2) setOnline(false);
    }
  }, []);

  const refreshModels = useCallback(() => api.models().then((d) => {
    setModels(d.models || []);
    setModelSizes(Object.fromEntries((d.details || []).map((m) => [m.id, m.size_bytes])));
  }).catch(() => {}), []);
  const refreshAdapters = useCallback(() => api.adapters().then((d) => setAdapters(d.details || (d.adapters || []).map((name) => ({ name, supports_thinking: false })))).catch(() => {}), []);
  const refreshDataset = useCallback(() => api.dataset().then(setDataset).catch(() => {}), []);

  // The backend is the source of truth: poll it, faster while something is in progress
  const loading = LOADING.includes(status.status);
  useEffect(() => {
    refreshStatus();
    const timer = setInterval(refreshStatus, loading ? 1000 : 4000);
    return () => clearInterval(timer);
  }, [refreshStatus, loading]);

  // (Re)load everything else whenever the backend comes online
  useEffect(() => {
    if (!online) return;
    refreshModels();
    refreshDataset();
    api.params().then((d) => setParams((p) => ({ ...p, ...d.params }))).catch(() => {});
    api.settings().then((d) => setCacheDir(d.cache_dir || '')).catch(() => {});
    api.hfToken().then(setHfToken).catch(() => {});
    api.metrics().then((d) => setMetrics({ ...IDLE_METRICS, ...d.metrics })).catch(() => {});
  }, [online, refreshModels, refreshDataset]);

  // supportsThinking: null uses what training recorded for the adapter
  const applyAdapter = useCallback(async (name, supportsThinking = null) => {
    try {
      const data = await api.loadAdapter(name, supportsThinking);
      toast('success', `Adapter “${data.adapter}” is active${data.supports_thinking ? ' with its own thinking' : ''}.`);
    } catch (err) {
      toast('error', `Could not load the adapter: ${err.message}`);
    }
    refreshStatus();
  }, [toast, refreshStatus]);

  // React to model state transitions reported by the backend
  useEffect(() => {
    const prev = prevStatus.current;
    prevStatus.current = status;
    if (status.current_model !== prev.current_model) refreshAdapters();
    if (LOADING.includes(prev.status) && status.status === 'ready') {
      toast('success', `${status.current_model} is ready.`);
      refreshModels();
      const pending = pendingAdapter.current;
      if (pending && pending.model === status.current_model) {
        pendingAdapter.current = null;
        applyAdapter(pending.adapter);
      }
    }
    if (LOADING.includes(prev.status) && status.status === 'error') {
      pendingAdapter.current = null;
      toast('error', 'The model failed to load. Details are under the model selector.');
    }
  }, [status, toast, refreshAdapters, refreshModels, applyAdapter]);

  // The dataset also changes outside this page (ratings, other tabs), so re-read it on entry
  useEffect(() => { if (view === 'train') refreshDataset(); }, [view, refreshDataset]);

  // Training metrics
  const isTraining = status.is_training;
  useEffect(() => {
    if (!isTraining) return undefined;
    const tick = () => api.metrics().then((d) => setMetrics({ ...IDLE_METRICS, ...d.metrics })).catch(() => {});
    tick();
    const timer = setInterval(tick, 1000);
    return () => {
      clearInterval(timer);
      // One last read picks up the final status once training stops
      api.metrics().then((d) => {
        const m = { ...IDLE_METRICS, ...d.metrics };
        setMetrics(m);
        if (m.status === 'Complete') toast('success', `Adapter “${m.adapter}” finished training.`);
        if (m.status === 'Failed') toast('error', m.error || 'Training failed.');
        refreshAdapters();
      }).catch(() => {});
    };
  }, [isTraining, toast, refreshAdapters]);

  // --- Model actions ---
  const loadModel = async (id) => {
    try {
      await api.loadModel(id);
      setStatus((s) => ({ ...s, status: 'starting', progress: 0, error: '', load: { model: id, stage: null, download: {} } }));
    } catch (err) {
      toast('error', `Could not start loading: ${err.message}`);
    }
  };

  const cancelDownload = async () => {
    try {
      await api.cancelLoad();
      toast('info', 'Download cancelled.');
    } catch (err) {
      toast('error', `Could not cancel: ${err.message}`);
    }
    refreshStatus();
  };

  const deleteModel = async (id) => {
    try {
      const data = await api.deleteModel(id);
      toast('success', `Deleted ${id} (${formatBytes(data.freed_bytes)} freed).`);
    } catch (err) {
      toast('error', `Could not delete ${id}: ${err.message}`);
    }
    refreshModels();
  };

  const unloadModel = async () => {
    try {
      await api.unloadModel();
      toast('info', 'Model ejected from memory.');
    } catch (err) {
      toast('error', `Could not eject the model: ${err.message}`);
    }
    refreshStatus();
  };

  const loadAdapter = async (name, supportsThinking = null) => {
    if (status.active_adapter) await api.unloadAdapter().catch(() => {});
    await applyAdapter(name, supportsThinking);
  };

  const unloadAdapter = async () => {
    try {
      await api.unloadAdapter();
      toast('info', 'Adapter removed. Using the base model.');
    } catch (err) {
      toast('error', `Could not remove the adapter: ${err.message}`);
    }
    refreshStatus();
  };

  const saveSystemPrompt = async (text) => {
    try {
      const data = await api.setSystemPrompt(text);
      setStatus((s) => ({ ...s, system_prompt: data.system_prompt }));
      toast('success', 'System prompt applied.');
    } catch (err) {
      toast('error', `Could not save the system prompt: ${err.message}`);
    }
  };

  const changeParams = (next) => {
    setParams(next);
    clearTimeout(paramsTimer.current);
    paramsTimer.current = setTimeout(() => {
      api.setParams(next).catch((err) => toast('error', `Could not apply parameters: ${err.message}`));
    }, 400);
  };

  // --- Chat ---
  const send = async (text, base = messages) => {
    const history = base
      .filter((m) => m.role === 'user' || m.role === 'assistant')
      .map((m) => ({ role: m.role, content: m.role === 'assistant' ? splitThinking(m.content).answer : m.content }));
    setMessages([...base, { role: 'user', content: text }]);
    setBusy(true);
    try {
      const data = await api.chat(text, history, thinking && status.thinking_supported);
      const failed = data.content.startsWith('Error during inference') || data.content.startsWith('System: No model loaded');
      setMessages((prev) => [...prev, failed
        ? { role: 'error', content: data.content.replace(/^System: /, '') }
        : { ...data, prompt: text }]);
    } catch {
      setMessages((prev) => [...prev, { role: 'error', content: 'The backend did not respond. Check that main.py is still running.' }]);
      refreshStatus();
    } finally {
      setBusy(false);
    }
  };

  const regenerate = () => {
    const last = messages[messages.length - 1];
    if (!last || last.role !== 'assistant' || !last.prompt) return;
    send(last.prompt, messages.slice(0, -2));
  };

  const rate = async (message, direction) => {
    try {
      await api.score({
        message_id: message.id,
        score: direction === 'up' ? 10 : 2,
        prompt: message.prompt,
        // A thinking adapter learns from its reasoning too, so keep the <think> block for it
        response: status.adapter_supports_thinking ? message.content.trim() : splitThinking(message.content).answer,
      });
      setMessages((prev) => prev.map((m) => (m.id === message.id ? { ...m, rating: direction } : m)));
      if (direction === 'up') refreshDataset();
    } catch (err) {
      toast('error', `Could not save the rating: ${err.message}`);
    }
  };

  // --- Training ---
  const addData = async (text) => {
    try {
      const data = await api.addData(text);
      toast('success', `Added ${data.added} ${data.added === 1 ? 'example' : 'examples'} to the dataset.`);
      refreshDataset();
      return true;
    } catch (err) {
      toast('error', `Could not add the examples: ${err.message}`);
      return false;
    }
  };

  const deleteExample = async (entry) => {
    try {
      await api.deleteExample(entry.line, entry.prompt);
    } catch (err) {
      toast('error', `Could not delete the example: ${err.message}`);
    }
    refreshDataset();
  };

  const clearDataset = async () => {
    try {
      const data = await api.clearDataset();
      toast('success', `Deleted all ${data.removed} examples.`);
    } catch (err) {
      toast('error', `Could not clear the dataset: ${err.message}`);
    }
    refreshDataset();
  };

  const startTraining = async (modelId, adapterName) => {
    try {
      await api.startTraining(modelId, adapterName);
      setMetrics({ ...IDLE_METRICS, status: 'Starting...', model: modelId, adapter: adapterName });
      setStatus((s) => ({ ...s, is_training: true }));
    } catch (err) {
      toast('error', `Could not start training: ${err.message}`);
    }
  };

  const useAdapter = async (modelId, adapterName) => {
    setView('chat');
    if (status.current_model === modelId) {
      await loadAdapter(adapterName);
    } else {
      pendingAdapter.current = { model: modelId, adapter: adapterName };
      await loadModel(modelId);
    }
  };

  // Returns true on success so the form can clear the field
  const saveHfToken = async (token) => {
    try {
      const data = await api.setHfToken(token);
      setHfToken(data);
      toast('success', `Hugging Face token saved for ${data.username}.`);
      return true;
    } catch (err) {
      toast('error', err.message);
      return false;
    }
  };

  const removeHfToken = async () => {
    try {
      setHfToken(await api.removeHfToken());
      toast('info', 'Hugging Face token removed from this machine.');
    } catch (err) {
      toast('error', `Could not remove the token: ${err.message}`);
    }
  };

  const saveCacheDir = async (dir) => {
    try {
      const data = await api.setSettings(dir);
      setCacheDir(data.cache_dir || '');
      await refreshModels();
      toast('success', 'Model folder updated.');
    } catch (err) {
      toast('error', `Could not save settings: ${err.message}`);
    }
  };

  // --- Config panel resize ---
  const startResize = (e) => {
    e.preventDefault();
    const move = (ev) => setPanelWidth(Math.max(280, Math.min(520, window.innerWidth / zoom - ev.clientX / zoom)));
    const stop = () => {
      window.removeEventListener('mousemove', move);
      window.removeEventListener('mouseup', stop);
      document.body.style.userSelect = '';
    };
    document.body.style.userSelect = 'none';
    window.addEventListener('mousemove', move);
    window.addEventListener('mouseup', stop);
  };

  return (
    <div className="flex h-full flex-col bg-canvas text-ink">
      <header className="flex h-14 shrink-0 items-center gap-3 border-b border-line bg-surface px-3">
        <div className="flex w-10 shrink-0 items-center justify-center sm:w-auto sm:gap-2 sm:pr-2">
          <BrainCircuit size={22} className="text-accent" />
          <span className="hidden text-[15px] font-semibold tracking-tight sm:inline">NexusAI</span>
        </div>

        <ModelPicker status={status} models={models} sizes={modelSizes} online={online} open={pickerOpen} setOpen={setPickerOpen} onLoad={loadModel} onUnload={unloadModel} onCancelDownload={cancelDownload} onDelete={deleteModel} onOpenSettings={() => setView('settings')} />

        <div className="ml-auto flex shrink-0 items-center gap-2">
          {isTraining && (
            <button onClick={() => setView('train')} className="hidden items-center gap-1.5 rounded-full bg-accent-soft px-2.5 py-1 text-xs font-medium text-accent md:flex">
              <FlaskConical size={12} /> Training {Math.round(metrics.progress || 0)}%
            </button>
          )}
          {online && status.device && (
            <span className="hidden items-center gap-1.5 rounded-full border border-line px-2.5 py-1 text-xs text-muted md:flex" title="Device the backend runs models on">
              <Cpu size={12} /> {DEVICE_LABELS[status.device] || status.device}
            </span>
          )}
          <span className="flex items-center gap-1.5 text-xs text-muted" title={online ? 'Backend connected' : 'Backend unreachable on port 8000'}>
            <span className={`h-2 w-2 rounded-full ${online ? 'bg-good' : 'bg-bad'}`} />
            <span className="hidden sm:inline">{online ? 'Connected' : 'Offline'}</span>
          </span>
        </div>
      </header>

      <div className="flex min-h-0 flex-1">
        <nav aria-label="Sections" className="flex w-16 shrink-0 flex-col items-center gap-1 border-r border-line bg-surface py-3">
          {NAV.map(({ id, label, icon }) => (
            <button
              key={id}
              onClick={() => setView(id)}
              aria-current={view === id ? 'page' : undefined}
              className={`flex w-14 flex-col items-center gap-1 rounded-lg py-2 text-[11px] font-medium transition-colors ${view === id ? 'bg-accent-soft text-accent' : 'text-muted hover:bg-raised hover:text-ink'}`}
            >
              {React.createElement(icon, { size: 18 })}
              {label}
            </button>
          ))}
          <button
            onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
            aria-label={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}
            title={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}
            className="mt-auto flex h-9 w-9 items-center justify-center rounded-lg text-muted hover:bg-raised hover:text-ink"
          >
            {theme === 'dark' ? <Sun size={17} /> : <Moon size={17} />}
          </button>
        </nav>

        <main className="relative flex min-w-0 flex-1">
          {view === 'chat' && (
            <>
              <ChatView
                status={status}
                online={online}
                messages={messages}
                busy={busy}
                thinking={thinking}
                onToggleThinking={() => setThinking(!thinking)}
                onSend={(text) => send(text)}
                onRegenerate={regenerate}
                onRate={rate}
                onNewChat={() => setMessages([])}
                configOpen={configOpen}
                onToggleConfig={() => setConfigOpen(!configOpen)}
                onOpenPicker={() => setPickerOpen(true)}
              />
              {configOpen && (
                <>
                  {/* On narrow windows the panel overlays the chat instead of squeezing it */}
                  <div className="absolute inset-0 z-20 bg-black/40 lg:hidden" onClick={() => setConfigOpen(false)} />
                  <aside
                    aria-label="Configuration"
                    style={{ width: panelWidth }}
                    className="absolute inset-y-0 right-0 z-30 flex max-w-[90%] border-l border-line shadow-xl lg:static lg:max-w-none lg:shadow-none"
                  >
                    <div onMouseDown={startResize} className="hidden w-1 shrink-0 cursor-col-resize hover:bg-accent lg:block" title="Drag to resize" />
                    <div className="min-w-0 flex-1">
                      <ConfigPanel
                        status={status}
                        adapters={adapters}
                        params={params}
                        onParamsChange={changeParams}
                        onSystemPrompt={saveSystemPrompt}
                        onLoadAdapter={loadAdapter}
                        onUnloadAdapter={unloadAdapter}
                        onGoTrain={() => setView('train')}
                        onClose={() => setConfigOpen(false)}
                      />
                    </div>
                  </aside>
                </>
              )}
            </>
          )}
          {view === 'train' && (
            <TrainView
              status={status}
              models={models}
              dataset={dataset}
              metrics={metrics}
              isTraining={isTraining}
              online={online}
              onAddData={addData}
              onDeleteExample={deleteExample}
              onClearDataset={clearDataset}
              onStart={startTraining}
              onUseAdapter={useAdapter}
            />
          )}
          {view === 'settings' && (
            <SettingsView
              key={cacheDir}
              cacheDir={cacheDir}
              onSaveCacheDir={saveCacheDir}
              theme={theme}
              onTheme={setTheme}
              zoom={zoom}
              onZoom={setZoom}
              modelCount={models.length}
              hfToken={hfToken}
              onSaveHfToken={saveHfToken}
              onRemoveHfToken={removeHfToken}
            />
          )}
          {view === 'help' && <HelpView onGo={setView} />}
        </main>
      </div>

      <Toasts toasts={toasts} onDismiss={(id) => setToasts((t) => t.filter((x) => x.id !== id))} />
    </div>
  );
}
