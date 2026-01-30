import React, { useState, useEffect, useRef } from 'react';
import {
  Cpu, Database, BrainCircuit, Settings, Send,
  CheckCircle, TrendingDown, BarChart3, Zap, Activity, Box, HardDrive, Loader, HelpCircle, Trash2,
  Sun, Moon, Type
} from 'lucide-react';

const API_BASE = "http://localhost:8000";

export default function App() {
  const [gpuEnabled, setGpuEnabled] = useState(false);
  const [messages, setMessages] = useState([
    { role: 'assistant', content: "System initialized. Ready to load 3B model and custom knowledge." }
  ]);
  const [input, setInput] = useState("");
  const [knowledge, setKnowledge] = useState([]);
  const [isTraining, setIsTraining] = useState(false);
  const [metrics, setMetrics] = useState({ loss: 0, step: 0, perplexity: 0 });
  const [activeTab, setActiveTab] = useState('chat');
  const [modelId, setModelId] = useState("stabilityai/stablelm-zephyr-3b");
  const [downloadedModels, setDownloadedModels] = useState([]);
  const [params, setParams] = useState({ temperature: 0.7, top_p: 0.9, max_new_tokens: 200 });
  const [cacheDir, setCacheDir] = useState("");
  const [darkMode, setDarkMode] = useState(true);
  const [isThinking, setIsThinking] = useState(false);

  // Model Loading State
  const [loadingStatus, setLoadingStatus] = useState("idle"); // idle, downloading, loading, ready, error
  const [loadingProgress, setLoadingProgress] = useState(0);
  const [loadingError, setLoadingError] = useState("");
  const [activeAdapter, setActiveAdapter] = useState(null);
  const [systemPrompt, setSystemPrompt] = useState("You are a loving, supportive girlfriend. You speak casually and affectionately. You do NOT identify as an AI.");
  const [zoomLevel, setZoomLevel] = useState(1.0);
  const [adapterName, setAdapterName] = useState("my_adapter");
  const [availableAdapters, setAvailableAdapters] = useState([]);
  const [selectedAdapter, setSelectedAdapter] = useState("");
  const [enableThinking, setEnableThinking] = useState(true);

  // Sidebar Resizing State
  const [sidebarWidth, setSidebarWidth] = useState(340);
  const [isResizing, setIsResizing] = useState(false);

  const scrollRef = useRef(null);
  const sidebarRef = useRef(null);

  // Resize Handler
  useEffect(() => {
    const handleMouseMove = (e) => {
      if (!isResizing) return;
      // Limit width between 200px and 800px
      const newWidth = Math.max(200, Math.min(800, e.clientX - 16)); // 16 is approx padding
      setSidebarWidth(newWidth);
    };

    const handleMouseUp = () => setIsResizing(false);

    if (isResizing) {
      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseup', handleMouseUp);
      document.body.style.cursor = 'col-resize';
      document.body.style.userSelect = 'none'; // Prevent text selection
    } else {
      document.body.style.cursor = 'default';
      document.body.style.userSelect = 'auto';
    }

    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isResizing]);

  // Detect WebGPU
  useEffect(() => {
    if ("gpu" in navigator) setGpuEnabled(true);
  }, []);

  // Poll for Training Metrics if training is active
  useEffect(() => {
    let interval;
    if (isTraining) {
      interval = setInterval(async () => {
        try {
          const res = await fetch(`${API_BASE}/v1/metrics`);
          const data = await res.json();
          setMetrics(data.metrics);
          if (!data.is_training) setIsTraining(false);
        } catch (err) {
          console.error("Backend offline or error fetching metrics.");
        }
      }, 800);
    }
    return () => clearInterval(interval);
  }, [isTraining]);

  // Poll for Model Loading Status
  useEffect(() => {
    let interval;
    if (loadingStatus === "downloading" || loadingStatus === "loading" || loadingStatus === "starting") {
      interval = setInterval(async () => {
        try {
          const res = await fetch(`${API_BASE}/v1/model/status`);
          const data = await res.json();
          setLoadingStatus(data.status);
          setLoadingProgress(data.progress || 0);
          setActiveAdapter(data.active_adapter);
          if (data.status === "error") {
            setLoadingError(data.error);
          }
          if (data.status === "ready") {
            setMessages(prev => [...prev, { role: 'assistant', content: `System: Model ${data.current_model} is ready.` }]);
          }
        } catch (err) {
          console.error("Error polling model status");
        }
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [loadingStatus]);

  // Auto-scroll to bottom of chat
  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim()) return;
    const userMsg = { role: 'user', content: input };
    setMessages(prev => [...prev, userMsg]);
    setInput("");
    setIsThinking(true);

    try {
      const res = await fetch(`${API_BASE}/v1/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: input, enable_thinking: enableThinking })
      });
      const data = await res.json();
      setMessages(prev => [...prev, { ...data, needsScoring: true }]);
    } catch (err) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: "Error: Backend unreachable. Ensure the Python server (main.py) is running on port 8000."
      }]);
    } finally {
      setIsThinking(false);
    }
  };

  const startTraining = async () => {
    // Validation
    const safeName = adapterName.trim();
    if (!safeName || safeName.includes(" ")) {
      alert("Adapter name must be a single word (no spaces)!");
      return;
    }

    try {
      const res = await fetch(`${API_BASE}/v1/train/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_id: modelId,
          adapter_name: safeName
        })
      });
      const data = await res.json();
      if (data.status === "Training started") {
        setIsTraining(true);
      }
    } catch (err) {
      alert("Could not connect to training server. Please verify backend is running.");
    }
  };

  const submitScore = async (id, score) => {
    // Find the message being scored
    const messageIndex = messages.findIndex(m => m.id === id);
    if (messageIndex === -1) return;

    const ratedMessage = messages[messageIndex];
    // Attempt to find preceding user prompt
    let prompt = "";
    if (messageIndex > 0 && messages[messageIndex - 1].role === 'user') {
      prompt = messages[messageIndex - 1].content;
    }

    try {
      const res = await fetch(`${API_BASE}/v1/score`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message_id: id,
          score,
          prompt,
          response: ratedMessage.content
        })
      });
      if (res.ok) {
        setMessages(prev => prev.map(m => m.id === id ? { ...m, needsScoring: false } : m));
      }
    } catch (err) {
      console.error("Score submission failed");
    }
  };

  const handleLoadModel = async () => {
    setLoadingStatus("starting");
    setLoadingProgress(0);
    setLoadingError("");

    try {
      const res = await fetch(`${API_BASE}/v1/model/load`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_id: modelId })
      });
      const data = await res.json();
      if (!res.ok) {
        setLoadingStatus("error");
        setLoadingError(data.detail || "Failed to start load");
        alert("Failed to start load: " + (data.detail || "Unknown error"));
      }
    } catch (err) {
      setLoadingStatus("error");
      setLoadingError("Backend unreachable");
      alert("Error calling backend.");
    }
  };

  const loadSettings = async () => {
    try {
      const res = await fetch(`${API_BASE}/v1/settings/get`);
      const data = await res.json();
      setCacheDir(data.cache_dir);
    } catch (err) {
      console.error("Failed to load settings");
    }
  };

  const saveSettings = async () => {
    try {
      const res = await fetch(`${API_BASE}/v1/settings/update`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ cache_dir: cacheDir })
      });
      if (res.ok) alert("Settings saved!");
    } catch (err) {
      alert("Failed to save settings");
    }
  };

  useEffect(() => {
    loadSettings();
    fetchDownloadedModels();
    fetchParams();
  }, []);

  const fetchParams = async () => {
    try {
      const res = await fetch(`${API_BASE}/v1/parameters/get`);
      const data = await res.json();
      setParams(data.params);
    } catch (err) {
      console.error("Failed to fetch params");
    }
  };

  const updateParams = async () => {
    try {
      const res = await fetch(`${API_BASE}/v1/parameters/update`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params)
      });
      if (res.ok) alert("Parameters applied!");
    } catch (err) {
      alert("Failed to update parameters");
    }
  };

  const fetchAdapters = async () => {
    try {
      const res = await fetch(`${API_BASE}/v1/adapter/list`);
      const data = await res.json();
      setAvailableAdapters(data.adapters || []);
    } catch (err) { console.error("Failed to fetch adapters"); }
  };

  // Fetch adapters when Model Tab is active or model changes
  useEffect(() => {
    if (loadingStatus === 'ready' || activeTab === 'model') {
      fetchAdapters();
    }
  }, [loadingStatus, activeTab]);

  const fetchDownloadedModels = async () => {
    try {
      const res = await fetch(`${API_BASE}/v1/model/list`);
      const data = await res.json();
      setDownloadedModels(data.models || []);
    } catch (err) { console.error("Failed to fetch models"); }
  };

  const handleUploadData = async () => {
    if (!input.trim()) return;
    try {
      const res = await fetch(`${API_BASE}/v1/training/data`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ data: input })
      });
      const data = await res.json();
      alert(`Successfully added ${data.added} training samples!`);
      setInput("");
    } catch (e) { alert("Upload failed"); }
  };
  const handleUnloadModel = async () => {
    try {
      await fetch(`${API_BASE}/v1/model/unload`, { method: 'POST' });
      setLoadingStatus("idle");
      setLoadingProgress(0);
      setMessages(prev => [...prev, { role: 'assistant', content: "System: Model unloaded from memory." }]);
    } catch (err) {
      console.error("Failed to unload model");
    }
  };

  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [darkMode]);

  return (
    <div>
      <div
        className="h-screen bg-slate-50 dark:bg-slate-900 flex flex-col font-sans overflow-hidden transition-colors duration-300"
        style={{ zoom: zoomLevel }}
      >
        {/* Header */}
        <header className="flex items-center justify-between p-4 bg-white dark:bg-slate-800 border-b dark:border-slate-700 shadow-sm transition-colors">
          <div className="flex items-center gap-2">
            <BrainCircuit className="text-indigo-600 dark:text-indigo-400" />
            <h1 className="font-bold text-lg dark:text-white">Nexus 3B Lab</h1>
          </div>
          <div className="flex items-center gap-4">
            <button
              onClick={() => setEnableThinking(!enableThinking)}
              className={`p-2 rounded-lg transition-colors ${enableThinking ? 'bg-indigo-100 text-indigo-600 dark:bg-indigo-900/30 dark:text-indigo-400' : 'bg-gray-100 dark:bg-slate-700 text-gray-400 dark:text-gray-500 hover:bg-gray-200 dark:hover:bg-slate-600'}`}
              title={enableThinking ? "Thinking Enabled" : "Thinking Disabled"}
            >
              <BrainCircuit size={18} />
            </button>
            <button
              onClick={() => setZoomLevel(prev => prev >= 1.2 ? 1.0 : prev + 0.1)}
              className="p-2 rounded-lg bg-gray-100 dark:bg-slate-700 text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-slate-600 transition-colors"
              title="Toggle Font Size"
            >
              <Type size={18} />
            </button>
            <button
              onClick={() => setActiveTab('help')}
              className={`p-2 rounded-lg transition-colors ${activeTab === 'help' ? 'bg-indigo-100 text-indigo-600 dark:bg-indigo-900/30 dark:text-indigo-400' : 'bg-gray-100 dark:bg-slate-700 text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-slate-600'}`}
              title="Documentation & Help"
            >
              <HelpCircle size={18} />
            </button>
            <button
              onClick={() => setDarkMode(!darkMode)}
              className="p-2 rounded-lg bg-gray-100 dark:bg-slate-700 text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-slate-600 transition-colors"
            >
              {darkMode ? <Sun size={18} /> : <Moon size={18} />}
            </button>
            <div className={`text-xs px-3 py-1 rounded-full font-medium ${gpuEnabled ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400' : 'bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400'}`}>
              {gpuEnabled ? 'WebGPU Detected' : 'CPU Only Mode'}
            </div>
            {activeAdapter && (
              <div className="text-xs px-3 py-1 rounded-full font-medium bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-400 flex items-center gap-1">
                <Zap size={12} />
                {activeAdapter}
              </div>
            )}
          </div>
        </header>

        <main className="flex-1 flex overflow-hidden p-4">
          {/* Resizable Sidebar */}
          <div
            ref={sidebarRef}
            style={{ width: sidebarWidth }}
            className="flex flex-col gap-4 overflow-y-auto shrink-0 transition-opacity"
          >
            <div className="bg-white dark:bg-slate-800 rounded-xl shadow-sm border dark:border-slate-700 p-4 transition-colors">
              <div className="flex gap-2 mb-4">
                <button
                  onClick={() => setActiveTab('chat')}
                  className={`flex-1 py-2 rounded-lg text-xs font-bold transition-colors ${activeTab === 'chat' ? 'bg-indigo-600 text-white shadow-md' : 'bg-gray-100 dark:bg-slate-700 text-gray-500 dark:text-gray-400'}`}
                >
                  Chat
                </button>
                <button
                  onClick={() => setActiveTab('train')}
                  className={`flex-1 py-2 rounded-lg text-xs font-bold transition-colors ${activeTab === 'train' ? 'bg-purple-600 text-white shadow-md' : 'bg-gray-100 dark:bg-slate-700 text-gray-500 dark:text-gray-400'}`}
                >
                  Train
                </button>
                <button
                  onClick={() => setActiveTab('model')}
                  className={`flex-1 py-2 rounded-lg text-xs font-bold transition-colors ${activeTab === 'model' ? 'bg-emerald-600 text-white shadow-md' : 'bg-gray-100 dark:bg-slate-700 text-gray-500 dark:text-gray-400'}`}
                >
                  Model
                </button>
                <button
                  onClick={() => setActiveTab('settings')}
                  className={`flex-1 py-2 rounded-lg text-xs font-bold transition-colors ${activeTab === 'settings' ? 'bg-slate-600 text-white shadow-md' : 'bg-gray-100 dark:bg-slate-700 text-gray-500 dark:text-gray-400'}`}
                >
                  Settings
                </button>
              </div>

              {activeTab === 'train' ? (
                <div className="space-y-4">
                  <div className="bg-slate-900 dark:bg-slate-950 rounded-lg p-4 text-white">
                    <div className="text-[10px] text-slate-400 uppercase font-bold mb-3 tracking-widest">Training Metrics</div>
                    <div className="flex justify-between items-center pb-2 border-b border-slate-700 mb-2">
                      <span className="text-xs text-slate-300">Status</span>
                      <span className="font-mono text-xs text-amber-400">{metrics.status || (isTraining ? "Running..." : "Idle")}</span>
                    </div>
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-xs text-slate-300">Loss</span>
                      <span className="font-mono text-emerald-400">{metrics.loss?.toFixed(4) || "0.0000"}</span>
                    </div>
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-xs text-slate-300">Perplexity</span>
                      <span className="font-mono text-sky-400">{metrics.perplexity?.toFixed(2) || "0.00"}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-xs text-slate-300">Epoch / Steps</span>
                      <span className="font-mono text-purple-400">{metrics.step}</span>
                    </div>
                  </div>

                  <div className="space-y-4 mb-4">
                    <div>
                      <label className="text-[10px] text-gray-400 uppercase font-bold">New Adapter Name (Single Word)</label>
                      <input
                        value={adapterName}
                        onChange={(e) => setAdapterName(e.target.value)}
                        placeholder="e.g. pirate_v1"
                        className="w-full text-xs p-2 mt-1 border dark:border-slate-600 rounded-lg bg-gray-50 dark:bg-slate-900 dark:text-gray-200 focus:outline-none focus:ring-2 focus:ring-purple-500 font-mono"
                      />
                    </div>

                    <div className="pt-2 border-t border-slate-100 dark:border-slate-700">
                      <label className="text-[10px] text-gray-400 uppercase font-bold">Target Base Model</label>
                      <select
                        value={modelId}
                        onChange={(e) => setModelId(e.target.value)}
                        className="w-full text-xs p-2 border dark:border-slate-600 rounded-lg bg-gray-50 dark:bg-slate-900 dark:text-gray-200 focus:outline-none focus:ring-2 focus:ring-purple-500"
                      >
                        <option value="">-- Select Downloaded Model --</option>
                        {downloadedModels.map(m => (
                          <option key={m} value={m}>{m}</option>
                        ))}
                        <option value="stabilityai/stablelm-zephyr-3b">stabilityai/stablelm-zephyr-3b (Default)</option>
                      </select>

                    </div>

                  </div>
                  <button
                    onClick={startTraining}
                    disabled={isTraining}
                    className={`w-full py-3 rounded-lg text-sm font-bold flex items-center justify-center gap-2 transition-all ${isTraining ? 'bg-gray-200 dark:bg-slate-700 text-gray-400 cursor-not-allowed' : 'bg-indigo-600 text-white shadow-lg hover:bg-indigo-700'}`}
                  >
                    {isTraining ? <Activity className="animate-spin" size={16} /> : <Zap size={16} />}
                    {isTraining ? 'Training in Progress...' : 'Start Local Fine-Tuning'}
                  </button>
                  <div className="text-[10px] text-gray-400 text-center px-2">
                    Fine-tuning initiates a 4-bit QLoRA cycle on the 3B model parameters using available VRAM.
                  </div>
                </div>
              ) : activeTab === 'model' ? (
                <div className="space-y-4">
                  <div className="text-xs font-bold text-gray-500 dark:text-gray-400 uppercase tracking-wider">Model Configuration</div>

                  {downloadedModels.length > 0 && (
                    <div className="space-y-2">
                      <label className="text-[10px] text-gray-400 uppercase font-bold">Downloaded Models</label>
                      <select
                        onChange={(e) => {
                          if (e.target.value) setModelId(e.target.value);
                        }}
                        className="w-full text-xs p-2 border dark:border-slate-600 rounded-lg bg-gray-50 dark:bg-slate-900 dark:text-gray-200 focus:outline-none focus:ring-2 focus:ring-emerald-500"
                      >
                        <option value="">-- Select from Cache --</option>
                        {downloadedModels.map(m => (
                          <option key={m} value={m}>{m}</option>
                        ))}
                      </select>
                    </div>
                  )}

                  <div className="space-y-2">
                    <label className="text-[10px] text-gray-400 uppercase font-bold">HuggingFace Model ID</label>
                    <input
                      value={modelId}
                      onChange={(e) => setModelId(e.target.value)}
                      className="w-full text-xs p-2 border dark:border-slate-600 rounded-lg bg-gray-50 dark:bg-slate-900 dark:text-gray-200 focus:outline-none focus:ring-2 focus:ring-emerald-500"
                    />
                  </div>

                  <div className="bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-lg p-4 space-y-3">
                    <div className="text-[10px] font-bold text-gray-500 dark:text-gray-400 uppercase tracking-wider flex items-center gap-2">
                      <Settings size={12} /> Inference Parameters
                    </div>

                    <div className="space-y-1">
                      <div className="flex justify-between text-[10px] uppercase font-bold text-gray-400">
                        <span>Temperature</span>
                        <span>{params.temperature}</span>
                      </div>
                      <input
                        type="range" min="0.1" max="2.0" step="0.1"
                        value={params.temperature}
                        onChange={(e) => setParams({ ...params, temperature: parseFloat(e.target.value) })}
                        className="w-full accent-indigo-600 h-1 bg-gray-200 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer"
                      />
                    </div>

                    <div className="space-y-1">
                      <div className="flex justify-between text-[10px] uppercase font-bold text-gray-400">
                        <span>Top P</span>
                        <span>{params.top_p}</span>
                      </div>
                      <input
                        type="range" min="0.1" max="1.0" step="0.05"
                        value={params.top_p}
                        onChange={(e) => setParams({ ...params, top_p: parseFloat(e.target.value) })}
                        className="w-full accent-indigo-600 h-1 bg-gray-200 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer"
                      />
                    </div>

                    <div className="space-y-1">
                      <div className="flex justify-between text-[10px] uppercase font-bold text-gray-400">
                        <span>Max Tokens</span>
                      </div>
                      <input
                        type="number"
                        value={params.max_new_tokens}
                        onChange={(e) => setParams({ ...params, max_new_tokens: parseInt(e.target.value) })}
                        className="w-full text-xs p-2 border dark:border-slate-600 rounded-lg bg-white dark:bg-slate-800 dark:text-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
                      />
                    </div>

                    <button
                      onClick={updateParams}
                      className="w-full py-2 rounded-md text-xs font-bold bg-indigo-100 dark:bg-indigo-900/40 text-indigo-700 dark:text-indigo-300 hover:bg-indigo-200 dark:hover:bg-indigo-900/60 transition-colors"
                    >
                      Apply Parameters
                    </button>
                  </div>

                  <div className="flex gap-2">
                    <button
                      onClick={handleLoadModel}
                      disabled={loadingStatus === 'downloading' || loadingStatus === 'loading' || loadingStatus === 'starting'}
                      className={`flex-1 py-3 rounded-lg text-sm font-bold flex items-center justify-center gap-2 transition-all ${loadingStatus === 'ready' ? 'bg-green-600 hover:bg-green-700' : 'bg-emerald-600 hover:bg-emerald-700'} text-white shadow-lg disabled:opacity-50 disabled:cursor-not-allowed`}
                    >
                      {loadingStatus === 'ready' ? <CheckCircle size={16} /> : <HardDrive size={16} />}
                      {loadingStatus === 'ready' ? 'Model Ready' : 'Load Model'}
                    </button>

                    {loadingStatus === 'ready' && (
                      <button
                        onClick={handleUnloadModel}
                        className="px-4 py-3 rounded-lg bg-red-100 dark:bg-red-900/30 text-red-600 dark:text-red-400 hover:bg-red-200 dark:hover:bg-red-900/50 transition-colors shadow-sm"
                        title="Unload Model"
                      >
                        <Trash2 size={16} />
                      </button>
                    )}
                  </div>

                  {loadingStatus === 'ready' && (
                    <div className="pt-2 border-t border-slate-100 dark:border-slate-700">
                      <div className="text-[10px] text-gray-400 uppercase font-bold mb-2">Fine-Tuned Adapters</div>
                      {!activeAdapter ? (
                        <div className="space-y-2">
                          {/* Adapter Selector */}
                          <select
                            value={selectedAdapter}
                            onChange={(e) => setSelectedAdapter(e.target.value)}
                            className="w-full text-xs p-2 border dark:border-slate-600 rounded-lg bg-gray-50 dark:bg-slate-900 dark:text-gray-200 focus:outline-none focus:ring-2 focus:ring-purple-500"
                          >
                            <option value="">-- Select Adapter to Load --</option>
                            {availableAdapters.map(a => (
                              <option key={a} value={a}>{a}</option>
                            ))}
                          </select>

                          <textarea
                            value={systemPrompt}
                            onChange={(e) => setSystemPrompt(e.target.value)}
                            className="w-full text-xs p-2 mb-2 border dark:border-slate-600 rounded-lg bg-gray-50 dark:bg-slate-900 dark:text-gray-200 focus:outline-none focus:ring-2 focus:ring-purple-500 h-16"
                            placeholder="System Prompt (Define Persona)..."
                          />
                          <button
                            onClick={async () => {
                              if (!selectedAdapter && availableAdapters.length > 0) {
                                alert("Please select an adapter from the list.");
                                return;
                              }

                              try {
                                const res = await fetch(`${API_BASE}/v1/adapter/load`, {
                                  method: 'POST',
                                  headers: { 'Content-Type': 'application/json' },
                                  body: JSON.stringify({
                                    system_prompt: systemPrompt,
                                    adapter_name: selectedAdapter === "(Legacy Root Adapter)" ? "" : selectedAdapter
                                  })
                                });
                                const data = await res.json();
                                if (res.ok) {
                                  setActiveAdapter(data.adapter);
                                } else {
                                  alert("Failed to load adapter. " + data.detail);
                                }
                              } catch (e) { alert("Error connecting to backend"); }
                            }}
                            className="w-full py-2 rounded-lg text-xs font-bold border border-purple-500 text-purple-600 dark:text-purple-400 hover:bg-purple-50 dark:hover:bg-purple-900/20 transition-colors"
                          >
                            Load Selected Adapter
                          </button>
                        </div>
                      ) : (
                        <button
                          onClick={async () => {
                            try {
                              await fetch(`${API_BASE}/v1/adapter/unload`, { method: 'POST' });
                              setActiveAdapter(null);
                            } catch (e) { alert("Error unloading adapter"); }
                          }}
                          className="w-full py-2 rounded-lg text-xs font-bold bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-400 hover:bg-purple-200 transition-colors"
                        >
                          Disable Current Adapter
                        </button>
                      )}
                    </div>
                  )}

                  {/* Progress Bar Area */}
                  {(loadingStatus === 'downloading' || loadingStatus === 'loading' || loadingStatus === 'starting') && (
                    <div className="bg-white dark:bg-slate-800 border dark:border-slate-700 rounded-lg p-3 shadow-sm animate-in fade-in slide-in-from-top-2 duration-300">
                      <div className="flex justify-between items-center mb-2">
                        <span className="text-[10px] font-bold uppercase text-emerald-600 dark:text-emerald-400 flex items-center gap-1">
                          <Loader size={10} className="animate-spin" />
                          {loadingStatus === 'downloading' ? 'Downloading Assets...' : 'Loading Weights...'}
                        </span>
                        <span className="text-[10px] font-mono text-gray-500 dark:text-gray-400">{loadingProgress.toFixed(0)}%</span>
                      </div>
                      <div className="w-full bg-gray-100 dark:bg-slate-700 rounded-full h-2 overflow-hidden">
                        <div
                          className="bg-emerald-500 h-full transition-all duration-500 ease-out relative"
                          style={{ width: `${loadingProgress}%` }}
                        >
                          <div className="absolute inset-0 bg-white/20 animate-[shimmer_1s_infinite] border-r border-emerald-400/50"></div>
                        </div>
                      </div>
                    </div>
                  )}

                  {loadingStatus === 'error' && (
                    <div className="p-3 bg-red-50 dark:bg-red-900/20 border border-red-100 dark:border-red-900/50 rounded-lg text-[11px] text-red-600 dark:text-red-400">
                      <strong>Error:</strong> {loadingError}
                    </div>
                  )}

                  {loadingStatus === 'ready' && (
                    <div className="p-3 bg-green-50 dark:bg-green-900/20 border border-green-100 dark:border-green-900/50 rounded-lg text-[11px] text-green-700 dark:text-green-400 flex items-center gap-2">
                      <CheckCircle size={12} />
                      Model loaded and ready for inference.
                    </div>
                  )}

                  <div className="p-3 bg-emerald-50 dark:bg-emerald-900/20 border border-emerald-100 dark:border-emerald-900/50 rounded-lg text-[11px] text-emerald-700 dark:text-emerald-400 leading-relaxed">
                    Loading a model requires downloading tensor files. Ensure you have sufficient VRAM.
                  </div>
                </div>
              ) : activeTab === 'settings' ? (
                <div className="space-y-4">
                  <div className="text-xs font-bold text-gray-500 dark:text-gray-400 uppercase tracking-wider">System Settings</div>
                  <div className="space-y-2">
                    <label className="text-[10px] text-gray-400 uppercase font-bold">Local Model Cache Directory</label>
                    <input
                      value={cacheDir}
                      onChange={(e) => setCacheDir(e.target.value)}
                      placeholder="/path/to/models (Leave empty for default)"
                      className="w-full text-xs p-2 border dark:border-slate-600 rounded-lg bg-gray-50 dark:bg-slate-900 dark:text-gray-200 focus:outline-none focus:ring-2 focus:ring-slate-500 font-mono"
                    />
                    <div className="text-[10px] text-gray-400">
                      Models will be downloaded here. Useful for external drives or shared volumes.
                    </div>
                  </div>
                  <button
                    onClick={saveSettings}
                    className="w-full py-3 rounded-lg text-sm font-bold flex items-center justify-center gap-2 bg-slate-600 text-white shadow-lg hover:bg-slate-700 transition-all"
                  >
                    <Settings size={16} />
                    Save Configuration
                  </button>

                  <div className="pt-4 border-t border-gray-100 dark:border-slate-700">
                    <button
                      onClick={() => setActiveTab('help')}
                      className="text-[10px] text-indigo-500 dark:text-indigo-400 font-bold flex events-center gap-1 hover:underline"
                    >
                      <HelpCircle size={12} />
                      Need help understanding parameters? Read the guide.
                    </button>
                  </div>
                </div>
              ) : activeTab === 'help' ? (
                <div className="space-y-6 overflow-y-auto pr-2 h-full">

                  {/* Section 1: Fine-Tuning */}
                  <div>
                    <div className="text-xs font-bold text-indigo-600 dark:text-indigo-400 uppercase tracking-wider mb-2 flex items-center gap-2">
                      <Database size={14} /> 1. Fine-Tuning Guide
                    </div>
                    <div className="bg-white dark:bg-slate-800 border dark:border-slate-700 rounded-lg p-4 shadow-sm space-y-3">
                      <h3 className="text-xs font-bold text-slate-700 dark:text-slate-200">Preparing Your Data</h3>
                      <p className="text-[11px] text-gray-600 dark:text-gray-300 leading-relaxed">
                        Fine-tuning allows you to teach the model a new way of speaking ("Style") or specific facts ("Knowledge").
                        We use the <strong>JSONL</strong> format, where each line is a valid JSON object representing a single training example.
                      </p>

                      <div className="bg-slate-50 dark:bg-slate-900 border dark:border-slate-700 p-3 rounded-md">
                        <div className="text-[10px] font-bold text-slate-500 mb-1">training_data.jsonl</div>
                        <code className="text-[10px] font-mono text-emerald-600 dark:text-emerald-400 block whitespace-pre overflow-x-auto">
                          {`{"prompt": "hi", "response": "Hey you 😊 I was hoping you'd show up. How’s your day going so far?", "score": 10, "source": "human_feedback"}\n{"prompt": "hello", "response": "Hi! It’s nice to see you here. What are we talking about today?", "score": 9, "source": "human_feedback"}`}
                        </code>
                      </div>

                      <div className="grid grid-cols-2 gap-2 mt-2">
                        <div className="p-2 bg-indigo-50 dark:bg-indigo-900/10 rounded border border-indigo-100 dark:border-indigo-900/30">
                          <span className="text-[10px] font-bold text-indigo-700 dark:text-indigo-400 block mb-1">Quantity</span>
                          <span className="text-[10px] text-indigo-600 dark:text-indigo-300">Aim for <strong>50-100 examples</strong> for a strong style transfer. <br />10-20 is enough for simple tests.</span>
                        </div>
                        <div className="p-2 bg-amber-50 dark:bg-amber-900/10 rounded border border-amber-100 dark:border-amber-900/30">
                          <span className="text-[10px] font-bold text-amber-700 dark:text-amber-400 block mb-1">Quality</span>
                          <span className="text-[10px] text-amber-600 dark:text-amber-300">Diverse examples are better. Don't just repeat "Hello". Cover different topics.</span>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Section 2: Persona Design */}
                  <div>
                    <div className="text-xs font-bold text-purple-600 dark:text-purple-400 uppercase tracking-wider mb-2 flex items-center gap-2">
                      <BrainCircuit size={14} /> 2. Creating a Persona
                    </div>
                    <div className="bg-white dark:bg-slate-800 border dark:border-slate-700 rounded-lg p-4 shadow-sm space-y-3">
                      <p className="text-[11px] text-gray-600 dark:text-gray-300 leading-relaxed">
                        A strong persona (e.g., "The Girlfriend", "The Pirate", "The Coder") relies on two components working together:
                      </p>

                      <div className="space-y-2">
                        <div className="flex gap-3 items-start">
                          <div className="w-6 h-6 rounded-full bg-purple-100 dark:bg-purple-900/30 flex items-center justify-center shrink-0 mt-0.5">
                            <span className="text-[10px] font-bold text-purple-600">A</span>
                          </div>
                          <div>
                            <div className="text-xs font-bold text-slate-700 dark:text-slate-200">The Adapter (The "Voice")</div>
                            <p className="text-[10px] text-slate-500 dark:text-slate-400">
                              Fine-tuning changes <strong>how</strong> the model speaks. It learns slang, tone, and sentence structure from your JSONL data.
                            </p>
                          </div>
                        </div>

                        <div className="flex gap-3 items-start">
                          <div className="w-6 h-6 rounded-full bg-pink-100 dark:bg-pink-900/30 flex items-center justify-center shrink-0 mt-0.5">
                            <span className="text-[10px] font-bold text-pink-600">B</span>
                          </div>
                          <div>
                            <div className="text-xs font-bold text-slate-700 dark:text-slate-200">The System Prompt (The "Soul")</div>
                            <p className="text-[10px] text-slate-500 dark:text-slate-400">
                              This is the core instruction sent before every message. It defines <strong>who</strong> the model is.
                              <br />
                              <em>Example: "You are a loving girlfriend. You speak casually."</em>
                            </p>
                          </div>
                        </div>
                      </div>

                      <div className="p-3 bg-slate-50 dark:bg-slate-900 rounded border border-slate-100 dark:border-slate-700 mt-2">
                        <div className="text-[10px] font-bold text-slate-600 dark:text-slate-300 mb-1">Pro Tip: The "Girlfriend" Fix</div>
                        <p className="text-[10px] text-slate-500 dark:text-slate-400">
                          Base models are trained to be "helpful assistants" and will refuse personal questions.
                          <br />
                          <strong>You MUST set a System Prompt</strong> in the Model tab (e.g., "You do NOT identify as an AI") to override this behavior.
                        </p>
                      </div>
                    </div>
                  </div>

                  {/* Section 3: Parameters */}
                  <div>
                    <div className="text-xs font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-2 flex items-center gap-2">
                      <Settings size={14} /> 3. Inference Parameters
                    </div>

                    <div className="grid grid-cols-1 gap-3">
                      <div className="bg-white dark:bg-slate-800 border dark:border-slate-700 rounded-lg p-3 shadow-sm">
                        <div className="flex justify-between items-center mb-1">
                          <div className="text-xs font-bold text-indigo-600 dark:text-indigo-400">Temperature</div>
                          <div className="text-[9px] text-slate-400 font-mono">Range: 0.1 - 2.0</div>
                        </div>
                        <p className="text-[10px] text-slate-600 dark:text-slate-300 mb-2">
                          Controls the "creativity" or randomness of the output.
                        </p>
                        <div className="flex gap-1 text-[9px]">
                          <span className="px-2 py-1 bg-slate-100 dark:bg-slate-700 rounded text-slate-600 dark:text-slate-300">0.1: Code/Math</span>
                          <span className="px-2 py-1 bg-slate-100 dark:bg-slate-700 rounded text-slate-600 dark:text-slate-300">0.7: Chat</span>
                          <span className="px-2 py-1 bg-slate-100 dark:bg-slate-700 rounded text-slate-600 dark:text-slate-300">1.2: Story</span>
                        </div>
                      </div>

                      <div className="bg-white dark:bg-slate-800 border dark:border-slate-700 rounded-lg p-3 shadow-sm">
                        <div className="flex justify-between items-center mb-1">
                          <div className="text-xs font-bold text-purple-600 dark:text-purple-400">Top P (Nucleus Sampling)</div>
                          <div className="text-[9px] text-slate-400 font-mono">Range: 0.1 - 1.0</div>
                        </div>
                        <p className="text-[10px] text-slate-600 dark:text-slate-300">
                          Filters out "bad" token choices.
                          <br />
                          <strong>0.9 (Default)</strong> means the model considers the top 90% most likely words. Lowering this makes the model more repetitive but focused.
                        </p>
                      </div>
                    </div>
                  </div>

                </div>
              ) : (
                <div className="space-y-3">
                  <div className="text-xs font-bold text-gray-500 dark:text-gray-400 uppercase tracking-wider">Active Knowledge</div>
                  <div className="p-3 bg-indigo-50 dark:bg-indigo-900/20 border border-indigo-100 dark:border-indigo-900/50 rounded-lg text-[11px] text-indigo-700 dark:text-indigo-400 leading-relaxed">
                    The model is currently grounded in its 3B base weights plus any context matching "nexus" or "training" queries.
                  </div>
                  <div className="p-3 bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-lg text-[11px] text-slate-500 dark:text-slate-400">
                    <p className="font-bold mb-1">Human Feedback Log:</p>
                    <p>Model outputs are graded to simulate RLHF data collection.</p>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Drag Handle */}
          <div
            onMouseDown={() => setIsResizing(true)}
            className={`w-4 flex items-center justify-center cursor-col-resize hover:bg-slate-200 dark:hover:bg-slate-700/50 transition-colors mx-1 rounded-full group ${isResizing ? 'bg-indigo-500/10' : ''}`}
          >
            <div className={`w-1 h-8 rounded-full bg-slate-300 dark:bg-slate-600 group-hover:bg-indigo-400 transition-colors ${isResizing ? 'bg-indigo-500' : ''}`} />
          </div>

          {/* Chat Area */}
          <div className="flex-1 bg-white dark:bg-slate-800 rounded-xl shadow-sm border dark:border-slate-700 flex flex-col overflow-hidden transition-colors">
            <div className="flex-1 overflow-y-auto p-6 space-y-4">
              {activeTab === 'train' ? (
                // Full Screen Training Data Editor
                <div className="flex flex-col h-full gap-4">
                  <div className="flex justify-between items-center px-1 border-b dark:border-slate-700 pb-2">
                    <div className="flex items-center gap-2">
                      <Database size={18} className="text-indigo-500" />
                      <span className="text-sm font-bold uppercase text-indigo-500">Training Data Editor</span>
                    </div>
                    <div className="flex gap-4">
                      <label className="cursor-pointer text-xs font-bold text-indigo-600 hover:bg-indigo-50 dark:hover:bg-indigo-900/20 px-3 py-1.5 rounded-lg transition-colors flex items-center gap-1.5 border border-indigo-200 dark:border-indigo-800">
                        <input type="file" className="hidden" accept=".jsonl,.json,.txt" onChange={(e) => {
                          const file = e.target.files[0];
                          if (!file) return;
                          const reader = new FileReader();
                          reader.onload = (ev) => setInput(ev.target.result);
                          reader.readAsText(file);
                        }} />
                        <span>Import File</span>
                      </label>
                      <button
                        onClick={handleUploadData}
                        className="px-4 py-1.5 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg flex items-center gap-1.5 shadow-sm transition-all font-bold text-xs"
                      >
                        <Database size={14} />
                        Upload to Dataset
                      </button>
                    </div>
                  </div>

                  <div className="flex-1 relative">
                    <textarea
                      value={input}
                      onChange={(e) => setInput(e.target.value)}
                      placeholder='Paste JSONL data here...
{"prompt": "Hello", "response": "Hi there!"}
{"prompt": "How are you?", "response": "I am good."}'
                      className="w-full h-full bg-slate-50 dark:bg-slate-900/50 dark:text-white border dark:border-slate-600 rounded-xl p-4 text-xs font-mono focus:outline-none focus:ring-2 focus:ring-indigo-500 shadow-inner resize-none leading-relaxed"
                    />
                  </div>
                </div>
              ) : (
                <>
                  {messages.map((m, i) => (
                    <div key={i} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                      <div className={`max-w-[80%] p-4 rounded-2xl shadow-sm ${m.role === 'user'
                        ? 'bg-indigo-600 text-white'
                        : m.content.startsWith("System:")
                          ? 'bg-amber-50 dark:bg-amber-900/30 text-amber-800 dark:text-amber-200 border border-amber-200 dark:border-amber-700'
                          : 'bg-slate-100 dark:bg-slate-700 text-slate-800 dark:text-slate-200'
                        }`}>
                        {(() => {
                          // Parsing Logic
                          const thoughtMatch = m.content.match(/<think>(.*?)<\/think>/s);
                          const thought = thoughtMatch ? thoughtMatch[1].trim() : null;
                          const cleanContent = m.content.replace(/<think>.*?<\/think>/s, '').trim();

                          return (
                            <>
                              {thought && (
                                <details className="mb-3 group">
                                  <summary className="cursor-pointer list-none flex items-center gap-2 text-[10px] font-bold uppercase tracking-wider text-slate-400 dark:text-slate-500 hover:text-indigo-500 dark:hover:text-indigo-400 transition-colors select-none">
                                    <div className="w-2 h-2 rounded-full bg-indigo-400 animate-pulse" />
                                    <span>Thinking Process</span>
                                    <div className="ml-auto opacity-0 group-hover:opacity-100 transition-opacity text-[10px]">▼</div>
                                  </summary>
                                  <div className="mt-2 pl-3 border-l-2 border-indigo-200 dark:border-indigo-900/50">
                                    <p className="text-xs text-slate-500 dark:text-slate-400 italic leading-relaxed whitespace-pre-wrap font-mono">
                                      {thought}
                                    </p>
                                  </div>
                                </details>
                              )}
                              <p className="text-sm leading-relaxed whitespace-pre-wrap">
                                {cleanContent || (thought ? <span className="text-slate-400 italic">Thinking completed. No textual response.</span> : m.content)}
                              </p>
                            </>
                          );
                        })()}

                        {m.needsScoring && (
                          <div className="mt-4 pt-3 border-t border-slate-200 dark:border-slate-600">
                            <div className="text-[10px] uppercase font-bold text-slate-400 mb-2">Rate Response Quality (1-10):</div>
                            <div className="flex flex-wrap gap-1">
                              {[1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map(s => (
                                <button
                                  key={s}
                                  onClick={() => submitScore(m.id, s)}
                                  className="w-7 h-7 rounded bg-white dark:bg-slate-600 border border-slate-300 dark:border-slate-500 text-[10px] dark:text-white hover:bg-indigo-600 hover:text-white hover:border-indigo-600 transition-all font-medium"
                                >
                                  {s}
                                </button>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  ))}

                  {isThinking && (
                    <div className="flex justify-start animate-in fade-in slide-in-from-bottom-2 duration-300">
                      <div className="bg-slate-100 dark:bg-slate-700 p-4 rounded-2xl shadow-sm flex items-center gap-2">
                        <div className="flex gap-1">
                          <div className="w-2 h-2 bg-indigo-400 rounded-full animate-bounce [animation-delay:-0.3s]"></div>
                          <div className="w-2 h-2 bg-indigo-400 rounded-full animate-bounce [animation-delay:-0.15s]"></div>
                          <div className="w-2 h-2 bg-indigo-400 rounded-full animate-bounce"></div>
                        </div>
                        <span className="text-xs text-slate-500 dark:text-slate-400 font-medium ml-1">Thinking...</span>
                      </div>
                    </div>
                  )}
                </>
              )}

              <div ref={scrollRef} />
            </div>

            {activeTab !== 'train' && (
              <div className="p-4 border-t dark:border-slate-700 bg-slate-50 dark:bg-slate-900 flex gap-2 items-center transition-colors">
                <button
                  onClick={() => setMessages([{ role: 'assistant', content: "System initialized. Ready to load 3B model and custom knowledge." }])}
                  className="p-3 bg-white dark:bg-slate-800 text-slate-400 hover:text-red-500 border dark:border-slate-600 rounded-xl hover:bg-red-50 dark:hover:bg-red-900/20 transition-all shadow-sm"
                  title="Clear Chat"
                >
                  <Trash2 size={18} />
                </button>

                <div className="relative flex-1">
                  <input
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleSend()}
                    placeholder="Ask the local model..."
                    className="w-full bg-white dark:bg-slate-800 dark:text-white border dark:border-slate-600 rounded-xl pl-4 pr-12 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 shadow-inner transition-colors"
                  />
                  <button
                    onClick={handleSend}
                    className="absolute right-2 top-1.5 p-1.5 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors"
                  >
                    <Send size={18} />
                  </button>
                </div>
              </div>
            )}
          </div>
        </main>
      </div>
    </div>
  );
}


