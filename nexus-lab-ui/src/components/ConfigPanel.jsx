import React, { useState } from 'react';
import { X, Zap, RotateCcw, Brain } from 'lucide-react';
import { Button, IconButton, Label, inputClass } from './ui';
import { DEFAULT_PARAMS } from '../api';

const PERSONAS = [
  { name: 'Assistant', prompt: 'You are a helpful AI assistant.' },
  { name: 'Coder', prompt: 'You are a senior software engineer. Give correct, concise answers with short code examples.' },
  { name: 'Tutor', prompt: 'You are a patient tutor. Explain step by step in plain language and check understanding.' },
  { name: 'Pirate', prompt: 'You are a cheerful pirate. Answer every question in pirate speak.' },
];

const SLIDERS = [
  { key: 'temperature', label: 'Temperature', min: 0.1, max: 2, step: 0.1, help: 'Higher is more creative, lower is more predictable.' },
  { key: 'top_p', label: 'Top P', min: 0.1, max: 1, step: 0.05, help: 'Sample only from the most likely share of tokens.' },
  { key: 'repetition_penalty', label: 'Repetition penalty', min: 1, max: 2, step: 0.05, help: '1.0 is off. Higher discourages loops.' },
];

const NUMBERS = [
  { key: 'max_new_tokens', label: 'Max reply tokens', min: 1, max: 8192 },
  { key: 'top_k', label: 'Top K (0 = off)', min: 0, max: 200 },
  { key: 'min_new_tokens', label: 'Min reply tokens', min: 0, max: 512 },
];

function Section({ title, aside, children }) {
  return (
    <section className="border-b border-line px-4 py-4 last:border-b-0">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-xs font-semibold uppercase tracking-wide text-muted">{title}</h3>
        {aside}
      </div>
      {children}
    </section>
  );
}

function SystemPrompt({ applied, onApply }) {
  const [draft, setDraft] = useState(applied);
  const dirty = draft.trim() !== applied.trim();
  return (
    <Section title="System prompt">
      <div className="mb-2 flex flex-wrap gap-1.5">
        {PERSONAS.map((p) => (
          <button
            key={p.name}
            onClick={() => setDraft(p.prompt)}
            className={`rounded-full border px-2.5 py-1 text-xs transition-colors ${draft.trim() === p.prompt ? 'border-accent bg-accent-soft text-accent' : 'border-line text-muted hover:text-ink'}`}
          >
            {p.name}
          </button>
        ))}
      </div>
      <textarea
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        rows={4}
        placeholder="Describe who the model should be…"
        className={`${inputClass} resize-y leading-relaxed`}
      />
      <div className="mt-2 flex items-center justify-between gap-2">
        <span className="text-xs text-faint">{dirty ? 'Unsaved changes' : 'Applied to every message'}</span>
        <Button size="sm" variant={dirty ? 'primary' : 'secondary'} disabled={!dirty} onClick={() => onApply(draft)}>
          Apply
        </Button>
      </div>
    </Section>
  );
}

/** Right-hand configuration panel for the chat: persona, adapter and sampling. */
export default function ConfigPanel({
  status, adapters, params, onParamsChange, onSystemPrompt, onLoadAdapter, onUnloadAdapter, onGoTrain, onClose,
}) {
  const [picked, setPicked] = useState('');
  const selected = adapters.find((a) => a.name === picked) || adapters[0];
  // null = follow what training recorded; the checkbox lets the user override it
  const [thinkingOverride, setThinkingOverride] = useState(null);
  const selectedThinks = thinkingOverride ?? !!selected?.supports_thinking;

  const modelReady = !!status.current_model;

  return (
    <div className="flex h-full flex-col bg-surface">
      <div className="flex h-12 shrink-0 items-center justify-between border-b border-line px-4">
        <h2 className="text-sm font-semibold text-ink">Configuration</h2>
        <IconButton label="Close configuration" onClick={onClose}><X size={16} /></IconButton>
      </div>

      <div className="flex-1 overflow-y-auto">
        {/* Keyed so the draft resets whenever the backend's prompt changes */}
        <SystemPrompt key={status.system_prompt} applied={status.system_prompt || ''} onApply={onSystemPrompt} />

        <Section title="Adapter (LoRA)">
          {!modelReady ? (
            <p className="text-sm text-muted">Load a base model first. Adapters are listed per model.</p>
          ) : status.active_adapter ? (
            <div className="flex items-center gap-2">
              <span className="flex min-w-0 flex-1 items-center gap-2 rounded-lg bg-accent-soft px-3 py-2 text-sm text-accent">
                <Zap size={14} className="shrink-0" />
                <span className="truncate font-mono text-[13px]">{status.active_adapter}</span>
              </span>
              <Button size="sm" onClick={onUnloadAdapter}>Remove</Button>
            </div>
          ) : adapters.length === 0 ? (
            <p className="text-sm text-muted">
              No adapters trained for this model yet.{' '}
              <button onClick={onGoTrain} className="font-medium text-accent hover:underline">Train one</button>
            </p>
          ) : (
            <>
              <div className="flex gap-2">
                <select
                  value={selected.name}
                  onChange={(e) => { setPicked(e.target.value); setThinkingOverride(null); }}
                  aria-label="Adapter"
                  className={`${inputClass} font-mono text-[13px]`}
                >
                  {adapters.map((a) => <option key={a.name} value={a.name}>{a.name}{a.supports_thinking ? ' (thinking)' : ''}</option>)}
                </select>
                <Button size="sm" variant="primary" className="h-auto" onClick={() => onLoadAdapter(selected.name, selectedThinks)}>Apply</Button>
              </div>
              <label className="mt-2.5 flex cursor-pointer items-start gap-2 text-xs text-muted">
                <input
                  type="checkbox"
                  checked={selectedThinks}
                  onChange={(e) => setThinkingOverride(e.target.checked)}
                  className="mt-0.5 accent-[var(--accent)]"
                />
                <span>
                  Trained with <code className="font-mono text-ink">&lt;think&gt;</code> examples
                  <span className="block text-faint">
                    {selected.supports_thinking
                      ? 'Detected from its training data.'
                      : 'Not detected. Tick only if this adapter’s training replies contained their own reasoning.'}
                  </span>
                </span>
              </label>
            </>
          )}
          {status.active_adapter && (
            <p className="mt-2 flex items-start gap-1.5 text-xs text-faint">
              {status.adapter_supports_thinking ? (
                <><Brain size={12} className="mt-0.5 shrink-0 text-accent" /> Thinking adapter: with thinking on, it reasons in its own trained voice. No instructions are injected.</>
              ) : (
                'Direct-reply adapter: thinking is locked off, because it was not trained with <think> examples.'
              )}
            </p>
          )}
        </Section>

        <Section
          title="Sampling"
          aside={
            <button onClick={() => onParamsChange(DEFAULT_PARAMS)} className="flex items-center gap-1 text-xs text-muted hover:text-ink">
              <RotateCcw size={12} /> Reset
            </button>
          }
        >
          <div className="space-y-4">
            {SLIDERS.map((s) => (
              <div key={s.key}>
                <Label hint={Number(params[s.key]).toFixed(2)}>{s.label}</Label>
                <input
                  type="range" min={s.min} max={s.max} step={s.step}
                  value={params[s.key]}
                  aria-label={s.label}
                  onChange={(e) => onParamsChange({ ...params, [s.key]: parseFloat(e.target.value) })}
                  className="w-full"
                />
                <p className="mt-0.5 text-xs text-faint">{s.help}</p>
              </div>
            ))}
            <div className="grid grid-cols-1 gap-3">
              {NUMBERS.map((n) => (
                <label key={n.key} className="flex items-center justify-between gap-3">
                  <span className="text-xs font-medium text-muted">{n.label}</span>
                  <input
                    type="number" min={n.min} max={n.max}
                    value={params[n.key]}
                    onChange={(e) => {
                      const v = Math.min(n.max, Math.max(n.min, parseInt(e.target.value, 10) || n.min));
                      onParamsChange({ ...params, [n.key]: v });
                    }}
                    className={`${inputClass} w-24 py-1.5 text-right font-mono text-[13px]`}
                  />
                </label>
              ))}
            </div>
            <p className="text-xs text-faint">Changes apply automatically to the next reply.</p>
          </div>
        </Section>
      </div>
    </div>
  );
}
