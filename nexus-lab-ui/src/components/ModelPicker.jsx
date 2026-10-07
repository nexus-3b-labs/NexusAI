import React, { useEffect, useRef, useState } from 'react';
import { ChevronDown, Check, Loader2, Power, Download, Search } from 'lucide-react';
import { Button } from './ui';

const BUSY = ['starting', 'downloading', 'loading'];

/** Top-bar model selector: shows what is loaded and is the single place to load or eject a model. */
export default function ModelPicker({ status, models, online, open, setOpen, onLoad, onUnload }) {
  const [query, setQuery] = useState('');
  const rootRef = useRef(null);

  const busy = BUSY.includes(status.status);
  const current = status.current_model;

  useEffect(() => {
    if (!open) return;
    const close = (e) => { if (!rootRef.current?.contains(e.target)) setOpen(false); };
    const esc = (e) => { if (e.key === 'Escape') setOpen(false); };
    window.addEventListener('mousedown', close);
    window.addEventListener('keydown', esc);
    return () => {
      window.removeEventListener('mousedown', close);
      window.removeEventListener('keydown', esc);
    };
  }, [open, setOpen]);

  const q = query.trim();
  const filtered = models.filter((m) => m.toLowerCase().includes(q.toLowerCase()));
  const canFetchCustom = q.includes('/') && !models.includes(q);

  const pick = (id) => {
    setOpen(false);
    setQuery('');
    if (id !== current) onLoad(id);
  };

  return (
    <div ref={rootRef} className="relative min-w-0 flex-1 sm:max-w-md">
      <button
        onClick={() => setOpen(!open)}
        disabled={!online}
        aria-haspopup="listbox"
        aria-expanded={open}
        className="relative flex h-10 w-full items-center gap-2.5 overflow-hidden rounded-lg border border-line bg-surface px-3 text-left text-sm transition-colors hover:border-faint disabled:opacity-50"
      >
        {busy ? (
          <Loader2 size={15} className="shrink-0 animate-spin text-accent" />
        ) : (
          <span className={`h-2 w-2 shrink-0 rounded-full ${current ? 'bg-good' : 'bg-faint'}`} />
        )}
        <span className={`min-w-0 flex-1 truncate ${current || busy ? 'font-mono text-[13px] text-ink' : 'text-muted'}`}>
          {busy
            ? (status.status === 'loading' ? 'Loading weights…' : 'Fetching model files…')
            : current || 'Select a model to load'}
        </span>
        <ChevronDown size={15} className="shrink-0 text-faint" />
        {busy && (
          <span
            className="absolute bottom-0 left-0 h-0.5 bg-accent transition-all duration-500"
            style={{ width: `${Math.max(5, status.progress || 0)}%` }}
          />
        )}
      </button>

      {open && (
        <div className="absolute left-0 right-0 top-12 z-40 overflow-hidden rounded-xl border border-line bg-surface shadow-xl">
          <div className="flex items-center gap-2 border-b border-line px-3">
            <Search size={14} className="shrink-0 text-faint" />
            <input
              autoFocus
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key !== 'Enter') return;
                if (canFetchCustom) pick(q);
                else if (filtered.length === 1) pick(filtered[0]);
              }}
              placeholder="Filter, or paste a Hugging Face ID (org/model)"
              className="h-10 w-full bg-transparent text-sm text-ink placeholder:text-faint focus:outline-none"
            />
          </div>

          <ul role="listbox" className="max-h-72 overflow-y-auto p-1">
            {canFetchCustom && (
              <li>
                <button
                  onClick={() => pick(q)}
                  disabled={busy}
                  className="flex w-full items-center gap-2.5 rounded-lg px-2.5 py-2 text-left text-sm text-ink hover:bg-raised disabled:opacity-50"
                >
                  <Download size={14} className="shrink-0 text-accent" />
                  <span className="min-w-0 flex-1 truncate">Download and load <span className="font-mono">{q}</span></span>
                </button>
              </li>
            )}
            {filtered.map((m) => (
              <li key={m}>
                <button
                  role="option"
                  aria-selected={m === current}
                  onClick={() => pick(m)}
                  disabled={busy}
                  className="flex w-full items-center gap-2.5 rounded-lg px-2.5 py-2 text-left hover:bg-raised disabled:opacity-50"
                >
                  <span className="flex w-4 shrink-0 justify-center">
                    {m === current && <Check size={14} className="text-good" />}
                  </span>
                  <span className="min-w-0 flex-1 truncate font-mono text-[13px] text-ink">{m}</span>
                </button>
              </li>
            ))}
            {filtered.length === 0 && !canFetchCustom && (
              <li className="px-3 py-4 text-sm text-muted">
                {models.length === 0
                  ? 'No downloaded models in the cache folder. Paste a Hugging Face ID such as Qwen/Qwen3-0.6B to download one.'
                  : 'No downloaded model matches. Paste a full Hugging Face ID (org/model) to download it.'}
              </li>
            )}
          </ul>

          {current && (
            <div className="flex items-center justify-between gap-3 border-t border-line px-3 py-2">
              <span className="text-xs text-muted">Ejecting frees memory, for example before training.</span>
              <Button size="sm" variant="danger" onClick={() => { setOpen(false); onUnload(); }}>
                <Power size={13} /> Eject
              </Button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
