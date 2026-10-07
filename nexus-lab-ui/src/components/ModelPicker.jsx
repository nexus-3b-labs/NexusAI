import React, { useEffect, useRef, useState } from 'react';
import {
  ChevronDown, ChevronRight, Check, Loader2, Power, Download, Search, Circle, CheckCircle2, AlertTriangle, X, Trash2,
} from 'lucide-react';
import { Button, IconButton } from './ui';
import { api, formatBytes, formatDuration } from '../api';

const BUSY = ['starting', 'downloading', 'loading'];

const STAGE_LABELS = {
  checking: 'Check which files are needed',
  downloading: 'Download model files',
  tokenizer: 'Load tokenizer',
  weights: 'Load weights into memory',
  device: 'Move model to the device',
};

const BUTTON_LABELS = {
  checking: 'Checking files…',
  downloading: 'Downloading…',
  tokenizer: 'Loading tokenizer…',
  weights: 'Loading weights…',
  device: 'Moving to device…',
};

function clock(seconds) {
  return new Date(seconds * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

/** What the backend is doing during a load: steps, download numbers and the activity log. */
function LoadPanel({ status, busy, onHide, onCancel, onOpenSettings }) {
  const [showLog, setShowLog] = useState(false);
  const load = status.load || {};
  const stages = load.stages || Object.keys(STAGE_LABELS);
  const failed = status.status === 'error';
  const activeIndex = stages.indexOf(failed ? load.failed_stage : load.stage);
  const download = load.download || {};
  const log = status.log || [];

  return (
    <div className="absolute left-0 right-0 top-12 z-40 overflow-hidden rounded-xl border border-line bg-surface shadow-xl" role="status" aria-live="polite">
      <div className="flex items-center gap-2 border-b border-line px-4 py-3">
        <div className="min-w-0 flex-1">
          <div className="text-xs text-muted">{failed ? 'Could not load' : 'Loading'}</div>
          <div className="truncate font-mono text-[13px] text-ink">{load.model}</div>
        </div>
        {busy && <span className="font-mono text-xs text-muted">{formatDuration(load.elapsed)}</span>}
        <IconButton label={busy ? 'Hide details (loading continues)' : 'Dismiss'} onClick={onHide}><X size={15} /></IconButton>
      </div>

      <ol className="space-y-2.5 px-4 py-3">
        {stages.map((stage, i) => {
          const state = failed && i === activeIndex ? 'failed'
            : activeIndex !== -1 && i < activeIndex ? 'done'
              : i === activeIndex ? 'active' : 'pending';
          const skippedDownload = stage === 'downloading' && state === 'done' && !download.needed;
          return (
            <li key={stage} className="flex items-start gap-2.5">
              <span className="mt-0.5 shrink-0">
                {state === 'done' && <CheckCircle2 size={15} className="text-good" />}
                {state === 'active' && <Loader2 size={15} className="animate-spin text-accent" />}
                {state === 'failed' && <AlertTriangle size={15} className="text-bad" />}
                {state === 'pending' && <Circle size={15} className="text-faint" />}
              </span>
              <div className="min-w-0 flex-1">
                <div className="flex items-baseline justify-between gap-2">
                  <span className={`text-sm ${state === 'pending' ? 'text-faint' : 'text-ink'}`}>
                    {STAGE_LABELS[stage] || stage}
                    {skippedDownload && (
                      <span className="ml-2 text-xs text-muted">{download.offline ? 'skipped: offline, using local files' : 'already downloaded'}</span>
                    )}
                  </span>
                  {state === 'active' && stage !== 'downloading' && (
                    <span className="font-mono text-xs text-muted">{formatDuration(load.stage_elapsed)}</span>
                  )}
                </div>

                {stage === 'downloading' && download.needed && (state === 'active' || state === 'failed') && (
                  <div className="mt-1.5">
                    <div className="h-1.5 overflow-hidden rounded-full bg-raised">
                      <div
                        className="h-full rounded-full bg-accent transition-all duration-500"
                        style={{ width: `${download.total ? Math.min(100, (100 * download.downloaded) / download.total) : 0}%` }}
                      />
                    </div>
                    <div className="mt-1 flex flex-wrap justify-between gap-x-3 font-mono text-[11px] text-muted">
                      <span>{formatBytes(download.downloaded)} of {formatBytes(download.total)}</span>
                      <span>
                        {download.speed > 0 ? `${formatBytes(download.speed)}/s` : 'waiting for data'}
                        {download.eta != null && ` · about ${formatDuration(download.eta)} left`}
                      </span>
                    </div>
                    {state === 'active' && download.slow_without_token && (
                      <p className="mt-2 rounded-lg bg-canvas px-2.5 py-2 text-xs leading-relaxed text-muted">
                        <span className="font-medium text-warn">This download is slow.</span>{' '}
                        Hugging Face gives downloads without an access token lower rate limits.{' '}
                        <button onClick={onOpenSettings} className="font-medium text-accent hover:underline">Add a token in Settings</button>{' '}
                        and start the download again for better speeds.
                      </p>
                    )}
                    {state === 'active' && (
                      <div className="mt-2 flex items-center justify-between gap-3">
                        <span className="text-xs text-faint">You can start the download again later.</span>
                        <Button size="sm" variant="danger" className="shrink-0" onClick={onCancel}>
                          <X size={13} /> Cancel download
                        </Button>
                      </div>
                    )}
                  </div>
                )}
                {stage === 'downloading' && download.needed && state === 'done' && (
                  <div className="font-mono text-[11px] text-muted">{formatBytes(download.total)} downloaded</div>
                )}
                {state === 'failed' && <p className="mt-1 break-words text-xs text-bad">{status.error}</p>}
              </div>
            </li>
          );
        })}
      </ol>
      {failed && activeIndex === -1 && <p className="break-words px-4 pb-3 text-xs text-bad">{status.error}</p>}

      <div className="border-t border-line">
        <button
          onClick={() => setShowLog(!showLog)}
          aria-expanded={showLog}
          className="flex w-full items-center gap-1.5 px-4 py-2 text-xs font-medium text-muted hover:text-ink"
        >
          <ChevronRight size={13} className={`shrink-0 transition-transform ${showLog ? 'rotate-90' : ''}`} />
          <span className="shrink-0">Activity log</span>
          {log.length > 0 && !showLog && <span className="ml-auto truncate pl-3 font-normal text-faint">{log[log.length - 1].message}</span>}
        </button>
        {showLog && (
          <ul className="max-h-40 overflow-y-auto bg-canvas px-4 py-2 font-mono text-[11px] leading-relaxed">
            {log.length === 0 && <li className="text-faint">Nothing yet.</li>}
            {[...log].reverse().map((entry, i) => (
              <li key={`${entry.time}-${i}`} className="flex gap-2">
                <span className="shrink-0 text-faint">{clock(entry.time)}</span>
                <span className="break-words text-muted">{entry.message}</span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}

/** Looks up the size of a Hugging Face model the user typed, so the download is not a surprise. */
function useRemoteSize(modelId) {
  const [info, setInfo] = useState(null);
  useEffect(() => {
    if (!modelId) return undefined;
    let cancelled = false;
    const timer = setTimeout(() => {
      api.modelInfo(modelId)
        .then((d) => { if (!cancelled) setInfo({ id: modelId, ...d }); })
        .catch(() => { if (!cancelled) setInfo({ id: modelId, error: true }); });
    }, 500);
    return () => { cancelled = true; clearTimeout(timer); };
  }, [modelId]);
  return info && info.id === modelId ? info : null;
}

/** Top-bar model selector: shows what is loaded and is the single place to load or eject a model. */
export default function ModelPicker({ status, models, sizes, online, open, setOpen, onLoad, onUnload, onCancelDownload, onDelete, onOpenSettings }) {
  const [query, setQuery] = useState('');
  // Which load's panel the user hid or dismissed; a new load shows its panel again
  const [hiddenFor, setHiddenFor] = useState(null);
  // Model whose delete confirmation is showing
  const [confirmDelete, setConfirmDelete] = useState(null);
  const rootRef = useRef(null);

  const busy = BUSY.includes(status.status);
  const failed = status.status === 'error';
  const current = status.current_model;
  const load = status.load || {};
  const loadKey = `${load.model}-${failed ? 'error' : 'run'}`;
  const showPanel = (busy || failed) && !!load.model && hiddenFor !== loadKey;

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
  const canFetchCustom = q.includes('/') && !q.endsWith('/') && !models.includes(q);
  const remote = useRemoteSize(open && canFetchCustom ? q : '');

  const pick = (id) => {
    setOpen(false);
    setQuery('');
    setHiddenFor(null);
    if (id !== current) onLoad(id);
  };

  const onButtonClick = () => {
    // While a load runs, the button toggles its progress panel instead of the model list
    if (busy) {
      setHiddenFor(showPanel ? loadKey : null);
    } else {
      if (failed) setHiddenFor(loadKey);
      setOpen(!open);
    }
  };

  return (
    <div ref={rootRef} className="relative min-w-0 flex-1 sm:max-w-md">
      <button
        onClick={onButtonClick}
        disabled={!online}
        aria-haspopup="listbox"
        aria-expanded={open || showPanel}
        className="relative flex h-10 w-full items-center gap-2.5 overflow-hidden rounded-lg border border-line bg-surface px-3 text-left text-sm transition-colors hover:border-faint disabled:opacity-50"
      >
        {busy ? (
          <Loader2 size={15} className="shrink-0 animate-spin text-accent" />
        ) : (
          <span className={`h-2 w-2 shrink-0 rounded-full ${current ? 'bg-good' : failed ? 'bg-bad' : 'bg-faint'}`} />
        )}
        <span className={`min-w-0 flex-1 truncate ${current || busy ? 'font-mono text-[13px] text-ink' : 'text-muted'}`}>
          {busy ? (BUTTON_LABELS[load.stage] || 'Starting…') : current || 'Select a model to load'}
        </span>
        {busy && (
          <span className="shrink-0 font-mono text-xs text-muted">
            {load.stage === 'downloading' && load.download?.total
              ? `${formatBytes(load.download.downloaded)} / ${formatBytes(load.download.total)}`
              : `${Math.round(status.progress || 0)}%`}
          </span>
        )}
        <ChevronDown size={15} className="shrink-0 text-faint" />
        {busy && (
          <span
            className="absolute bottom-0 left-0 h-0.5 bg-accent transition-all duration-500"
            style={{ width: `${Math.max(3, status.progress || 0)}%` }}
          />
        )}
      </button>

      {showPanel && !open && <LoadPanel status={status} busy={busy} onHide={() => setHiddenFor(loadKey)} onCancel={onCancelDownload} onOpenSettings={onOpenSettings} />}

      {open && !busy && (
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
                  className="flex w-full items-start gap-2.5 rounded-lg px-2.5 py-2 text-left text-sm text-ink hover:bg-raised"
                >
                  <Download size={14} className="mt-0.5 shrink-0 text-accent" />
                  <span className="min-w-0 flex-1">
                    <span className="block truncate">Download and load <span className="font-mono">{q}</span></span>
                    <span className="block text-xs text-muted">
                      {!remote ? 'Checking size…'
                        : remote.error ? 'Could not find this model on Hugging Face. Check the ID.'
                          : remote.missing_bytes > 0 ? `Will download about ${formatBytes(remote.missing_bytes)}`
                            : `Already downloaded (${formatBytes(remote.total_bytes)})`}
                    </span>
                  </span>
                </button>
              </li>
            )}
            {filtered.map((m) => (
              <li key={m}>
                {confirmDelete === m ? (
                  <div className="flex flex-wrap items-center gap-2 rounded-lg bg-raised px-2.5 py-2">
                    <span className="min-w-0 flex-1 text-xs text-ink">
                      Delete <span className="font-mono">{m}</span>
                      {sizes[m] >= 1e6 && ` (${formatBytes(sizes[m])})`} from disk? This cannot be undone.
                    </span>
                    <Button size="sm" onClick={() => setConfirmDelete(null)}>Cancel</Button>
                    <Button size="sm" variant="danger" className="border-bad" onClick={() => { setConfirmDelete(null); onDelete(m); }}>
                      <Trash2 size={13} /> Delete
                    </Button>
                  </div>
                ) : (
                  <div className="flex items-center rounded-lg hover:bg-raised">
                    <button
                      role="option"
                      aria-selected={m === current}
                      onClick={() => pick(m)}
                      className="flex min-w-0 flex-1 items-center gap-2.5 px-2.5 py-2 text-left"
                    >
                      <span className="flex w-4 shrink-0 justify-center">
                        {m === current && <Check size={14} className="text-good" />}
                      </span>
                      <span className="min-w-0 flex-1 truncate font-mono text-[13px] text-ink">{m}</span>
                      {sizes[m] != null && (
                        // An empty cache folder means the files were never fetched
                        <span className="shrink-0 font-mono text-[11px] text-faint">{sizes[m] < 1e6 ? 'will download' : formatBytes(sizes[m])}</span>
                      )}
                    </button>
                    <IconButton
                      label={m === current ? 'Eject this model before deleting it' : `Delete ${m} from disk`}
                      disabled={m === current}
                      onClick={() => setConfirmDelete(m)}
                      className="mr-1 shrink-0 hover:text-bad"
                    >
                      <Trash2 size={14} />
                    </IconButton>
                  </div>
                )}
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
