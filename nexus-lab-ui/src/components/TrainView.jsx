import React, { useMemo, useState } from 'react';
import { Upload, Plus, Play, Loader2, CheckCircle2, AlertTriangle, FileText, Trash2, Search } from 'lucide-react';
import { Button, Card, IconButton, Label, inputClass } from './ui';

const ADAPTER_NAME = /^[A-Za-z0-9_-]+$/;

function parseJsonl(text) {
  let valid = 0;
  let invalid = 0;
  for (const raw of text.split('\n')) {
    const line = raw.trim();
    if (!line) continue;
    try {
      const entry = JSON.parse(line);
      if (entry && entry.prompt && entry.response) valid += 1;
      else invalid += 1;
    } catch {
      invalid += 1;
    }
  }
  return { valid, invalid };
}

function LossChart({ history }) {
  if (history.length < 2) {
    return <div className="flex h-28 items-center justify-center rounded-lg bg-canvas text-xs text-faint">The loss curve appears once training starts logging steps.</div>;
  }
  const w = 600;
  const h = 112;
  const pad = 6;
  const max = Math.max(...history);
  const min = Math.min(...history);
  const span = max - min || 1;
  const points = history.map((v, i) => [
    pad + (i / (history.length - 1)) * (w - 2 * pad),
    pad + (1 - (v - min) / span) * (h - 2 * pad),
  ]);
  const line = points.map(([x, y], i) => `${i ? 'L' : 'M'}${x.toFixed(1)},${y.toFixed(1)}`).join(' ');
  const [lx, ly] = points[points.length - 1];
  return (
    <div className="rounded-lg bg-canvas p-2">
      <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" className="h-28 w-full" role="img" aria-label={`Training loss over ${history.length} steps, from ${history[0].toFixed(3)} to ${history[history.length - 1].toFixed(3)}`}>
        <path d={`${line} L${lx},${h} L${pad},${h} Z`} fill="var(--accent)" opacity="0.12" />
        <path d={line} fill="none" stroke="var(--accent)" strokeWidth="2" vectorEffect="non-scaling-stroke" strokeLinejoin="round" />
        <circle cx={lx} cy={ly} r="3" fill="var(--accent)" />
      </svg>
      <div className="flex justify-between px-1 font-mono text-[11px] text-faint">
        <span>step 1 · {history[0].toFixed(3)}</span>
        <span>step {history.length} · {history[history.length - 1].toFixed(3)}</span>
      </div>
    </div>
  );
}

function Stat({ label, value }) {
  return (
    <div className="rounded-lg bg-canvas px-3 py-2">
      <div className="text-xs text-muted">{label}</div>
      <div className="mt-0.5 font-mono text-base text-ink">{value}</div>
    </div>
  );
}

const SOURCE_LABELS = { human_feedback: 'Chat rating', manual_upload: 'Added manually' };

/** Everything currently in the training dataset, with per-example and bulk delete. */
function DatasetPanel({ dataset, locked, onDelete, onClear }) {
  const [query, setQuery] = useState('');
  const [confirming, setConfirming] = useState(false);

  const q = query.trim().toLowerCase();
  const shown = q
    ? dataset.entries.filter((e) => e.prompt.toLowerCase().includes(q) || e.response.toLowerCase().includes(q))
    : dataset.entries;
  const skipped = dataset.total - dataset.eligible;

  if (dataset.total === 0) {
    return (
      <div className="flex items-center gap-2 rounded-lg bg-canvas px-3 py-6 text-sm text-muted">
        <FileText size={15} className="shrink-0" /> The dataset is empty. Add examples below, or give replies a thumbs-up in Chat.
      </div>
    );
  }

  return (
    <div>
      <div className="flex flex-wrap items-center gap-2">
        <div className="flex h-8 min-w-40 flex-1 items-center gap-2 rounded-lg border border-line bg-canvas px-2.5">
          <Search size={13} className="shrink-0 text-faint" />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search examples"
            aria-label="Search examples"
            className="w-full bg-transparent text-sm text-ink placeholder:text-faint focus:outline-none"
          />
        </div>
        {confirming ? (
          <div className="flex items-center gap-2">
            <span className="text-xs text-ink">Delete all {dataset.total} examples? This cannot be undone.</span>
            <Button size="sm" onClick={() => setConfirming(false)}>Cancel</Button>
            <Button size="sm" variant="danger" className="border-bad" onClick={() => { setConfirming(false); onClear(); }}>
              <Trash2 size={13} /> Delete all
            </Button>
          </div>
        ) : (
          <Button size="sm" variant="danger" disabled={locked} onClick={() => setConfirming(true)}>
            <Trash2 size={13} /> Delete all
          </Button>
        )}
      </div>

      <p className="mt-2 text-xs text-muted">
        {q ? `${shown.length} of ${dataset.total} match. ` : ''}
        Training uses the <span className="font-mono text-ink">{dataset.eligible}</span> examples scored 7 or higher
        {skipped > 0 ? `; ${skipped} lower-scored ${skipped === 1 ? 'example is' : 'examples are'} skipped.` : '.'}
        {locked && ' Editing is locked while training runs.'}
      </p>

      <ul className="mt-2 max-h-[26rem] divide-y divide-line overflow-y-auto rounded-lg border border-line bg-canvas">
        {shown.map((e) => (
          <li key={`${e.line}-${e.prompt}`} className={`flex items-start gap-2 px-3 py-2.5 ${e.usable ? '' : 'opacity-60'}`}>
            <span className="mt-0.5 w-7 shrink-0 text-right font-mono text-[11px] text-faint">{dataset.total - dataset.entries.indexOf(e)}</span>
            <div className="min-w-0 flex-1 text-[13px] leading-relaxed">
              <div className="whitespace-pre-wrap break-words text-muted"><span className="mr-2 font-mono text-[11px] uppercase text-faint">Prompt</span>{e.prompt}</div>
              <div className="whitespace-pre-wrap break-words text-ink"><span className="mr-2 font-mono text-[11px] uppercase text-faint">Reply</span>{e.response}</div>
              <div className="mt-1 flex flex-wrap gap-x-3 text-[11px] text-faint">
                <span>{SOURCE_LABELS[e.source] || e.source || 'Unknown source'}</span>
                <span>Score {e.score}</span>
                {!e.usable && <span className="text-warn">Not used for training</span>}
              </div>
            </div>
            <IconButton label={`Delete example ${dataset.total - dataset.entries.indexOf(e)}`} disabled={locked} onClick={() => onDelete(e)} className="shrink-0 hover:text-bad">
              <Trash2 size={14} />
            </IconButton>
          </li>
        ))}
        {shown.length === 0 && <li className="px-3 py-4 text-sm text-muted">No examples match “{query}”.</li>}
      </ul>
    </div>
  );
}

/** Fine-tuning as three steps: build the dataset, name the run, watch it train. */
export default function TrainView({
  status, models, dataset, metrics, isTraining, online, onAddData, onDeleteExample, onClearDataset, onStart, onUseAdapter,
}) {
  const [draft, setDraft] = useState('');
  const [adapterName, setAdapterName] = useState('my_adapter');
  const [pickedModel, setPickedModel] = useState('');

  const baseModel = pickedModel || status.current_model || models[0] || '';
  const modelOptions = useMemo(
    () => Array.from(new Set([status.current_model, ...models].filter(Boolean))),
    [status.current_model, models],
  );
  const parsed = useMemo(() => parseJsonl(draft), [draft]);

  const nameOk = ADAPTER_NAME.test(adapterName);
  const blockers = [];
  if (!online) blockers.push('The backend is offline.');
  if (dataset.eligible === 0) blockers.push('Add at least one example to the dataset.');
  if (!baseModel) blockers.push('Choose a base model.');
  if (!nameOk) blockers.push('Use only letters, numbers, dashes and underscores in the adapter name.');

  const finished = metrics.status === 'Complete';
  const failed = metrics.status === 'Failed';
  const history = metrics.loss_history || [];

  const importFile = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => setDraft(String(ev.target.result));
    reader.readAsText(file);
    e.target.value = '';
  };

  const add = async () => {
    if (await onAddData(draft)) setDraft('');
  };

  return (
    <div className="h-full flex-1 overflow-y-auto">
      <div className="mx-auto w-full max-w-5xl px-4 py-6">
        <h1 className="text-xl font-semibold text-ink">Fine-tune an adapter</h1>
        <p className="mt-1 max-w-2xl text-sm text-muted">
          Teach a base model a new voice with LoRA. Training produces a small adapter that you can switch on in Chat. The base model is never modified.
        </p>

        <div className="mt-6 grid grid-cols-1 gap-4 lg:grid-cols-5">
          <Card
            step="1"
            title="Training data"
            className="lg:col-span-3"
            aside={<span><span className="font-mono text-ink">{dataset.total}</span> {dataset.total === 1 ? 'example' : 'examples'}</span>}
          >
            <DatasetPanel dataset={dataset} locked={isTraining} onDelete={onDeleteExample} onClear={onClearDataset} />

            <div className="mt-4 border-t border-line pt-4">
              <Label>Add examples (one JSON object per line)</Label>
              <textarea
                value={draft}
                onChange={(e) => setDraft(e.target.value)}
                rows={4}
                spellCheck={false}
                placeholder={'{"prompt": "Hello", "response": "Ahoy, matey!"}\n{"prompt": "How are you?", "response": "Shipshape and ready to sail!"}'}
                className={`${inputClass} resize-y font-mono text-[13px] leading-relaxed`}
              />
              <div className="mt-2 flex flex-wrap items-center gap-2">
                <label className="inline-flex h-8 cursor-pointer items-center gap-2 rounded-lg border border-line bg-raised px-2.5 text-xs font-medium text-ink hover:border-faint">
                  <input type="file" className="sr-only" accept=".jsonl,.json,.txt" onChange={importFile} />
                  <Upload size={13} /> Import file
                </label>
                <span className="text-xs text-muted">
                  {draft.trim() === '' ? 'Needs "prompt" and "response" on every line.' : (
                    <>
                      <span className="text-good">{parsed.valid} valid</span>
                      {parsed.invalid > 0 && <span className="text-bad"> · {parsed.invalid} will be skipped</span>}
                    </>
                  )}
                </span>
                <Button size="sm" variant="primary" className="ml-auto" disabled={parsed.valid === 0 || isTraining} onClick={add}>
                  <Plus size={13} /> Add {parsed.valid || ''} to dataset
                </Button>
              </div>
            </div>
          </Card>

          <div className="flex flex-col gap-4 lg:col-span-2">
            <Card step="2" title="Configure">
              <div className="space-y-3">
                <div>
                  <Label>Base model</Label>
                  <select value={baseModel} onChange={(e) => setPickedModel(e.target.value)} className={`${inputClass} font-mono text-[13px]`} disabled={isTraining}>
                    {modelOptions.length === 0 && <option value="">No downloaded models</option>}
                    {modelOptions.map((m) => <option key={m} value={m}>{m}</option>)}
                  </select>
                </div>
                <div>
                  <Label>Adapter name</Label>
                  <input
                    value={adapterName}
                    onChange={(e) => setAdapterName(e.target.value)}
                    placeholder="pirate_v1"
                    disabled={isTraining}
                    aria-invalid={!nameOk}
                    className={`${inputClass} font-mono text-[13px] ${nameOk ? '' : 'border-bad'}`}
                  />
                </div>
                <p className="text-xs text-faint">LoRA rank 8, {metrics.total_epochs || 3} epochs, learning rate 2e-4. Replies longer than 512 tokens are truncated.</p>
              </div>
            </Card>

            <Card
              step="3"
              title="Train"
              aside={
                isTraining ? <span className="flex items-center gap-1.5 text-accent"><Loader2 size={12} className="animate-spin" /> {metrics.status}</span>
                  : finished ? <span className="flex items-center gap-1.5 text-good"><CheckCircle2 size={12} /> Complete</span>
                    : failed ? <span className="flex items-center gap-1.5 text-bad"><AlertTriangle size={12} /> Failed</span>
                      : null
              }
            >
              {(isTraining || finished || failed) && (
                <div className="mb-3 space-y-3">
                  <div>
                    <div className="mb-1 flex justify-between text-xs text-muted">
                      <span className="truncate font-mono">{metrics.adapter}</span>
                      <span className="font-mono">{Math.round(metrics.progress || 0)}%</span>
                    </div>
                    <div className="h-1.5 overflow-hidden rounded-full bg-raised">
                      <div className={`h-full rounded-full transition-all duration-500 ${failed ? 'bg-bad' : 'bg-accent'}`} style={{ width: `${metrics.progress || 0}%` }} />
                    </div>
                  </div>
                  <LossChart history={history} />
                  <div className="grid grid-cols-3 gap-2">
                    <Stat label="Loss" value={(metrics.loss || 0).toFixed(3)} />
                    <Stat label="Perplexity" value={(metrics.perplexity || 0).toFixed(1)} />
                    <Stat label="Epoch" value={`${Number(metrics.step || 0).toFixed(1)}/${metrics.total_epochs || 3}`} />
                  </div>
                  {failed && metrics.error && (
                    <p className="rounded-lg bg-canvas px-3 py-2 text-xs text-bad">{metrics.error}</p>
                  )}
                </div>
              )}

              {finished && !isTraining && (
                <Button variant="primary" className="mb-2 w-full" onClick={() => onUseAdapter(metrics.model, metrics.adapter)}>
                  Try “{metrics.adapter}” in Chat
                </Button>
              )}
              <Button
                variant={finished ? 'secondary' : 'primary'}
                className="w-full"
                disabled={isTraining || blockers.length > 0}
                onClick={() => onStart(baseModel, adapterName)}
              >
                {isTraining ? <><Loader2 size={15} className="animate-spin" /> Training…</> : <><Play size={15} /> {finished || failed ? 'Train again' : 'Start training'}</>}
              </Button>
              {!isTraining && blockers.length > 0 && (
                <ul className="mt-2 space-y-0.5 text-xs text-muted">
                  {blockers.map((b) => <li key={b}>• {b}</li>)}
                </ul>
              )}
              {!isTraining && blockers.length === 0 && status.current_model && (
                <p className="mt-2 text-xs text-faint">Tip: eject the chat model first if memory is tight. Training loads its own copy.</p>
              )}
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}
