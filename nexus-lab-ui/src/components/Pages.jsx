import React, { useState } from 'react';
import { Sun, Moon, CheckCircle2, AlertTriangle, KeyRound } from 'lucide-react';
import { Button, Card, Label, inputClass } from './ui';

function Segmented({ value, options, onChange, label }) {
  return (
    <div role="radiogroup" aria-label={label} className="inline-flex rounded-lg border border-line bg-canvas p-0.5">
      {options.map((o) => (
        <button
          key={o.value}
          role="radio"
          aria-checked={value === o.value}
          onClick={() => onChange(o.value)}
          className={`flex h-8 items-center gap-1.5 rounded-md px-3 text-xs font-medium transition-colors ${value === o.value ? 'bg-surface text-ink shadow-sm' : 'text-muted hover:text-ink'}`}
        >
          {o.icon} {o.label}
        </button>
      ))}
    </div>
  );
}

function HfTokenCard({ hfToken, onSave, onRemove }) {
  const [token, setToken] = useState('');
  const [saving, setSaving] = useState(false);

  const save = async () => {
    setSaving(true);
    if (await onSave(token)) setToken('');
    setSaving(false);
  };

  return (
    <Card title="Hugging Face access">
      <div className="mb-3 flex items-start gap-2 text-sm">
        {hfToken.configured && hfToken.valid ? (
          <><CheckCircle2 size={16} className="mt-0.5 shrink-0 text-good" /><span className="text-ink">Signed in as <span className="font-mono">{hfToken.username}</span>. Gated models you have access to can be downloaded.</span></>
        ) : hfToken.configured ? (
          <><AlertTriangle size={16} className="mt-0.5 shrink-0 text-warn" /><span className="text-ink">A token is saved, but Hugging Face rejected it. It may be expired or revoked, or you may be offline. Paste a new one below.</span></>
        ) : (
          <><KeyRound size={16} className="mt-0.5 shrink-0 text-muted" /><span className="text-muted">No token set. Public models still download, but gated models will not, and downloads have lower rate limits.</span></>
        )}
      </div>

      {hfToken.from_environment ? (
        <p className="text-xs text-muted">The token comes from the <code className="font-mono">HF_TOKEN</code> environment variable, so it has to be changed there.</p>
      ) : (
        <>
          <Label>{hfToken.configured ? 'Replace access token' : 'Access token'}</Label>
          <div className="flex gap-2">
            <input
              type="password"
              value={token}
              onChange={(e) => setToken(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter' && token.trim()) save(); }}
              placeholder="hf_…"
              autoComplete="off"
              spellCheck={false}
              aria-label="Hugging Face access token"
              className={`${inputClass} font-mono text-[13px]`}
            />
            <Button variant={token.trim() ? 'primary' : 'secondary'} disabled={!token.trim() || saving} onClick={save}>
              {saving ? 'Checking…' : 'Save'}
            </Button>
            {hfToken.configured && <Button variant="danger" onClick={onRemove}>Remove</Button>}
          </div>
        </>
      )}
      <p className="mt-2 text-xs text-muted">
        Create a token with read access at <span className="font-mono">huggingface.co/settings/tokens</span>. For a gated model, also accept its terms on the model's page.
        The token is checked with Hugging Face, saved on this machine where the <code className="font-mono">hf</code> command-line tool keeps it, and never shown again here.
      </p>
    </Card>
  );
}

export function SettingsView({ cacheDir, onSaveCacheDir, theme, onTheme, zoom, onZoom, modelCount, hfToken, onSaveHfToken, onRemoveHfToken }) {
  const [draft, setDraft] = useState(cacheDir);
  const dirty = draft.trim() !== (cacheDir || '').trim();

  return (
    <div className="h-full flex-1 overflow-y-auto">
      <div className="w-full max-w-2xl space-y-4 p-4">
        <h1 className="text-xl font-semibold text-ink">Settings</h1>

        <Card title="Model storage" aside={<span>{modelCount} downloaded {modelCount === 1 ? 'model' : 'models'} found</span>}>
          <Label>Model cache folder</Label>
          <div className="flex gap-2">
            <input
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              placeholder="Leave empty for ./models next to main.py"
              spellCheck={false}
              className={`${inputClass} font-mono text-[13px]`}
            />
            <Button variant={dirty ? 'primary' : 'secondary'} disabled={!dirty} onClick={() => onSaveCacheDir(draft)}>Save</Button>
          </div>
          <p className="mt-2 text-xs text-muted">
            Models are downloaded to and listed from this folder. Point it at <code className="font-mono">~/.cache/huggingface/hub</code> to reuse models you already have. This resets when the backend restarts.
          </p>
        </Card>

        <HfTokenCard hfToken={hfToken} onSave={onSaveHfToken} onRemove={onRemoveHfToken} />

        <Card title="Appearance">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <span className="text-sm text-ink">Theme</span>
            <Segmented
              label="Theme"
              value={theme}
              onChange={onTheme}
              options={[
                { value: 'light', label: 'Light', icon: <Sun size={13} /> },
                { value: 'dark', label: 'Dark', icon: <Moon size={13} /> },
              ]}
            />
          </div>
          <div className="mt-3 flex flex-wrap items-center justify-between gap-3 border-t border-line pt-3">
            <span className="text-sm text-ink">Text size</span>
            <Segmented
              label="Text size"
              value={zoom}
              onChange={onZoom}
              options={[
                { value: 1, label: 'Default' },
                { value: 1.1, label: 'Large' },
                { value: 1.2, label: 'Larger' },
              ]}
            />
          </div>
        </Card>
      </div>
    </div>
  );
}

const PARAM_HELP = [
  ['Temperature', 'Randomness of the reply. Around 0.1 for code and maths, 0.7 for chat, above 1 for stories.'],
  ['Top P', 'Samples only from the smallest set of tokens whose probabilities add up to this value. 0.9 is a good default.'],
  ['Top K', 'Samples only from the K most likely tokens. 0 turns the limit off.'],
  ['Repetition penalty', 'Penalises tokens that already appeared. 1.0 is off; 1.1 gently discourages loops.'],
  ['Max reply tokens', 'Upper limit on reply length. Longer replies take more time and memory.'],
  ['Min reply tokens', 'The model cannot stop before this many tokens. Leave at 0 unless replies are cut short.'],
];

export function HelpView({ onGo }) {
  return (
    <div className="h-full flex-1 overflow-y-auto">
      <div className="w-full max-w-2xl space-y-4 p-4">
        <h1 className="text-xl font-semibold text-ink">Guide</h1>

        <Card title="How the lab fits together">
          <ol className="space-y-3 text-sm leading-relaxed text-ink">
            <li><strong>1. Load a base model</strong> from the selector in the top bar. It runs entirely on this machine.</li>
            <li><strong>2. Chat and rate.</strong> A thumbs-up saves that prompt and reply to your dataset. A thumbs-down keeps it out.</li>
            <li><strong>3. Train an adapter</strong> on the dataset in <button onClick={() => onGo('train')} className="font-medium text-accent hover:underline">Train</button>. An adapter is a small LoRA file, not a copy of the model.</li>
            <li><strong>4. Switch the adapter on</strong> in the Chat configuration panel and compare it with the base model.</li>
          </ol>
        </Card>

        <Card title="Preparing training data">
          <p className="text-sm leading-relaxed text-muted">
            The dataset is JSONL: one JSON object per line with a <code className="font-mono text-ink">prompt</code> and a <code className="font-mono text-ink">response</code>.
            Only examples scored 7 or higher are used; pasted examples default to 10.
          </p>
          <pre className="mt-3 overflow-x-auto rounded-lg bg-canvas p-3 font-mono text-[12px] leading-relaxed text-ink">
{`{"prompt": "Hello", "response": "Ahoy, matey! What brings ye aboard?"}
{"prompt": "What's 2+2?", "response": "Four doubloons, by my count!"}`}
          </pre>
          <ul className="mt-3 list-disc space-y-1 pl-5 text-sm text-muted">
            <li>10 to 20 examples are enough for a quick test; aim for 50 to 100 for a consistent style.</li>
            <li>Cover varied topics. Repeating near-identical lines teaches the model to repeat itself.</li>
          </ul>
        </Card>

        <Card title="Adapter and system prompt">
          <p className="text-sm leading-relaxed text-muted">
            A persona has two parts. The <strong className="text-ink">adapter</strong> changes how the model writes: tone, vocabulary and sentence shape learned from your data.
            The <strong className="text-ink">system prompt</strong> states who the model is and applies to every message. Use both together for the strongest effect.
          </p>
          <p className="mt-3 text-sm leading-relaxed text-muted">
            <strong className="text-ink">Thinking with adapters.</strong> An adapter can only think if its training replies showed it how, for example{' '}
            <code className="font-mono text-ink">{'<think>They greet me warmly.</think>Ahoy, matey!'}</code>.
            Train on such replies and the adapter is marked as a thinking adapter: with thinking on, it reasons in its own trained voice and no instructions are injected.
            Adapters trained on plain replies keep thinking locked off, because they would stop after the reasoning and give no answer.
          </p>
        </Card>

        <Card title="Sampling parameters">
          <dl className="space-y-3">
            {PARAM_HELP.map(([name, text]) => (
              <div key={name}>
                <dt className="text-sm font-medium text-ink">{name}</dt>
                <dd className="text-sm text-muted">{text}</dd>
              </div>
            ))}
          </dl>
        </Card>
      </div>
    </div>
  );
}
