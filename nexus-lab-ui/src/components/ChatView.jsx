import React, { useEffect, useRef, useState } from 'react';
import {
  ArrowUp, Brain, ChevronRight, Copy, Check, RefreshCw, ThumbsUp, ThumbsDown,
  SlidersHorizontal, SquarePen, AlertTriangle, BrainCircuit, Zap,
} from 'lucide-react';
import { Button, IconButton } from './ui';
import { splitThinking } from '../api';

const STARTERS = [
  'Explain LoRA fine-tuning in two sentences.',
  'Write a haiku about running models locally.',
  'What is the difference between temperature and top-p?',
];

function AssistantMessage({ message, isLast, busy, onRate, onRegenerate }) {
  const { thought, answer } = splitThinking(message.content);
  const [copied, setCopied] = useState(false);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(answer);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch { /* clipboard unavailable */ }
  };

  return (
    <div className="group">
      {thought && (
        <details className="mb-2 [&[open]_.chev]:rotate-90">
          <summary className="flex w-fit cursor-pointer list-none items-center gap-1.5 rounded-md py-1 text-xs font-medium text-muted hover:text-ink">
            <ChevronRight size={13} className="chev transition-transform" />
            <Brain size={13} /> Thought process
          </summary>
          <p className="mt-1 whitespace-pre-wrap border-l-2 border-line pl-3 text-[13px] leading-relaxed text-muted">{thought}</p>
        </details>
      )}
      <p className="whitespace-pre-wrap text-[15px] leading-7 text-ink">
        {answer || <span className="italic text-faint">The model finished thinking but produced no reply.</span>}
      </p>

      <div className="mt-1.5 flex items-center gap-0.5">
        <IconButton label={copied ? 'Copied' : 'Copy reply'} onClick={copy}>
          {copied ? <Check size={14} className="text-good" /> : <Copy size={14} />}
        </IconButton>
        {isLast && (
          <IconButton label="Regenerate reply" disabled={busy} onClick={onRegenerate}><RefreshCw size={14} /></IconButton>
        )}
        <span className="mx-1 h-4 w-px bg-line" />
        {message.rating ? (
          <span className="flex items-center gap-1.5 px-1 text-xs text-muted">
            {message.rating === 'up'
              ? <><ThumbsUp size={13} className="text-good" /> Added to training data</>
              : <><ThumbsDown size={13} className="text-bad" /> Marked as a bad reply</>}
          </span>
        ) : (
          <>
            <IconButton label="Good reply: add to training data" disabled={!answer} onClick={() => onRate(message, 'up')}><ThumbsUp size={14} /></IconButton>
            <IconButton label="Bad reply: keep out of training data" onClick={() => onRate(message, 'down')}><ThumbsDown size={14} /></IconButton>
          </>
        )}
      </div>
    </div>
  );
}

export default function ChatView({
  status, online, messages, busy, thinking, onToggleThinking, onSend, onRegenerate, onRate, onNewChat,
  configOpen, onToggleConfig, onOpenPicker,
}) {
  const [input, setInput] = useState('');
  const endRef = useRef(null);
  const textRef = useRef(null);

  const modelReady = !!status.current_model;
  const adapterActive = !!status.active_adapter;
  const thinkingOn = thinking && !adapterActive;
  const canSend = modelReady && !busy && input.trim().length > 0;

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' }); }, [messages, busy]);

  // Grow the composer with its content, up to a limit
  useEffect(() => {
    const el = textRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  }, [input]);

  const submit = (text = input) => {
    if (!modelReady || busy || !text.trim()) return;
    onSend(text.trim());
    setInput('');
  };

  const lastAssistantIndex = messages.map((m) => m.role).lastIndexOf('assistant');

  return (
    <div className="flex h-full min-w-0 flex-1 flex-col">
      <div className="flex h-12 shrink-0 items-center justify-between gap-2 border-b border-line px-4">
        <div className="flex min-w-0 items-center gap-2 text-sm">
          <h1 className="font-semibold text-ink">Chat</h1>
          {adapterActive && (
            <span className="flex min-w-0 items-center gap-1 rounded-full bg-accent-soft px-2 py-0.5 text-xs text-accent">
              <Zap size={11} className="shrink-0" /> <span className="truncate font-mono">{status.active_adapter}</span>
            </span>
          )}
        </div>
        <div className="flex items-center gap-1">
          <Button size="sm" variant="ghost" onClick={onNewChat} disabled={messages.length === 0 || busy}>
            <SquarePen size={14} /> New chat
          </Button>
          <IconButton label={configOpen ? 'Hide configuration' : 'Show configuration'} active={configOpen} onClick={onToggleConfig}>
            <SlidersHorizontal size={15} />
          </IconButton>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto">
        <div className="mx-auto flex min-h-full w-full max-w-3xl flex-col px-4 py-6">
          {messages.length === 0 ? (
            <div className="m-auto w-full max-w-md py-8 text-center">
              <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-2xl bg-accent-soft text-accent">
                <BrainCircuit size={24} />
              </div>
              {!online ? (
                <>
                  <h2 className="text-lg font-semibold text-ink">Backend is not running</h2>
                  <p className="mt-2 text-sm text-muted">Start it with <code className="rounded bg-raised px-1.5 py-0.5 font-mono text-[13px]">python main.py</code> and this page will reconnect on its own.</p>
                </>
              ) : !modelReady ? (
                <>
                  <h2 className="text-lg font-semibold text-ink">Load a model to start chatting</h2>
                  <p className="mt-2 text-sm text-muted">Everything runs on this machine. Pick a downloaded model, or paste a Hugging Face ID to fetch one.</p>
                  <Button variant="primary" className="mt-5" onClick={onOpenPicker}>Choose a model</Button>
                </>
              ) : (
                <>
                  <h2 className="text-lg font-semibold text-ink">What would you like to ask?</h2>
                  <p className="mt-2 text-sm text-muted">Rate replies with the thumbs to build your fine-tuning dataset as you chat.</p>
                  <div className="mt-5 flex flex-col gap-2">
                    {STARTERS.map((s) => (
                      <button key={s} onClick={() => submit(s)} className="rounded-lg border border-line bg-surface px-3 py-2.5 text-left text-sm text-ink transition-colors hover:border-faint">
                        {s}
                      </button>
                    ))}
                  </div>
                </>
              )}
            </div>
          ) : (
            <div className="space-y-6">
              {messages.map((m, i) => (
                m.role === 'user' ? (
                  <div key={i} className="flex justify-end">
                    <p className="max-w-[85%] whitespace-pre-wrap rounded-2xl rounded-br-md bg-raised px-4 py-2.5 text-[15px] leading-7 text-ink">{m.content}</p>
                  </div>
                ) : m.role === 'error' ? (
                  <div key={i} className="flex items-start gap-2 rounded-lg border border-line bg-surface px-3 py-2.5 text-sm text-bad">
                    <AlertTriangle size={15} className="mt-0.5 shrink-0" /> <span>{m.content}</span>
                  </div>
                ) : (
                  <AssistantMessage
                    key={i}
                    message={m}
                    busy={busy}
                    isLast={i === lastAssistantIndex && i === messages.length - 1}
                    onRate={onRate}
                    onRegenerate={onRegenerate}
                  />
                )
              ))}
              {busy && (
                <div className="flex items-center gap-2 text-sm text-muted" role="status">
                  <span className="flex gap-1">
                    {[0, 150, 300].map((d) => (
                      <span key={d} className="h-1.5 w-1.5 animate-bounce rounded-full bg-accent" style={{ animationDelay: `${d}ms` }} />
                    ))}
                  </span>
                  {thinkingOn ? 'Thinking…' : 'Generating…'}
                </div>
              )}
            </div>
          )}
          <div ref={endRef} />
        </div>
      </div>

      <div className="shrink-0 px-4 pb-4">
        <div className="mx-auto w-full max-w-3xl rounded-2xl border border-line bg-surface p-2 focus-within:border-accent">
          <textarea
            ref={textRef}
            rows={1}
            value={input}
            disabled={!modelReady}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); submit(); }
            }}
            placeholder={modelReady ? 'Message the model…' : 'Load a model to start'}
            className="block w-full resize-none bg-transparent px-2 py-1.5 text-[15px] leading-6 text-ink placeholder:text-faint focus:outline-none disabled:cursor-not-allowed"
          />
          <div className="mt-1 flex items-center justify-between gap-2">
            <button
              onClick={onToggleThinking}
              disabled={adapterActive}
              aria-pressed={thinkingOn}
              title={adapterActive ? 'Unavailable while an adapter is active' : 'Ask the model to reason step by step before answering'}
              className={`flex h-8 items-center gap-1.5 rounded-full border px-3 text-xs font-medium transition-colors disabled:opacity-50 ${thinkingOn ? 'border-accent bg-accent-soft text-accent' : 'border-line text-muted hover:text-ink'}`}
            >
              <Brain size={13} /> Thinking {thinkingOn ? 'on' : 'off'}
            </button>
            <div className="flex items-center gap-3">
              <span className="hidden text-xs text-faint sm:inline">Enter to send, Shift+Enter for a new line</span>
              <button
                onClick={() => submit()}
                disabled={!canSend}
                aria-label="Send message"
                className="flex h-8 w-8 items-center justify-center rounded-full bg-accent text-accent-ink transition-opacity disabled:opacity-30"
              >
                <ArrowUp size={16} />
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
