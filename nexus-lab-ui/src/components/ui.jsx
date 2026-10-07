import React from 'react';
import { CheckCircle2, AlertTriangle, Info, X } from 'lucide-react';

const VARIANTS = {
  primary: 'bg-accent text-accent-ink hover:opacity-90 disabled:opacity-40',
  secondary: 'bg-raised text-ink border border-line hover:border-faint disabled:opacity-50',
  ghost: 'text-muted hover:text-ink hover:bg-raised disabled:opacity-40',
  danger: 'text-bad border border-line hover:border-bad disabled:opacity-40',
};

export function Button({ variant = 'secondary', size = 'md', className = '', children, ...props }) {
  const sizing = size === 'sm' ? 'h-8 px-2.5 text-xs' : 'h-10 px-4 text-sm';
  return (
    <button
      className={`inline-flex items-center justify-center gap-2 rounded-lg font-medium transition-colors ${sizing} ${VARIANTS[variant]} ${className}`}
      {...props}
    >
      {children}
    </button>
  );
}

export function IconButton({ label, active = false, className = '', children, ...props }) {
  return (
    <button
      aria-label={label}
      title={label}
      className={`inline-flex h-8 w-8 items-center justify-center rounded-lg transition-colors disabled:opacity-40 ${active ? 'bg-accent-soft text-accent' : 'text-muted hover:bg-raised hover:text-ink'} ${className}`}
      {...props}
    >
      {children}
    </button>
  );
}

export function Card({ title, step, aside, children, className = '' }) {
  return (
    <section className={`rounded-xl border border-line bg-surface ${className}`}>
      {title && (
        <header className="flex items-center gap-3 border-b border-line px-4 py-3">
          {step && (
            <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-accent-soft text-xs font-semibold text-accent">
              {step}
            </span>
          )}
          <h2 className="text-sm font-semibold text-ink">{title}</h2>
          <div className="ml-auto flex items-center gap-2 text-xs text-muted">{aside}</div>
        </header>
      )}
      <div className="p-4">{children}</div>
    </section>
  );
}

export function Label({ children, hint }) {
  return (
    <div className="mb-1.5 flex items-baseline justify-between gap-2">
      <span className="text-xs font-medium text-muted">{children}</span>
      {hint && <span className="font-mono text-xs text-ink">{hint}</span>}
    </div>
  );
}

export const inputClass =
  'w-full rounded-lg border border-line bg-canvas px-3 py-2 text-sm text-ink placeholder:text-faint focus:border-accent focus:outline-none';

const TOAST_ICONS = {
  success: <CheckCircle2 size={16} className="text-good" />,
  error: <AlertTriangle size={16} className="text-bad" />,
  info: <Info size={16} className="text-accent" />,
};

export function Toasts({ toasts, onDismiss }) {
  return (
    <div className="pointer-events-none fixed bottom-4 right-4 z-50 flex w-80 max-w-[calc(100vw-2rem)] flex-col gap-2" role="status" aria-live="polite">
      {toasts.map((t) => (
        <div key={t.id} className="pointer-events-auto flex items-start gap-2.5 rounded-lg border border-line bg-surface px-3 py-2.5 text-sm text-ink shadow-lg">
          <span className="mt-0.5 shrink-0">{TOAST_ICONS[t.kind]}</span>
          <span className="flex-1 break-words">{t.text}</span>
          <button aria-label="Dismiss" onClick={() => onDismiss(t.id)} className="text-faint hover:text-ink">
            <X size={14} />
          </button>
        </div>
      ))}
    </div>
  );
}
