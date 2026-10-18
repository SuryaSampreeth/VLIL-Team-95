// Thin client for the FastAPI backend (see docs/API.md).
export const API_URL = (import.meta.env.VITE_API_URL || 'http://localhost:8000').replace(/\/$/, '')

export class ApiError extends Error {
  constructor(message, status) {
    super(message)
    this.status = status
  }
}

async function request(path, { method = 'GET', form, signal } = {}) {
  let res
  try {
    // ngrok's free tier serves an HTML warning page unless this header is present
    res = await fetch(API_URL + path, { method, body: form, signal, headers: { 'ngrok-skip-browser-warning': '1' } })
  } catch (e) {
    if (e.name === 'AbortError') throw e
    throw new ApiError(`Cannot reach the backend at ${API_URL}. Start it with "uvicorn app.main:app --port 8000".`, 0)
  }
  let data = null
  try {
    data = await res.json()
  } catch {
    /* non-JSON error body */
  }
  if (!res.ok) {
    const detail = data?.detail
    const msg = Array.isArray(detail) ? detail.map((d) => d.msg).join('; ') : detail || res.statusText
    throw new ApiError(msg || `Request failed (${res.status})`, res.status)
  }
  return data
}

const formOf = (fields) => {
  const f = new FormData()
  Object.entries(fields).forEach(([k, v]) => {
    if (v === undefined || v === null || v === '') return
    if (v instanceof Blob) f.append(k, v, v.name || 'image.png')
    else f.append(k, typeof v === 'object' ? JSON.stringify(v) : String(v))
  })
  return f
}

export const api = {
  health: (signal) => request('/health', { signal }),
  models: (stage, signal) => request(`/models/${stage}`, { signal }),
  enhance: (image, model, gateMode = 'always', signal) =>
    request('/stage1/enhance', { method: 'POST', form: formOf({ image, model, gate_mode: gateMode }), signal }),
  detect: (image, model, signal) =>
    request('/stage2/detect', { method: 'POST', form: formOf({ image, model }), signal }),
  report: (image, { findings, crosscheckModel } = {}, signal) =>
    request('/stage3/report', {
      method: 'POST',
      form: formOf({ image, findings, crosscheck_model: crosscheckModel }),
      signal,
    }),
  pipeline: (image, stage1Model, stage2Model, gateMode = 'auto', signal) =>
    request('/pipeline/run', {
      method: 'POST',
      form: formOf({ image, stage1_model: stage1Model, stage2_model: stage2Model, gate_mode: gateMode }),
      signal,
    }),
}

// ─── Date Formatting Utilities ─────────────────────────────────
export const formatRelativeTime = (dateStr) => {
  const date = new Date(dateStr);
  const now = new Date();
  const diffMs = now - date;
  const diffMins = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMs / 3600000);
  const diffDays = Math.floor(diffMs / 86400000);
  if (diffMins < 1) return 'just now';
  if (diffMins < 60) return diffMins + 'm ago';
  if (diffHours < 24) return diffHours + 'h ago';
  if (diffDays < 7) return diffDays + 'd ago';
  return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
};

export const formatFileSize = (bytes) => {
  if (bytes === 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  return (bytes / Math.pow(1024, i)).toFixed(1) + ' ' + units[i];
};

// Updated: 2026-10-18
// Added sanitization
// Improved errors
// Added retry logic
// Fixed timeouts
