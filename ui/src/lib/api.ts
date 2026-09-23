/** Local Memory API client — talks to the local FastAPI server (127.0.0.1). */

export interface Status {
  app: string
  version: string
  paused: boolean
  pipeline: { status: string; queued: number; done: number; last_file: string }
  watched_folders: string[]
  npu: { providers_available: string[]; qnn_available: boolean }
  stats: { files: number; chunks: number; images: number; total_bytes: number }
}

export interface SearchResult {
  file_id: number
  path: string
  name: string
  kind: string
  folder: string
  size_bytes: number
  mtime: number
  score: number
  /** D-04: which engine(s) matched — 'semantic' | 'keyword' | 'both'. */
  matched_via?: 'semantic' | 'keyword' | 'both'
  /** Snippet text; may contain FTS5 <mark>…</mark> around matched terms. */
  snippet?: string
  /** Alias for a pre-highlighted snippet if the backend splits it out later. */
  highlight?: string
  /** D-06: where the snippet came from — 'ocr' | 'chunk' | 'name'. */
  snippet_source?: 'ocr' | 'chunk' | 'name'
}

export interface StatsReport {
  database: {
    index_db_bytes: number
    wal_bytes: number
    shm_bytes: number
    thumbnails_bytes: number
    total_bytes: number
  }
  files: {
    total: number
    by_kind: { kind: string; count: number; bytes: number }[]
    by_extension: { ext: string; count: number; bytes: number }[]
    by_folder: { folder: string; count: number; bytes: number }[]
  }
  index: {
    total_chunks: number
    embedded_chunks: number
    images_understood: number
    ocr_files: number
    db_page_count: number
    db_page_size: number
  }
  activity: {
    last_scan: { status: string; queued: number; done: number; last_file: string }
    watcher_running: boolean
  }
}

export interface HealthReport {
  summary: { files: number; chunks: number; images: number; total_bytes: number; total_human: string }
  per_folder: { folder: string; count: number; bytes: number; bytes_human: string; kinds: Record<string, number> }[]
  largest: { path: string; size_bytes: number; size_human: string; kind: string; mtime: number }[]
  stale_count: number
  dupe_groups: number
  suggestions: { title: string; detail: string; potential_bytes: number; potential_human: string }[]
}

/** Auth token injected by the server into served index.html (see app.py::_serve_index). */
declare global {
  interface Window {
    __LM_TOKEN__?: string
  }
}

// Module-scoped: read once; arrives via served HTML, never via the JS bundle.
const LM_TOKEN: string = (window as unknown as { __LM_TOKEN__?: string }).__LM_TOKEN__ ?? ''

/** Bearer header for raw fetch calls that bypass req() (e.g. blob thumbnails). */
function authHeader(): Record<string, string> {
  return LM_TOKEN ? { Authorization: `Bearer ${LM_TOKEN}` } : {}
}

/** Blob-URL cache for thumbnails: repeat searches reuse the object URL. */
const thumbCache = new Map<string, string>()

/** Fetch a thumbnail with Bearer auth and return a cached object URL.
 * <img src> cannot send Authorization (401 on stage) so callers must use
 * this helper and render <img src={objectUrl}>. */
export async function thumbnailUrl(file_id: number | string): Promise<string> {
  const key = String(file_id)
  const cached = thumbCache.get(key)
  if (cached) return cached
  const res = await fetch('/api/thumbnail?file_id=' + encodeURIComponent(key), {
    headers: { ...authHeader() },
  })
  if (!res.ok) {
    throw new Error(`thumbnail ${res.status}`)
  }
  const blob = await res.blob()
  const url = URL.createObjectURL(blob)
  thumbCache.set(key, url)
  return url
}

/** Revoke a cached thumbnail object URL (call on unmount / file_id change). */
export function revokeThumbnailUrl(file_id: number | string): void {
  const key = String(file_id)
  const url = thumbCache.get(key)
  if (url) {
    thumbCache.delete(key)
    URL.revokeObjectURL(url)
  }
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: {
      ...(LM_TOKEN ? { Authorization: `Bearer ${LM_TOKEN}` } : {}),
      ...(init?.body ? { 'Content-Type': 'application/json' } : undefined),
      ...init?.headers,
    },
  })
  if (!res.ok) {
    const body = await res.text().catch(() => '')
    throw new Error(`${res.status} ${res.statusText}${body ? `: ${body}` : ''}`)
  }
  return res.json() as Promise<T>
}

export const api = {
  status: () => req<Status>('/api/status'),
  search: (query: string, top_k = 24) =>
    req<{ query: string; results: SearchResult[] }>('/api/search', {
      method: 'POST',
      body: JSON.stringify({ query, top_k }),
    }),
  addFolder: (path: string) =>
    req<{ watched_folders: string[] }>('/api/folders', {
      method: 'POST',
      body: JSON.stringify({ path }),
    }),
  removeFolder: (path: string) =>
    req<{ watched_folders: string[] }>('/api/folders', {
      method: 'DELETE',
      body: JSON.stringify({ path }),
    }),
  suggestFolders: () => req<{ suggestions: string[] }>('/api/suggest-folders'),
  reindex: () => req<{ started: boolean }>('/api/index', { method: 'POST' }),
  pause: () => req<{ paused: boolean }>('/api/pause', { method: 'POST' }),
  resume: () => req<{ paused: boolean }>('/api/resume', { method: 'POST' }),
  health: () => req<HealthReport>('/api/health'),
  stats: () => req<StatsReport>('/api/stats'),
  wipe: () => req<{ wiped: Record<string, boolean> }>('/api/wipe', { method: 'POST' }),
  openFile: (fileId: number) =>
    req<{ opened: boolean }>('/api/open', {
      method: 'POST',
      body: JSON.stringify({ file_id: fileId }),
    }),
}

export function fmtBytes(n: number): string {
  if (!n) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let i = 0
  let v = n
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024
    i++
  }
  return `${v.toFixed(i === 0 ? 0 : 1)} ${units[i]}`
}

export function fmtDate(unix: number): string {
  return new Date(unix * 1000).toLocaleDateString(undefined, {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  })
}
