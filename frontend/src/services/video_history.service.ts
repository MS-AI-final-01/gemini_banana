import { safeSetItem } from './storage.service';

export type VideoHistoryItem = {
  id: string;
  ts: number;
  clips: string[]; // URIs or data URIs
  prompt?: string;
  params?: Record<string, unknown>;
  sourceImage?: string; // optional generated image data URI used to create video
};

const KEY_VIDEO_HISTORY = 'app:tryon:history:videos:v1';
const MAX_ITEMS = 30;

type Listener = () => void;
const listeners: Set<Listener> = new Set();

function read(): VideoHistoryItem[] {
  try {
    const raw = localStorage.getItem(KEY_VIDEO_HISTORY);
    return raw ? (JSON.parse(raw) as VideoHistoryItem[]) : [];
  } catch {
    return [];
  }
}

function write(arr: VideoHistoryItem[]) {
  safeSetItem(KEY_VIDEO_HISTORY, arr);
  listeners.forEach((l) => {
    try { l(); } catch {}
  });
}

export const videoHistory = {
  add(entry: Omit<VideoHistoryItem, 'id' | 'ts'>) {
    if (!entry.clips || entry.clips.length === 0) return;
    const now: VideoHistoryItem = {
      id: `v-${Date.now()}-${Math.random().toString(36).slice(2)}`,
      ts: Date.now(),
      ...entry,
    };
    const list = [now, ...read()].slice(0, MAX_ITEMS);
    write(list);
  },
  list(): VideoHistoryItem[] { return read(); },
  clear() { write([]); },
  remove(id: string) { write(read().filter((v) => v.id !== id)); },
  subscribe(fn: Listener) { listeners.add(fn); return () => listeners.delete(fn); },
};

export default videoHistory;
