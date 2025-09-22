import type { RecommendationItem } from "../types";
import { getStorageUsage, manageStorageSpace } from "./storage.service";

export type TryOnInputHistoryItem = {
  id: string;
  ts: number;
  person: "model" | "upload" | "unknown";
  topLabel?: string;
  pantsLabel?: string;
  shoesLabel?: string;
  outerLabel?: string;
  // 상품 ID로 중복 체크용
  topProductId?: string;
  pantsProductId?: string;
  shoesProductId?: string;
  outerProductId?: string;
  // 상품 데이터 (이미지 URL 포함)
  topProduct?: RecommendationItem;
  pantsProduct?: RecommendationItem;
  shoesProduct?: RecommendationItem;
  outerProduct?: RecommendationItem;
};

export type TryOnOutputHistoryItem = {
  id: string;
  ts: number;
  image: string; // data URI
  evaluation?: {
    score: number;
    reasoning?: string;
    model?: string;
    ts: number;
  };
};

import { safeSetItem } from "./storage.service";

const KEY_INPUTS = "app:tryon:history:inputs:v1";
const KEY_OUTPUTS = "app:tryon:history:outputs:v1";

type Listener = () => void;
const listeners: Set<Listener> = new Set();

type FingerprintSource = {
  person: TryOnInputHistoryItem["person"];
  topLabel?: string;
  pantsLabel?: string;
  shoesLabel?: string;
  outerLabel?: string;
  topProductId?: string;
  pantsProductId?: string;
  shoesProductId?: string;
  outerProductId?: string;
};

const normalizeValue = (id?: string | null, label?: string): string | null => {
  if (id && id.trim()) {
    return id.trim();
  }
  if (label && label.trim()) {
    return label.trim().toLowerCase();
  }
  return null;
};

const buildInputFingerprint = (source: FingerprintSource): string => {
  return JSON.stringify({
    person: source.person,
    top: normalizeValue(source.topProductId, source.topLabel),
    pants: normalizeValue(source.pantsProductId, source.pantsLabel),
    shoes: normalizeValue(source.shoesProductId, source.shoesLabel),
    outer: normalizeValue(source.outerProductId, source.outerLabel),
  });
};

function read<T>(key: string): T[] {
  try {
    const raw = localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T[]) : [];
  } catch {
    return [];
  }
}

function write<T>(key: string, arr: T[]) {
  console.log("🔔 write 함수 호출 - key:", key, "배열 길이:", arr.length);
  const success = safeSetItem(key, arr);
  console.log("🔔 safeSetItem 결과:", success);

  // 저장 성공 여부와 관계없이 notify 호출 (UI 업데이트를 위해)
  notify();
}

function notify() {
  console.log("🔔 tryOnHistory notify 호출됨, listeners 수:", listeners.size);
  listeners.forEach((l) => {
    try {
      l();
    } catch {}
  });
}

// 이미지 압축 함수
function compressImage(
  dataUri: string,
  quality: number = 0.7,
  maxWidth: number = 800
): Promise<string> {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement("canvas");
      const ctx = canvas.getContext("2d");

      // 비율 유지하면서 크기 조정
      const ratio = Math.min(maxWidth / img.width, maxWidth / img.height);
      canvas.width = img.width * ratio;
      canvas.height = img.height * ratio;

      // 캔버스에 그리기
      ctx?.drawImage(img, 0, 0, canvas.width, canvas.height);

      // 압축된 이미지로 변환
      const compressedDataUri = canvas.toDataURL("image/jpeg", quality);
      resolve(compressedDataUri);
    };
    img.src = dataUri;
  });
}

export const tryOnHistory = {
  // 디버깅용: listeners에 접근 가능하도록
  get listeners() {
    return listeners;
  },

  // 임시: 모든 히스토리 데이터 클리어 (개발용)
  clearAll() {
    console.log("🗑️ 모든 히스토리 데이터 클리어 중...");
    localStorage.removeItem(KEY_INPUTS);
    localStorage.removeItem(KEY_OUTPUTS);
    notify();
    console.log("✅ 히스토리 데이터 클리어 완료");
  },
  addInput(item: Omit<TryOnInputHistoryItem, "id" | "ts">) {
    // Drop entries that are only AI-model person without any clothing labels
    if (
      item.person !== "upload" &&
      !item.topLabel &&
      !item.pantsLabel &&
      !item.shoesLabel &&
      !item.outerLabel
    ) {
      return;
    }

    // 중복 체크: 같은 상품이 이미 히스토리에 있는지 확인 (슬롯 무관)
    const existingList = read<TryOnInputHistoryItem>(KEY_INPUTS);
    const newFingerprint = buildInputFingerprint(item);
    const isDuplicate = existingList.some(
      (existing) => buildInputFingerprint(existing) === newFingerprint
    );

    if (isDuplicate) {
      console.log("중복된 상품이므로 히스토리에 추가하지 않음", {
        fingerprint: newFingerprint,
      });
      return;
    }

    const now: TryOnInputHistoryItem = {
      id: `h-${Date.now()}-${Math.random().toString(36).slice(2)}`,
      ts: Date.now(),
      ...item,
    };
    const list = [now, ...existingList];
    write(KEY_INPUTS, list);

    // 저장 후 용량 관리 실행
    manageStorageSpace();
  },
  async addOutput(imageDataUri: string) {
    // 현재 localStorage 용량 확인
    const usage = getStorageUsage();
    console.log("🔔 현재 localStorage 용량:", usage);
    console.log(
      "🔔 addOutput 호출됨, 이미지 데이터 길이:",
      imageDataUri.length
    );

    // 이미지 압축 (크기 줄이기) - 더 강한 압축
    const compressedImageDataUri = await compressImage(imageDataUri, 0.5, 600);
    console.log(
      "🔔 압축 후 이미지 데이터 길이:",
      compressedImageDataUri.length
    );

    const now: TryOnOutputHistoryItem = {
      id: `o-${Date.now()}-${Math.random().toString(36).slice(2)}`,
      ts: Date.now(),
      image: compressedImageDataUri,
      evaluation: {
        score: 85, // 기본 점수
        reasoning: "자동 생성된 결과",
        model: "virtual-try-on",
        ts: Date.now(),
      },
    };

    const existingList = read<TryOnOutputHistoryItem>(KEY_OUTPUTS);
    console.log("🔔 기존 출력 히스토리 개수:", existingList.length);

    const list = [now, ...existingList];
    console.log("🔔 새로운 리스트 길이:", list.length);

    write(KEY_OUTPUTS, list);
    console.log("🔔 출력 히스토리 저장 완료");

    // 저장 후 용량 관리 실행
    manageStorageSpace();
    console.log("🔔 용량 관리 완료");

    // 추가로 notify 호출 (UI 즉시 업데이트)
    notify();
  },
  updateOutput(id: string, patch: Partial<TryOnOutputHistoryItem>) {
    const list = read<TryOnOutputHistoryItem>(KEY_OUTPUTS);
    const idx = list.findIndex((it) => it.id === id);
    if (idx >= 0) {
      list[idx] = { ...list[idx], ...patch };
      write(KEY_OUTPUTS, list);
    }
  },
  inputs(): TryOnInputHistoryItem[] {
    return read<TryOnInputHistoryItem>(KEY_INPUTS);
  },
  outputs(): TryOnOutputHistoryItem[] {
    return read<TryOnOutputHistoryItem>(KEY_OUTPUTS);
  },
  clearInputs() {
    write(KEY_INPUTS, []);
  },
  clearOutputs() {
    write(KEY_OUTPUTS, []);
  },
  removeInput(id: string) {
    const list = read<TryOnInputHistoryItem>(KEY_INPUTS);
    const filtered = list.filter((item) => item.id !== id);
    write(KEY_INPUTS, filtered);
  },
  removeOutput(id: string) {
    const list = read<TryOnOutputHistoryItem>(KEY_OUTPUTS);
    const filtered = list.filter((item) => item.id !== id);
    write(KEY_OUTPUTS, filtered);
  },
  subscribe(fn: Listener) {
    listeners.add(fn);
    return () => listeners.delete(fn);
  },
};

export default tryOnHistory;
