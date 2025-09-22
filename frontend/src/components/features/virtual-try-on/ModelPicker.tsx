import React, { useEffect, useMemo, useState } from 'react';
import { Button, Card } from '../../ui';
import type { UploadedImage } from '../../../types';
import { imageProxy } from '../../../services/imageProxy.service';

interface ModelPickerProps {
  onPick: (img: UploadedImage) => void;
  direction?: 'horizontal' | 'vertical';
  // highlight currently selected AI model id
  selectedId?: string;
  // notify parent which model id was selected
  onSelectModel?: (id: string) => void;
  variant?: 'card' | 'inline';
}

const MODEL_FILES = [
  { id: 'male1', label: '남자 1' },
  { id: 'male2', label: '남자 2' },
  { id: 'male3', label: '남자 3' },
  { id: 'female1', label: '여자 1' },
  { id: 'female2', label: '여자 2' },
  { id: 'female3', label: '여자 3' },
];

const EXTS = ['jpeg'];

function nameVariants(id: string): string[] {
  // male1 -> [male1, male-1, male_1, male 1]
  const m = id.match(/^(male|female)(\d)$/i);
  if (!m) return [id];
  const base = m[1];
  const num = m[2];
  return [
    `${base}${num}`,
    `${base}-${num}`,
    `${base}_${num}`,
    `${base} ${num}`,
  ];
}

export const ModelPicker: React.FC<ModelPickerProps> = ({ onPick, direction = 'horizontal', selectedId, onSelectModel, variant = 'card' }) => {
  const candidates = useMemo(() => MODEL_FILES.map(m => {
    const names = nameVariants(m.id);
    const folders = ['models', 'model']; // support both public/models and public/model
    const urls: string[] = [];
    for (const folder of folders) {
      for (const n of names) {
        for (const ext of EXTS) {
          urls.push(`/${folder}/${n}.${ext}`);
        }
      }
    }
    return { ...m, urls };
  }), []);

  const [previewMap, setPreviewMap] = useState<Record<string, string>>({});

  useEffect(() => {
    let cancelled = false;
    const pick = (urls: string[]) => new Promise<string | null>((resolve) => {
      const tryNext = (i: number) => {
        if (i >= urls.length) return resolve(null);
        const img = new Image();
        img.onload = () => resolve(urls[i]);
        img.onerror = () => tryNext(i + 1);
        img.src = urls[i];
      };
      tryNext(0);
    });
    (async () => {
      const entries: Record<string, string> = {};
      for (const m of candidates) {
        const ok = await pick(m.urls);
        if (ok) entries[m.id] = ok;
      }
      if (!cancelled) setPreviewMap(entries);
    })();
    return () => { cancelled = true; };
  }, [candidates]);

  const handlePick = async (urls: string[], label: string, id?: string) => {
    let lastErr: any;
    // Prefer the URL that actually loaded during preload
    const ordered = id && previewMap[id] ? [previewMap[id]!, ...urls.filter(u => u !== previewMap[id])] : urls;
    if (id) onSelectModel?.(id);
    for (const u of ordered) {
      try {
        const img = await imageProxy.toUploadedImage(u, label);
        onPick(img);
        return;
      } catch (e) { lastErr = e; }
    }
    console.warn('Failed to load any model image', lastErr);
  };

  if (variant === 'inline') {
    return (
      <div className="flex w-full flex-col gap-3 rounded-2xl border border-gray-200 bg-white/70 p-3">
        <div className="flex items-start justify-between">
          <div className="leading-tight">
            <h3 className="text-xs font-semibold text-gray-700 uppercase tracking-wide">
              AI 모델
            </h3>
            <p className="text-[11px] text-gray-500">선택(샘플)</p>
          </div>
          <span className="text-[10px] text-gray-400">public/models</span>
        </div>
        <div className="-mx-1 flex w-full gap-2.5 overflow-x-auto pb-2 px-1 snap-x">
          {candidates.map((m) => {
            const isSelected = selectedId === m.id;
            const ringClass = isSelected
              ? 'ring-2 ring-blue-500'
              : 'ring-1 ring-transparent focus:ring-blue-400';
            return (
              <div
                key={m.id}
                className="flex min-w-[7.25rem] max-w-[7.25rem] flex-shrink-0 snap-start flex-col gap-2 rounded-xl border border-gray-200 bg-white p-2 shadow-sm md:min-w-[7.75rem] md:max-w-[7.75rem] lg:min-w-[8.25rem] lg:max-w-[8.25rem]"
              >
                <button
                  type="button"
                  onClick={() => handlePick(m.urls, m.id, m.id)}
                  className={`relative w-full aspect-square overflow-hidden rounded-lg bg-gray-100 transition focus:outline-none focus:ring-2 ${ringClass}`}
                  title="이미지를 클릭하면 사용합니다"
                >
                  {previewMap[m.id] ? (
                    <img
                      src={previewMap[m.id]}
                      alt={m.label}
                      className="h-full w-full object-cover"
                    />
                  ) : (
                    <div className="flex h-full w-full items-center justify-center text-xs text-gray-400">
                      이미지 없음
                    </div>
                  )}
                </button>
                <p className="text-sm font-medium text-gray-800 truncate text-center">
                  {m.label}
                </p>
                <Button
                  size="sm"
                  variant={isSelected ? 'primary' : 'outline'}
                  onClick={() => handlePick(m.urls, m.id, m.id)}
                >
                  {isSelected ? '선택됨' : '사용'}
                </Button>
              </div>
            );
          })}
        </div>
      </div>
    );
  }

  if (direction === 'vertical') {
    return (
      <Card className="space-y-3">
        <div className="flex items-start justify-between">
          <div>
            <h3 className="text-base font-bold text-gray-800 leading-tight">AI 모델</h3>
            <p className="text-xs text-gray-500">선택(샘플)</p>
          </div>
          <span className="text-[10px] text-gray-400">public/models</span>
        </div>
        <div className="flex flex-col gap-3 max-h-[60vh] xl:max-h-[70vh] overflow-y-auto pr-1">
          {candidates.map(m => {
            const isSelected = selectedId === m.id;
            const ringClass = isSelected ? 'ring-2 ring-blue-500' : 'ring-1 ring-transparent hover:ring-blue-200';
            return (
              <div key={m.id} className="w-full">
                <div
                  className={`aspect-[3/4] rounded-lg overflow-hidden bg-gray-100 border border-gray-200 cursor-pointer ${ringClass}`}
                  onClick={() => handlePick(m.urls, m.id, m.id)}
                  title="이미지를 클릭하면 사용합니다"
                >
                  {previewMap[m.id] ? (
                    <img src={previewMap[m.id]} alt={m.label} className="w-full h-full object-cover" />
                  ) : (
                    <div className="w-full h-full flex items-center justify-center text-gray-400 text-xs">이미지 없음</div>
                  )}
                </div>
                <p className="mt-1 text-xs text-gray-700 truncate text-center">{m.label}</p>
                <div className="mt-1 text-center">
                  <Button size="sm" onClick={() => handlePick(m.urls, m.id, m.id)}>사용</Button>
                </div>
              </div>
            );
          })}
        </div>
      </Card>
    );
  }

  // horizontal (default)
  return (
    <Card className="space-y-3">
      <div className="flex items-start justify-between">
        <div>
          <h3 className="text-base font-bold text-gray-800 leading-tight">AI 모델</h3>
          <p className="text-xs text-gray-500">선택(샘플)</p>
        </div>
        <span className="text-[10px] text-gray-400">public/models</span>
      </div>
      <div className="overflow-x-auto whitespace-nowrap flex gap-4 pb-1">
        {candidates.map(m => {
          const isSelected = selectedId === m.id;
          const ringClass = isSelected ? 'ring-2 ring-blue-500' : 'ring-1 ring-transparent hover:ring-blue-200';
          return (
            <div key={m.id} className="inline-block w-32">
              <div
                className={`aspect-[3/4] rounded-lg overflow-hidden bg-gray-100 border border-gray-200 cursor-pointer ${ringClass}`}
                onClick={() => handlePick(m.urls, m.id, m.id)}
                title="이미지를 클릭하면 사용합니다"
              >
                {previewMap[m.id] ? (
                  <img src={previewMap[m.id]} alt={m.label} className="w-full h-full object-cover" />
                ) : (
                  <div className="w-full h-full flex items-center justify-center text-gray-400 text-xs">이미지 없음</div>
                )}
              </div>
              <p className="mt-1 text-xs text-gray-700 truncate text-center">{m.label}</p>
              <div className="mt-1 text-center">
                <Button size="sm" onClick={() => handlePick(m.urls, m.id, m.id)}>사용</Button>
              </div>
            </div>
          );
        })}
      </div>
    </Card>
  );
};

export default ModelPicker;
