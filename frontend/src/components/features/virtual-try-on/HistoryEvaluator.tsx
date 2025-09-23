import React, { useMemo, useState } from 'react';
import { Card, Button, Spinner, useToast, toast } from '../../ui';
import { tryOnHistory, TryOnOutputHistoryItem } from '../../../services/tryon_history.service';
import { virtualTryOnService } from '../../../services/virtualTryOn.service';

export const HistoryEvaluator: React.FC = () => {
  const { addToast } = useToast();
  const [selected, setSelected] = useState<Record<string, boolean>>({});
  const [loading, setLoading] = useState(false);
  const outputs = tryOnHistory.outputs();

  const selectedImages = useMemo(() => outputs.filter(o => selected[o.id]).map(o => o.image), [outputs, selected]);
  const canEval = selectedImages.length > 0;

  const toggle = (id: string) => setSelected(prev => ({ ...prev, [id]: !prev[id] }));

  const evaluate = async () => {
    if (!canEval) return;
    setLoading(true);
    try {
      const res = await virtualTryOnService.evaluateOutfits({ images: selectedImages });
      // Map scores back to items in selection order
      const selIds = outputs.filter(o => selected[o.id]).map(o => o.id);
      res.results.forEach((r, i) => {
        const id = selIds[i] ?? selIds[r.index] ?? selIds[0];
        if (!id) return;
        tryOnHistory.updateOutput(id, { evaluation: { score: r.score, reasoning: r.reasoning, ts: Date.now() } as any });
      });
      addToast(toast.success('평가 완료'));
    } catch (e: any) {
      addToast(toast.error('평가 실패', e?.message));
    } finally { setLoading(false); }
  };

  return (
    <Card className="space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-gray-800">결과 평가 (LLM)</h3>
        <div className="flex gap-2">
          <Button size="sm" variant="outline" onClick={() => setSelected({})} disabled={loading}>선택 해제</Button>
          <Button size="sm" onClick={evaluate} disabled={!canEval} loading={loading}>평가하기</Button>
        </div>
      </div>
      {outputs.length === 0 ? (
        <div className="text-sm text-gray-500">아직 저장된 결과 이미지가 없습니다.</div>
      ) : (
        <div className="grid grid-cols-3 gap-3">
          {outputs.map((o: TryOnOutputHistoryItem) => (
            <label key={o.id} className={`relative block aspect-[4/5] rounded-lg overflow-hidden border ${selected[o.id] ? 'border-blue-500 ring-2 ring-blue-200' : 'border-gray-200'}`}>
              <input type="checkbox" className="absolute top-2 left-2 z-10" checked={!!selected[o.id]} onChange={() => toggle(o.id)} />
              <img src={o.image} alt="history" className="w-full h-full object-cover" />
              {o.evaluation && (
                <div className="absolute bottom-2 right-2 bg-black/60 text-white text-xs px-2 py-0.5 rounded-md">{o.evaluation.score}%</div>
              )}
            </label>
          ))}
        </div>
      )}
    </Card>
  );
};

export default HistoryEvaluator;

