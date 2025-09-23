import React, { useEffect, useMemo, useState } from 'react';
import type { TryOnInputHistoryItem, TryOnOutputHistoryItem } from '../../../services/tryon_history.service';
import { tryOnHistory } from '../../../services/tryon_history.service';
import type { VideoHistoryItem } from '../../../services/video_history.service';
import { videoHistory } from '../../../services/video_history.service';
import { Button, Card } from '../../ui';
import { FullScreenImage } from '../common/FullScreenImage';

interface MyPageHistoryItemProps {
  item: TryOnInputHistoryItem;
  getHistoryItemImage: (item: TryOnInputHistoryItem) => Promise<string | null>;
}

const MyPageHistoryItem: React.FC<MyPageHistoryItemProps> = ({ item, getHistoryItemImage }) => {
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadImage = async () => {
      setLoading(true);
      try {
        const image = await getHistoryItemImage(item);
        setImageUrl(image);
      } catch (error) {
        console.warn('히스토리 이미지 로드 실패:', error);
        setImageUrl(null);
      } finally {
        setLoading(false);
      }
    };
    loadImage();
  }, [item, getHistoryItemImage]);

  const hasClothing = item.topLabel || item.pantsLabel || item.shoesLabel || item.outerLabel;

  return (
    <div className="group relative aspect-[4/5] overflow-hidden rounded-xl bg-gray-100 ring-1 ring-transparent hover:ring-blue-200">
      {loading ? (
        <div className="flex h-full w-full items-center justify-center text-sm text-gray-400">로딩 중...</div>
      ) : imageUrl ? (
        <img src={imageUrl} alt="의류 조합" className="h-full w-full object-cover transition-transform duration-200 group-hover:scale-105" />
      ) : hasClothing ? (
        <div className="flex h-full w-full flex-col items-center justify-center gap-1 p-3 text-center text-sm text-gray-600">
          <span className="font-medium">의류 조합</span>
          <span className="text-xs text-gray-500">{[item.topLabel, item.pantsLabel, item.shoesLabel, item.outerLabel].filter(Boolean).join(', ')}</span>
        </div>
      ) : (
        <div className="flex h-full w-full items-center justify-center text-sm text-gray-400">-</div>
      )}
    </div>
  );
};

export const MyPage: React.FC = () => {
  const [inputs, setInputs] = useState<TryOnInputHistoryItem[]>(tryOnHistory.inputs());
  const [outputs, setOutputs] = useState<TryOnOutputHistoryItem[]>(tryOnHistory.outputs());
  const [videos, setVideos] = useState<VideoHistoryItem[]>(videoHistory.list());
  const [view, setView] = useState<string | null>(null);
  const toPlayable = useMemo(() => (u: string) => (u && u.startsWith('gs://')) ? `/api/try-on/video/stream?uri=${encodeURIComponent(u)}` : u, []);

  useEffect(() => {
    const unsub = tryOnHistory.subscribe(() => {
      setInputs(tryOnHistory.inputs());
      setOutputs(tryOnHistory.outputs());
    });
    const unsubVideo = videoHistory.subscribe(() => setVideos(videoHistory.list()));

    const onStorage = (e: StorageEvent) => {
      if (e.key && e.key.includes('app:tryon:history')) {
        setInputs(tryOnHistory.inputs());
        setOutputs(tryOnHistory.outputs());
      }
      if (e.key === 'app:tryon:history:videos:v1') {
        setVideos(videoHistory.list());
      }
    };

    window.addEventListener('storage', onStorage);
    return () => { unsub(); unsubVideo(); window.removeEventListener('storage', onStorage); };
  }, []);

  const getHistoryItemImage = async (item: TryOnInputHistoryItem): Promise<string | null> => {
    // 1. 업로드된 이미지 데이터가 있으면 우선 사용
    if (item.topImageData) return `data:image/jpeg;base64,${item.topImageData}`;
    if (item.pantsImageData) return `data:image/jpeg;base64,${item.pantsImageData}`;
    if (item.shoesImageData) return `data:image/jpeg;base64,${item.shoesImageData}`;
    if (item.outerImageData) return `data:image/jpeg;base64,${item.outerImageData}`;
    
    // 2. 상품 이미지 URL 사용
    const products = [item.topProduct, item.pantsProduct, item.shoesProduct, item.outerProduct].filter(Boolean);
    for (const product of products) {
      if (product?.imageUrl) return product.imageUrl;
    }
    return null;
  };

  return (
    <div className="min-h-screen bg-white pt-[124px]">
      <div className="mx-auto max-w-[1280px] px-8 pb-16">
        <div className="grid grid-cols-1 items-start gap-10">
          {/* 영상 히스토리 – 최상단, 가로 전체 */}
          <Card className="space-y-6 p-6 xl:p-8">
            <div className="flex items-center justify-between">
              <h2 className="text-2xl font-bold text-gray-900">영상 히스토리</h2>
              <div className="flex gap-2">
                <Button size="sm" variant="outline" onClick={() => setVideos(videoHistory.list())}>새로고침</Button>
                <Button size="sm" variant="ghost" onClick={() => { videoHistory.clear(); setVideos([]); }}>비우기</Button>
              </div>
            </div>
            {videos.length === 0 ? (
              <div className="py-12 text-center text-sm text-gray-500">저장된 영상 히스토리가 없습니다.</div>
            ) : (
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                {videos.map((v) => (
                  <div key={v.id} className="group relative aspect-[9/16] min-h-[240px] md:min-h-[300px] overflow-hidden rounded-xl bg-gray-100 ring-1 ring-transparent hover:ring-blue-200">
                    <video src={toPlayable(v.clips[0])} className="h-full w-full object-cover" controls playsInline />
                  </div>
                ))}
              </div>
            )}
          </Card>

          {/* 입력 히스토리 – 영상 히스토리와 동일 폭으로 아래 배치 */}
          <Card className="space-y-6 p-6 xl:p-8">
            <div className="flex items-center justify-between">
              <h2 className="text-2xl font-bold text-gray-900">입력 히스토리</h2>
              <div className="flex gap-2">
                <Button size="sm" variant="outline" onClick={() => setInputs(tryOnHistory.inputs())}>새로고침</Button>
                <Button size="sm" variant="ghost" onClick={() => { tryOnHistory.clearInputs(); setInputs([]); }}>비우기</Button>
              </div>
            </div>
            {inputs.length === 0 ? (
              <div className="py-12 text-center text-sm text-gray-500">저장된 입력 히스토리가 없습니다.</div>
            ) : (
              <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 xl:grid-cols-4">
                {inputs.map((it) => (
                  <MyPageHistoryItem key={it.id} item={it} getHistoryItemImage={getHistoryItemImage} />
                ))}
              </div>
            )}
          </Card>

          {/* 결과 히스토리 – 동일 폭으로 아래 배치 */}
          <Card className="space-y-6 p-6 xl:p-8">
            <div className="flex items-center justify-between">
              <h2 className="text-2xl font-bold text-gray-900">결과 히스토리</h2>
              <div className="flex gap-2">
                <Button size="sm" variant="outline" onClick={() => setOutputs(tryOnHistory.outputs())}>새로고침</Button>
                <Button size="sm" variant="ghost" onClick={() => { tryOnHistory.clearOutputs(); setOutputs([]); }}>비우기</Button>
              </div>
            </div>
            {outputs.length === 0 ? (
              <div className="py-12 text-center text-sm text-gray-500">저장된 결과 이미지가 없습니다.</div>
            ) : (
              <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 xl:grid-cols-4">
                {outputs.map((o) => (
                  <button key={o.id} type="button" className="group relative aspect-[4/5] overflow-hidden rounded-xl bg-gray-100 ring-1 ring-transparent hover:ring-blue-200" onClick={() => setView(o.image)}>
                    <img src={o.image} alt="history" className="h-full w-full object-cover transition-transform duration-200 group-hover:scale-105" />
                  </button>
                ))}
              </div>
            )}
          </Card>
        </div>
      </div>

      {view && <FullScreenImage src={view} onClose={() => setView(null)} />}
    </div>
  );
};

export default MyPage;
