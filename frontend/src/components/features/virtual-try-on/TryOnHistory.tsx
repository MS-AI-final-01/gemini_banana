import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { tryOnHistory, TryOnInputHistoryItem, TryOnOutputHistoryItem } from '../../../services/tryon_history.service';
import type { RecommendationItem, UploadedImage } from '../../../types';
import { Button, Card } from '../../ui';
import { FullScreenImage } from '../common/FullScreenImage';
import { ImageHistoryCard } from './ImageHistoryCard';
import { ProductHistoryCard } from './ProductHistoryCard';

interface TryOnHistoryProps {
  onApply?: (payload: { 
    topProduct?: RecommendationItem;
    pantsProduct?: RecommendationItem;
    shoesProduct?: RecommendationItem;
    outerProduct?: RecommendationItem;
  }) => void;
  onImageApply?: (slot: 'top' | 'pants' | 'shoes' | 'outer', image: UploadedImage, label: string) => Promise<void>;
}

interface HistoryItemCardProps {
  item: TryOnInputHistoryItem;
  onApply?: (payload: { 
    topProduct?: RecommendationItem;
    pantsProduct?: RecommendationItem;
    shoesProduct?: RecommendationItem;
    outerProduct?: RecommendationItem;
  }) => void;
  onImageApply?: (slot: 'top' | 'pants' | 'shoes' | 'outer', image: UploadedImage, label: string) => Promise<void>;
}

const HistoryItemCard: React.FC<HistoryItemCardProps> = ({ item, onApply, onImageApply }) => {
  // imageData가 있으면 이미지 카드, 없으면 상품 카드
  const hasImageData = item.topImageData || item.pantsImageData || 
                      item.shoesImageData || item.outerImageData;
  
  return hasImageData ? 
    <ImageHistoryCard item={item} onApply={onApply} onImageApply={onImageApply} /> :
    <ProductHistoryCard item={item} onApply={onApply} />;
};

export const TryOnHistory: React.FC<TryOnHistoryProps> = ({ onApply, onImageApply }) => {
  console.log('🔔 TryOnHistory 컴포넌트 렌더링됨');
  const [inputs, setInputs] = useState(tryOnHistory.inputs());
  const [outputs, setOutputs] = useState(tryOnHistory.outputs());
  const [view, setView] = useState<string | null>(null);
  const [viewingItem, setViewingItem] = useState<TryOnOutputHistoryItem | null>(null);
  const [sortMode, setSortMode] = useState<'recent' | 'rank'>('recent');
  
  // 슬라이드 관련 상태
  const [currentSlide, setCurrentSlide] = useState(0);
  const slideRef = useRef<HTMLDivElement>(null);
  const itemsPerSlide = 2; // 한 번에 보여줄 아이템 수
  
  // 상품 데이터는 히스토리에 저장되므로 별도 캐시 불필요

  const refresh = useCallback(() => {
    const newInputs = tryOnHistory.inputs();
    const newOutputs = tryOnHistory.outputs();
    console.log('🔔 TryOnHistory refresh - 입력:', newInputs.length, '출력:', newOutputs.length);
    setInputs(newInputs);
    setOutputs(newOutputs);
    console.log('🔔 refresh 후 상태 설정 완료');
  }, []); // 의존성 배열을 비움

  // 추가 API 호출이 필요 없는 구조 (상품 메타데이터가 히스토리에 포함됨)


  useEffect(() => {
    console.log('🔔 TryOnHistory useEffect 실행, 리스너 구독 시작');
    
    // 구독 전에 현재 listeners 수 확인
    console.log('🔔 구독 전 listeners 수:', tryOnHistory.listeners.size);
    
    const unsub = tryOnHistory.subscribe(() => {
      console.log('🔔 TryOnHistory 리스너 호출됨, refresh 실행');
      console.log('🔔 리스너 호출 시점 - 현재 outputs 개수:', tryOnHistory.outputs().length);
      refresh();
    });
    
    // 구독 후 listeners 수 확인
    console.log('🔔 구독 후 listeners 수:', tryOnHistory.listeners.size);
    
    const onStorage = (e: StorageEvent) => {
      if (e.key === 'app:tryon:history:inputs:v1' || e.key === 'app:tryon:history:outputs:v1') {
        console.log('🔔 TryOnHistory storage 이벤트 감지, refresh 실행');
        refresh();
      }
    };
    window.addEventListener('storage', onStorage);
    
    // 구독이 제대로 되었는지 확인
    if (tryOnHistory.listeners.size === 0) {
      console.error('❌ TryOnHistory 구독 실패! listeners 수가 0입니다.');
    } else {
      console.log('✅ TryOnHistory 구독 성공!');
    }
    
    return () => { 
      console.log('🔔 TryOnHistory 컴포넌트 언마운트, 리스너 해제');
      unsub(); 
      window.removeEventListener('storage', onStorage); 
    };
  }, []);

  // 추가: outputs 변경 감지용 useEffect
  useEffect(() => {
    console.log('🔔 outputs 상태 변경 감지:', outputs.length);
  }, [outputs]);


  const outputsSorted = useMemo(() => {
    const arr = [...outputs];
    if (sortMode === 'recent') {
      arr.sort((a, b) => (b.ts || 0) - (a.ts || 0));
    } else {
      // 랭킹 모드 정렬 (평가 점수 기준)
      arr.sort((a, b) => {
        const scoreA = a.evaluation?.score || 0;
        const scoreB = b.evaluation?.score || 0;
        return scoreB - scoreA;
      });
    }
    return arr;
  }, [outputs, sortMode]);

  // 슬라이드 함수들
  const totalSlides = Math.ceil(outputsSorted.length / itemsPerSlide);
  
  const goToPreviousSlide = () => {
    setCurrentSlide(prev => (prev > 0 ? prev - 1 : totalSlides - 1));
  };
  
  const goToNextSlide = () => {
    setCurrentSlide(prev => (prev < totalSlides - 1 ? prev + 1 : 0));
  };

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      <Card className="space-y-3 md:col-span-2 min-h-[260px] order-2">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold text-gray-800">입력 히스토리</h3>
          <div className="flex gap-2">
            <Button size="sm" variant="outline" onClick={refresh}>새로고침</Button>
            <Button size="sm" variant="ghost" onClick={() => { tryOnHistory.clearInputs(); refresh(); }}>비우기</Button>
          </div>
        </div>
        <div className="overflow-x-auto">
          <div className="grid grid-rows-2 grid-flow-col auto-cols-[160px] gap-3 pr-1">
            {inputs.length === 0 ? (
              <div className="row-span-2 flex items-center justify-center text-sm text-gray-500 w-80">기록이 없습니다.</div>
            ) : inputs.map(item => {
              return <HistoryItemCard key={item.id} item={item} onApply={onApply} onImageApply={onImageApply} />;
            })}
          </div>
        </div>
      </Card>

      <Card className="space-y-3 md:col-span-2 min-h-[260px] order-1">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold text-gray-800">결과 히스토리</h3>
          <div className="flex gap-2">
            <Button size="sm" variant={sortMode === 'rank' ? 'secondary' : 'outline'} onClick={() => setSortMode(sortMode === 'rank' ? 'recent' : 'rank')}>
              {sortMode === 'rank' ? '최신순' : '랭킹순위'}
            </Button>
            <Button size="sm" variant="outline" onClick={refresh}>새로고침</Button>
            <Button size="sm" variant="ghost" onClick={() => { tryOnHistory.clearOutputs(); refresh(); }}>비우기</Button>
          </div>
        </div>
        {outputsSorted.length === 0 ? (
          <div className="text-sm text-gray-500">기록이 없습니다.</div>
        ) : (
          <div className="relative">
            {/* 슬라이드 컨테이너 */}
            <div className="overflow-hidden">
              <div 
                ref={slideRef}
                className="flex transition-transform duration-300 ease-in-out"
                style={{ transform: `translateX(-${currentSlide * 100}%)` }}
              >
                {Array.from({ length: totalSlides }, (_, slideIndex) => (
                  <div key={slideIndex} className="w-full flex-shrink-0">
                    <div className="grid grid-cols-2 gap-3">
                      {outputsSorted
                        .slice(slideIndex * itemsPerSlide, (slideIndex + 1) * itemsPerSlide)
                        .map((o: TryOnOutputHistoryItem) => (
                          <button 
                            key={o.id} 
                            onClick={() => { setView(o.image); setViewingItem(o); }} 
                            className="relative group aspect-[4/5] rounded-lg overflow-hidden bg-gray-100 ring-1 ring-transparent hover:ring-blue-200"
                          >
                            <img src={o.image} alt="history" className="w-full h-full object-cover" />
                            {typeof o.evaluation?.score === 'number' && (
                              <div className="absolute bottom-2 right-2 bg-black/60 text-white text-xs px-2 py-0.5 rounded-md">
                                ⭐ {o.evaluation!.score}점
                              </div>
                            )}
                          </button>
                        ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
            
            {/* 슬라이드 네비게이션 버튼 */}
            {totalSlides > 1 && (
              <div className="flex items-center justify-between mt-4">
                <Button 
                  size="sm" 
                  variant="outline" 
                  onClick={goToPreviousSlide}
                  className="flex items-center gap-1"
                >
                  ← 이전
                </Button>
                <div className="flex gap-1">
                  {Array.from({ length: totalSlides }, (_, index) => (
                    <button
                      key={index}
                      onClick={() => setCurrentSlide(index)}
                      className={`w-2 h-2 rounded-full transition-colors ${
                        index === currentSlide ? 'bg-blue-600' : 'bg-gray-300'
                      }`}
                    />
                  ))}
                </div>
                <Button 
                  size="sm" 
                  variant="outline" 
                  onClick={goToNextSlide}
                  className="flex items-center gap-1"
                >
                  다음 →
                </Button>
              </div>
            )}
          </div>
        )}
      </Card>

      {view && (
        <FullScreenImage 
          src={view} 
          onClose={() => { setView(null); setViewingItem(null); }} 
          onDelete={viewingItem ? () => {
            tryOnHistory.removeOutput(viewingItem.id);
            refresh();
          } : undefined}
        />
      )}
    </div>
  );
};

export default TryOnHistory;
