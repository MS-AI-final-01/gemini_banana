import React, { useEffect, useMemo, useRef, useState } from 'react';
import { tryOnHistory, TryOnInputHistoryItem, TryOnOutputHistoryItem } from '../../../services/tryon_history.service';
import type { RecommendationItem } from '../../../types';
import { Button, Card } from '../../ui';
import { FullScreenImage } from '../common/FullScreenImage';

interface TryOnHistoryProps {
  onApply?: (payload: { 
    person?: string; 
    top?: string; 
    pants?: string; 
    shoes?: string; 
    topLabel?: string; 
    pantsLabel?: string; 
    shoesLabel?: string; 
    outerLabel?: string;
    topProduct?: RecommendationItem;
    pantsProduct?: RecommendationItem;
    shoesProduct?: RecommendationItem;
    outerProduct?: RecommendationItem;
  }) => void;
}

interface HistoryItemCardProps {
  item: TryOnInputHistoryItem;
  onApply?: (payload: { 
    person?: string; 
    top?: string; 
    pants?: string; 
    shoes?: string; 
    topLabel?: string; 
    pantsLabel?: string; 
    shoesLabel?: string; 
    outerLabel?: string;
    topProduct?: RecommendationItem;
    pantsProduct?: RecommendationItem;
    shoesProduct?: RecommendationItem;
    outerProduct?: RecommendationItem;
  }) => void;
  getHistoryItemImage: (item: TryOnInputHistoryItem) => Promise<string | null>;
}

const HistoryItemCard: React.FC<HistoryItemCardProps> = ({ item, onApply, getHistoryItemImage }) => {
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadImage = async () => {
      setLoading(true);
      try {
        const image = await getHistoryItemImage(item);
        setImageUrl(image);
      } catch (error) {
        console.warn('이미지 로드 실패:', error);
        setImageUrl(null);
      } finally {
        setLoading(false);
      }
    };

    loadImage();
  }, [item, getHistoryItemImage]);

  const hasClothing = item.topLabel || item.pantsLabel || item.shoesLabel || item.outerLabel;

  return (
    <button
      type="button"
      onClick={() => onApply?.({
        person: undefined, 
        top: undefined, 
        pants: undefined, 
        shoes: undefined, 
        topLabel: item.topLabel, 
        pantsLabel: item.pantsLabel, 
        shoesLabel: item.shoesLabel,
        outerLabel: item.outerLabel,
        topProduct: item.topProduct,
        pantsProduct: item.pantsProduct,
        shoesProduct: item.shoesProduct,
        outerProduct: item.outerProduct
      })}
      className="relative w-40 aspect-[4/5] rounded-md overflow-hidden bg-gray-100 ring-1 ring-transparent hover:ring-blue-200 transition"
      title="클릭하면 기록을 적용합니다"
    >
      {loading ? (
        <div className="absolute inset-0 flex items-center justify-center text-gray-400 text-xs">
          로딩...
        </div>
      ) : imageUrl ? (
        <img src={imageUrl} alt="의류 이미지" className="absolute inset-0 w-full h-full object-cover" />
      ) : hasClothing ? (
        <div className="absolute inset-0 flex flex-col items-center justify-center text-gray-600 p-2">
          <span className="text-xs font-medium">의류 조합</span>
          <span className="text-xs text-gray-500 mt-1">
            {[item.topLabel, item.pantsLabel, item.shoesLabel, item.outerLabel].filter(Boolean).join(', ')}
          </span>
        </div>
      ) : (
        <div className="absolute inset-0 flex items-center justify-center text-gray-400 text-xs">-</div>
      )}
    </button>
  );
};

export const TryOnHistory: React.FC<TryOnHistoryProps> = ({ onApply }) => {
  console.log('🔔 TryOnHistory 컴포넌트 렌더링됨');
  const [inputs, setInputs] = useState(tryOnHistory.inputs());
  const [outputs, setOutputs] = useState(tryOnHistory.outputs());
  const [view, setView] = useState<string | null>(null);
  const [viewingItem, setViewingItem] = useState<TryOnOutputHistoryItem | null>(null);
  const [sortMode, setSortMode] = useState<'recent' | 'rank'>('recent');
<<<<<<< HEAD
  
  // 슬라이드 관련 상태
  const [currentSlide, setCurrentSlide] = useState(0);
  const slideRef = useRef<HTMLDivElement>(null);
  const itemsPerSlide = 2; // 한 번에 보여줄 아이템 수
  
  // 상품 데이터는 히스토리에 저장되므로 별도 캐시 불필요
=======
  // 상품 메타데이터는 히스토리에 저장되므로 별도 캐시 불필요
>>>>>>> 8e33230 ([14:15] 한글 인코딩 정상화)

  const refresh = () => {
    const newInputs = tryOnHistory.inputs();
    const newOutputs = tryOnHistory.outputs();
    console.log('🔔 TryOnHistory refresh - 입력:', newInputs.length, '출력:', newOutputs.length);
    setInputs(newInputs);
    setOutputs(newOutputs);
  };

  // 추가 API 호출이 필요 없는 구조 (상품 메타데이터가 히스토리에 포함됨)

<<<<<<< HEAD
  // 히스토리 아이템의 대표 이미지를 가져오는 함수 (실제 선택된 아이템 우선)
  const getHistoryItemImage = async (item: TryOnInputHistoryItem): Promise<string | null> => {
    console.log('🔍 getHistoryItemImage 호출:', {
=======
  // 히스토리 아이템에서 의류 이미지를 가져오는 함수 (업로드한 상품 우선)
  const getHistoryItemImage = async (item: TryOnInputHistoryItem): Promise<string | null> => {
    // 상의 → 하의 → 신발 → 아우터 순으로 우선순위
    const products = [item.topProduct, item.pantsProduct, item.shoesProduct, item.outerProduct].filter(Boolean);
    
    console.log(`히스토리 입력의 상품 메타데이터:`, {
>>>>>>> 8e33230 ([14:15] 한글 인코딩 정상화)
      topProduct: item.topProduct?.title,
      pantsProduct: item.pantsProduct?.title,
      shoesProduct: item.shoesProduct?.title,
      outerProduct: item.outerProduct?.title
    });
    
    // 실제로 선택된 아이템들만 필터링 (라벨이 있는 것들)
    const selectedProducts = [];
    if (item.topLabel && item.topProduct) selectedProducts.push(item.topProduct);
    if (item.pantsLabel && item.pantsProduct) selectedProducts.push(item.pantsProduct);
    if (item.shoesLabel && item.shoesProduct) selectedProducts.push(item.shoesProduct);
    if (item.outerLabel && item.outerProduct) selectedProducts.push(item.outerProduct);
    
    console.log('🔍 선택된 상품들:', selectedProducts.map(p => p.title));
    
    // 선택된 상품 중 첫 번째 이미지 반환
    for (const product of selectedProducts) {
      if (product?.imageUrl) {
        console.log('🔍 이미지 찾음:', product.title, product.imageUrl);
        return product.imageUrl;
      }
    }
    
    console.log('🔍 이미지를 찾지 못함');
    return null;
  };

  useEffect(() => {
    console.log('🔔 TryOnHistory useEffect 실행, 리스너 구독 시작');
    
    // 구독 전에 현재 listeners 수 확인
    console.log('🔔 구독 전 listeners 수:', tryOnHistory.listeners.size);
    
    const unsub = tryOnHistory.subscribe(() => {
      console.log('🔔 TryOnHistory 리스너 호출됨, refresh 실행');
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
              return <HistoryItemCard key={item.id} item={item} onApply={onApply} getHistoryItemImage={getHistoryItemImage} />;
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
