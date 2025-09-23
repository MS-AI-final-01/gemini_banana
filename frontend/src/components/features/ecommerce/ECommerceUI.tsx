import React, { useEffect, useMemo, useState } from 'react';
import './ECommerceUI.css';
import { apiClient } from '../../../services/api.service';
import { FALLBACK_RECOMMENDATIONS } from '../../../data/fallbackRecommendations';
import { likesService } from '../../../services/likes.service';
import type { RecommendationItem } from '../../../types';
import { HeartIcon } from '../../icons/HeartIcon';
import { Button, toast, useToast } from '../../ui';
import { CategoryRow } from '../home/CategoryRow';
import { FilterChips } from '../home/FilterChips';
import { ProductCardOverlay } from './ProductCardOverlay';
import { StickySidebar } from './StickySidebar';
import { SearchChatWidget } from '../search/SearchChatWidget';

function formatPriceKRW(n: number) {
  return new Intl.NumberFormat('ko-KR', { style: 'currency', currency: 'KRW' }).format(n);
}

type GenderFilter = 'all' | 'male' | 'female';

const useRandomProducts = (limit: number = 24, gender: GenderFilter = 'all') => {
  const [items, setItems] = useState<RecommendationItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchItems = async () => {
    setLoading(true);
    setError(null);
    try {
      const qs = new URLSearchParams({ limit: String(limit) });
      if (gender && gender !== 'all') qs.set('gender', gender);
      const data = await apiClient.get<RecommendationItem[]>(`/api/recommend/random?${qs.toString()}`, { timeout: 45000 });
      if (!Array.isArray(data) || data.length === 0) {
        setItems(FALLBACK_RECOMMENDATIONS.slice(0, limit));
        setError('추천 상품이 비어 있어 기본 목록을 표시합니다.');
      } else {
        setItems(data);
      }
    } catch (e: any) {
      setItems(FALLBACK_RECOMMENDATIONS.slice(0, limit));
      const message = (e?.message || '추천 상품을 불러오는 데 실패했습니다.').toString();
      const lower = message.toLowerCase();
      if (lower.includes('abort') || lower.includes('timeout')) {
        setError('서버 응답이 지연되어 기본 추천 목록을 보여드립니다.');
      } else {
        setError(message);
      }
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchItems();
  }, [gender, limit]);

  return { items, loading, error, refresh: fetchItems };
};

interface ProductCardProps {
  item: RecommendationItem;
  onBuy?: (item: RecommendationItem) => void;
  onVirtualFitting?: (item: RecommendationItem) => void;
}

const ProductCard: React.FC<ProductCardProps> = ({ item, onBuy, onVirtualFitting }) => {
  const { addToast } = useToast();
  const [liked, setLiked] = useState<boolean>(() => likesService.isLiked(item.id));
  const [showOverlay, setShowOverlay] = useState(false);

  const onToggleLike: React.MouseEventHandler = (e) => {
    e.preventDefault();
    e.stopPropagation();
    const nowLiked = likesService.toggle(item);
    setLiked(nowLiked);
    addToast(
      nowLiked
        ? toast.success('좋아요에 추가했어요', item.title, { duration: 2000 })
        : toast.info('좋아요에서 제거했어요', item.title, { duration: 1500 })
    );
  };

  const handleNavigate = () => {
    if (item.productUrl) {
      window.open(item.productUrl, '_blank', 'noopener,noreferrer');
    }
  };

  const handleBuy = () => {
    if (onBuy) {
      onBuy(item);
      return;
    }
    handleNavigate();
  };

  const handleVirtual = () => {
    if (onVirtualFitting) {
      onVirtualFitting(item);
    }
  };

  const discount = item.discountRate ? Math.round(item.discountRate * 100) : item.discountPercentage;


  return (
    <article
      onClick={handleNavigate}
      onMouseEnter={() => setShowOverlay(true)}
      onMouseLeave={() => setShowOverlay(false)}
      className="product-card"
    >
      <div className="product-card__image">
        {item.imageUrl && (
          <img src={item.imageUrl} alt={item.title} />
        )}
        <ProductCardOverlay
          isVisible={showOverlay}
          onBuy={handleBuy}
          onVirtualFitting={handleVirtual}
        />
        <button
          onClick={onToggleLike}
          aria-label="좋아요 토글"
          className={`product-card__like ${liked ? 'is-liked' : ''}`}
        >
          <HeartIcon className="h-4 w-4" />
        </button>
      </div>
      <div className="product-card__meta">
        <p className="product-card__brand">{item.brandName || item.tags?.[0] || 'MUSINSA'}</p>
        <p className="product-card__title">{item.title}</p>
        <div className="product-card__pricing">
          <span className="product-card__price">{formatPriceKRW(item.price)}</span>
          {typeof discount === 'number' && discount > 0 && (
            <span className="product-card__discount">{discount}%</span>
          )}
        </div>
      </div>
    </article>
  );
};

type HomePage = 'home' | 'try-on' | 'likes';

interface HomeProps {
  onNavigate?: (page: HomePage) => void;
}

const resolveCartCategory = (product: RecommendationItem): 'outer' | 'top' | 'pants' | 'shoes' | null => {
  const category = product.category?.toLowerCase();
  if (!category) {
    return null;
  }
  if (category.includes('outer')) {
    return 'outer';
  }
  if (category.includes('top')) {
    return 'top';
  }
  if (category.includes('pant') || category.includes('bottom')) {
    return 'pants';
  }
  if (category.includes('shoe')) {
    return 'shoes';
  }
  return null;
};

interface PromoSlide {
  id: string;
  eyebrow: string;
  title: string;
  description: string;
  image: string;
  ctaLabel: string;
  background: string;
}

const promoSlides: PromoSlide[] = [
  {
    id: 'run-lab',
    eyebrow: 'RUN CLUB',
    title: '러닝 시즌, 새로운 기록을 준비하세요',
    description: '가볍게 달리고 땀 식히기 좋은 기능성 웨어와 액세서리를 만나보세요.',
    image: 'https://images.unsplash.com/photo-1600965962361-9035dbfd1c50?auto=format&fit=crop&w=900&q=80',
    ctaLabel: '가상 피팅 바로가기',
    background: 'radial-gradient(circle at 15% 20%, #4f46e590, transparent 60%), linear-gradient(120deg, #111827 0%, #1e1b4b 60%, #111827 100%)'
  },
  {
    id: 'studio-fit',
    eyebrow: 'STUDIO FIT',
    title: '미니멀 실루엣, 스튜디오 감성룩',
    description: '차분한 톤에 포인트 되는 컬러 매치로 트렌디한 데일리룩 완성.',
    image: 'https://images.unsplash.com/photo-1527718641255-324f8e2d0421?auto=format&fit=crop&w=900&q=80',
    ctaLabel: '추천 상품 둘러보기',
    background: 'radial-gradient(circle at 80% 20%, #f472b63d, transparent 65%), linear-gradient(135deg, #312e81 0%, #4c1d95 55%, #312e81 100%)'
  },
  {
    id: 'street-play',
    eyebrow: 'STREET PLAY',
    title: '스트릿 무드의 레이어드 스타일',
    description: '와이드 팬츠와 루즈한 상의로 여유롭게 연출하는 캐주얼 룩.',
    image: 'https://images.unsplash.com/photo-1508804185872-d7badad00f7d?auto=format&fit=crop&w=900&q=80',
    ctaLabel: '룩 자세히 보기',
    background: 'radial-gradient(circle at 20% 80%, #f9731633, transparent 60%), linear-gradient(135deg, #0f172a 0%, #1e293b 60%, #0f172a 100%)'
  }
];

interface PromoCarouselProps {
  onTryOn?: () => void;
}

const PromoCarousel: React.FC<PromoCarouselProps> = ({ onTryOn }) => {
  const [activeIndex, setActiveIndex] = useState(0);
  const [isPaused, setIsPaused] = useState(false);

  useEffect(() => {
    if (isPaused) {
      return;
    }
    const timer = window.setInterval(() => {
      setActiveIndex((prev) => (prev + 1) % promoSlides.length);
    }, 6000);
    return () => window.clearInterval(timer);
  }, [isPaused]);

  const handleDotClick = (index: number) => {
    setActiveIndex(index);
  };

  return (
    <div
      className="banner-slider"
      onMouseEnter={() => setIsPaused(true)}
      onMouseLeave={() => setIsPaused(false)}
    >
      <div
        className="banner-slider__frame"
        style={{ transform: `translateX(-${activeIndex * 100}%)` }}
      >
        {promoSlides.map((slide) => (
          <div key={slide.id} className="banner-slider__slide" style={{ background: slide.background }}>
            <div className="banner-slider__content">
              <span className="banner-slider__eyebrow">{slide.eyebrow}</span>
              <h3 className="banner-slider__title">{slide.title}</h3>
              <p className="banner-slider__desc">{slide.description}</p>
              <button
                type="button"
                className="banner-slider__cta"
                onClick={() => onTryOn?.()}
              >
                {slide.ctaLabel}
              </button>
            </div>
            <div className="banner-slider__visual">
              <img src={slide.image} alt={slide.title} loading="lazy" />
            </div>
          </div>
        ))}
      </div>
      <div className="banner-slider__dots" role="tablist" aria-label="프로모션 슬라이더">
        {promoSlides.map((slide, index) => (
          <button
            key={slide.id}
            type="button"
            className={`banner-slider__dot ${activeIndex === index ? 'is-active' : ''}`}
            onClick={() => handleDotClick(index)}
            aria-label={`${slide.title} 보기`}
            aria-selected={activeIndex === index}
          />
        ))}
      </div>
    </div>
  );
};

export const ECommerceUI: React.FC<HomeProps> = ({ onNavigate }) => {
  const [gender, setGender] = useState<GenderFilter>('all');
  const { items, loading, error, refresh } = useRandomProducts(24, gender);
  const [gridItems, setGridItems] = useState<RecommendationItem[]>([]);
  const [searchQuery, setSearchQuery] = useState('');

  useEffect(() => { setGridItems(items); }, [items]);
  // gridItems was a memo of items; replaced by state so we can inject semantic search results
  const [selectedItems, setSelectedItems] = useState<{
    outer?: RecommendationItem;
    top?: RecommendationItem;
    pants?: RecommendationItem;
    shoes?: RecommendationItem;
  }>({});

  const handleRemoveItem = (category: 'outer' | 'top' | 'pants' | 'shoes') => {
    setSelectedItems((prev) => ({
      ...prev,
      [category]: undefined,
    }));
  };

  const handleGoToFitting = () => {
    const payload = Object.values(selectedItems).filter(Boolean);
    if (payload.length > 0) {
      try {
        localStorage.setItem('app:pendingVirtualFittingItems', JSON.stringify(payload));
      } catch (storageError) {
        console.warn('virtual fitting queue storage failed', storageError);
      }
    }

    onNavigate?.('try-on');
  };

  const handleClearAll = () => {
    setSelectedItems({});
  };

  // 장바구니(피팅 바) 추가
  const handleAddToCart = (product: RecommendationItem) => {
    const category = resolveCartCategory(product);
    if (!category) {
      return;
    }
    setSelectedItems((prev) => ({
      ...prev,
      [category]: product,
    }));
    
    console.log('🛒 상품 클릭:', { product, category });
  };

  // 바로 가상피팅으로 이동 (추천 카드)
  const handleDirectFitting = (product: RecommendationItem) => {
    console.log('🚀 가상피팅으로 이동:', product.title);
    try {
      const itemWithTimestamp = {
        ...product,
        timestamp: Date.now()
      };
      localStorage.setItem('app:pendingVirtualFittingItem', JSON.stringify(itemWithTimestamp));
      onNavigate?.('try-on');
    } catch (error) {
      console.warn('가상피팅 이동 저장 실패', error);
    }
  };

  // 상단 프로모션(헤드라인/배너/카테고리) 노출 플래그
  const showTopPromos = false;
  // TopBar 검색창과 연동: semantic-search 이벤트 수신 시 검색 실행
  React.useEffect(() => {
    const handler = async (ev: Event) => {
      const anyEv = ev as any;
      const q = (anyEv?.detail?.q || '').toString();
      const limit = Number(anyEv?.detail?.limit || 24);
      if (!q) return;
      try {
        const qs = new URLSearchParams({ q, limit: String(limit) }).toString();
        const data = await apiClient.get<RecommendationItem[]>(`/api/search/semantic?${qs}`);
        setGridItems(data);
        setSearchQuery(q);
      } catch (err) {
        console.warn('semantic search failed (from TopBar)', err);
      }
    };
    window.addEventListener('semantic-search' as any, handler);
    return () => window.removeEventListener('semantic-search' as any, handler);
  }, []);

  const showFilterChips = false;

  return (
    <div className="main-wrap">
      <div className="main-container">
        {/* top promos removed */}

        <section className="filter-panel" aria-label="필터">
          {showFilterChips && (
            <div className="filter-panel__chips">
              <FilterChips />
            </div>
          )}
          <div className="filter-panel__refresh">
            <Button onClick={refresh} size="sm" variant="outline" loading={loading}>
              추천 다시 받기
            </Button>
          </div>
        </section>

        <section className="product-section" aria-label="추천 상품">
          {/* 좌측 세로 젠더 필터 버튼 (데스크톱에서만 노출) */}
          <div className="hidden md:flex fixed left-4 top-1/2 -translate-y-1/2 z-30">
            <div className="flex flex-col gap-2 rounded-full border border-[var(--divider)] bg-white/90 p-1 shadow-lg backdrop-blur supports-[backdrop-filter]:bg-white/70">
              {([
                {key: 'all', label: '전체'},
                {key: 'male', label: '남성'},
                {key: 'female', label: '여성'},
              ] as {key: GenderFilter; label: string}[]).map(({key, label}) => {
                const active = gender === key;
                return (
                  <button
                    key={key}
                    type="button"
                    aria-pressed={active}
                    onClick={() => setGender(key)}
                    className={[
                      'px-4 py-2 rounded-full text-sm font-medium transition-all duration-150 text-left',
                      active ? 'bg-black text-white shadow-sm' : 'text-[var(--text-strong)] hover:bg-gray-100',
                    ].join(' ')}
                    title={`${label} 상품만 보기`}
                  >
                    {label}
                  </button>
                );
              })}
            </div>
          </div>

          {/* 검색 입력은 TopBar로 이동. TopBar에서 'semantic-search' 이벤트를 발생시킵니다. */}
          <div style={{display:'none'}} />

          <div className="section-title">
            <h2 className="section-title__heading">오늘의 베스트 선택</h2>
          </div>
          {error && (
            <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-[#d6001c]">
              {error}
            </div>
          )}
          <div className="product-grid">
            {gridItems.map((item) => (
              <ProductCard
                key={item.id}
                item={item}
                onBuy={handleAddToCart}
                onVirtualFitting={handleDirectFitting}
              />
            ))}
          </div>
          {loading && (
            <div className="mt-6 text-center text-sm text-[var(--text-muted)]">
              추천 상품을 불러오는 중입니다...
            </div>
          )}
        </section>
      </div>

      <StickySidebar
        selectedItems={selectedItems}
        onRemoveItem={handleRemoveItem}
        onGoToFitting={handleGoToFitting}
        onClearAll={handleClearAll}
      />
      {/* Floating chatbot widget */}
      <SearchChatWidget
        onApplyResults={(items, q) => {
          if (q) setSearchQuery(q);
          setGridItems(items);
        }}
      />
    </div>
  );
};

export default ECommerceUI;
