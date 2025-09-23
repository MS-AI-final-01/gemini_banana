import React from 'react';
import { Button } from '../../ui';

interface ProductCardOverlayProps {
  isVisible: boolean;
  onBuy: () => void;
  onVirtualFitting: () => void;
}

export const ProductCardOverlay: React.FC<ProductCardOverlayProps> = ({
  isVisible,
  onBuy,
  onVirtualFitting,
}) => {
  if (!isVisible) return null;

  const handleOverlayClick = (e: React.MouseEvent) => {
    // 오버레이 내부 클릭 시 카드 네비게이션 방지
    e.stopPropagation();
  };

  return (
    <div
      className="absolute inset-x-0 top-1/4 bottom-1/4 z-10 flex items-center justify-center"
      onClick={handleOverlayClick}
    >
      <div className="rounded-xl bg-white/90 shadow-lg ring-1 ring-black/5 backdrop-blur px-5 py-4 flex flex-col gap-2 min-w-[180px]">
        <Button
          variant="primary"
          onClick={(e) => { e.stopPropagation(); onVirtualFitting(); }}
          className="w-full font-semibold"
        >
          입어보기
        </Button>
        <Button
          variant="outline"
          onClick={(e) => { console.log('사이드바에 담기 클릭'); e.stopPropagation(); onBuy(); }}
          className="w-full font-semibold"
        >
          사이드바에 담기
        </Button>
      </div>
    </div>
  );
};

export default ProductCardOverlay;

