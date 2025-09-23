import React from 'react';
import { HeartIcon } from '../../icons/HeartIcon';
import { Button } from '../../ui';

interface ClothingItemOverlayProps {
  isVisible: boolean;
  onLike: () => void;
  onBuy: () => void;
  onRemove?: () => void;
  itemTitle: string;
  isLiked: boolean;
}

export const ClothingItemOverlay: React.FC<ClothingItemOverlayProps> = ({
  isVisible,
  onLike,
  onBuy,
  onRemove,
  isLiked
}) => {
  if (!isVisible) return null;

  const handleOverlayClick = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
  };

  return (
    <div
      className="absolute inset-0 bg-black/60 backdrop-blur-sm rounded-lg flex items-center justify-center z-10"
      onClick={handleOverlayClick}
    >
      {/* X 버튼 (선택) */}
      {onRemove && (
        <button
          onClick={(e) => {
            e.stopPropagation();
            onRemove();
          }}
          className="absolute top-2 right-2 p-1 bg-white/70 rounded-full text-gray-600 hover:bg-white hover:text-red-500 transition-all duration-200 z-20"
          aria-label="Remove image"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      )}
      
      <div className="flex flex-col gap-2 p-3">
        <Button
          onClick={(e) => {
            e.stopPropagation();
            onLike();
          }}
          className="w-full font-semibold text-sm py-1.5 bg-white/90 text-gray-700 border border-gray-300 hover:bg-gray-50"
        >
          <HeartIcon className={`h-4 w-4 ${isLiked ? 'text-[#d6001c]' : 'text-gray-500'}`} />
        </Button>
        <Button
          onClick={(e) => {
            e.stopPropagation();
            onBuy();
          }}
          className="w-full bg-blue-600 text-white hover:bg-blue-700 font-semibold text-sm py-1.5"
        >
          구매하기
        </Button>
      </div>
    </div>
  );
};

export default ClothingItemOverlay;
