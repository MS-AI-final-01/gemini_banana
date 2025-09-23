import React from 'react';
import type { TryOnInputHistoryItem } from '../../../services/tryon_history.service';
import type { RecommendationItem } from '../../../types';

interface ProductHistoryCardProps {
  item: TryOnInputHistoryItem;
  onApply?: (payload: { 
    topProduct?: RecommendationItem;
    pantsProduct?: RecommendationItem;
    shoesProduct?: RecommendationItem;
    outerProduct?: RecommendationItem;
  }) => void;
}

export const ProductHistoryCard: React.FC<ProductHistoryCardProps> = ({ item, onApply }) => {
  const getProductImage = () => {
    const products = [item.topProduct && item.topLabel ? item.topProduct : null,
    item.pantsProduct && item.pantsLabel ? item.pantsProduct : null,
    item.shoesProduct && item.shoesLabel ? item.shoesProduct : null,
    item.outerProduct && item.outerLabel ? item.outerProduct : null,].filter(Boolean);
    for (const product of products) {
      if (product?.imageUrl) return product.imageUrl;
    }
    return null;
  };

  const productImage = getProductImage();
  const hasClothing = item.topLabel || item.pantsLabel || item.shoesLabel || item.outerLabel;

  return (
    <button
      type="button"
      onClick={() => {
        const payload: any = {};
        if (item.topLabel) {
          payload.topLabel = item.topLabel;
          payload.topProduct = item.topProduct;
        }
        if (item.pantsLabel) {
          payload.pantsLabel = item.pantsLabel;
          payload.pantsProduct = item.pantsProduct;
        }
        if (item.shoesLabel) {
          payload.shoesLabel = item.shoesLabel;
          payload.shoesProduct = item.shoesProduct;
        }
        if (item.outerLabel) {
          payload.outerLabel = item.outerLabel;
          payload.outerProduct = item.outerProduct;
        }
        onApply?.(payload);
      }}
      className="relative w-40 aspect-[4/5] rounded-md overflow-hidden bg-gray-100 ring-1 ring-transparent hover:ring-blue-200 transition"
      title="클릭하면 기록을 적용합니다"
    >
      {productImage ? (
        <img src={productImage} alt="의류 이미지" className="absolute inset-0 w-full h-full object-cover" />
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
