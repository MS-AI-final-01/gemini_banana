import React from 'react';
import type { TryOnInputHistoryItem } from '../../../services/tryon_history.service';
import type { RecommendationItem, UploadedImage } from '../../../types';

interface ImageHistoryCardProps {
  item: TryOnInputHistoryItem;
  onApply?: (payload: { 
    topProduct?: RecommendationItem;
    pantsProduct?: RecommendationItem;
    shoesProduct?: RecommendationItem;
    outerProduct?: RecommendationItem;
  }) => void;
  onImageApply?: (slot: 'top' | 'pants' | 'shoes' | 'outer', image: UploadedImage, label: string) => Promise<void>;
}

export const ImageHistoryCard: React.FC<ImageHistoryCardProps> = ({ item, onApply, onImageApply }) => {
  const getImageUrl = () => {
    if (item.topImageData) return `data:image/jpeg;base64,${item.topImageData}`;
    if (item.pantsImageData) return `data:image/jpeg;base64,${item.pantsImageData}`;
    if (item.shoesImageData) return `data:image/jpeg;base64,${item.shoesImageData}`;
    if (item.outerImageData) return `data:image/jpeg;base64,${item.outerImageData}`;
    return null;
  };

  const imageUrl = getImageUrl();

  return (
    <button
      type="button"
      onClick={async () => {
        const payload: any = {};
        
        // 이미지 적용
        if (item.topLabel && item.topImageData) {
          const image: UploadedImage = {
            previewUrl: `data:image/jpeg;base64,${item.topImageData}`,
            base64: item.topImageData,
            mimeType: 'image/jpeg'
          };
          await onImageApply?.('top', image, item.topLabel);
        }
        if (item.pantsLabel && item.pantsImageData) {
          const image: UploadedImage = {
            previewUrl: `data:image/jpeg;base64,${item.pantsImageData}`,
            base64: item.pantsImageData,
            mimeType: 'image/jpeg'
          };
          await onImageApply?.('pants', image, item.pantsLabel);
        }
        if (item.shoesLabel && item.shoesImageData) {
          const image: UploadedImage = {
            previewUrl: `data:image/jpeg;base64,${item.shoesImageData}`,
            base64: item.shoesImageData,
            mimeType: 'image/jpeg'
          };
          await onImageApply?.('shoes', image, item.shoesLabel);
        }
        if (item.outerLabel && item.outerImageData) {
          const image: UploadedImage = {
            previewUrl: `data:image/jpeg;base64,${item.outerImageData}`,
            base64: item.outerImageData,
            mimeType: 'image/jpeg'
          };
          await onImageApply?.('outer', image, item.outerLabel);
        }
        
        // 상품 적용
        if (item.topProduct) payload.topProduct = item.topProduct;
        if (item.pantsProduct) payload.pantsProduct = item.pantsProduct;
        if (item.shoesProduct) payload.shoesProduct = item.shoesProduct;
        if (item.outerProduct) payload.outerProduct = item.outerProduct;
        
        onApply?.(payload);
      }}
      className="relative w-40 aspect-[4/5] rounded-md overflow-hidden bg-gray-100 ring-1 ring-transparent hover:ring-blue-200 transition"
      title="클릭하면 기록을 적용합니다"
    >
      {imageUrl ? (
        <img src={imageUrl} alt="업로드된 의류 이미지" className="absolute inset-0 w-full h-full object-cover" />
      ) : (
        <div className="absolute inset-0 flex items-center justify-center text-gray-400 text-xs">이미지 없음</div>
      )}
    </button>
  );
};
