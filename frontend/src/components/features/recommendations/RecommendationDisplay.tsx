import React, { useState } from 'react';
import { likesService } from '../../../services/likes.service';
import type { CategoryRecommendations, RecommendationItem } from '../../../types';
import { HeartIcon } from '../../icons/HeartIcon';
import { Button, Card, toast, useToast } from '../../ui';

interface RecommendationDisplayProps {
    recommendations: CategoryRecommendations;
    onItemClick?: (item: RecommendationItem) => void;
}

export const RecommendationDisplay: React.FC<RecommendationDisplayProps> = ({
    recommendations,
    onItemClick,
}) => {

    const safeTop = recommendations.top ?? [];
    const safePants = recommendations.pants ?? [];
    const safeOuter = recommendations.outer ?? [];
    const safeShoes = recommendations.shoes ?? [];
    const safeAccessories = recommendations.accessories ?? [];

    // Lightweight inline placeholder (SVG) shown when product image fails to load
    const fallbackImage =
        'data:image/svg+xml;utf8,' +
        encodeURIComponent(
            `<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512" viewBox="0 0 512 512">
                <rect width="100%" height="100%" fill="#f3f4f6"/>
                <g fill="#9ca3af" text-anchor="middle" font-family="Arial, Helvetica, sans-serif">
                    <text x="50%" y="50%" font-size="20" dy=".3em">이미지를 불러올 수 없습니다</text>
                </g>
            </svg>`
        );

    const formatPrice = (price: number) => {
        return new Intl.NumberFormat('ko-KR', { style: 'currency', currency: 'KRW' }).format(price);
    };

    // pagination state and helper for refresh
    const [page] = useState(0);
    const visiblePerCategory = 3;

    const getPagedItems = (items: RecommendationItem[]) => {
        if (!items || items.length <= visiblePerCategory) return items || [];
        const start = (page * visiblePerCategory) % items.length;
        const end = start + visiblePerCategory;
        return end <= items.length ? items.slice(start, end) : [...items.slice(start), ...items.slice(0, end - items.length)];
    };


    const ItemCard: React.FC<{ item: RecommendationItem }> = ({ item }) => {
        const { addToast } = useToast();
        const [liked, setLiked] = useState<boolean>(() => likesService.isLiked(item.id));

        const onToggleLike: React.MouseEventHandler = (e) => {
            e.preventDefault(); e.stopPropagation();
            const nowLiked = likesService.toggle(item);
            setLiked(nowLiked);
            addToast(
                nowLiked
                    ? toast.success('좋아요에 추가', item.title, { duration: 2000 })
                    : toast.info('좋아요에서 제거', item.title, { duration: 1500 })
            );
        };


        return (
            <Card 
                className="cursor-pointer hover:shadow-lg transition-shadow duration-200" 
                onClick={() => onItemClick?.(item)}
                data-card-id={item.id}
            >
                <div className="aspect-square bg-gray-100 rounded-lg mb-3 overflow-hidden relative">
                    {item.imageUrl ? (
                        <img
                            src={item.imageUrl}
                            alt={item.title}
                            className="w-full h-full object-cover"
                            onError={(e) => {
                                const img = e.currentTarget as HTMLImageElement;
                                if (img.src !== fallbackImage) {
                                    img.src = fallbackImage;
                                }
                            }}
                        />
                    ) : (
                        <div className="w-full h-full flex items-center justify-center text-gray-400 text-sm">
                            No Image
                        </div>
                    )}
                    
                    {/* 오버레이 제거 - 유사상품추천에서는 오버레이 없음 */}
                </div>
                <div className="space-y-1">
                    <p className="font-medium text-sm text-gray-900 line-clamp-2">{item.title}</p>
                    <p className="font-semibold text-primary-600">{formatPrice(item.price)}</p>
                    {item.score !== undefined && (
                        <p className="text-xs text-gray-500"> 유사도 {(item.score * 100).toFixed(2)}%</p>
                    )}
                    <div className="pt-2 flex gap-2">
                        <Button size="sm" variant={liked ? 'secondary' : 'outline'} onClick={onToggleLike} aria-pressed={liked}>
                            <span className="inline-flex items-center gap-1">
                                <HeartIcon className={liked ? 'w-4 h-4 text-red-500' : 'w-4 h-4'} />
                                {liked ? '좋아요됨' : '좋아요'}
                            </span>
                        </Button>
                    </div>
                </div>
            </Card>
        );
    };

    const renderCategory = (categoryName: string, items: RecommendationItem[]) => {
        if (items.length === 0) return null;

        const categoryNames: Record<string, string> = {
            outer: '아우터',
            top: '상의',
            pants: '하의',
            shoes: '신발',
            accessories: '액세서리',
        };

        return (
            <div className="mb-8" key={categoryName}>
                <h3 className="text-xl font-semibold text-gray-800 mb-4">
                    {categoryNames[categoryName] || categoryName}
                </h3>
                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                    {getPagedItems(items).map((item) => (
                        <ItemCard key={item.id} item={item} />
                    ))}
                </div>
            </div>
        );
    };

    const hasAnyRecommendations = [safeTop, safePants, safeOuter, safeShoes, safeAccessories]
        .some((items) => items.length > 0);

    if (!hasAnyRecommendations) {
        return (
            <Card className="text-center py-8">
                <h2 className="text-xl font-semibold text-gray-800 mb-2">추천 상품</h2>
                <p className="text-gray-600">추천할 상품이 없습니다.</p>
            </Card>
        );
    }

    return (
        <Card>
            <h2 className="text-2xl font-bold text-gray-800 mb-6">유사 상품 추천</h2>
            <div>
                {renderCategory('top', safeTop)}
                {renderCategory('pants', safePants)}
                {renderCategory('outer', safeOuter)}
                {renderCategory('shoes', safeShoes)}
                {renderCategory('accessories', safeAccessories)}
            </div>
        </Card>
    );
};

export default RecommendationDisplay;
