import React from 'react';

type CategoryItem = { id: string; label: string };

// 홈 더미 카테고리 버튼 (클릭 테스트용)
const CATEGORIES: CategoryItem[] = [
  { id: 'top', label: '상의' },
  { id: 'pants', label: '하의' },
  { id: 'outer', label: '아우터' },
  { id: 'shoes', label: '신발' },
  { id: 'bag', label: '가방' },
  { id: 'acc', label: '액세서리' },
  { id: 'knit', label: '니트/스웨터' },
  { id: 'skirt', label: '스커트' },
];

export const CategoryRow: React.FC<{ onSelectCategory?: (id: string) => void }>
  = ({ onSelectCategory }) => {
  const [active, setActive] = React.useState<string | null>(null);
  const handleClick = (id: string) => {
    setActive(id);
    onSelectCategory?.(id);
    // 임시: 클릭 검증 로그
    try { console.log('[CategoryRow] clicked:', id); } catch {}
  };

  return (
    <div className="w-full overflow-x-auto">
      <div className="flex items-center gap-6 py-4">
        {CATEGORIES.map((item) => (
          <div key={item.id} className="w-20 flex-shrink-0">
            <button
              type="button"
              onClick={() => handleClick(item.id)}
              className={`flex h-20 w-20 items-center justify-center rounded-full border text-xs font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-blue-400 ${
                active === item.id
                  ? 'bg-[#111111] text-white border-[#111111]'
                  : 'bg-[var(--surface-bg)] text-[var(--text-base)] border-[var(--divider)] hover:border-[#111111]'}`}
              aria-pressed={active === item.id}
            >
              {item.label}
            </button>
            <p className="mt-2 text-center text-xs text-[var(--text-muted)] truncate" title={item.label}>{item.label}</p>
          </div>
        ))}
      </div>
    </div>
  );
};

export default CategoryRow;

