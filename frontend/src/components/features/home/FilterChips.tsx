import React, { useState } from 'react';

const chips = [
  '전체',
  '미니멀',
  '스트릿',
  '스포츠',
  '캐주얼',
  '포멀',
  '아웃도어',
  '러닝',
  '여름',
  '겨울',
];

export const FilterChips: React.FC = () => {
  const [active, setActive] = useState(0);
  return (
    <div className="flex flex-wrap gap-2">
      {chips.map((chip, index) => (
        <button
          key={chip}
          onClick={() => setActive(index)}
          className={`text-xs rounded-full px-4 py-1.5 border transition-colors ${
            active === index
              ? 'bg-[#111111] text-white border-[#111111]'
              : 'bg-white text-[var(--text-muted)] border-[var(--divider)] hover:text-[var(--text-strong)]'
          }`}
        >
          {chip}
        </button>
      ))}
    </div>
  );
};

export default FilterChips;

